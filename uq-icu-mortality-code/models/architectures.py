"""
architectures.py -- network families, the loss and the Monte Carlo dropout passes

Paper:    Methods (model families GRU, LSTM, Transformer; Monte Carlo dropout by sampling the
          trained models with every dropout layer that was active in training switched on)
Origin:   src/lukas_methodik_nachbau.py (library module without SKRIPT_STAND; header
          "Nachbau der Methodik aus Kapitel 4") for one_hot_gcs, batches, RekurrentesModell,
          PositionsKodierung, baue_modell, the loss, _dropout_an and _det_vorhersage;
          src/rerun_pipeline.py v5.7 (lines 492-565) for EncoderSchicht and TransformerNatural,
          the Transformer that is actually trained (deviation A2 of the original protocol);
          the pass loop mc_passes follows mc_det_dropout.py / mc_base_stage.py of the review
          tooling (DECISIONS.md D7 / D9, 2026-10-09 / 2026-10-11), which replaced the separately
          trained MC-dropout twins with a heteroscedastic head of earlier drafts
Reads:    nothing
Writes:   nothing
Run:      library module; imported by common.py (model factory), uncertainty.py, mc_dropout.py

Conventions kept from the original:
  * the models return LOGITS (one per patient);
  * input_size is 25 (12 standardised continuous variables + 13 one-hot GCS columns);
  * Monte Carlo dropout uses T = config.MC_PASSES passes of the trained model with every dropout
    layer that was active in training in train mode (config.MC_MASKS = "full": the nn.Dropout
    modules and the inter-layer dropout of nn.GRU / nn.LSTM); the masks are drawn from torch's
    global RNG, which the caller seeds with config.MC_SEED immediately before the passes of a cohort.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SEED = 42            # module seed of the original (also the TPE sampler seed; equals config.MC_SEED)
T_MC = config.MC_PASSES
GCS_INDEX = 12       # column index of gcs_total among the 13 variables
GCS_MIN, GCS_MAX = 3, 15
GERAET = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import-time seeding as in the original module. Every training run re-seeds with
# common.seed_alles() before the first stochastic object, so this has no effect on results.
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================================
# 1) DATA
# ============================================================================

def one_hot_gcs(X, mittelwert, std):
    """13 variables -> 25 features: gcs_total (index 12) is one-hot encoded.
    gcs_total arrives standardised; it is de-standardised, rounded, clipped to
    [3, 15] and encoded into 13 columns. The other 12 (standardised) continuous
    columns are left unchanged."""
    X = np.asarray(X, dtype=np.float32)
    stetig = np.delete(X, GCS_INDEX, axis=2)                      # (N,48,12)
    gcs_std = X[:, :, GCS_INDEX]
    gcs = gcs_std * std[GCS_INDEX] + mittelwert[GCS_INDEX]        # de-standardise
    gcs = np.clip(np.round(gcs), GCS_MIN, GCS_MAX).astype(int)
    stufen = GCS_MAX - GCS_MIN + 1                                # 13
    onehot = np.zeros((*gcs.shape, stufen), dtype=np.float32)
    idx = gcs - GCS_MIN
    np.put_along_axis(onehot, idx[..., None], 1.0, axis=2)
    return np.concatenate([stetig, onehot], axis=2)              # (N,48,25)


def batches(X, y, batch_size, shuffle=True):
    """Mini-batches; the shuffled permutation comes from torch's global RNG."""
    n = len(X)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield X[j], y[j]


# ============================================================================
# 2) MODELS
# ============================================================================

class RekurrentesModell(nn.Module):
    """LSTM/GRU with h0 = 0 (PyTorch default). Last hidden state -> linear head.
    Dropout at the input, at the output and between the recurrent layers (the latter
    inside nn.GRU / nn.LSTM, active only when num_layers > 1); all three are sampled
    by the Monte Carlo passes (mc_passes)."""

    def __init__(self, input_size, hidden_size, num_layers, dropout, zelle="LSTM"):
        super().__init__()
        zellen = {"LSTM": nn.LSTM, "GRU": nn.GRU}
        rec_dropout = dropout if num_layers > 1 else 0.0   # recurrent (inter-layer) dropout
        self.rnn = zellen[zelle](input_size, hidden_size, num_layers,
                                 batch_first=True, dropout=rec_dropout)
        self.drop_ein = nn.Dropout(dropout)                # input dropout
        self.drop_aus = nn.Dropout(dropout)                # output dropout
        self.kopf = nn.Linear(hidden_size, 1)              # logit

    def forward(self, x):
        x = self.drop_ein(x)
        ausgabe, zustand = self.rnn(x)                     # h0 implicitly 0
        h = zustand[0] if isinstance(zustand, tuple) else zustand  # LSTM -> (h, c)
        h_final = self.drop_aus(h[-1])                     # last layer
        return self.kopf(h_final).squeeze(-1)


class PositionsKodierung(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=48):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderSchicht(nn.Module):
    """Post-LN encoder layer (Vaswani) with EXPLICIT Q/K/V/O linears instead of
    nn.MultiheadAttention (deviation A2 of the original protocol: the layer
    definition was shared with the sigmaReparam arm, which is not part of this
    package; the layer itself is unchanged)."""

    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model not divisible by nhead")
        self.nhead, self.dk = nhead, d_model // nhead
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, dim_ff)
        self.ff2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_attn = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop_ff = nn.Dropout(dropout)

    def _mha(self, x):
        B, T, D = x.shape
        form = lambda z: z.view(B, T, self.nhead, self.dk).transpose(1, 2)
        q, k, v = form(self.wq(x)), form(self.wk(x)), form(self.wv(x))
        a = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dk), dim=-1)
        z = (self.drop_attn(a) @ v).transpose(1, 2).reshape(B, T, D)
        return self.wo(z)

    def forward(self, x):
        x = self.norm1(x + self.drop1(self._mha(x)))
        x = self.norm2(x + self.drop2(self.ff2(self.drop_ff(
            torch.relu(self.ff1(x))))))
        return x


class TransformerNatural(nn.Module):
    """Encoder Transformer of the family T1_det: norm_ein -> projection
    * sqrt(d_model) -> positional encoding -> layers (post-LN, ReLU) -> norm_aus
    -> mean over time -> dropout -> head. Every dropout (four per encoder layer and
    drop_kopf) is an nn.Dropout module and is sampled by the Monte Carlo passes."""

    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward,
                 dropout):
        super().__init__()
        self.d_model = d_model
        self.norm_ein = nn.LayerNorm(input_size)
        self.projektion = nn.Linear(input_size, d_model)
        self.posenc = PositionsKodierung(d_model)
        self.schichten = nn.ModuleList([
            EncoderSchicht(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)])
        self.norm_aus = nn.LayerNorm(d_model)
        self.drop_kopf = nn.Dropout(dropout)
        self.kopf = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.norm_ein(x)
        x = self.projektion(x) * math.sqrt(self.d_model)
        x = self.posenc(x)
        for s in self.schichten:
            x = s(x)
        x = self.norm_aus(x)
        h = self.drop_kopf(x.mean(dim=1))
        return self.kopf(h).squeeze(-1)


def baue_modell(hp):
    """Factory for the recurrent models. hp carries 'arch' (LSTM | GRU) and the
    hyperparameters; the Transformer is built directly in common.baue_modell_natural."""
    arch = hp["arch"]
    if arch in ("LSTM", "GRU"):
        m = RekurrentesModell(hp["input_size"], hp["hidden_size"], hp["num_layers"],
                              hp["dropout"], zelle=arch)
    else:
        raise ValueError(arch)
    return m.to(GERAET)


# ============================================================================
# 3) LOSS  (BCE with logits)
# ============================================================================

def bce_loss(logit, y):
    return F.binary_cross_entropy_with_logits(logit, y)


def verlust_fabrik(task):
    """Loss for the in-hospital-mortality task ('ihm'); the LOS losses and the
    heteroscedastic loss of the original module are not part of the package."""
    if task != "ihm":
        raise ValueError(f"task {task!r} not supported by the package (only 'ihm').")
    return bce_loss


# ============================================================================
# 4) MONTE CARLO DROPOUT PASSES of a trained model, and the deterministic forward pass
# ============================================================================

MC_MASKS = config.MC_MASKS   # "full" (the published convention) or "module"
MC_EPS = 1e-12               # clipping of the probabilities inside the binary entropy


def _dropout_an(modell, masks=MC_MASKS):
    """Switch the dropout that was active in training to train mode for the Monte
    Carlo passes; everything else stays in eval mode. "module": the nn.Dropout
    modules only; "full": additionally the recurrent encoder nn.GRU / nn.LSTM, whose
    inter-layer dropout is applied inside the RNN kernel and is active only in train
    mode (for the Transformer "full" and "module" coincide: it has no recurrent encoder)."""
    modell.eval()
    for m in modell.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    if masks == "full" and hasattr(modell, "rnn"):
        modell.rnn.train()


@torch.no_grad()
def mc_passes(modell, X, T=T_MC, batch_size=512, masks=MC_MASKS):
    """T stochastic forward passes of a trained model with its dropout switched on.
    Returns the logits as a (T, N) float64 array. The dropout masks come from torch's
    global RNG: seed it with config.MC_SEED immediately before calling, once per
    cohort, as the original run did. Batch 512, no shuffling, the model is returned
    to eval mode -- NO deviation here, otherwise the passes of the reported run are
    not reproduced."""
    _dropout_an(modell, masks)
    out = []
    for _ in range(T):
        ms = []
        for xb, _yb in batches(X, torch.zeros(len(X)), batch_size, shuffle=False):
            ms.append(modell(xb.to(GERAET)).detach().cpu())
        out.append(torch.cat(ms))
    modell.eval()
    return torch.stack(out).double().numpy()


def binaere_entropie_mc(p, eps=MC_EPS):
    """Binary entropy in nats of probabilities clipped to [eps, 1 - eps]."""
    p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def mc_signale(M):
    """The Monte Carlo measures from the (T, N) logits of mc_passes, on the probability
    scale as for the deep ensembles: p_t = sigmoid(logit_t); p_mean; U_total =
    H(p_mean) (predictive entropy); U_alea = mean_t H(p_t) (aleatoric part);
    MI = U_total - U_alea (mutual information, epistemic part); and the logit variance
    across the passes (ddof 0; the measure 'mc_epistemisch' of the result files).
    Checks the decomposition identity and the non-negativity of the MI."""
    P = 1.0 / (1.0 + np.exp(-M))
    p_mean = P.mean(0)
    u_total = binaere_entropie_mc(p_mean)
    u_alea = binaere_entropie_mc(P).mean(0)
    mi = u_total - u_alea
    if np.max(np.abs(u_total - (u_alea + mi))) > 1e-9 or mi.min() < -1e-9:
        raise SystemExit("ABORT (fail-fast): Monte Carlo decomposition violated "
                         f"(min MI {mi.min():.2e}).")
    return {"p_mean": p_mean, "u_total": u_total, "u_alea": u_alea, "mi": mi,
            "logit_var": M.var(0)}


@torch.no_grad()
def _det_vorhersage(modell, X, batch_size=512):
    """One deterministic pass (eval mode); returns the logits on the CPU."""
    modell.eval()
    out = []
    for xb, _ in batches(X, torch.zeros(len(X)), batch_size, shuffle=False):
        out.append(modell(xb.to(GERAET)).cpu())
    return torch.cat(out)
