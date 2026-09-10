"""
model_zoo.py -- architectures with genuinely different inductive biases.

The sink / same-family finding was measured on IQNet, a 1-D CNN. The honest
caveat was that it might be a property of that model class rather than of
modulation. To test that, the finding has to be reproduced on architectures
whose bias is *not* convolutional locality:

    ResNet1D    still convolutional, but residual and deeper -- a control that
                changes depth and gradient flow while keeping the CNN family
    GRUNet      recurrent: processes the IQ stream as a sequence, sequential
                bias, no fixed receptive field
    TransformerNet  attention over patches: no locality prior at all

Each maps (B, 2, 1024) -> (B, n_classes) and exposes the same interface as
cnn.IQNet, so cnn.train_model / cnn.predict work unchanged. Sizes are kept
modest so a leave-one-out sweep is affordable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ==========================================================================
# 1-D ResNet
# ==========================================================================


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 7, stride=stride, padding=3)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 7, padding=3)
        self.bn2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Sequential() if stride == 1 and cin == cout
                     else nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride),
                                        nn.BatchNorm1d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, n_classes: int, width: int = 48):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, width, 7, padding=3), nn.BatchNorm1d(width),
            nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            ResBlock1D(width, width, 2),
            ResBlock1D(width, width * 2, 2),
            ResBlock1D(width * 2, width * 2, 2),
            ResBlock1D(width * 2, width * 4, 2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(width * 4, n_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


# ==========================================================================
# Recurrent (GRU)
# ==========================================================================


class GRUNet(nn.Module):
    """
    A short conv front-end downsamples the 1024-sample stream, then a
    bidirectional GRU reads it as a sequence. The recurrence gives a sequential
    inductive bias with no fixed receptive field, unlike a CNN.
    """

    def __init__(self, n_classes: int, hidden: int = 96):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(2, 64, 7, stride=4, padding=3), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 7, stride=4, padding=3), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )  # 1024 -> 64 timesteps
        self.gru = nn.GRU(64, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(nn.Dropout(0.3),
                                  nn.Linear(hidden * 2, n_classes))

    def forward(self, x):
        z = self.frontend(x).transpose(1, 2)   # (B, T, C)
        out, _ = self.gru(z)
        return self.head(out.mean(dim=1))       # mean over time


# ==========================================================================
# Transformer
# ==========================================================================


class TransformerNet(nn.Module):
    """
    The IQ stream is cut into patches, each linearly embedded, and processed by
    a small transformer encoder. Attention has no locality prior -- any patch
    can attend to any other -- which is the bias furthest from a CNN.
    """

    def __init__(self, n_classes: int, dim: int = 128, patch: int = 16,
                 depth: int = 3, heads: int = 4, n_patches: int = 64):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(2 * patch, dim)
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout=0.1, batch_first=True,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.3),
                                  nn.Linear(dim, n_classes))

    def forward(self, x):
        b, _, n = x.shape
        p = x.reshape(b, 2, n // self.patch, self.patch)
        p = p.permute(0, 2, 1, 3).reshape(b, n // self.patch, 2 * self.patch)
        z = self.proj(p)
        cls = self.cls.expand(b, -1, -1)
        z = torch.cat([cls, z], dim=1) + self.pos[:, : z.shape[1] + 1]
        z = self.encoder(z)
        return self.head(z[:, 0])


ARCHITECTURES = {
    "IQNet (1D CNN)": None,     # filled from cnn.IQNet at call time
    "ResNet1D": ResNet1D,
    "GRU": GRUNet,
    "Transformer": TransformerNet,
}


if __name__ == "__main__":
    x = torch.randn(8, 2, 1024)
    import cnn
    zoo = {"IQNet": cnn.IQNet, "ResNet1D": ResNet1D, "GRU": GRUNet,
           "Transformer": TransformerNet}
    print(f"{'model':<14} {'params':>12} {'out shape':>12}")
    for name, cls in zoo.items():
        m = cls(24)
        p = sum(v.numel() for v in m.parameters())
        y = m(x)
        print(f"{name:<14} {p:>12,} {str(tuple(y.shape)):>12}")


# ==========================================================================
# ICRNNA -- an outside reproduction, brought in as a cross-check
# ==========================================================================
#
# Provenance, stated plainly because it matters for what this model can be
# used to claim. This is transcribed from a peer's `train_amc_standalone.py`
# / `train_amc2018_standalone.py`, which is itself a reproduction of
# El-Haryqy 2025 (ICRNNA: conv front-end + BiLSTM + additive attention) with
# three additions its author labels rather than hides: dropout after each
# conv block, BatchNorm after the BiLSTM, and BatchNorm inside the classifier.
#
# So it is a reproduction of a reproduction. It does not make the project's
# results "published-architecture" results. What it does give is an
# independent implementation with a number attached -- ~63.0% on RadioML
# 2016.10a (11 classes, 128 samples) and ~44% on 2018.01A under a 128-sample
# crop -- so our data pipeline and training loop can be checked against
# someone else's rather than only against themselves.
#
# Note on input length: every layer here is length-agnostic (conv, pool, GRU
# over time, attention pooling over time), so the same module takes 128- and
# 1024-sample frames with an identical parameter count. At 1024 the BiLSTM
# runs over 256 timesteps instead of 32, which is where the cost lives:
# recurrence does not parallelise over time.


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention pooling over the time axis."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, x):                      # (B, T, H)
        w = torch.softmax(self.v(torch.tanh(self.W(x))), dim=1)
        return (x * w).sum(dim=1)              # (B, H)


class BiLSTMWithBN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.bn = nn.BatchNorm1d(hidden_size * 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)


class ICRNNA(nn.Module):
    """
    Convolutional front-end -> bidirectional LSTM -> additive attention.

    The regularisation is the interesting difference from IQNet: dropout after
    every conv block, inside the LSTM, and at 0.5 in the classifier, against
    IQNet's single 0.3 before the output layer. If the overfitting measured on
    IQNet is a regularisation problem rather than a data-volume or
    LR-schedule problem, it should be visibly weaker here.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 64, 5, padding=2), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.MaxPool1d(2), nn.Dropout(0.3))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.MaxPool1d(2), nn.Dropout(0.3))
        self.bilstm = BiLSTMWithBN(128, 128, num_layers=2, dropout=0.3)
        self.attention = AdditiveAttention(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(128, n_classes))

    def forward(self, x):                      # (B, 2, L)
        x = self.conv2(self.conv1(x))          # (B, 128, L/4)
        x = x.permute(0, 2, 1)                 # (B, L/4, 128)
        x = self.bilstm(x)                     # (B, L/4, 256)
        return self.classifier(self.attention(x))


ARCHITECTURES["ICRNNA"] = ICRNNA
