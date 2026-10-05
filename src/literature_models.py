"""
literature_models.py -- four published AMC architectures, for a like-for-like
comparison with ICRNNA on RML2016.10a.

Ported from the Keras code of the AMR-Benchmark repository
(github.com/Richardzhangxx/AMR-Benchmark, folder RML201610a), which
accompanies Zhang, Luo, Xu, Luo & Zheng, "Deep learning based automatic
modulation recognition: models, datasets, and challenges", Digital Signal
Processing 129 (2022) 103650. Layer for layer, sizes and activations as there;
the benchmark's own choices are kept where they differ from the original
papers, and noted.

    VTCNN2    O'Shea, Corgan & Clancy, "Convolutional radio modulation
              recognition networks", EANN 2016. (CNN1 in the benchmark.)
    LSTM2     Rajendran et al., "Deep learning models for wireless signal
              classification with distributed low-cost spectrum sensors",
              IEEE TCCN 2018. Amplitude and phase in, not I/Q.
    MCLDNN    Xu et al., "A spatiotemporal multi-channel learning framework
              for automatic modulation recognition", IEEE WCL 2020. I/Q, I and
              Q as three inputs; CNN then LSTM.
    PETCGDNN  Zhang et al., "An efficient deep learning model for automatic
              modulation recognition based on parameter estimation and
              transformation", IEEE Commun. Lett. 2021. Estimates a phase,
              rotates the frame by it, then CNN then GRU.

Every model takes the (B, 2, 128) I/Q frame this repository uses and builds
any other input it needs (I alone, Q alone, amplitude/phase) inside forward(),
so cnn.train_model and cnn.predict work unchanged and every model sees the
same frames, split and seeds. Outputs are logits; the softmax is in the loss.

Keras 'same' padding with an even kernel pads one more on the right; PyTorch's
padding='same' does the same, so the shapes match the original.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VTCNN2(nn.Module):
    """Conv(50, 1x8, same) -> Conv(50, 2x8, valid) -> Dense 256 -> Dense n; dropout 0.5."""

    def __init__(self, n_classes: int, length: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, (1, 8), padding="same")
        self.conv2 = nn.Conv2d(50, 50, (2, 8))
        self.fc1 = nn.Linear(50 * (length - 7), 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):                       # (B, 2, L)
        x = x.unsqueeze(1)                      # (B, 1, 2, L)
        x = self.drop(F.relu(self.conv1(x)))
        x = self.drop(F.relu(self.conv2(x)))    # (B, 50, 1, L-7)
        # Keras flattens channels-last: (1, L-7, 50)
        x = x.permute(0, 2, 3, 1).flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


class LSTM2(nn.Module):
    """Two LSTM layers of 128 over (amplitude, phase/pi); dense to n classes.

    As the benchmark's rmldataset2016.to_amp_phase / norm_pad_zeros: amplitude
    |I + jQ| scaled to unit L2 norm per frame, phase atan2(Q, I) / pi.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(2, 128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):                       # (B, 2, L)
        amp = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        amp = amp / (amp.norm(dim=1, keepdim=True) + 1e-12)
        ph = torch.atan2(x[:, 1], x[:, 0]) / torch.pi
        h, _ = self.lstm(torch.stack([amp, ph], dim=-1))   # (B, L, 128)
        return self.fc(h[:, -1])


class MCLDNN(nn.Module):
    """Three inputs (I/Q as 2xL, I, Q) -> convs -> 2 LSTM(128) -> 2 Dense(128, selu)."""

    def __init__(self, n_classes: int, length: int = 128):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 50, (2, 8), padding="same")
        self.conv1_2 = nn.Conv1d(1, 50, 8)      # causal: padded on the left
        self.conv1_3 = nn.Conv1d(1, 50, 8)
        self.conv2 = nn.Conv2d(50, 50, (1, 8), padding="same")
        self.conv4 = nn.Conv2d(100, 100, (2, 5))
        self.lstm = nn.LSTM(100, 128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, n_classes)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):                       # (B, 2, L)
        x1 = F.relu(self.conv1_1(x.unsqueeze(1)))                    # (B, 50, 2, L)
        i = F.pad(x[:, :1], (7, 0))
        q = F.pad(x[:, 1:], (7, 0))
        x2 = F.relu(self.conv1_2(i))                                 # (B, 50, L)
        x3 = F.relu(self.conv1_3(q))
        x23 = torch.stack([x2, x3], dim=2)                           # (B, 50, 2, L)
        x23 = F.relu(self.conv2(x23))
        h = torch.cat([x1, x23], dim=1)                              # (B, 100, 2, L)
        h = F.relu(self.conv4(h)).squeeze(2).permute(0, 2, 1)        # (B, L-4, 100)
        h, _ = self.lstm(h)
        h = self.drop(F.selu(self.fc1(h[:, -1])))
        h = self.drop(F.selu(self.fc2(h)))
        return self.out(h)


class PETCGDNN(nn.Module):
    """Estimate one phase from the whole frame, rotate by it, Conv x2, GRU(128)."""

    def __init__(self, n_classes: int, length: int = 128):
        super().__init__()
        self.phase = nn.Linear(2 * length, 1)
        self.conv1 = nn.Conv2d(1, 75, (8, 2))
        self.conv2 = nn.Conv2d(75, 25, (5, 1))
        self.gru = nn.GRU(25, 128, batch_first=True)
        self.out = nn.Linear(128, n_classes)

    def forward(self, x):                       # (B, 2, L)
        i, q = x[:, 0], x[:, 1]
        theta = self.phase(x.permute(0, 2, 1).flatten(1))           # Keras flattens (L, 2)
        c, s = torch.cos(theta), torch.sin(theta)
        y1 = i * c + q * s
        y2 = q * c - i * s
        h = torch.stack([y1, y2], dim=-1).unsqueeze(1)               # (B, 1, L, 2)
        h = F.relu(self.conv1(h))                                    # (B, 75, L-7, 1)
        h = F.relu(self.conv2(h))                                    # (B, 25, L-11, 1)
        h = h.squeeze(3).permute(0, 2, 1)                            # (B, L-11, 25)
        _, hn = self.gru(h)
        return self.out(hn[-1])


LITERATURE = {"VTCNN2": VTCNN2, "LSTM2": LSTM2, "MCLDNN": MCLDNN, "PETCGDNN": PETCGDNN}


if __name__ == "__main__":
    import model_zoo
    x = torch.randn(4, 2, 128)
    zoo = dict(LITERATURE, ICRNNA=model_zoo.ICRNNA)
    for name, cls in zoo.items():
        m = cls(11)
        p = sum(t.numel() for t in m.parameters())
        print(f"{name:<10} {p:>10,} params  out {tuple(m(x).shape)}")
