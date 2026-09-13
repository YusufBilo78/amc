"""
cnn.py -- Week 3-4: two CNN variants, and the experiment that separates them.

Week 1 produced a falsifiable prediction from looking at figures/02:

    BPSK, QPSK, 8PSK, PAM4, 16QAM and 64QAM have near-identical spectrograms,
    because identical pulse shaping gives identical occupied bandwidth. So a
    CNN fed spectrograms should fail on exactly those classes, while a CNN fed
    raw IQ should not.

This file trains both and checks. That is a better experiment than picking
whichever representation happens to score higher.

Run:
    python cnn.py --input iq
    python cnn.py --input spectrogram
    python cnn.py --input iq --data radioml
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

# Line-buffer stdout so progress is visible immediately even when the output is
# redirected to a file or piped. Python block-buffers non-tty output by default,
# which makes a long training run look frozen until it finishes.
sys.stdout.reconfigure(line_buffering=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

import modem

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_LEN = 1024


# ==========================================================================
# Data
# ==========================================================================


def synthetic_dataset(
    frames_per_cell: int = 500,
    snrs: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Frames from modem.py. Returns (X, y, z, class_names), X as (n, 2, 1024)."""
    snrs = np.arange(-20, 21, 4) if snrs is None else snrs
    rng = np.random.default_rng(seed)

    X, y, z = [], [], []
    t0 = time.time()
    for i, name in enumerate(modem.MODULATIONS):
        for snr in snrs:
            frames = np.stack(
                [modem.generate(name, FRAME_LEN, snr_db=float(snr), rng=rng)
                 for _ in range(frames_per_cell)]
            )
            X.append(np.stack([frames.real, frames.imag], axis=1).astype(np.float32))
            y.append(np.full(frames_per_cell, i, dtype=np.int64))
            z.append(np.full(frames_per_cell, snr, dtype=np.int16))
        print(f"  {name:<8} ({time.time() - t0:5.1f}s)")

    return (np.concatenate(X), np.concatenate(y), np.concatenate(z),
            modem.MODULATIONS)


def radioml_dataset(frames_per_cell: int = 512, seed: int = 0):
    """Same interface, backed by the real dataset."""
    import radioml

    with radioml.RadioML() as ds:
        data = ds.load(frames_per_cell=frames_per_cell, test_fraction=0.0, seed=seed)
    X = np.transpose(data["X_train"], (0, 2, 1)).astype(np.float32)  # -> (n, 2, 1024)
    return X, data["y_train"].astype(np.int64), data["z_train"], radioml.CLASSES


def split(X, y, z, test_fraction=0.3, val_fraction=0.0, seed=0):
    """
    Stratified split over (class, SNR) cells.

    Frames here are independently generated, so a random split is honest. On
    *captured* data it would not be -- there you must split by recording.

    `val_fraction=0` returns (train, test) and reproduces the earlier two-way
    behaviour exactly, index for index, so results already measured stay
    comparable. A non-zero `val_fraction` returns (train, val, test) instead.

    The point of the third set: validation is what training is allowed to look
    at -- early stopping, LR schedule, which checkpoint to keep -- so that the
    test set is untouched until the single final measurement. Without it the
    epoch count is chosen by hand against the same set that is then reported.
    """
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for cls in np.unique(y):
        for snr in np.unique(z):
            idx = rng.permutation(np.flatnonzero((y == cls) & (z == snr)))
            n_train = int(len(idx) * (1 - test_fraction - val_fraction))
            n_val = int(len(idx) * val_fraction)
            train.append(idx[:n_train])
            val.append(idx[n_train:n_train + n_val])
            test.append(idx[n_train + n_val:])
    if val_fraction == 0.0:
        return np.concatenate(train), np.concatenate(test)
    return np.concatenate(train), np.concatenate(val), np.concatenate(test)


def normalize_frames(X: np.ndarray) -> np.ndarray:
    """Unit average power per frame, so nothing can be won by reading gain."""
    power = np.mean(X[:, 0] ** 2 + X[:, 1] ** 2, axis=1, keepdims=True)
    return X / (np.sqrt(power)[:, None] + 1e-12)


# ==========================================================================
# Spectrogram front-end (differentiable, runs on GPU)
# ==========================================================================


class SpectrogramLayer(nn.Module):
    """
    (batch, 2, N) real/imag -> (batch, 1, F, T) log-magnitude spectrogram.

    Kept as a layer rather than a preprocessing step so both variants read the
    exact same input tensors; the only difference between the two experiments
    is what happens after this point.
    """

    def __init__(self, n_fft: int = 64, hop: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x[:, 0], x[:, 1])
        spec = torch.stft(
            z, n_fft=self.n_fft, hop_length=self.hop, window=self.window,
            return_complex=True, center=False, onesided=False,
        )
        spec = torch.fft.fftshift(spec, dim=1)  # DC in the middle
        mag = 20 * torch.log10(spec.abs() + 1e-8)
        mag = (mag - mag.mean(dim=(1, 2), keepdim=True)) / (
            mag.std(dim=(1, 2), keepdim=True) + 1e-8
        )
        return mag.unsqueeze(1)


# ==========================================================================
# Models
# ==========================================================================


class IQNet(nn.Module):
    """1D CNN over raw IQ. Sees the waveform, so constellation order is visible."""

    def __init__(self, n_classes: int, width: int = 64):
        super().__init__()
        channels = [2, width, width, width * 2, width * 2, width * 4, width * 4]
        blocks = []
        for i in range(len(channels) - 1):
            blocks += [
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=7, padding=3),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(channels[-1], n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class SpecNet(nn.Module):
    """2D CNN over the spectrogram. Sees bandwidth and time structure only."""

    def __init__(self, n_classes: int, width: int = 32):
        super().__init__()
        self.stft = SpectrogramLayer()
        channels = [1, width, width * 2, width * 4]
        blocks = []
        for i in range(len(channels) - 1):
            blocks += [
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(channels[-1], n_classes),
        )

    def forward(self, x):
        return self.head(self.features(self.stft(x)))


# ==========================================================================
# Training
# ==========================================================================


def train_model(model, Xtr, ytr, Xmon, ymon, epochs=25, batch_size=256, lr=1e-3,
                augment=None, patience=None, grad_clip=None):
    """
    `augment` is an optional callable applied to each training batch on the GPU.
    It is deliberately not applied at evaluation time: the point is to make the
    model invariant to those nuisance parameters, not to average over them at
    test time.

    `Xmon`/`ymon` is the set watched during training. Two protocols, chosen by
    `patience`:

      patience=None   Fixed-length training: OneCycleLR across exactly `epochs`
                      epochs, final model returned, `Xmon` only printed. The
                      original protocol, kept so earlier results reproduce.

      patience=int    ReduceLROnPlateau driven by `Xmon` accuracy, early stop
                      after `patience` epochs without improvement, and the
                      best-`Xmon` checkpoint returned instead of the last.

    In the second mode `Xmon` **must be a validation set disjoint from the test
    set**. Pass the test set there and the reported number is selected on
    itself. OneCycleLR cannot be used in that mode: it has to know its total
    step count up front and must run the schedule to completion, so stopping
    early would leave the learning rate mid-cycle.
    """
    model = model.to(DEVICE)

    # Mixed precision. Profiling showed the GPU pinned at 95% utilisation with
    # VRAM barely touched and the CPU idle, i.e. genuinely compute-bound rather
    # than starved for data. Ampere tensor cores sit unused in fp32, so fp16
    # autocast is the one software lever that actually reduces the work.
    # Disabled automatically on CPU, where it would only add overhead.
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    early = patience is not None
    if early:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.2, patience=max(1, patience // 3),
            min_lr=1e-6)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=epochs * (len(Xtr) // batch_size + 1)
        )

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xte_t = torch.from_numpy(Xmon).to(DEVICE)
    yte_t = torch.from_numpy(ymon).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters, training on {DEVICE}"
          + (f", early stopping on val (patience {patience})" if early else ""))

    best_acc, best_state, best_epoch, stale = -1.0, None, 0, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        total_loss = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            xb = Xtr_t[idx].to(DEVICE, non_blocking=True)
            yb = ytr_t[idx].to(DEVICE, non_blocking=True)
            if augment is not None:
                with torch.no_grad():
                    xb = augment(xb)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(model(xb), yb)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if not early:
                sched.step()          # OneCycleLR steps per batch
            total_loss += loss.item() * len(idx)

        if early:
            acc = evaluate_accuracy(model, Xte_t, yte_t, batch_size)
            sched.step(acc)           # ReduceLROnPlateau steps per epoch
            if acc > best_acc:
                best_acc, best_epoch, stale = acc, epoch + 1, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                stale += 1
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  epoch {epoch+1:>3}/{epochs}  "
                      f"loss {total_loss/len(perm):.4f}  val acc {acc:.4f}"
                      f"  best {best_acc:.4f} (ep {best_epoch})")
            if stale >= patience:
                print(f"  early stop at epoch {epoch+1}; best val "
                      f"{best_acc:.4f} at epoch {best_epoch}")
                break
        elif (epoch + 1) % 5 == 0 or epoch == 0:
            acc = evaluate_accuracy(model, Xte_t, yte_t, batch_size)
            print(f"  epoch {epoch+1:>3}/{epochs}  loss {total_loss/len(perm):.4f}"
                  f"  test acc {acc:.4f}")

    if early and best_state is not None:
        model.load_state_dict(best_state)

    # Which epoch the returned weights came from, and whether the run stopped
    # because it stopped improving or because it ran out of budget. Attached to
    # the model rather than returned, so that every existing caller is
    # unaffected -- they all ignore it.
    #
    # This is worth recording because a run that ends at its ceiling has not
    # converged, and reporting its accuracy as the model's is reporting a floor.
    # The paper-faithful ICRNNA looked 1.49 points worse than published for
    # exactly that reason: at a 58-epoch ceiling it peaked at epoch 58, and at
    # 150 it peaked at 107 and matched the paper.
    model.best_epoch = best_epoch if early else epochs
    model.best_val = best_acc if early else float("nan")
    model.stopped_early = bool(early and stale >= patience)
    model.epoch_ceiling = epochs
    return model


@torch.no_grad()
def evaluate_accuracy(model, Xte_t, yte_t, batch_size=512) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(Xte_t), batch_size):
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            pred = model(Xte_t[i : i + batch_size]).argmax(1)
        correct += (pred == yte_t[i : i + batch_size]).sum().item()
    return correct / len(Xte_t)


@torch.no_grad()
def predict(model, X, batch_size=512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(DEVICE)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


# ==========================================================================
# Reporting
# ==========================================================================


def report(pred, yte, zte, class_names, tag: str, source: str = "synthetic") -> dict:
    # `tag` alone is not a unique filename: an iq/radioml run would silently
    # overwrite the iq/synthetic figures. Both belong in the name.
    tag = f"{tag}_{source}"
    snrs = np.unique(zte)
    accs = [float((pred[zte == s] == yte[zte == s]).mean()) for s in snrs]

    print(f"\n  SNR (dB)   accuracy")
    for s, a in zip(snrs, accs):
        print(f"  {s:>7}     {a:.3f}")

    baseline_path = ROOT / "baseline_results.npz"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snrs, accs, "o-", lw=2, label=f"CNN ({tag})")
    if baseline_path.exists():
        b = np.load(baseline_path, allow_pickle=True)
        if len(b["modulations"]) == len(class_names):
            ax.plot(b["snrs"], b["accuracies"], "s--", lw=1.5, color="gray",
                    label="classical (cumulants + SVM)")
    ax.axhline(1 / len(class_names), ls=":", color="red", lw=1,
               label=f"chance ({1/len(class_names):.3f})")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"CNN on {tag} input vs classical baseline")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"06_cnn_{tag}_accuracy.png", dpi=140)
    plt.close(fig)

    high = zte >= 12
    cm = confusion_matrix(yte[high], pred[high], labels=range(len(class_names)))
    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    size = max(7, len(class_names) * 0.55)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"CNN ({tag}), SNR >= 12 dB  "
                 f"(acc {(pred[high] == yte[high]).mean():.3f})")
    if len(class_names) <= 14:
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                if cm[r, c] > 0.02:
                    ax.text(c, r, f"{cm[r,c]:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if cm[r, c] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES / f"07_cnn_{tag}_confusion.png", dpi=140)
    plt.close(fig)

    print(f"\nwrote figures/06_cnn_{tag}_accuracy.png")
    print(f"wrote figures/07_cnn_{tag}_confusion.png")
    return {"snrs": snrs, "accuracies": np.array(accs)}


# ==========================================================================


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", choices=("iq", "spectrogram"), default="iq")
    p.add_argument("--data", choices=("synthetic", "radioml"), default="synthetic")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--frames-per-cell", type=int, default=500)
    args = p.parse_args()

    print(f"building {args.data} dataset...")
    if args.data == "synthetic":
        X, y, z, class_names = synthetic_dataset(args.frames_per_cell)
    else:
        X, y, z, class_names = radioml_dataset(args.frames_per_cell)

    X = normalize_frames(X)
    tr, te = split(X, y, z)
    print(f"\n{len(tr)} train / {len(te)} test frames, {len(class_names)} classes")

    model = (IQNet if args.input == "iq" else SpecNet)(len(class_names))
    print(f"\ntraining {model.__class__.__name__} on {args.input} input...")
    t0 = time.time()
    model = train_model(model, X[tr], y[tr], X[te], y[te], epochs=args.epochs)
    print(f"  trained in {time.time() - t0:.1f}s")

    pred = predict(model, X[te])
    results = report(pred, y[te], z[te], class_names, args.input, args.data)

    np.savez(ROOT / f"cnn_{args.input}_{args.data}_results.npz", **results,
             class_names=np.array(class_names))
    torch.save(model.state_dict(), ROOT / f"cnn_{args.input}_{args.data}.pt")


if __name__ == "__main__":
    main()
