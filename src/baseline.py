"""
baseline.py -- Week 2: the classical AMC baseline the CNN must beat.

Pipeline: synthetic frames -> hand-designed features -> standardize -> RBF SVM.

The output that matters is accuracy *as a function of SNR*, not a single
averaged number. A single number hides the only interesting part of the curve:
the SNR at which the problem stops being solvable.

Run:
    python baseline.py
"""

from __future__ import annotations

import pathlib
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import features
import modem

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# 1024 samples per frame matches RadioML 2018.01A, so the same feature code
# runs unchanged on the real dataset once it finishes downloading.
FRAME_LEN = 1024
SNRS = np.arange(-20, 21, 4)
FRAMES_PER_CELL = 200  # frames per (modulation, SNR) pair
SEED = 0


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features, labels, snrs). Each frame is generated independently."""
    rng = np.random.default_rng(SEED)
    X, y, z = [], [], []

    total = len(modem.MODULATIONS) * len(SNRS)
    t0 = time.time()
    for i, name in enumerate(modem.MODULATIONS):
        for snr in SNRS:
            frames = np.stack(
                [
                    modem.generate(name, FRAME_LEN, snr_db=float(snr), rng=rng)
                    for _ in range(FRAMES_PER_CELL)
                ]
            )
            X.append(features.extract_batch(frames))
            y.append(np.full(FRAMES_PER_CELL, i))
            z.append(np.full(FRAMES_PER_CELL, snr))
        done = (i + 1) * len(SNRS)
        print(f"  {name:<8} {done:>4}/{total} cells  ({time.time() - t0:5.1f}s)")

    return np.concatenate(X), np.concatenate(y), np.concatenate(z)


def main() -> None:
    print("building dataset...")
    X, y, z = build_dataset()
    print(f"\n{X.shape[0]} frames, {X.shape[1]} features, {len(modem.MODULATIONS)} classes")

    # Every frame carries independent random symbols and independent noise, so
    # a random split is safe here. It will NOT be safe on captured data --
    # there, split by recording, never within one.
    Xtr, Xte, ytr, yte, _, zte = train_test_split(
        X, y, z, test_size=0.3, random_state=SEED, stratify=y
    )

    print("\ntraining SVM (RBF)...")
    t0 = time.time()
    clf = make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", cache_size=1000))
    clf.fit(Xtr, ytr)
    print(f"  done in {time.time() - t0:.1f}s")

    pred = clf.predict(Xte)
    print(f"\noverall test accuracy: {accuracy_score(yte, pred):.3f}")

    # ---------------------------------------------------------------- curve
    accs = []
    print("\n  SNR (dB)   accuracy")
    for snr in SNRS:
        mask = zte == snr
        acc = accuracy_score(yte[mask], pred[mask])
        accs.append(acc)
        print(f"  {snr:>7}     {acc:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(SNRS, accs, "o-", lw=2, color="tab:blue", label="cumulants + SVM")
    ax.axhline(1.0 / len(modem.MODULATIONS), ls="--", color="gray", lw=1,
               label=f"chance ({1/len(modem.MODULATIONS):.2f})")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("classification accuracy")
    ax.set_title("Classical baseline: accuracy vs SNR\n"
                 f"{len(modem.MODULATIONS)} classes, {FRAME_LEN}-sample frames")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "04_baseline_accuracy_vs_snr.png", dpi=140)
    plt.close(fig)
    print("\nwrote figures/04_baseline_accuracy_vs_snr.png")

    # ------------------------------------------------------------ confusion
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, snr in zip(axes, (16, 0)):
        mask = zte == snr
        cm = confusion_matrix(yte[mask], pred[mask], labels=range(len(modem.MODULATIONS)))
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(modem.MODULATIONS)))
        ax.set_yticks(range(len(modem.MODULATIONS)))
        ax.set_xticklabels(modem.MODULATIONS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(modem.MODULATIONS, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"SNR = {snr} dB   (acc {accuracy_score(yte[mask], pred[mask]):.3f})")
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                if cm[r, c] > 0.02:
                    ax.text(c, r, f"{cm[r, c]:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if cm[r, c] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Classical baseline confusion matrices (row-normalized)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "05_baseline_confusion.png", dpi=140)
    plt.close(fig)
    print("wrote figures/05_baseline_confusion.png")

    np.savez(
        ROOT / "baseline_results.npz",
        snrs=SNRS, accuracies=np.array(accs),
        modulations=np.array(modem.MODULATIONS),
    )
    print("wrote baseline_results.npz  (the curve the CNN has to beat)")


if __name__ == "__main__":
    main()
