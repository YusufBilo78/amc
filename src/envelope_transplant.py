"""
envelope_transplant.py -- can the envelope alone make the model say the wrong class?

Every result so far shows that *changing* the spectral envelope changes the
answer. That is suggestive but weak: changing the envelope might be disturbing
something else at the same time.

This asks the sharp version. A signal's spectrum factors as

    X(f) = |X(f)| . exp(j.phi(f))

where |X(f)| is the envelope -- set by symbol rate and pulse shaping, carrying
no information about constellation order -- and phi(f) is where the symbol
sequence actually lives. The two can be separated and recombined across
signals.

So: take frames that really are 16QAM, keep their phase spectrum untouched,
and give them the average *magnitude* spectrum of 64QAM. The symbols are still
16QAM. Only the costume changed.

    model says 64QAM  ->  the envelope drives the decision. The shortcut is
                          demonstrated directly, not inferred.
    model says 16QAM  ->  structure drives it, and the story needs rework.

Run over all ordered pairs to get a 5x5 matrix: row = donor of phase (the true
modulation), column = donor of envelope. If the envelope wins, columns are
constant. If structure wins, the matrix is diagonal.

Both models are probed. Prediction, recorded before running: the alpha=0 model
should follow the transplanted envelope substantially more than the alpha=0.75
model, because whitening removes the envelope from the input before the network
ever sees it -- so for the whitened model the transplant should be close to a
no-op. If both follow the envelope equally, whitening is not doing what the
account claims.
"""

from __future__ import annotations

import pathlib
import sys

sys.stdout.reconfigure(line_buffering=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import cnn
import domains
from narrow_the_search import mean_magnitude_spectrum, spectral_match
from whitening_sweep import spectral_whiten, to_model_input

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

CLASSES = list(domains.SHARED_CLASSES)
TRAIN_SNRS = list(range(-20, 31, 2))
EVAL_SNRS = list(range(10, 31, 2))
EPOCHS = 15
SEED = 0


def train_one(alpha, a, a_iq, tr, te):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X = to_model_input(spectral_whiten(a_iq, alpha))
    model = cnn.IQNet(len(CLASSES))
    return cnn.train_model(model, X[tr], a["y"][tr], X[te], a["y"][te],
                           epochs=EPOCHS)


def main() -> None:
    src = domains.RadioMLDomain()
    a = src.load(CLASSES, TRAIN_SNRS, frames_per_cell=768, seed=0)
    ev = src.load(CLASSES, EVAL_SNRS, frames_per_cell=300, seed=9)
    src.close()

    a_iq = (a["X"][:, 0] + 1j * a["X"][:, 1]).astype(np.complex64)
    ev_iq = (ev["X"][:, 0] + 1j * ev["X"][:, 1]).astype(np.complex64)
    tr, te = cnn.split(a["X"], a["y"], a["z"], test_fraction=0.3, seed=SEED)

    # Everything stays inside RadioML: the question is what the model learned,
    # not how it transfers. Using training-distribution frames removes domain
    # shift as a confound.
    by_class = {i: ev_iq[ev["y"] == i] for i in range(len(CLASSES))}
    envelopes = {i: mean_magnitude_spectrum(by_class[i]) for i in by_class}

    n = len(CLASSES)
    follow_env = {}   # alpha -> (n, n) P(predict = envelope donor)
    follow_str = {}   # alpha -> (n, n) P(predict = phase donor)

    for alpha in (0.0, 0.75):
        print(f"\n--- alpha = {alpha} ---")
        model = train_one(alpha, a, a_iq, tr, te)

        env_mat = np.zeros((n, n))
        str_mat = np.zeros((n, n))
        for i in range(n):          # phase donor = true modulation
            for j in range(n):      # envelope donor
                swapped = spectral_match(by_class[i], envelopes[j])
                X = to_model_input(spectral_whiten(swapped, alpha))
                pred = cnn.predict(model, X)
                env_mat[i, j] = float((pred == j).mean())
                str_mat[i, j] = float((pred == i).mean())
        follow_env[alpha] = env_mat
        follow_str[alpha] = str_mat

        off = ~np.eye(n, dtype=bool)
        print(f"{'phase\\envelope':<10}" + "".join(f"{c:>9}" for c in CLASSES))
        for i in range(n):
            print(f"{CLASSES[i]:<10}" + "".join(f"{env_mat[i,j]:>9.2f}"
                                                for j in range(n)))
        print(f"\noff-diagonal: follows envelope {env_mat[off].mean():.3f}   "
              f"keeps true class {str_mat[off].mean():.3f}")

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8),
                             gridspec_kw={"width_ratios": [1, 1, 0.7]})

    for ax, alpha, title in [
        (axes[0], 0.0, "alpha = 0  (no whitening)"),
        (axes[1], 0.75, "alpha = 0.75  (whitened)"),
    ]:
        m = follow_env[alpha]
        im = ax.imshow(m, cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(CLASSES, fontsize=9)
        ax.set_xlabel("envelope donated by", fontsize=10)
        ax.set_ylabel("true modulation (phase kept)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if m[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)

    off = ~np.eye(n, dtype=bool)
    ax = axes[2]
    x = np.arange(2)
    env_vals = [follow_env[0.0][off].mean(), follow_env[0.75][off].mean()]
    str_vals = [follow_str[0.0][off].mean(), follow_str[0.75][off].mean()]
    ax.bar(x - 0.2, env_vals, 0.4, label="predicts the donated envelope",
           color="tab:red")
    ax.bar(x + 0.2, str_vals, 0.4, label="keeps the true class",
           color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\alpha=0$", r"$\alpha=0.75$"], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of mismatched pairs")
    ax.set_title("Who wins when they disagree", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    for xi, (e, s) in enumerate(zip(env_vals, str_vals)):
        ax.text(xi - 0.2, e + 0.02, f"{e:.2f}", ha="center", fontsize=9)
        ax.text(xi + 0.2, s + 0.02, f"{s:.2f}", ha="center", fontsize=9)

    fig.suptitle(
        "Envelope transplant: phase from one modulation, magnitude spectrum "
        "from another", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "19_envelope_transplant.png", dpi=140)
    plt.close(fig)

    np.savez(ROOT / "envelope_transplant.npz", classes=np.array(CLASSES),
             follow_env_a0=follow_env[0.0], follow_env_a75=follow_env[0.75],
             follow_str_a0=follow_str[0.0], follow_str_a75=follow_str[0.75])

    print("\n" + "=" * 68)
    print(f"{'':<14}{'follows envelope':>18}{'keeps true class':>18}")
    for alpha in (0.0, 0.75):
        print(f"alpha={alpha:<8}{follow_env[alpha][off].mean():>18.3f}"
              f"{follow_str[alpha][off].mean():>18.3f}")
    print("=" * 68)
    print("\nwrote figures/19_envelope_transplant.png")


if __name__ == "__main__":
    main()
