"""
whitening_seeds.py -- repeat the whitening sweep with error bars.

whitening_sweep.py ran one model per alpha and found alpha=0.75 best, with the
gap falling from 0.158 to 0.052. But alpha=0.25 came out non-monotonic
(cross-domain 0.793, 16QAM 0.035) and with a single run there is no way to tell
a real effect from seed noise.

Nothing about the shape of that curve is claimable until it is repeated. This
runs every alpha across several seeds, varying both the train/test split and
the weight initialisation, and reports mean +- standard deviation.

Two questions to settle:

  1. Is the alpha=0.75 improvement larger than run-to-run variation?
  2. Is the alpha=0.25 dip real, or was it one unlucky model?

Whitening is deterministic given alpha, so it is computed once per alpha and
reused across seeds -- only the split and the initialisation change.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import cnn
import model_zoo
import domains
from whitening_sweep import accuracy_by_snr, spectral_whiten, to_model_input

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

CLASSES = list(domains.SHARED_CLASSES)
SNRS = list(range(-20, 31, 2))
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]
SEEDS = [0, 1, 2, 3]
EPOCHS = 15


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None,
                   help="where the .npz and the figure are written, and where "
                        "a partial run is resumed from. Defaults to the "
                        "repository root; point it at durable storage when the "
                        "machine is not (a Colab runtime is reclaimed after 12 "
                        "hours).")
    p.add_argument("--tag", default="",
                   help="appended to the output filename, so a run with a "
                        "different epoch ceiling does not overwrite or resume "
                        "into one that stopped somewhere else")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--patience", type=int, default=None,
                   help="switch to early stopping on a validation split. "
                        "Without it the original fixed-length recipe runs, "
                        "which is what the numbers this rerun replaces used.")
    p.add_argument("--seeds", type=int, default=len(SEEDS))
    args = p.parse_args()

    seeds = list(range(args.seeds))
    epochs = args.epochs
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else ROOT
    fig_dir = out_dir / "figures" if args.out_dir else FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"whitening_seeds{'_' + args.tag if args.tag else ''}.npz"

    print(f"{len(ALPHAS)} alphas x {len(seeds)} seeds = "
          f"{len(ALPHAS) * len(seeds)} models")
    print(f"max {epochs} epochs, protocol "
          f"{'70/15/15 + early stopping' if args.patience else '70/30 fixed'}")
    print(f"-> {npz_path}\n")

    src = domains.RadioMLDomain()
    a = src.load(CLASSES, SNRS, frames_per_cell=768, seed=0)
    src.close()
    dst = domains.SyntheticDomain()
    b = dst.load(CLASSES, SNRS, frames_per_cell=400, seed=1)

    a_iq = (a["X"][:, 0] + 1j * a["X"][:, 1]).astype(np.complex64)
    b_iq = (b["X"][:, 0] + 1j * b["X"][:, 1]).astype(np.complex64)

    high = np.array(SNRS) >= 10
    q = CLASSES.index("16QAM")

    # [alpha, seed] -> metric
    in_domain = np.full((len(ALPHAS), len(seeds)), np.nan)
    cross = np.full((len(ALPHAS), len(seeds)), np.nan)
    qam16 = np.full((len(ALPHAS), len(seeds)), np.nan)
    # See the note in compare_methods.py: this sweep reports differences
    # between cells, so a cell that stopped at the ceiling rather than because
    # it converged makes the comparison meaningless rather than merely low.
    best_epochs = np.full((len(ALPHAS), len(seeds)), np.nan)

    # Resume. The original wrote after every cell but never read the file back,
    # so a run cut short restarted from zero -- against this repository's own
    # rule that a sweep checkpoints *and* resumes. A cell counts as done when
    # its in-domain entry is no longer NaN; the shape check refuses to resume
    # into a file written by a differently-shaped configuration.
    if npz_path.exists():
        old = np.load(npz_path, allow_pickle=True)
        if old["in_domain"].shape == in_domain.shape:
            in_domain, cross = old["in_domain"], old["cross"]
            qam16 = old["qam16"]
            if "best_epochs" in old.files:
                best_epochs = old["best_epochs"]
            done = int(np.count_nonzero(~np.isnan(in_domain)))
            if done:
                print(f"resuming: {done}/{in_domain.size} models already done\n")
        else:
            print(f"{npz_path.name} has a different shape; starting over\n")

    header = (f"{'alpha':>6} {'seed':>5} {'in-dom':>8} {'cross':>8} "
              f"{'gap':>8} {'16QAM':>8} {'time':>7}")
    print(header)
    print("-" * len(header))

    t_start = time.time()
    for i, alpha in enumerate(ALPHAS):
        # Whitening depends only on alpha, so do it once and reuse.
        Xa = to_model_input(spectral_whiten(a_iq, alpha))
        Xb = to_model_input(spectral_whiten(b_iq, alpha))

        for j, seed in enumerate(seeds):
            if not np.isnan(in_domain[i, j]):
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)

            # With early stopping the monitored set must be a validation set:
            # the test set is what gets reported, and choosing the stopping
            # epoch on it would be selecting on the number being reported.
            if args.patience:
                tr, va, te = cnn.split(a["X"], a["y"], a["z"],
                                       test_fraction=0.15, val_fraction=0.15,
                                       seed=seed)
                mon_X, mon_y = Xa[va], a["y"][va]
            else:
                tr, te = cnn.split(a["X"], a["y"], a["z"], test_fraction=0.3,
                                   seed=seed)
                mon_X, mon_y = Xa[te], a["y"][te]

            model = model_zoo.backbone(len(CLASSES))
            model = cnn.train_model(model, Xa[tr], a["y"][tr], mon_X, mon_y,
                                    epochs=epochs, patience=args.patience,
                                    grad_clip=5.0 if args.patience else None)

            pred_in = cnn.predict(model, Xa[te])
            pred_cross = cnn.predict(model, Xb)
            acc_in = accuracy_by_snr(pred_in, a["y"][te], a["z"][te], SNRS)
            acc_cross = accuracy_by_snr(pred_cross, b["y"], b["z"], SNRS)

            mask = (b["z"] >= 10) & (b["y"] == q)
            best_epochs[i, j] = getattr(model, "best_epoch", np.nan)
            if args.patience and not getattr(model, "stopped_early", True):
                print(f"  NOT converged: peaked at epoch "
                      f"{getattr(model, 'best_epoch', '?')} and the "
                      f"{epochs}-epoch budget ran out before early stopping "
                      f"could fire. This cell's accuracy is a floor.")
            in_domain[i, j] = float(np.nanmean(acc_in[high]))
            cross[i, j] = float(np.nanmean(acc_cross[high]))
            qam16[i, j] = float((pred_cross[mask] == q).mean())

            print(f"{alpha:>6.2f} {seed:>5} {in_domain[i,j]:>8.3f} "
                  f"{cross[i,j]:>8.3f} {in_domain[i,j]-cross[i,j]:>+8.3f} "
                  f"{qam16[i,j]:>8.3f} {time.time()-t0:>6.0f}s")

            np.savez(npz_path, alphas=np.array(ALPHAS),
                     seeds=np.array(seeds), in_domain=in_domain,
                     cross=cross, qam16=qam16, best_epochs=best_epochs,
                     epoch_ceiling=epochs)
        print()

    print(f"total {(time.time()-t_start)/60:.1f} min\n")

    # ------------------------------------------------------------- summary
    gap = in_domain - cross
    print(f"{'alpha':>6} {'in-domain':>16} {'cross-domain':>16} "
          f"{'gap':>16} {'16QAM':>16}")
    print("-" * 74)
    for i, alpha in enumerate(ALPHAS):
        print(f"{alpha:>6.2f} "
              f"{in_domain[i].mean():>9.3f} +-{in_domain[i].std():<5.3f} "
              f"{cross[i].mean():>9.3f} +-{cross[i].std():<5.3f} "
              f"{gap[i].mean():>+9.3f} +-{gap[i].std():<5.3f} "
              f"{qam16[i].mean():>9.3f} +-{qam16[i].std():<5.3f}")

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, data, title, color in [
        (axes[0], cross, "cross-domain accuracy", "tab:red"),
        (axes[1], gap, "generalization gap", "tab:purple"),
        (axes[2], qam16, "16QAM, cross-domain", "tab:green"),
    ]:
        mean, std = data.mean(axis=1), data.std(axis=1)
        ax.errorbar(ALPHAS, mean, yerr=std, fmt="o-", lw=2, capsize=5,
                    color=color, markersize=7)
        for j in range(len(seeds)):
            ax.plot(ALPHAS, data[:, j], "o", ms=3, alpha=0.35, color="gray")
        ax.set_xlabel(r"whitening strength  $\alpha$")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)

    axes[0].plot(ALPHAS, in_domain.mean(axis=1), "s--", lw=1.5, color="tab:blue",
                 alpha=0.7, label="in-domain (for reference)")
    axes[0].legend(fontsize=9)
    axes[0].set_ylabel("accuracy at SNR >= 10 dB")
    axes[1].axhline(0, ls=":", color="gray", lw=1)

    fig.suptitle(
        f"Spectral whitening, {len(seeds)} seeds per point "
        "(bars = 1 s.d., grey dots = individual runs)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / f"16_whitening_seeds{'_' + args.tag if args.tag else ''}.png",
                dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------- verdict
    base, best_i = 0, int(cross.mean(axis=1).argmax())
    diff = cross[best_i].mean() - cross[base].mean()
    pooled = np.sqrt((cross[best_i].std() ** 2 + cross[base].std() ** 2) / 2)
    print("\n" + "=" * 70)
    print(f"best alpha = {ALPHAS[best_i]}")
    print(f"cross-domain: {cross[base].mean():.3f} +-{cross[base].std():.3f}"
          f"  ->  {cross[best_i].mean():.3f} +-{cross[best_i].std():.3f}"
          f"   ({diff:+.3f})")
    print(f"effect size vs run-to-run spread: {diff / (pooled + 1e-9):.1f} "
          f"pooled standard deviations")
    print(f"\nalpha=0.25 question: cross-domain "
          f"{cross[1].mean():.3f} +-{cross[1].std():.3f} vs "
          f"alpha=0 {cross[0].mean():.3f} +-{cross[0].std():.3f}")
    print("=" * 70)
    print(f"\nwrote {npz_path}")


if __name__ == "__main__":
    main()
