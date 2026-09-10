"""
compare_with_dann.py -- whitening against adversarial domain adaptation.

compare_methods.py already measured, over four seeds:

    none                                gap 0.182
    standard AMC augmentation           gap 0.141
    spectral whitening (alpha=0.75)     gap 0.056
    whitening + standard augmentation   gap 0.038

The obvious missing reference is DANN, which is the method the AMC
domain-adaptation literature is actually built on. This adds two rows on the
same data, the same seeds and the same schedule:

    DANN
    DANN + whitening

Protocol note. DANN is the only method here that **uses target-domain data
during training** (unlabelled, but present). To keep that honest the target set
is split in half: DANN adapts on one half and is scored on the other, so it is
never evaluated on frames it adapted to. The other methods were scored on the
full target set, which is statistically the same thing for them -- they never
touched target data at any point, and the frames are i.i.d. from one generator.

The comparison is therefore deliberately generous to DANN. If whitening merely
matches it, that is the stronger practical result: a receiver meeting an
unfamiliar transmitter has no target data to adapt to.
"""

from __future__ import annotations

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
import dann as dann_mod
import domains
from whitening_sweep import accuracy_by_snr, spectral_whiten, to_model_input

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

CLASSES = list(domains.SHARED_CLASSES)
SNRS = list(range(-20, 31, 2))
SEEDS = [0, 1, 2, 3]
EPOCHS = 15
BEST_ALPHA = 0.75

CONFIGS = [("DANN", 0.0), ("DANN + whitening", BEST_ALPHA)]


def main() -> None:
    print(f"{len(CONFIGS)} configs x {len(SEEDS)} seeds = "
          f"{len(CONFIGS) * len(SEEDS)} models\n")

    src = domains.RadioMLDomain()
    a = src.load(CLASSES, SNRS, frames_per_cell=768, seed=0)
    src.close()
    dst = domains.SyntheticDomain()
    b = dst.load(CLASSES, SNRS, frames_per_cell=400, seed=1)

    a_iq = (a["X"][:, 0] + 1j * a["X"][:, 1]).astype(np.complex64)
    b_iq = (b["X"][:, 0] + 1j * b["X"][:, 1]).astype(np.complex64)

    # Target split: half for adaptation, half for scoring. Stratified by
    # (class, SNR) so both halves cover the whole grid.
    rng = np.random.default_rng(123)
    adapt, score = [], []
    for c in np.unique(b["y"]):
        for s in np.unique(b["z"]):
            idx = rng.permutation(np.flatnonzero((b["y"] == c) & (b["z"] == s)))
            cut = len(idx) // 2
            adapt.append(idx[:cut])
            score.append(idx[cut:])
    adapt = np.concatenate(adapt)
    score = np.concatenate(score)
    print(f"target: {len(adapt)} frames for adaptation, {len(score)} for scoring")

    cache = {alpha: (to_model_input(spectral_whiten(a_iq, alpha)),
                     to_model_input(spectral_whiten(b_iq, alpha)))
             for _, alpha in CONFIGS}

    high = np.array(SNRS) >= 10
    q = CLASSES.index("16QAM")
    in_dom = np.full((len(CONFIGS), len(SEEDS)), np.nan)
    cross = np.full((len(CONFIGS), len(SEEDS)), np.nan)
    qam16 = np.full((len(CONFIGS), len(SEEDS)), np.nan)

    header = (f"{'config':<20} {'seed':>5} {'in-dom':>8} {'cross':>8} "
              f"{'gap':>8} {'16QAM':>8} {'time':>7}")
    print("\n" + header)
    print("-" * len(header))

    t_all = time.perf_counter()
    for i, (name, alpha) in enumerate(CONFIGS):
        Xa, Xb = cache[alpha]
        for j, seed in enumerate(SEEDS):
            t0 = time.perf_counter()
            torch.manual_seed(seed)
            np.random.seed(seed)

            tr, te = cnn.split(a["X"], a["y"], a["z"], test_fraction=0.3, seed=seed)
            model = dann_mod.DANNNet(len(CLASSES))
            model = dann_mod.train_dann(
                model, Xa[tr], a["y"][tr], Xb[adapt], Xa[te], a["y"][te],
                epochs=EPOCHS, verbose_every=EPOCHS)

            pred_in = cnn.predict(model, Xa[te])
            pred_cr = cnn.predict(model, Xb[score])
            acc_in = accuracy_by_snr(pred_in, a["y"][te], a["z"][te], SNRS)
            acc_cr = accuracy_by_snr(pred_cr, b["y"][score], b["z"][score], SNRS)

            mask = (b["z"][score] >= 10) & (b["y"][score] == q)
            in_dom[i, j] = float(np.nanmean(acc_in[high]))
            cross[i, j] = float(np.nanmean(acc_cr[high]))
            qam16[i, j] = float((pred_cr[mask] == q).mean())

            print(f"{name:<20} {seed:>5} {in_dom[i,j]:>8.3f} {cross[i,j]:>8.3f} "
                  f"{in_dom[i,j]-cross[i,j]:>+8.3f} {qam16[i,j]:>8.3f} "
                  f"{time.perf_counter()-t0:>6.0f}s")

            np.savez(ROOT / "compare_dann.npz",
                     configs=np.array([c[0] for c in CONFIGS]),
                     in_domain=in_dom, cross=cross, qam16=qam16)
        print()

    print(f"total {(time.perf_counter()-t_all)/60:.1f} min")

    # ------------------------------------------------- merge with earlier runs
    prev = np.load(ROOT / "compare_methods.npz", allow_pickle=True)
    names = [str(m) for m in prev["methods"]] + [c[0] for c in CONFIGS]
    all_in = np.vstack([prev["in_domain"], in_dom])
    all_cr = np.vstack([prev["cross"], cross])
    all_q = np.vstack([prev["qam16"], qam16])
    gap = all_in - all_cr

    print("\n" + "=" * 92)
    print(f"{'method':<24} {'in-domain':>15} {'cross-domain':>15} "
          f"{'gap':>15} {'16QAM':>15}")
    print("-" * 92)
    for k, nm in enumerate(names):
        print(f"{nm:<24} {all_in[k].mean():>8.3f} ±{all_in[k].std():<5.3f} "
              f"{all_cr[k].mean():>8.3f} ±{all_cr[k].std():<5.3f} "
              f"{gap[k].mean():>+8.3f} ±{gap[k].std():<5.3f} "
              f"{all_q[k].mean():>8.3f} ±{all_q[k].std():<5.3f}")
    print("=" * 92)

    def pooled(x, y):
        return np.sqrt((x.std() ** 2 + y.std() ** 2) / 2) + 1e-9

    i_wh = names.index("whitening a=0.75")
    i_dn = names.index("DANN")
    i_best = names.index("whitening + standard")
    i_dw = names.index("DANN + whitening")
    d = all_cr[i_wh].mean() - all_cr[i_dn].mean()
    print(f"\nwhitening vs DANN: {d:+.3f}  ({d/pooled(all_cr[i_wh], all_cr[i_dn]):+.1f} s.d.)")
    d = all_cr[i_dw].mean() - all_cr[i_dn].mean()
    print(f"whitening adds to DANN: {d:+.3f}  ({d/pooled(all_cr[i_dw], all_cr[i_dn]):+.1f} s.d.)")
    d = all_cr[i_best].mean() - all_cr[i_dn].mean()
    print(f"best non-adversarial vs DANN: {d:+.3f}  "
          f"({d/pooled(all_cr[i_best], all_cr[i_dn]):+.1f} s.d.)")
    print("\nDANN is the only row above that used target-domain data in training.")

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(names))
    colors = ["#95a5a6", "#e67e22", "#2980b9", "#27ae60", "#8e44ad", "#6c3483"]
    ax.barh(y, gap.mean(1), 0.6, xerr=gap.std(1), capsize=4, color=colors[:len(names)])
    for k in range(len(names)):
        ax.text(gap[k].mean() + gap[k].std() + 0.006, k, f"{gap[k].mean():.3f}",
                va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([n + ("  *" if "DANN" in n else "") for n in names],
                       fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("generalization gap  (lower is better)")
    ax.set_title("Spectral whitening against adversarial domain adaptation\n"
                 "* uses target-domain data during training; the others do not",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIGURES / "23_dann_comparison.png", dpi=140)
    plt.close(fig)
    print("\nwrote figures/23_dann_comparison.png")


if __name__ == "__main__":
    main()
