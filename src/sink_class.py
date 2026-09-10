"""
sink_class.py -- when the classifier is shown something it cannot be right
about, where does it go?

Three separate observations in this project pointed the same way and were never
followed up:

    cross-domain test   75% of our 16QAM predicted as 64QAM
    sweep.py, sps=16    64QAM 0.999, QPSK 0.042 -- far below the 0.20 chance
                        level, so not random error
    spectrogram CNN     four classes collapsed into a single block

Under out-of-distribution input the network does not spread its predictions.
It falls onto one class. Each time, that class was the densest constellation
available. Whether that is a rule or a coincidence has never been tested.

Design
------
The cleanest probe is a modulation the model has never seen, drawn from the
*training* domain. It is in-distribution as a signal and out-of-distribution as
a class, so nothing about domain shift confounds the answer, and the model
cannot be right no matter what it says.

Nested removals, each dropping the densest remaining class:

    train {BPSK QPSK 8PSK 16QAM}  probe 64QAM          -> expect 16QAM
    train {BPSK QPSK 8PSK}        probe 16QAM 64QAM    -> expect 8PSK
    train {BPSK QPSK}             probe 8PSK ...       -> expect QPSK

Two controls, both of which the hypothesis could fail:

1. DENSEST vs NEAREST. Train on {QPSK 8PSK 16QAM 64QAM} and probe with BPSK.
   BPSK's nearest neighbour is QPSK; the densest class is 64QAM. "Defaults to
   the densest" and "defaults to the nearest" make opposite predictions here,
   so this single cell separates them.

2. LABEL PERMUTATION. If the sink is really an output-index artifact -- the
   last unit, or the one that happened to be trained on most recently -- then
   permuting which index means which class should move the sink. If the sink
   follows the *class* rather than the index, it is a property of the signals.

Gaussian noise is included as a third probe: maximally out of distribution,
carrying no modulation at all.
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
import domains

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

# Ordered sparsest -> densest. This ordering is the hypothesis under test.
BY_DENSITY = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]

TRAIN_SNRS = list(range(-20, 31, 2))
PROBE_SNRS = list(range(10, 31, 2))   # high SNR: the model should be confident
EPOCHS = 15
SEEDS = [0, 1]

CONFIGS: list[tuple[str, list[str], list[str]]] = [
    # label,                    trained on,                       probed with
    ("drop 64QAM",              BY_DENSITY[:4],                   ["64QAM"]),
    ("drop 64QAM, 16QAM",       BY_DENSITY[:3],                   ["16QAM", "64QAM"]),
    ("drop 64/16QAM, 8PSK",     BY_DENSITY[:2],                   ["8PSK", "16QAM", "64QAM"]),
    ("drop BPSK  [control]",    BY_DENSITY[1:],                   ["BPSK"]),
]


def load_split(rml, classes, snrs, frames, seed):
    data = rml.load(classes, snrs, frames_per_cell=frames, seed=seed)
    return data


def noise_frames(n: int, rng) -> np.ndarray:
    """Unit-power complex Gaussian: no modulation, maximally out of distribution."""
    z = (rng.standard_normal((n, 1024)) + 1j * rng.standard_normal((n, 1024)))
    z /= np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True))
    return np.stack([z.real, z.imag], axis=1).astype(np.float32)


def sink_report(pred: np.ndarray, classes: list[str]) -> tuple[str, float, np.ndarray]:
    """Which class absorbs the predictions, and how concentrated is it."""
    counts = np.bincount(pred, minlength=len(classes)).astype(float)
    share = counts / counts.sum()
    i = int(share.argmax())
    return classes[i], float(share[i]), share


def run(rml, train_classes, probe_classes, seed, permute=False):
    """Train on train_classes, return the predicted distribution for each probe."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    order = list(train_classes)
    if permute:
        order = list(rng.permutation(order))

    tr_data = rml.load(order, TRAIN_SNRS, frames_per_cell=768, seed=seed)
    idx_tr, idx_te = cnn.split(tr_data["X"], tr_data["y"], tr_data["z"],
                               test_fraction=0.25, seed=seed)
    model = cnn.IQNet(len(order))
    model = cnn.train_model(model, tr_data["X"][idx_tr], tr_data["y"][idx_tr],
                            tr_data["X"][idx_te], tr_data["y"][idx_te],
                            epochs=EPOCHS)

    in_dist = float((cnn.predict(model, tr_data["X"][idx_te])
                     == tr_data["y"][idx_te]).mean())

    out = {}
    for probe in probe_classes:
        pdata = rml.load([probe], PROBE_SNRS, frames_per_cell=300, seed=seed + 50)
        out[probe] = sink_report(cnn.predict(model, pdata["X"]), order)

    out["<noise>"] = sink_report(
        cnn.predict(model, noise_frames(3000, rng)), order)
    return order, in_dist, out


def main() -> None:
    rml = domains.RadioMLDomain()
    t0 = time.time()

    print("=" * 78)
    print("NESTED REMOVAL  (each row drops the densest remaining class)")
    print("=" * 78)

    results = {}
    for label, train_classes, probes in CONFIGS:
        expected = train_classes[-1]  # densest remaining, under the hypothesis
        print(f"\n{label}")
        print(f"  trained on: {', '.join(train_classes)}")
        print(f"  hypothesis predicts the sink is: {expected}")
        for seed in SEEDS:
            order, in_dist, out = run(rml, train_classes, probes, seed)
            for probe, (sink, share, _) in out.items():
                tick = "OK " if sink == expected else "-- "
                print(f"    seed {seed}  probe {probe:<8} -> sink {sink:<7} "
                      f"({share:.1%})  {tick}[in-dist acc {in_dist:.3f}]")
            results[(label, seed)] = (order, in_dist, out, expected)

    print("\n" + "=" * 78)
    print("LABEL PERMUTATION CONTROL  (all five classes, output indices shuffled)")
    print("=" * 78)
    perm_rows = []
    for seed in [0, 1, 2]:
        order, in_dist, out = run(rml, BY_DENSITY, ["64QAM"], seed, permute=True)
        # 64QAM is in the training set here, so probe only with noise.
        sink, share, _ = out["<noise>"]
        perm_rows.append((order, sink, share))
        print(f"  seed {seed}  output order {order}")
        print(f"           noise -> sink {sink} ({share:.1%})")

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(4.1 * len(CONFIGS), 4.2),
                             sharey=True)
    for ax, (label, train_classes, probes) in zip(axes, CONFIGS):
        order, _, out, expected = results[(label, SEEDS[0])]
        probe_names = list(out.keys())
        width = 0.8 / len(probe_names)
        x = np.arange(len(order))
        for k, probe in enumerate(probe_names):
            _, _, share = out[probe]
            ax.bar(x + k * width - 0.4 + width / 2, share, width,
                   label=f"probe: {probe}")
        hi = order.index(expected)
        ax.axvspan(hi - 0.45, hi + 0.45, color="gold", alpha=0.22, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("share of predictions")
    fig.suptitle("Where do predictions go when the model cannot be right?\n"
                 "gold band = class the 'densest sink' hypothesis predicts",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGURES / "20_sink_class.png", dpi=140)
    plt.close(fig)

    np.savez(ROOT / "sink_class.npz",
             configs=np.array([c[0] for c in CONFIGS]),
             permutation_sinks=np.array([r[1] for r in perm_rows]))

    # ------------------------------------------------------------- verdict
    hits = tot = 0
    control_sinks = []
    for (label, seed), (order, _, out, expected) in results.items():
        for probe, (sink, _, _) in out.items():
            if probe == "<noise>":
                continue
            if "control" in label:
                control_sinks.append(sink)
                continue
            tot += 1
            hits += sink == expected

    print("\n" + "=" * 78)
    print(f"nested removals: sink matched the densest remaining class "
          f"in {hits}/{tot} cells")
    print(f"densest-vs-nearest control (BPSK probe): sinks were {control_sinks}")
    print("   -> 64QAM would support 'densest'; QPSK would support 'nearest'")
    perm_sinks = [r[1] for r in perm_rows]
    print(f"label permutation, noise probe: sinks were {perm_sinks}")
    print("   -> same class each time means the sink is not an output-index artifact")
    print(f"\ntotal {(time.time()-t0)/60:.1f} min")
    print("=" * 78)
    rml.close()
    print("\nwrote figures/20_sink_class.png")


if __name__ == "__main__":
    main()
