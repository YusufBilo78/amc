"""
sink_class_24.py -- the adjacency rule on all 24 classes.

sink_class.py established, on five classes, that a classifier shown a
modulation it has never seen does not spread its predictions: it lands on one
class, and the discriminating control said that class is the *nearest* in
constellation order, not the densest.

Five classes is a weak test of that, because they form a single density
ordering (BPSK < QPSK < 8PSK < 16QAM < 64QAM) in which "nearest" and "next
densest" almost always point at the same place. RadioML's full set does not
have that problem. It has four digital families that each carry their own
internal order:

    ASK   OOK  4ASK  8ASK
    PSK   BPSK  QPSK  8PSK  16PSK  32PSK
    APSK  16APSK  32APSK  64APSK  128APSK
    QAM   16QAM  32QAM  64QAM  128QAM  256QAM

So the sharp question becomes: hold out 32PSK, and does it land on **16PSK**
(its neighbour inside its own family) or on a **QAM/APSK of similar order**
(comparable density, different family)? Those are different answers and five
classes could not separate them.

Method: leave-one-class-out. Train on 23, probe with the held-out class drawn
from RadioML itself, so the probe is in-distribution as a signal and
out-of-distribution only as a class. Ten held-out classes are used rather than
all 24 -- chosen to cover top-of-family, middle-of-family, and two awkward
cases -- because each one costs a full training run.

Data is loaded once and masked per run; only the label mapping changes.
"""

from __future__ import annotations

import json
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
import radioml

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

FAMILIES: dict[str, list[str]] = {
    "ASK": ["OOK", "4ASK", "8ASK"],
    "PSK": ["BPSK", "QPSK", "8PSK", "16PSK", "32PSK"],
    "APSK": ["16APSK", "32APSK", "64APSK", "128APSK"],
    "QAM": ["16QAM", "32QAM", "64QAM", "128QAM", "256QAM"],
    "analog": ["AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "FM"],
    "other": ["GMSK", "OQPSK"],
}

# Every class, one at a time. The first pass used a chosen subset of ten to
# keep the cost down; this is the complete sweep, and results already in the
# checkpoint are reused rather than recomputed.
#
# Caveat on the taxonomy: "order within family" is well defined for the four
# digital families, where it is constellation size. For `analog` and `other` the
# ordering in FAMILIES is a convention, not a physical progression, so
# ORDER-ADJACENT means less there and only the same-family verdict should be
# read.
HELD_OUT = [c for fam in FAMILIES.values() for c in fam]

TRAIN_SNRS = list(range(-20, 31, 2))
PROBE_SNRS = list(range(10, 31, 2))
FRAMES_TRAIN = 256
FRAMES_PROBE = 300
EPOCHS = 15
SEED = 0


def family_of(name: str) -> str:
    for fam, members in FAMILIES.items():
        if name in members:
            return fam
    return "?"


def neighbours_of(name: str) -> list[str]:
    """Immediate neighbours in the family's own order."""
    fam = family_of(name)
    members = FAMILIES[fam]
    i = members.index(name)
    out = []
    if i > 0:
        out.append(members[i - 1])
    if i < len(members) - 1:
        out.append(members[i + 1])
    return out


def main() -> None:
    classes = list(radioml.CLASSES)
    ds = radioml.RadioML()
    print(f"backend: {ds.backend}")

    idx_of = {c: i for i, c in enumerate(classes)}

    # Load every class once; each run masks out one label.
    print(f"\nloading all {len(classes)} classes "
          f"({FRAMES_TRAIN}/cell) ...")
    t0 = time.perf_counter()
    train_rows, train_y, train_z = [], [], []
    rng = np.random.default_rng(SEED)
    for ci, name in enumerate(classes):
        for snr in TRAIN_SNRS:
            rows = ds.cell_indices(ci, snr)
            rows = rng.choice(rows, FRAMES_TRAIN, replace=False)
            train_rows.append(np.sort(rows))
            train_y.append(np.full(FRAMES_TRAIN, ci, dtype=np.int64))
            train_z.append(np.full(FRAMES_TRAIN, snr, dtype=np.int16))
    train_rows = np.concatenate(train_rows)
    order = np.argsort(train_rows)
    train_rows = train_rows[order]
    y_all = np.concatenate(train_y)[order]
    z_all = np.concatenate(train_z)[order]
    X_all = np.transpose(ds.X[train_rows], (0, 2, 1)).astype(np.float32)
    power = np.mean(X_all[:, 0] ** 2 + X_all[:, 1] ** 2, axis=1, keepdims=True)
    X_all /= np.sqrt(power)[:, None] + 1e-12
    print(f"  {X_all.shape} in {time.perf_counter()-t0:.0f}s "
          f"({X_all.nbytes/1e9:.1f} GB)")

    # Probe frames, high SNR only.
    probes = {}
    for name in HELD_OUT:
        ci = idx_of[name]
        rows = np.concatenate([
            rng.choice(ds.cell_indices(ci, s), FRAMES_PROBE, replace=False)
            for s in PROBE_SNRS])
        rows = np.sort(rows)
        P = np.transpose(ds.X[rows], (0, 2, 1)).astype(np.float32)
        p = np.mean(P[:, 0] ** 2 + P[:, 1] ** 2, axis=1, keepdims=True)
        probes[name] = P / (np.sqrt(p)[:, None] + 1e-12)
    ds.close()

    # Each run costs several minutes, so results are written to disk as soon
    # as they exist and completed classes are skipped on a restart. A dead
    # battery or a closed lid then costs one run, not the whole sweep.
    ckpt = ROOT / "sink_class_24_partial.json"
    results = json.loads(ckpt.read_text()) if ckpt.exists() else {}
    if results:
        print(f"\nresuming: {len(results)} of {len(HELD_OUT)} already done "
              f"({', '.join(results)})")

    t_start = time.perf_counter()
    for run, held in enumerate(HELD_OUT, 1):
        if held in results:
            print(f"\n[{run}/{len(HELD_OUT)}] {held} -- already done, skipping")
            continue
        held_i = idx_of[held]
        keep = y_all != held_i
        kept_classes = [c for c in classes if c != held]
        remap = {idx_of[c]: j for j, c in enumerate(kept_classes)}

        X = X_all[keep]
        y = np.array([remap[v] for v in y_all[keep]], dtype=np.int64)
        z = z_all[keep]

        print(f"\n[{run}/{len(HELD_OUT)}] hold out {held} "
              f"({family_of(held)})  -- training on {len(kept_classes)} classes")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        tr, te = cnn.split(X, y, z, test_fraction=0.25, seed=SEED)
        model = model_zoo.backbone(len(kept_classes))
        model = cnn.train_model(model, X[tr], y[tr], X[te], y[te], epochs=EPOCHS)
        in_dist = float((cnn.predict(model, X[te]) == y[te]).mean())

        pred = cnn.predict(model, probes[held])
        share = np.bincount(pred, minlength=len(kept_classes)).astype(float)
        share /= share.sum()
        top = np.argsort(share)[::-1][:3]

        sink = kept_classes[top[0]]
        nb = neighbours_of(held)
        verdict = ("ORDER-ADJACENT" if sink in nb else
                   "same family" if family_of(sink) == family_of(held) else
                   "different family")
        print(f"    in-dist acc {in_dist:.3f}   sink -> {sink} "
              f"({share[top[0]]:.1%})   [{verdict}]")
        print(f"    top-3: " + ", ".join(
            f"{kept_classes[i]} {share[i]:.1%}" for i in top))
        print(f"    family neighbours of {held}: {', '.join(nb) or 'none'}")

        results[held] = dict(sink=sink, share=float(share[top[0]]),
                             verdict=verdict, in_dist=in_dist,
                             top3=[(kept_classes[i], float(share[i])) for i in top],
                             neighbours=nb)
        ckpt.write_text(json.dumps(results, indent=2))

    print(f"\ntotal {(time.perf_counter()-t_start)/60:.1f} min")

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 88)
    print(f"{'held out':<12} {'family':<8} {'sink':<12} {'share':>7}  verdict")
    print("-" * 88)
    counts = {"ORDER-ADJACENT": 0, "same family": 0, "different family": 0}
    for held in HELD_OUT:
        r = results[held]
        counts[r["verdict"]] += 1
        print(f"{held:<12} {family_of(held):<8} {r['sink']:<12} "
              f"{r['share']:>6.1%}  {r['verdict']}")
    n = len(HELD_OUT)
    print("-" * 88)
    print(f"order-adjacent within family : {counts['ORDER-ADJACENT']}/{n}")
    print(f"same family (not adjacent)   : {counts['same family']}/{n}")
    print(f"different family             : {counts['different family']}/{n}")
    print(f"same family overall          : "
          f"{counts['ORDER-ADJACENT'] + counts['same family']}/{n}")
    print("=" * 88)

    # ------------------------------------------------------------------ plot
    fig, ax = plt.subplots(figsize=(11, max(6, 0.42 * len(HELD_OUT))))
    colors = {"ORDER-ADJACENT": "#1e8449", "same family": "#f39c12",
              "different family": "#c0392b"}
    y_pos = np.arange(len(HELD_OUT))
    for i, held in enumerate(HELD_OUT):
        r = results[held]
        left = 0.0
        for j, (name, s) in enumerate(r["top3"]):
            c = colors[r["verdict"]] if j == 0 else "#d5d8dc"
            ax.barh(i, s, left=left, color=c, edgecolor="white")
            if s > 0.06:
                ax.text(left + s / 2, i, f"{name}\n{s:.0%}", ha="center",
                        va="center", fontsize=7,
                        color="white" if j == 0 else "#2c3e50")
            left += s
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{h}  ({family_of(h)})" for h in HELD_OUT], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of predictions (top 3 shown)")
    ax.set_title("Leave-one-class-out on RadioML's 24 classes\n"
                 "where does an unseen modulation land?",
                 fontsize=12, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()]
    ax.legend(handles, colors.keys(), fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(FIGURES / "22_sink_class_24.png", dpi=140)
    plt.close(fig)

    np.savez(ROOT / "sink_class_24.npz",
             held_out=np.array(HELD_OUT),
             sinks=np.array([results[h]["sink"] for h in HELD_OUT]),
             shares=np.array([results[h]["share"] for h in HELD_OUT]),
             verdicts=np.array([results[h]["verdict"] for h in HELD_OUT]))
    print("\nwrote figures/22_sink_class_24.png")


if __name__ == "__main__":
    main()
