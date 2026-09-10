"""
sink_class_24_control.py -- does the sink structure survive proper training?

sink_class_24.py ran ten leave-one-class-out models and found that an unseen
modulation lands inside its own family 9 times out of 10. But those models were
deliberately cheap -- 256 frames per cell, 15 epochs -- so that ten of them
would fit in an evening. Their in-distribution accuracy sat at 0.53-0.55 across
23 classes.

That is well above the 0.043 chance level and the sink structure was consistent
across all ten, which argues against it being one model's quirk. It does not
rule out the structure being an artifact of *undertraining* in general: a
weak model might fall back on coarse family-level features simply because it
never learned the finer ones.

So four cases are repeated at 512 frames per cell and 25 epochs -- the settings
that previously reached 0.88 on 24 classes. Chosen because they carry the
argument:

    32APSK, 32QAM   the family-versus-density pair. Same order, different
                    family. If the sink crosses families here, "family beats
                    density" fails.
    64QAM           where order-adjacency broke (went to 256QAM, skipping
                    128QAM). Is that real or undertraining?
    OQPSK           the one class that left its family. Does a stronger model
                    find it a home?

If the sinks match the cheap runs, the finding stands and the caveat can be
dropped. If they move, the finding was about weak models and must be reported
that way.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

import cnn
import model_zoo
import radioml
from sink_class_24 import FAMILIES, family_of, neighbours_of

ROOT = pathlib.Path(__file__).resolve().parent.parent

CASES = ["32APSK", "32QAM", "64QAM", "OQPSK"]

TRAIN_SNRS = list(range(-20, 31, 2))
PROBE_SNRS = list(range(10, 31, 2))
FRAMES_TRAIN = 512     # was 256
FRAMES_PROBE = 300
EPOCHS = 25            # was 15
SEED = 0

CKPT = ROOT / "sink_class_24_control.json"
WEAK = ROOT / "sink_class_24_partial.json"


def main() -> None:
    weak = json.loads(WEAK.read_text()) if WEAK.exists() else {}
    results = json.loads(CKPT.read_text()) if CKPT.exists() else {}
    if results:
        print(f"resuming: {len(results)}/{len(CASES)} done "
              f"({', '.join(results)})\n")

    classes = list(radioml.CLASSES)
    idx_of = {c: i for i, c in enumerate(classes)}
    ds = radioml.RadioML()
    print(f"backend: {ds.backend}")
    print(f"training at {FRAMES_TRAIN} frames/cell, {EPOCHS} epochs "
          f"(cheap run used 256 / 15)\n")

    print("loading all 24 classes ...")
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    rows, ys, zs = [], [], []
    for ci in range(len(classes)):
        for snr in TRAIN_SNRS:
            r = rng.choice(ds.cell_indices(ci, snr), FRAMES_TRAIN, replace=False)
            rows.append(np.sort(r))
            ys.append(np.full(FRAMES_TRAIN, ci, dtype=np.int64))
            zs.append(np.full(FRAMES_TRAIN, snr, dtype=np.int16))
    rows = np.concatenate(rows)
    order = np.argsort(rows)
    rows = rows[order]
    y_all = np.concatenate(ys)[order]
    z_all = np.concatenate(zs)[order]
    X_all = np.transpose(ds.X[rows], (0, 2, 1)).astype(np.float32)
    p = np.mean(X_all[:, 0] ** 2 + X_all[:, 1] ** 2, axis=1, keepdims=True)
    X_all /= np.sqrt(p)[:, None] + 1e-12
    print(f"  {X_all.shape}  {X_all.nbytes/1e9:.1f} GB  "
          f"in {time.perf_counter()-t0:.0f}s")

    probes = {}
    for name in CASES:
        if name in results:
            continue
        ci = idx_of[name]
        r = np.sort(np.concatenate([
            rng.choice(ds.cell_indices(ci, s), FRAMES_PROBE, replace=False)
            for s in PROBE_SNRS]))
        P = np.transpose(ds.X[r], (0, 2, 1)).astype(np.float32)
        pp = np.mean(P[:, 0] ** 2 + P[:, 1] ** 2, axis=1, keepdims=True)
        probes[name] = P / (np.sqrt(pp)[:, None] + 1e-12)
    ds.close()

    t_start = time.perf_counter()
    for n, held in enumerate(CASES, 1):
        if held in results:
            print(f"\n[{n}/{len(CASES)}] {held} -- already done, skipping")
            continue

        held_i = idx_of[held]
        keep = y_all != held_i
        kept = [c for c in classes if c != held]
        remap = {idx_of[c]: j for j, c in enumerate(kept)}
        X = X_all[keep]
        y = np.array([remap[v] for v in y_all[keep]], dtype=np.int64)
        z = z_all[keep]

        print(f"\n[{n}/{len(CASES)}] hold out {held} ({family_of(held)})")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        tr, te = cnn.split(X, y, z, test_fraction=0.25, seed=SEED)
        model = model_zoo.backbone(len(kept))
        model = cnn.train_model(model, X[tr], y[tr], X[te], y[te], epochs=EPOCHS)
        in_dist = float((cnn.predict(model, X[te]) == y[te]).mean())

        pred = cnn.predict(model, probes[held])
        share = np.bincount(pred, minlength=len(kept)).astype(float)
        share /= share.sum()
        top = np.argsort(share)[::-1][:3]
        sink = kept[top[0]]
        nb = neighbours_of(held)
        verdict = ("ORDER-ADJACENT" if sink in nb else
                   "same family" if family_of(sink) == family_of(held) else
                   "different family")

        w = weak.get(held, {})
        agree = "MATCHES" if w.get("sink") == sink else "DIFFERS"
        print(f"    in-dist acc {in_dist:.3f}  (cheap run: {w.get('in_dist', float('nan')):.3f})")
        print(f"    sink -> {sink} ({share[top[0]]:.1%})   [{verdict}]")
        print(f"    cheap run said: {w.get('sink','?')} "
              f"({w.get('share',0):.1%})   -> {agree}")
        print(f"    top-3: " + ", ".join(f"{kept[i]} {share[i]:.1%}" for i in top))

        results[held] = dict(sink=sink, share=float(share[top[0]]),
                             verdict=verdict, in_dist=in_dist, agree=agree,
                             top3=[(kept[i], float(share[i])) for i in top])
        CKPT.write_text(json.dumps(results, indent=2))

    print(f"\ntotal {(time.perf_counter()-t_start)/60:.1f} min")

    print("\n" + "=" * 84)
    print(f"{'held out':<10} {'cheap sink':<12} {'strong sink':<12} "
          f"{'strong acc':>10}  {'verdict':<16} agree")
    print("-" * 84)
    n_match = 0
    for held in CASES:
        r = results[held]
        w = weak.get(held, {})
        n_match += r["agree"] == "MATCHES"
        print(f"{held:<10} {w.get('sink','?'):<12} {r['sink']:<12} "
              f"{r['in_dist']:>10.3f}  {r['verdict']:<16} {r['agree']}")
    print("-" * 84)
    print(f"sinks unchanged under proper training: {n_match}/{len(CASES)}")
    same_fam = sum(r["verdict"] != "different family" for r in results.values())
    print(f"still inside own family: {same_fam}/{len(CASES)}")
    print("=" * 84)


if __name__ == "__main__":
    main()
