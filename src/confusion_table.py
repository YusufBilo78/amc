"""
confusion_table.py -- the decision table, printed from a finished run.

The question this answers is the one that gets asked first in a review and is
usually answered with a single averaged number instead: we put N signals of a
known modulation in, and how often did the classifier say each of the possible
answers? Row = what was transmitted, column = what was decided, and every row
sums to the number of frames of that class that were tested. That is a
confusion matrix, and it is what an accuracy figure is an average over.

Nothing here trains anything. It reads the confusion counts that
`train_backbone.py` already stores in its .npz, pools the seeds, and prints
them. So the table for a run that has finished costs nothing to produce.

    python confusion_table.py ../train_backbone_rml2016_f1000_colab.npz
    python confusion_table.py ../train_backbone_rml2018_f512_colab.npz \
        --classes BPSK,QPSK,16QAM,64QAM
    python confusion_table.py ../train_backbone_rml2016_f1000_colab.npz \
        --counts --csv ../confusion_2016.csv
    python confusion_table.py ../train_backbone_rml2018_f2048_c4_colab.npz \
        --snr 0                      # one level
    python confusion_table.py ../train_backbone_rml2018_f2048_c4_colab.npz \
        --snr=-4:4                   # a range, inclusive. Note the "=": a
                                     # value starting with "-" is read as a
                                     # flag otherwise

**On --snr.** The default table pools every level at or above the run's
threshold, which is the headline. On an easy problem that table saturates --
four classes at 1024 samples make two errors in forty thousand decisions above
10 dB -- and the table that shows where decisions go is the one at 0 dB.
Runs made since `confusions_by_snr` was added carry every level; older runs
carry only the pooled matrix, and `eval_by_snr.py` rebuilds the rest from
their checkpoints.

**On --classes.** Selecting four rows out of a twenty-four class run is not the
same experiment as training a four-class classifier. The model printed here
still had 24 answers available to it, so the four rows do not sum to 100% over
the four columns; what falls outside goes to the `other` column, and it is
reported rather than renormalised away. A real four-class experiment is
`train_backbone.py --classes ...`, which gives the model four answers to choose
between. Both are worth having and they are not interchangeable.

The counts are pooled over seeds, which is deliberate: three seeds of 750
frames is 2,250 decisions for that row, and the frequency of a rare confusion
is exactly what needs the larger denominator. Per-seed spread is in the
`overall` and `high` arrays of the same file.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def load(path: pathlib.Path):
    z = np.load(path, allow_pickle=True)
    names = [str(s) for s in z["class_names"]]
    conf = z["confusions"]
    return conf, names, z


def render(conf, names, rows, counts, threshold, n_seeds):
    """The table as a list of lines. `rows` indexes which classes to print."""
    subset = len(rows) < len(names)
    cols = rows if subset else list(range(len(names)))
    head = [names[j] for j in cols] + (["other"] if subset else [])

    w = max(len(names[i]) for i in rows)
    cw = max(7, max(len(h) for h in head) + 1)

    out = []
    out.append("".ljust(w + 2) + "".join(f"{h:>{cw}s}" for h in head)
               + "   recall        n")
    for i in rows:
        total = conf[i].sum()
        vals = [conf[i, j] for j in cols]
        if subset:
            vals.append(total - sum(vals))
        if counts:
            cells = "".join(f"{v:{cw}d}" for v in vals)
        else:
            cells = "".join(f"{v / total * 100:{cw}.1f}" for v in vals)
        recall = conf[i, i] / total * 100
        out.append(f"{names[i]:>{w}s}  {cells}   {recall:5.1f}%  {total:7,d}")
    return out


def sentences(conf, names, rows, limit=6):
    """The largest off-diagonal entries, said in words rather than read off."""
    said = []
    for i in rows:
        total = conf[i].sum()
        for j in np.argsort(conf[i])[::-1]:
            if j == i or conf[i, j] == 0:
                continue
            said.append((conf[i, j] / total, names[i], names[j],
                         int(conf[i, j]), int(total)))
    said.sort(reverse=True)
    return [f"  {frac * 100:5.1f}%  {a} decided as {b}  ({k:,} of {n:,})"
            for frac, a, b, k, n in said[:limit]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="a .npz written by train_backbone.py")
    p.add_argument("--classes", default=None,
                   help="comma-separated subset of class names to print as "
                        "rows. Read the note in the module docstring first: "
                        "this narrows the table, it does not narrow the "
                        "decision the model was making")
    p.add_argument("--counts", action="store_true",
                   help="print integer decision counts instead of row "
                        "percentages")
    p.add_argument("--csv", default=None, help="also write the counts here")
    p.add_argument("--snr", default=None,
                   help="print the table at one SNR level (\"0\") or over an "
                        "inclusive range instead of pooled above the run's "
                        "threshold. A range starting below zero needs the "
                        "\"=\" form, --snr=-4:4, or the value is read as a "
                        "flag. Needs confusions_by_snr in the file")
    args = p.parse_args()

    path = pathlib.Path(args.npz)
    conf_per_seed, names, z = load(path)
    snrs = z["snrs"].astype(int)
    threshold = int(z["high_snr_threshold"])

    if args.snr is not None:
        if "confusions_by_snr" not in z.files:
            raise SystemExit(
                f"{path.name} has no per-SNR confusions -- it predates the "
                "key. Rebuild them from the run's checkpoints:\n"
                f"    python eval_by_snr.py {path} --data-path <dir>")
        lo, _, hi = args.snr.partition(":")
        lo = int(lo)
        hi = int(hi) if hi else lo
        pick = (snrs >= lo) & (snrs <= hi)
        if not pick.any():
            raise SystemExit(f"--snr {args.snr}: no level in "
                             f"{snrs.min()}..{snrs.max()} matches")
        conf_per_seed = z["confusions_by_snr"][:, pick].sum(axis=1)
        which = (f"SNR = {lo} dB" if lo == hi
                 else f"SNR {lo}..{hi} dB ({pick.sum()} levels)")
    else:
        which = f"SNR >= {threshold} dB"

    conf = conf_per_seed.sum(axis=0)
    n_seeds = conf_per_seed.shape[0]

    if args.classes:
        wanted = [c.strip() for c in args.classes.split(",")]
        missing = [c for c in wanted if c not in names]
        if missing:
            raise SystemExit(
                f"not in this run: {', '.join(missing)}\n"
                f"available: {', '.join(names)}")
        rows = [names.index(c) for c in wanted]
    else:
        rows = list(range(len(names)))

    print(f"\n{path.name}")
    print(f"{str(z['dataset'])}, {len(names)} classes, {n_seeds} seeds pooled, "
          f"{which}")
    print(f"{conf.sum():,} decisions in total, "
          f"{conf.sum(axis=1)[rows[0]]:,} per class\n")

    # A table from an under-trained run is a table of floors, and that is not
    # visible in the counts. Say it here rather than leaving it in a log
    # nobody kept.
    if "best_epochs" in z.files and "patience" in z.files:
        best = z["best_epochs"]
        ceiling, patience = int(z["epochs"]), int(z["patience"])
        short = [int(b) for b in best
                 if not np.isnan(b) and b + patience > ceiling]
        peaks = ", ".join("?" if np.isnan(b) else str(int(b)) for b in best)
        if short:
            print(f"WARNING: {len(short)} of {len(best)} seeds did not "
                  f"converge -- peaks at {peaks} under a\n"
                  f"{ceiling}-epoch ceiling with patience {patience}, so "
                  f"{max(short)} needed {max(short) + patience}. Read every "
                  f"number below as a floor.\n")
        else:
            print(f"converged: peaks at epoch {peaks}, ceiling {ceiling}, "
                  f"patience {patience}\n")

    print("row = transmitted, column = decided\n")

    for line in render(conf, names, rows, args.counts, threshold, n_seeds):
        print(line)

    print("\nwhere the errors go:")
    for line in sentences(conf, names, rows):
        print(line)

    correct = sum(conf[i, i] for i in rows)
    tested = sum(conf[i].sum() for i in rows)
    print(f"\nover these {len(rows)} classes: {correct:,} of {tested:,} "
          f"correct = {correct / tested * 100:.2f}%")

    if args.csv:
        out = pathlib.Path(args.csv)
        with out.open("w", encoding="utf-8") as fh:
            fh.write("transmitted," + ",".join(names) + "\n")
            for i in rows:
                fh.write(names[i] + "," +
                         ",".join(str(int(v)) for v in conf[i]) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
