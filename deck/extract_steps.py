"""Numbers for the step-by-step deck (answers to Moshe's questions of the
2026-10 meeting), in one file:

    python deck/extract_steps.py      -> deck/steps_data.json

Everything is read from committed result files; nothing is retrained.
  four-signal network  train_backbone_rml2016_f1000_c4_colab_s14.npz (4 outputs, 14 seeds)
  eleven-signal network train_backbone_rml2016_f1000_colab.npz       (11 outputs, 3 seeds)
The eleven-signal rows for BPSK/QPSK/QAM16/QAM64 are that network's test
decisions on those four signals: what happens when the four are fed to a
detector that has all eleven outputs.
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).parent
FOUR = ["BPSK", "QPSK", "QAM16", "QAM64"]


def main():
    d16 = json.loads((HERE / "deck2016_data.json").read_text())
    c4 = d16["c4"]
    m10 = c4["mats"]["10"]

    z = np.load(ROOT / "train_backbone_rml2016_f1000_colab.npz", allow_pickle=True)
    cls = [str(c) for c in z["class_names"]]
    snrs = z["snrs"].astype(int).tolist()
    m11 = z["confusions_by_snr"].sum(0)[snrs.index(10)]
    rows11 = []
    for c in FOUR:
        r = m11[cls.index(c)]
        inside = [int(r[cls.index(k)]) for k in FOUR]
        outside = {cls[j]: int(v) for j, v in enumerate(r) if v and cls[j] not in FOUR}
        rows11.append({"name": c, "inside": inside, "outside": outside, "total": int(r.sum())})
    errors11 = sum(r["total"] - r["inside"][i] for i, r in enumerate(rows11))
    left11 = sum(sum(r["outside"].values()) for r in rows11)

    q = FOUR.index("QAM16")
    col = sum(row[q] for row in m10)
    det = {"sent": sum(m10[q]), "hit": m10[q][q], "miss": sum(m10[q]) - m10[q][q],
           "fa": col - m10[q][q], "called": col, "others": sum(map(sum, m10)) - sum(m10[q])}

    frames = 4 * 20 * 1000
    out = {
        "c4": {"m10": m10, "classes": FOUR, "seeds": c4["seeds"], "best_epochs": c4["lengths"]["128"]["best_epochs"],
               "high_total": c4["lengths"]["128"]["high_total"], "params": d16["params"]["total4"],
               "frames": frames, "train": int(frames * 0.70), "val": int(frames * 0.15), "test": int(frames * 0.15)},
        "c11": {"rows": rows11, "seeds": int(len(z["overall"])), "per_row": rows11[0]["total"],
                "errors": int(errors11), "left": int(left11), "decisions": int(sum(r["total"] for r in rows11))},
        "qam16_10db": det,
    }
    (HERE / "steps_data.json").write_text(json.dumps(out, indent=1))
    print("wrote steps_data.json", out["c11"]["errors"], out["c11"]["left"], det)


if __name__ == "__main__":
    main()
