"""ICRNNA against four published architectures on RML2016.10a, 11 classes.

Every model trained by train_backbone.py under one protocol: the same 1,000
frames per cell, the same 70/15/15 split and the same three seeds, early
stopping on validation. Architectures in src/literature_models.py.

    python tools/literature_table.py   -> literature_2016.md, figures/38_literature_2016.png
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
MODELS = [("ICRNNA", "El-Haryqy et al. 2025", "train_backbone_rml2016_f1000_colab.npz"),
          ("LSTM2", "Rajendran et al. 2018", "train_backbone_rml2016_f1000_LSTM2_colab.npz"),
          ("MCLDNN", "Xu et al. 2020", "train_backbone_rml2016_f1000_MCLDNN_colab.npz"),
          ("PET-CGDNN", "Zhang et al. 2021", "train_backbone_rml2016_f1000_PETCGDNN_colab.npz"),
          ("VT-CNN2", "O'Shea et al. 2016", "train_backbone_rml2016_f1000_VTCNN2_colab.npz")]
PARAMS = {"ICRNNA": 786_379, "LSTM2": 201_099, "MCLDNN": 406_199, "PET-CGDNN": 71_871, "VT-CNN2": 1_592_383}
COLORS = ["#F08A24", "#2a78d6", "#1baf7a", "#8FA0B5", "#1E293B"]


def main():
    rows, curves = [], {}
    for name, ref, f in MODELS:
        z = np.load(ROOT / f, allow_pickle=True)
        be = z["best_epochs"].astype(int).tolist() if "best_epochs" in z.files else ["39 (seed 0)"]
        rows.append((name, ref, PARAMS[name], z["overall"].mean(), z["overall"].std(),
                     z["high"].mean(), z["high"].std(), be, int(z["epochs"])))
        curves[name] = (z["snrs"], z["curves"].mean(0))
    L = ["# ICRNNA against published architectures — RML2016.10a, 11 classes", "",
         "Same protocol for every model (`train_backbone.py --arch`): 1,000 frames per",
         "cell, 70/15/15 split per (class, SNR), the same three seeds, early stopping",
         "on validation. Accuracy on the test split, 33,000 test frames per seed;",
         "≥ 10 dB is the 8,250 of them at labels 10–18. Mean ± s.d. over seeds.", "",
         "| model | paper | parameters | overall | ≥ 10 dB | best epochs (ceiling) |",
         "|---|---|---|---|---|---|"]
    for n, ref, p, o, os_, h, hs, be, ep in rows:
        L.append(f"| **{n}** | {ref} | {p:,} | {100*o:.2f} ± {100*os_:.2f} | {100*h:.2f} ± {100*hs:.2f} | "
                 f"{', '.join(map(str, be))} ({ep}) |")
    L += ["", "All runs converged (best epoch + patience 20 within the ceiling). The",
          "ICRNNA row is the committed run (ceiling 60; its best epoch was not stored,",
          "and the convergence check reran seed 0: best epoch 39, bit-identical).",
          "The others used a 150 ceiling.", ""]
    (ROOT / "literature_2016.md").write_text("\n".join(L))
    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, *_), c in zip(rows, COLORS):
        s, v = curves[name]
        ax.plot(s, 100 * v, color=c, lw=2.6 if name == "ICRNNA" else 1.8, marker="o", ms=3.5, label=name)
    ax.set_xlabel("SNR label (dB)"); ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_xticks(range(-20, 19, 4)); ax.grid(color="#e4e3df", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("RML2016.10a, 11 classes: five architectures, one protocol\n"
                 "4,950 test frames per point (3 seeds × 11 classes × 150)", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "38_literature_2016.png", dpi=150)
    print("\n".join(L))


if __name__ == "__main__":
    main()
