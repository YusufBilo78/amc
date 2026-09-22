"""Pull every number the deck quotes out of the committed result files.

Run from the repository root:

    python deck/extract_data.py

Writes deck/deck_data.json. build.js reads only that file, so a number on a
slide can always be traced to the .npz it came from. Nothing here is typed in
by hand except the faithful-build figures, which live in JSON files that are
quoted verbatim.
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent / "deck_data.json"


def load(name):
    return np.load(ROOT / name, allow_pickle=True)


D = {}

# Method table: the 100/150-ceiling run, 20/20 converged.
z = load("compare_methods_ICRNNA_es_e100.npz")
D["methods"] = [dict(name=str(m), in_dom=float(z["in_domain"][i].mean()),
                     cross=float(z["cross"][i].mean()), cross_sd=float(z["cross"][i].std()),
                     qam16=float(z["qam16"][i].mean()),
                     epochs=[int(e) for e in z["best_epochs"][i]])
                for i, m in enumerate(z["methods"])]

# The alpha sweep, 25/25.
z = load("whitening_seeds_e100.npz")
D["alpha"] = [dict(alpha=float(a), cross=float(z["cross"][i].mean()),
                   cross_sd=float(z["cross"][i].std()), qam16=float(z["qam16"][i].mean()))
              for i, a in enumerate(z["alphas"])]

# Four-class decision table on 2018, converged at the 150 ceiling.
z = load("train_backbone_rml2018_f2048_c4_colab_e150.npz")
names = [str(c) for c in z["class_names"]]
snrs = z["snrs"].astype(int).tolist()
by = z["confusions_by_snr"]           # (seeds, n_snr, n, n)


def mat(lo, hi):
    pick = [k for k, s in enumerate(snrs) if lo <= s <= hi]
    return by[:, pick].sum(axis=(0, 1)).astype(int).tolist()


pick14 = [k for k, s_ in enumerate(snrs) if s_ >= 14]
M14 = by[:, pick14].sum(axis=(0, 1))
D["c4_from14"] = dict(threshold=14, decisions=int(M14.sum()), errors=int(M14.sum() - np.trace(M14)), levels=len(pick14))
D["c4"] = dict(classes=names, snrs=snrs, curve=z["curves"].mean(0).round(4).tolist(),
               overall=float(z["overall"].mean()), overall_sd=float(z["overall"].std()),
               high=float(z["high"].mean()), best_epochs=[int(e) for e in z["best_epochs"]],
               mats={"high": mat(10, 30), "m8": mat(-8, -8), "m4": mat(-4, -4),
                     "z0": mat(0, 0), "p4": mat(4, 4)},
               recall_by_snr={str(s): [float(by[:, k, i, i].sum() / by[:, k, i].sum())
                                       for i in range(4)] for k, s in enumerate(snrs)})

# 24-class plain classification on 2018.
z = load("train_backbone_rml2018_f512_colab.npz")
D["c24"] = dict(overall=float(z["overall"].mean()), overall_sd=float(z["overall"].std()),
                high=float(z["high"].mean()), high_sd=float(z["high"].std()),
                snrs=z["snrs"].astype(int).tolist(), curve=z["curves"].mean(0).round(4).tolist(),
                classes=[str(c) for c in z["class_names"]])
C = z["confusions"].sum(0)
D["c24"]["recall"] = [float(C[i, i] / C[i].sum()) for i in range(len(C))]

# Paper-faithful build on 2016, read from its two JSON files.
e58 = json.loads((ROOT / "icrnna_faithful_results.json").read_text())
e150 = json.loads((ROOT / "icrnna_faithful_e150_results.json").read_text())
D["faithful"] = dict(e58=round(e58["summary"]["mean_test_acc"], 2),
                     e58_sd=round(e58["summary"]["std_test_acc"], 2),
                     e150=round(e150["summary"]["mean_test_acc"], 2),
                     paper=float(e150["paper_target"]),
                     e150_peak=int(e150["per_seed"][0]["best_epoch"]))

# 11-class run on 2016, per-class recall above 10 dB, for the dataset slide.
z = load("train_backbone_rml2016_f1000_colab.npz")
C = z["confusions"].sum(0)
D["c11_2016"] = dict(classes=[str(c) for c in z["class_names"]], high=float(z["high"].mean()),
                     recall=[float(C[i, i] / C[i].sum()) for i in range(len(C))],
                     n_per_class=int(C[0].sum()))
D["c24"]["n_per_class"] = int(np.load(ROOT / "train_backbone_rml2018_f512_colab.npz", allow_pickle=True)["confusions"].sum(0)[0].sum())

# Four-class rehearsal on 2016.
z = load("train_backbone_rml2016_f1000_c4_colab.npz")
C = z["confusions"].sum(0)
D["c4_2016"] = dict(classes=[str(c) for c in z["class_names"]], mat=C.astype(int).tolist(),
                    high=float(z["high"].mean()), recall=[float(C[i, i] / C[i].sum()) for i in range(4)])

# The sink thread on the current backbone: 24-class leave-one-out and the
# six-class architecture control.
FAMILIES = {"ASK": ["OOK", "4ASK", "8ASK"], "PSK": ["BPSK", "QPSK", "8PSK", "16PSK", "32PSK"],
            "APSK": ["16APSK", "32APSK", "64APSK", "128APSK"],
            "QAM": ["16QAM", "32QAM", "64QAM", "128QAM", "256QAM"],
            "analog": ["AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "FM"],
            "other": ["GMSK", "OQPSK"]}
fam = {c: f for f, cs in FAMILIES.items() for c in cs}
s24 = json.loads((ROOT / "sink_class_24_partial_e60.json").read_text())
held = [c for cs in FAMILIES.values() for c in cs]
rows = [dict(held=h, family=fam[h], sink=s24[h]["sink"], share=s24[h]["share"],
             same=fam[h] == fam[s24[h]["sink"]], second=s24[h]["top3"][1][0],
             second_share=s24[h]["top3"][1][1], best_epoch=s24[h]["best_epoch"],
             converged=s24[h]["converged"]) for h in held]
merged = dict(fam)
for c in FAMILIES["APSK"] + FAMILIES["QAM"]:
    merged[c] = "APSK+QAM"
D["sink24"] = dict(rows=rows, same=sum(r["same"] for r in rows), n=len(rows),
                   chance=sum((len(FAMILIES[fam[h]]) - 1) / 23 for h in held),
                   same_if_apsk_qam_merged=sum(merged[h] == merged[s24[h]["sink"]] for h in held),
                   ceiling=60, patience=10, frames=256)
arch = json.loads((ROOT / "sink_across_arch_e60.json").read_text())
archs = ["IQNet (1D CNN)", "ResNet1D", "GRU", "Transformer", "ICRNNA"]
six = ["8ASK", "32PSK", "128APSK", "256QAM", "FM", "OQPSK"]
D["sink_arch"] = dict(archs=archs, held=six,
                      cells={a: {h: dict(sink=arch[f"{a} | {h}"]["sink"], share=arch[f"{a} | {h}"]["share"],
                                         same=arch[f"{a} | {h}"]["same_family"], converged=arch[f"{a} | {h}"]["converged"])
                                 for h in six} for a in archs},
                      same={a: sum(arch[f"{a} | {h}"]["same_family"] for h in six) for a in archs},
                      chance=sum((len(FAMILIES[fam[h]]) - 1) / 23 for h in six),
                      unconverged=[k for k, r in arch.items() if not r["converged"]])

OUT.write_text(json.dumps(D, indent=1))
print("methods:", [(m["name"], round(m["cross"], 3)) for m in D["methods"]])
print("alpha  :", [(a["alpha"], round(a["cross"], 3)) for a in D["alpha"]])
print("c4 high:", D["c4"]["mats"]["high"], " 0 dB:", D["c4"]["mats"]["z0"])
print("sink24:", D["sink24"]["same"], "/", D["sink24"]["n"], " arch:", D["sink_arch"]["same"])
print("wrote", OUT)
