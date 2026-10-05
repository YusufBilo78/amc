"""Numbers for the 2016 deck, read from the committed result files.

    python deck/extract_2016.py      -> deck/deck2016_data.json

build2016.js reads only that file, so every number on a slide traces back to
an .npz or .json in the repository.
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = {128: "train_backbone_rml2016_f1000_c4_colab_s14.npz",
        64: "train_backbone_rml2016_f1000_c4_L64_colab_s14.npz",
        32: "train_backbone_rml2016_f1000_c4_L32_colab_s14.npz"}
LEVELS = [10, 6, 4, 2, 0, -2, -4, -6]


def c4():
    out = {"lengths": {}}
    for L, f in RUNS.items():
        z = np.load(ROOT / f, allow_pickle=True)
        by = z["confusions_by_snr"]                      # seeds, snr, n, n
        pooled = by.sum(0)
        snrs = z["snrs"].astype(int).tolist()
        wrong = [int(m.sum() - np.trace(m)) for m in pooled]
        high = pooled[[i for i, s in enumerate(snrs) if s >= 10]].sum(0)
        out["lengths"][L] = {
            "wrong_by_snr": wrong,
            "high_wrong": int(high.sum() - np.trace(high)),
            "high_total": int(high.sum()),
            "high_recall": (np.diag(high) / high.sum(1)).round(4).tolist(),
            "high_matrix": high.astype(int).tolist(),
            "floor_pct": round(100 * np.mean([w / pooled[i].sum() for i, w in
                                               enumerate(wrong) if snrs[i] >= 2]), 1),
            "best_epochs": [int(v) for v in z["best_epochs"]],
        }
        if L == 128:
            out["classes"] = [str(c) for c in z["class_names"]]
            out["snrs"] = snrs
            out["seeds"] = int(len(z["overall"]))
            out["per_snr_decisions"] = int(pooled[0].sum())
            out["mats"] = {str(lv): pooled[snrs.index(lv)].astype(int).tolist()
                           for lv in LEVELS}
            m20 = pooled[0]
            dec = m20.sum(0)
            out["m20"] = {
                "decided_share": (dec / dec.sum()).round(4).tolist(),
                "recall": (np.diag(m20) / m20.sum(1)).round(4).tolist(),
                "precision": (np.diag(m20) / np.maximum(dec, 1)).round(4).tolist(),
                "bpsk_share_by_seed": (by[:, 0].sum(1)[:, 0]
                                       / by[:, 0].sum((1, 2))).round(3).tolist(),
            }
    return out


def measured_snr():
    d = json.loads((ROOT / "rml2016_measured_snr.json").read_text())
    lab = np.array(d["labels"])
    m = {c: np.array([d["measured_snr_db"][c][str(s)] for s in lab])
         for c in d["measured_snr_db"]}
    win = (lab >= -6) & (lab <= 4)
    slopes = {c: round(float(np.polyfit(lab[win], m[c][win], 1)[0]), 2)
              for c in ("BPSK", "QPSK", "8PSK")}
    psk = np.mean([m["BPSK"], m["QPSK"], m["8PSK"]], 0)
    pick = [list(lab).index(s) for s in (-8, -6, -4)]
    offsets = {c: round(float(np.mean(m[c][pick] - psk[pick])), 1)
               for c in ("PAM4", "QAM16", "QAM64")}
    at = {str(s): {c: round(float(m[c][list(lab).index(s)]), 1)
                   for c in ("QPSK", "QAM16", "QAM64")} for s in (-6, 0, 2, 10)}
    return {"slopes": slopes, "offsets": offsets, "at": at,
            "unscaled": {"PAM4": 7.0, "QAM16": 10.0, "QAM64": 16.2}}


def methods():
    z = np.load(ROOT / "compare_methods_ICRNNA_es_rml2016_e100.npz")
    rows = []
    for i, name in enumerate(z["methods"]):
        rows.append({"name": str(name), "in": round(float(z["in_domain"][i].mean()), 3),
                     "cross": round(float(z["cross"][i].mean()), 3),
                     "qam16": round(float(z["qam16"][i].mean()), 3)})
    w = np.load(ROOT / "whitening_smoothing_rml2016_e100.npz")
    smooth = {"bins": [int(b) for b in w["bins"][1:]],
              "in": w["in_domain"][1:].mean(1).round(4).tolist(),
              "cross": w["cross"][1:].mean(1).round(4).tolist(),
              "none_in": round(float(w["in_domain"][0].mean()), 4),
              "none_cross": round(float(w["cross"][0].mean()), 4)}
    return {"rows": rows, "seeds": int(len(z["seeds"])), "smooth": smooth}


def faithful():
    a = json.loads((ROOT / "icrnna_faithful_results.json").read_text())
    b = json.loads((ROOT / "icrnna_faithful_e150_results.json").read_text())
    o = json.loads((ROOT / "overfit_2x2.json").read_text())
    icr = next(v for k, v in o.items() if k.startswith("ICRNNA"))
    iq = o["IQNet|fixed"]
    return {"paper": a["paper_target"], "e58": a["summary"]["mean_test_acc"],
            "e150": b["summary"]["mean_test_acc"],
            "e150_best": b["per_seed"][0]["best_epoch"],
            "iqnet_train": round(iq["final_train"], 4), "iqnet_test": round(iq["final_test"], 4),
            "icrnna_train": round(icr["final_train"], 4), "icrnna_test": round(icr["final_test"], 4)}


def literature():
    rows = []
    for name, year, f, params in (
            ("ICRNNA", 2025, "train_backbone_rml2016_f1000_colab.npz", 786_379),
            ("LSTM2", 2018, "train_backbone_rml2016_f1000_LSTM2_colab.npz", 201_099),
            ("MCLDNN", 2020, "train_backbone_rml2016_f1000_MCLDNN_colab.npz", 406_199),
            ("PET-CGDNN", 2021, "train_backbone_rml2016_f1000_PETCGDNN_colab.npz", 71_871),
            ("VT-CNN2", 2016, "train_backbone_rml2016_f1000_VTCNN2_colab.npz", 1_592_383)):
        z = np.load(ROOT / f, allow_pickle=True)
        rows.append({"name": name, "year": year, "params": params,
                     "overall": round(float(z["overall"].mean()), 4),
                     "high": round(float(z["high"].mean()), 4)})
    return rows


if __name__ == "__main__":
    data = {"c4": c4(), "snr": measured_snr(), "methods": methods(), "lit": literature(),
            "faithful": faithful(), "params": {"total4": 784960 + 129 * 4,
                                               "lstm": 659456 + 512}}
    out = pathlib.Path(__file__).parent / "deck2016_data.json"
    out.write_text(json.dumps(data, indent=1))
    print("wrote", out)
    c = data["c4"]
    print({L: (v["high_wrong"], v["high_total"], v["floor_pct"]) for L, v in c["lengths"].items()})
    print(data["snr"], data["faithful"], data["methods"]["rows"])
