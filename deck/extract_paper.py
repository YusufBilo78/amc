"""Numbers for the paper deck (El-Haryqy et al., Results in Engineering 26 (2025)
104783), in one file:

    python deck/extract_paper.py      -> deck/paper_data.json

PAPER holds values transcribed from the paper itself, each with the table,
figure or section it comes from. OURS is read from the committed result files.
build_paper.js reads only paper_data.json.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).parent

PAPER = {
    "cite": "El-Haryqy, Kharbouche, Ouamna, Madini, Zouine. Results in Engineering 26 (2025) 104783",
    # Abstract, Table 3, Table 7
    "headline": {"acc_10a": 63.24, "acc_10b": 65.39, "mflops": 48.42, "params_m": 0.79, "infer_s": 0.2313},
    # Table 2, RML2016.10a column; Section 4.1 for the split
    "training": {"lr": 0.001, "lr_drop": 0.2, "lr_patience": 5, "min_lr": 1e-6, "batch": 32,
                 "dropout": 0.3, "l2": 1e-4, "epochs": 58, "split": [70, 15, 15]},
    # Table 3, RML2016.10a column
    "table3_10a": [["ICRNNA", 63.24], ["CNN-BiLSTM-DNN", 62.73], ["MCLDNN", 61.42], ["MAMCA", 60.79],
                   ["PET-CGDNN", 60.58], ["DAE", 60.22], ["ResNet", 57.10], ["CVCNN", 56.37],
                   ["IC-AMCNet", 56.30], ["CGDNet", 56.21], ["MCNet", 56.20], ["Transformer", 55.34]],
    # Table 5 (RML2016.10a), SNR -10..4 dB
    "table5_snr": [-10, -8, -6, -4, -2, 0, 2, 4],
    "table5": {"ICRNNA": [25.32, 37.45, 55.26, 71.13, 85.11, 92.21, 93.40, 93.02],
               "CNN-BiLSTM-DNN": [27.73, 39.55, 56.18, 69.23, 80.05, 88.68, 91.32, 93.18],
               "MCLDNN": [23.50, 38.09, 56.32, 67.27, 81.05, 88.36, 89.59, 90.86]},
    # Table 6, RML2016.10a: accuracy, F1, precision
    "table6_10a": [["CNN only", 0.5993, 0.6193, 0.7486], ["BiLSTM only", 0.6313, 0.6537, 0.7676],
                   ["No attention", 0.6280, 0.6532, 0.7526], ["Full ICRNNA", 0.6324, 0.6567, 0.7536]],
    # Table 7: parameter column as printed, in "millions"
    "table7_params_m": {"MCLDNN": 406, "PET-CGDNN": 79.2, "ICRNN (Ours)": 0.79},
    # Section 4.2.2 text: overall accuracy quoted there, and Fig. 7 / text: validation accuracy at epoch 58
    "sec422_overall": 63.39, "val_acc_ep58": 63.59,
    # Fig. 8 / Section 4.2.2: accuracy at -20 dB
    "acc_m20": 9.0,
    # Fig. 11, ICRNNA panel, 0 dB: WBFM -> AM-DSB, QAM16 -> QAM64, QAM64 -> QAM16 (percent of row)
    # 8PSK row: 91.9 right, 4.3 QPSK, 1.9 QAM16, 1.2 AM-SSB, 0.6 QAM64; 161 is the only row size
    # that reproduces all five printed values (148 / 7 / 3 / 2 / 1 frames)
    "fig11": {"wbfm_as_amdsb": 57.9, "wbfm": 41.4, "q16_as_q64": 5.7, "q64_as_q16": 6.1, "acc": 92.21,
              "psk8": 91.9, "psk8_row": {"n": 161, "8PSK": 148, "QPSK": 7, "QAM16": 3, "AM-SSB": 2, "QAM64": 1}},
}

# The authors' own code (OFFICIAL_CODE.md), commit ac000a0: layer shapes and
# parameter counts printed from build_improved_model(11) itself, BatchNorm
# statistics included (Keras count); 792,396 of them trainable.
OFFICIAL = {
    "released": "2026-08-08", "commit": "ac000a0",
    "params": 794_188, "trainable": 792_396,
    "blocks": {"CNN": 25_920, "BiLSTM": 659_456, "Attention": 66_177, "Dense": 42_635},
    "shapes": ["2 × 128", "128 × 64", "128 × 128", "64 × 128", "64 × 256", "64 × 256", "256", "128", "64", "11"],
    "train_frames": 154_000, "batch": 32, "max_epochs": 100, "es_patience": 6,
    "lr_patience": 5, "clipnorm": 1.0,
}

# Independently verified parameter counts (LITERATURE_CHECK.md, section 1)
VERIFIED_PARAMS = {"MCLDNN": 406_199, "PET-CGDNN": 71_871}


def ours():
    a = json.loads((ROOT / "icrnna_faithful_results.json").read_text())
    b = json.loads((ROOT / "icrnna_faithful_e150_results.json").read_text())
    snr = json.loads((ROOT / "rml2016_measured_snr.json").read_text())["measured_snr_db"]
    lit = json.loads((HERE / "deck2016_data.json").read_text())["lit"]
    e150 = b["per_seed"][0]
    return {
        "e58_mean": round(a["summary"]["mean_test_acc"], 2), "e58_std": round(a["summary"]["std_test_acc"], 2),
        "e58_best_epochs": [p["best_epoch"] for p in a["per_seed"]], "e58_seeds": len(a["per_seed"]),
        "e150_acc": round(e150["test_acc"], 2), "e150_best_epoch": e150["best_epoch"],
        "e150_ceiling": b["config"]["epochs"], "e150_patience": b["config"]["early_stop_patience"],
        "params": e150["n_params"],
        "per_snr": {int(k): v for k, v in e150["per_snr_acc"].items()},
        "snr_at_label0": {c: round(snr[c]["0"], 1) for c in ("BPSK", "QPSK", "QAM16", "QAM64")},
        "lit": lit,
    }


def official_runs():
    """Our runs of the authors' code (Colab task official_2016), if the
    results file has been copied into the repository yet."""
    f = ROOT / "official_icrnna_2016_results.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    runs = d["per_seed"]
    m0 = [r["confusions_by_snr"][r["snrs"].index(0)] for r in runs]      # 0 dB, every run
    hit0 = sum(m[i][i] for m in m0 for i in range(len(m)))
    all0 = sum(sum(row) for m in m0 for row in m)
    return {"n": len(runs), "mean": d["summary"]["mean_test_acc"], "std": d["summary"]["std_test_acc"],
            "acc0": round(100 * hit0 / all0, 2),
            "accs": [r["test_acc"] for r in runs], "stopped": [r["stopped_epoch"] for r in runs],
            "best": [r["best_epoch"] for r in runs], "split_id": d.get("split_id"),
            # validation accuracy at the restored (best-loss) epoch; the paper's Fig. 7 ends at 63.59
            "val_at_best": [r["history"][r["best_epoch"] - 1]["val_acc"] for r in runs]}


if __name__ == "__main__":
    out = {"paper": PAPER, "verified_params": VERIFIED_PARAMS, "official": OFFICIAL,
           "official_runs": official_runs(), "ours": ours()}
    (HERE / "paper_data.json").write_text(json.dumps(out, indent=1))
    o = out["ours"]
    print("wrote paper_data.json", o["e58_mean"], o["e150_acc"], o["params"], o["snr_at_label0"])
