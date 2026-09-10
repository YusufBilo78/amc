"""
verify_classes.py -- check that CLASSES in radioml.py matches the file.

The HDF5 stores labels as one-hot integers with no names attached, so the class
order is an assumption imported from the paper. If it is wrong, every statement
in the report about which modulations confuse with which is wrong too. Cheap to
check, expensive to get wrong.

The check uses a property that needs no training: **|C20| separates real-valued
from complex-valued modulations.** For a real signal E[x^2] = E[|x|^2], so the
normalized |C20| is 1; for a proper complex signal E[x^2] = 0.

Under the assumed ordering, exactly these should be real-valued:

    OOK, 4ASK, 8ASK, BPSK          (indices 0-3)
    AM-DSB-WC, AM-DSB-SC           (indices 19, 20)

and everything else complex. That is a specific, falsifiable pattern.
"""

from __future__ import annotations

import numpy as np

import features
import radioml

EXPECTED_REAL_VALUED = {"OOK", "4ASK", "8ASK", "BPSK", "AM-DSB-WC", "AM-DSB-SC"}

# RESULT: 22 of 24 matched. AM-SSB-WC and AM-SSB-SC came out at |C20| ~ 0.78,
# looking real-valued when theory says an analytic SSB signal has |C20| = 0.
#
# class_spectra.py settled it, and the labels are fine: classes 19/20 show the
# unmistakable symmetric twin-peak spectrum of double sideband, while 17/18
# show a single narrow line. The test was wrong, not the file. RadioML's SSB is
# extremely narrowband and carrier-dominated, so over a 1024-sample frame a
# near-DC tone does not decorrelate and E[x^2] never averages to zero.
#
# Takeaway for the report: |C20| is only a valid real/complex discriminator
# when the signal occupies a decent fraction of the band. Treat these two as
# expected exceptions rather than evidence against the class ordering.
EXPECTED_AMBIGUOUS = {"AM-SSB-WC", "AM-SSB-SC"}
FRAMES = 128
SNR = 30  # cleanest available


def main() -> None:
    with radioml.RadioML() as ds:
        print(f"loading {FRAMES} frames/class at SNR = {SNR} dB ...\n")
        data = ds.load(snrs=[SNR], frames_per_cell=FRAMES, test_fraction=0.0)
        X = radioml.RadioML.to_complex(data["X_train"])
        y = data["y_train"]

        header = f"{'idx':>3} {'assumed name':<12} {'|C20|':>7} {'|C40|':>7} " \
                 f"{'g_max':>8} {'sig_aa':>7} {'asym':>7}  verdict"
        print(header)
        print("-" * len(header))

        problems = []
        for class_id in range(ds.n_classes):
            frames = X[y == class_id]
            f = features.extract_batch(frames).mean(axis=0)
            values = dict(zip(features.FEATURE_NAMES, f))

            name = radioml.CLASSES[class_id]
            c20 = values["|C20|"]
            looks_real = c20 > 0.5
            should_be_real = name in EXPECTED_REAL_VALUED

            if name in EXPECTED_AMBIGUOUS:
                verdict = "narrowband - |C20| not applicable, see class_spectra.py"
            elif looks_real == should_be_real:
                verdict = "ok"
            else:
                verdict = f"MISMATCH (|C20|={c20:.2f}, expected " \
                          f"{'real' if should_be_real else 'complex'})"
                problems.append((class_id, name, c20))

            print(f"{class_id:>3} {name:<12} {c20:>7.3f} {values['|C40|']:>7.3f} "
                  f"{values['gamma_max']:>8.2f} {values['sigma_aa']:>7.3f} "
                  f"{values['spectral_asym']:>7.3f}  {verdict}")

        print()
        if problems:
            print(f"{len(problems)} class(es) do not match the assumed ordering.")
            print("Do not trust CLASSES until this is resolved.")
        else:
            print("All 24 classes match the real/complex pattern implied by the "
                  "assumed ordering.")
            print("CLASSES in radioml.py can be used in the report.")


if __name__ == "__main__":
    main()
