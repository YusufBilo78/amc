"""
run_experiment.py -- run the open-work sweeps on Colab.

Usage:
  CELL 1:  from google.colab import drive; drive.mount('/content/drive')
  CELL 2:  paste this file, set TASK below, run. Pick a GPU runtime.

Three tasks, matching the numbered list in README.md "Open work":

  "compare_methods"   item 1 -- finish the last 3 cells of
                      `compare_methods --arch ICRNNA`, which stands at 17/20
  "whitening_seeds"   item 2 -- rerun the alpha sweep with error bars, because
                      alpha=0.75 is unverified for the current backbone
  "faithful_2016"     item 5 -- run the paper-faithful ICRNNA against 63.24%
  "converged_2018"    the open question: the 2018 classification run again,
                      with a ceiling high enough that early stopping decides.
                      2016 turned out to have converged, but with one epoch of
                      margin against the 60-epoch ceiling, and 2018 is the
                      harder problem -- 24 classes and 1024-sample frames
                      against 11 and 128. If it ran to the ceiling its 0.5699
                      is a floor, and so is anything else measured under that
                      ceiling.
  "converged_2016"    done -- the 2016 classification run again, with a
                      ceiling high enough that early stopping decides and the
                      stopping epoch recorded. The existing result used a
                      60-epoch ceiling and never recorded where it stopped, so
                      whether it converged is unknown and its accuracy may be a
                      floor. Writes to its own file; the existing one is left
                      alone either way.
  "faithful_budget"   the question item 5 left open: two of its three seeds
                      peaked at the final epoch and early stopping never
                      fired, so the paper's 58-epoch ceiling was binding.
                      Same build, same everything else, a ceiling high enough
                      for early stopping to decide. One seed, because the
                      faithful run measured a seed spread of 0.07 points
                      against a 1.49-point deficit -- one run is already 21x
                      the noise, and a second buys nothing until the first
                      says the gap moved.

Item 5 needs only the 2016 pickle. The other two read RadioML 2018 and share
the arrangements `run_training.py` already makes: repository on local disk,
dataset staged to local disk, results written to Drive so that a reclaimed
runtime costs nothing but the cell that was in flight.

One thing here is not just plumbing. `compare_methods` is being *resumed*: 17
of its 20 cells were computed on another machine against the complete 21 GB
file. For the 3 that remain to belong in the same table they have to see the
same frames, and `RadioML.load` picks frames by position inside each
(class, SNR) cell -- so the staging keeps **every** frame of the five shared
classes rather than a subsample of them, and the existing .npz is copied into
the output directory first so the finished cells are recognised as done rather
than recomputed. Get either wrong and the run completes, reports numbers, and
quietly puts two different experiments in one file.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

# ==========================================================================
# Config
# ==========================================================================
TASK = "compare_methods"   # "compare_methods" | "whitening_seeds"
                           # "faithful_2016" | "faithful_budget"
                           # "converged_2016" | "converged_2018"

# converged_2016 only. 300 is a ceiling, not a run length. One seed: this asks
# where training stops, not what the accuracy is to three decimals -- if the
# answer is "well short of 60" the existing result stands and nothing more is
# needed, and if it is "at the ceiling" one seed has already settled it.
CONVERGE_EPOCHS = 300
CONVERGE_SEEDS = 1

# converged_2018 only. Lower ceiling than 2016's because an epoch there costs
# about 0.18 min against 0.07, so 200 is already 36 minutes of worst case. 2016
# peaked at 39; if 2018 needs more than 200 that is itself the answer.
CONVERGE_EPOCHS_2018 = 200
FRAMES_PER_CELL_2018_CHECK = 512   # must match the committed run to compare

# faithful_budget only. 150 is a ceiling, not a run length: early stopping at
# patience 15 decides where it actually stops, which is the whole point.
BUDGET_EPOCHS = 150
BUDGET_SEEDS = 1

SEEDS = 5                  # compare_methods: the table is 4 methods x 5 seeds
EPOCHS = 60
PATIENCE = 20

BRANCH = "claude/iacs-modulation-classification-gwwv4m"
REPO = "https://github.com/YusufBilo78/amc.git"
DRIVE = pathlib.Path("/content/drive/MyDrive")
REPO_DIR = pathlib.Path("/content/amc")
OUT_DIR = DRIVE / "amc-results"
STAGE_DIR = pathlib.Path("/content/amc-shared5")

PKL_HINTS_2016 = ("RadioML/RML2016.10a_dict.pkl",
                  "RML2016.10a_dict.pkl",
                  "RadioML2016/RML2016.10a_dict.pkl")
H5_HINTS = ("RadioML/GOLD_XYZ_OSC.0001_1024.hdf5",
            "RadioML2018/GOLD_XYZ_OSC.0001_1024.hdf5",
            "GOLD_XYZ_OSC.0001_1024.hdf5")

# The five classes that mean the same thing in every domain, as ids in
# RadioML 2018's own 24-class numbering. Both sweeps use exactly these.
SHARED_IDS = [3, 4, 5, 12, 14]     # BPSK, QPSK, 8PSK, 16QAM, 64QAM


def hr(title):
    print(f"\n{'=' * 68}\n {title}\n{'=' * 68}")


def sh(cmd, cwd=None):
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def main():
    hr("1. Environment")
    if not DRIVE.exists():
        raise SystemExit("Drive is not mounted. Run CELL 1 first:\n"
                         "  from google.colab import drive; "
                         "drive.mount('/content/drive')")
    import torch

    print(f"torch {torch.__version__}   cuda: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("No GPU. Runtime > Change runtime type > GPU.")
    gpu = torch.cuda.get_device_properties(0)
    print(f"gpu  : {gpu.name}, {gpu.total_memory / 1e9:.1f} GB")
    st = os.statvfs("/content")
    free_gb = st.f_bavail * st.f_frsize / 1e9
    print(f"local disk free: {free_gb:.0f} GB")

    hr("2. Repository")
    if (REPO_DIR / ".git").exists():
        sh(["git", "fetch", "origin", BRANCH], cwd=REPO_DIR)
        sh(["git", "checkout", BRANCH], cwd=REPO_DIR)
        sh(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=REPO_DIR)
    else:
        sh(["git", "clone", "--branch", BRANCH, "--single-branch", REPO,
            str(REPO_DIR)])
    sh(["git", "log", "--oneline", "-1"], cwd=REPO_DIR)
    SRC = REPO_DIR / "src"
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(REPO_DIR / "colab"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    if TASK == "converged_2016":
        hr("3. Did the 2016 run converge, or did the ceiling stop it?")
        print("Same protocol as the committed result -- 11 classes, all 1000")
        print(f"frames per cell, 70/15/15, patience 20 -- with the ceiling at")
        print(f"{CONVERGE_EPOCHS} instead of 60 and the stopping epoch recorded.")
        print("\nIt writes to train_backbone_rml2016_f1000_colab_e"
              f"{CONVERGE_EPOCHS}.npz, so the")
        print("existing result is untouched whichever way this comes out.")
        print("\nAbout 4 s an epoch on 2016, so the full ceiling is ~20 min.\n")

        pkl = None
        for hint in PKL_HINTS_2016:
            if (DRIVE / hint).is_file():
                pkl = DRIVE / hint
                break
        if pkl is None:
            found = list(DRIVE.rglob("RML2016.10a_dict.pkl"))
            if not found:
                raise SystemExit("RML2016.10a_dict.pkl not found in Drive.")
            pkl = found[0]
        print(f"data: {pkl}\n")

        cmd = [sys.executable, "train_backbone.py", "--data", "rml2016",
               "--data-path", str(pkl.parent), "--frames-per-cell", "1000",
               "--epochs", str(CONVERGE_EPOCHS), "--patience", "20",
               "--seeds", str(CONVERGE_SEEDS),
               "--tag", f"colab_e{CONVERGE_EPOCHS}", "--out-dir", str(OUT_DIR)]
        print(" ".join(cmd) + "\n")
        t0 = time.time()
        sh(cmd, cwd=SRC)
        hr(f"Done in {(time.time() - t0) / 60:.1f} min")
        print("Read the per-seed line: 'converged' means the existing 60-epoch")
        print("result stands. 'NOT converged' means it was a floor.")
        return

    if TASK == "faithful_budget":
        hr("3. Is the 58-epoch ceiling what costs the 1.49 points?")
        print(f"Same faithful build, ceiling raised to {BUDGET_EPOCHS}, "
              f"{BUDGET_SEEDS} seed.")
        print("A different budget is no longer the paper's protocol, so this")
        print(f"writes to *_e{BUDGET_EPOCHS}.* and leaves the faithful run's")
        print("results untouched.\n")
        print("Worst case is the full ceiling at about 0.74 min an epoch, so")
        print(f"budget up to {BUDGET_EPOCHS * 0.74 / 60:.1f} h; early stopping")
        print("should land well short of that.\n")
        env = dict(os.environ, AMC_FAITHFUL_EPOCHS=str(BUDGET_EPOCHS),
                   AMC_FAITHFUL_SEEDS=str(BUDGET_SEEDS))
        proc = subprocess.Popen(
            [sys.executable, str(REPO_DIR / "colab" / "icrnna_faithful_2016.py")],
            cwd=REPO_DIR / "colab", env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        if proc.returncode != 0:
            raise SystemExit(f"failed ({proc.returncode})")
        return

    if TASK == "faithful_2016":
        hr("3. Open work item 5 -- the paper-faithful ICRNNA on RML2016.10a")
        print("Needs only the 2016 pickle; it finds it in Drive itself.")
        print("Three seeds, 58 epochs each at batch 32 -- budget about an hour")
        print("a seed. Results and checkpoints go to Drive, and a rerun of this")
        print("cell skips seeds that already finished.\n")
        sh([sys.executable, str(REPO_DIR / "colab" / "icrnna_faithful_2016.py")],
           cwd=REPO_DIR / "colab")
        return

    if TASK == "converged_2018":
        hr("3. Did the 2018 run converge, or did the ceiling stop it?")
        print("Same protocol as the committed result -- 24 classes, "
              f"{FRAMES_PER_CELL_2018_CHECK} frames per cell,")
        print(f"native 1024 samples, 70/15/15, patience 20 -- with the ceiling "
              f"at\n{CONVERGE_EPOCHS_2018} instead of 60 and the stopping "
              f"epoch recorded.\n")
        print("2016 converged at epoch 39, one epoch inside its ceiling. This")
        print("is the harder problem, so it is the one that might not have.\n")

        h5 = None
        for hint in H5_HINTS:
            if (DRIVE / hint).is_file():
                h5 = DRIVE / hint
                break
        if h5 is None:
            raise SystemExit(
                "GOLD_XYZ_OSC.0001_1024.hdf5 not found in Drive. Looked at:\n  "
                + "\n  ".join(str(DRIVE / h) for h in H5_HINTS))
        print(f"found: {h5}  ({h5.stat().st_size / 1e9:.0f} GB)\n")

        import run_training

        stage = pathlib.Path("/content/amc-data")
        run_training.stage_from_hdf5(h5, stage,
                                     FRAMES_PER_CELL_2018_CHECK, free_gb)
        os.environ["AMC_DATA_DIR"] = str(stage)

        cmd = [sys.executable, "train_backbone.py", "--data", "rml2018",
               "--data-path", str(stage),
               "--frames-per-cell", str(FRAMES_PER_CELL_2018_CHECK),
               "--epochs", str(CONVERGE_EPOCHS_2018), "--patience", "20",
               "--seeds", str(CONVERGE_SEEDS),
               "--tag", f"colab_e{CONVERGE_EPOCHS_2018}",
               "--out-dir", str(OUT_DIR)]
        print(" ".join(cmd) + "\n")
        t0 = time.time()
        sh(cmd, cwd=SRC)
        hr(f"Done in {(time.time() - t0) / 60:.1f} min")
        print("Read the per-seed line. 'converged' and the committed 0.5699")
        print("stands; 'NOT converged' and it is a floor, as is anything else")
        print("measured under the same ceiling.")
        return

    # ------------------------------------------------- 2018 for both sweeps
    hr("3. Dataset -- every frame of the five shared classes")
    h5 = None
    for hint in H5_HINTS:
        if (DRIVE / hint).is_file():
            h5 = DRIVE / hint
            break
    if h5 is None:
        raise SystemExit(
            "GOLD_XYZ_OSC.0001_1024.hdf5 not found in Drive. Looked at:\n  "
            + "\n  ".join(str(DRIVE / h) for h in H5_HINTS))
    print(f"found: {h5}  ({h5.stat().st_size / 1e9:.0f} GB)")
    print(f"staging to {STAGE_DIR} -- all 4096 frames per cell for "
          f"{len(SHARED_IDS)} classes, about 4.4 GB.")
    print("Not a subsample: these sweeps must select the same frames a run")
    print("against the complete file would.\n")

    import run_training

    run_training.stage_from_hdf5(h5, STAGE_DIR, None, free_gb,
                                 class_ids=SHARED_IDS)

    # The sweeps run as subprocesses, so an in-process edit of
    # radioml.SEARCH_DIRS would not reach them. AMC_DATA_DIR does, and is
    # inherited by anything launched from here.
    os.environ["AMC_DATA_DIR"] = str(STAGE_DIR)
    print(f"AMC_DATA_DIR={STAGE_DIR}")

    # ----------------------------------------------------------------------
    if TASK == "compare_methods":
        hr("4. Open work item 1 -- the last 3 cells of compare_methods")
        name = "compare_methods_ICRNNA_es.npz"
        src_npz, dst_npz = REPO_DIR / name, OUT_DIR / name

        # Seed the output directory from the repository's copy, unless a
        # previous run of this cell already left a further-along one there.
        if dst_npz.exists():
            import numpy as np

            a = np.load(dst_npz)["in_domain"]
            b = np.load(src_npz)["in_domain"]
            done_a = int(np.count_nonzero(~np.isnan(a)))
            done_b = int(np.count_nonzero(~np.isnan(b)))
            print(f"{name}: {done_a}/{a.size} cells in Drive, "
                  f"{done_b}/{b.size} in the repository")
            if done_b > done_a:
                shutil.copyfile(src_npz, dst_npz)
                print("  repository copy is further along; using that")
        else:
            shutil.copyfile(src_npz, dst_npz)
            print(f"seeded {dst_npz} from the repository "
                  f"({src_npz.stat().st_size / 1024:.0f} KB)")

        cmd = [sys.executable, "compare_methods.py", "--arch", "ICRNNA",
               "--seeds", str(SEEDS), "--epochs", str(EPOCHS),
               "--patience", str(PATIENCE), "--out-dir", str(OUT_DIR)]

    elif TASK == "whitening_seeds":
        hr("4. Open work item 2 -- the alpha sweep, with error bars")
        print("alpha=0.75 was measured on the superseded backbone and one seed")
        print("per point. This reruns all five alphas across several seeds on")
        print("the current backbone, which is what makes the claim checkable.\n")
        cmd = [sys.executable, "whitening_seeds.py",
               "--seeds", str(SEEDS), "--epochs", str(EPOCHS),
               "--patience", str(PATIENCE), "--out-dir", str(OUT_DIR)]
    else:
        raise SystemExit(f"unknown TASK {TASK!r}")

    print(f"results -> {OUT_DIR}  (Drive: survives the runtime)")
    print("Written after every cell. If this runtime dies, rerun this same")
    print("cell -- finished cells are skipped, not repeated.\n")
    print(" ".join(cmd) + "\n")
    t0 = time.time()
    sh(cmd, cwd=SRC)
    hr(f"Done in {(time.time() - t0) / 60:.1f} min")
    print(f"results : {OUT_DIR}")
    print("The .npz is the record worth committing to the repository.")


if __name__ == "__main__":
    main()
