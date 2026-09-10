"""
overfit_2x2.py -- what actually caused the epoch-18 overfitting?

check_overfit.py measured IQNet reaching train 1.000 / test 0.676 by epoch 30
on RadioML, 5 shared classes, 512 frames per cell. That is one number from one
configuration, and it does not say *why*. Three explanations were live:

  1. architecture     IQNet carries one dropout (0.3) immediately before the
                      output layer and nothing else. ICRNNA drops out after
                      every conv block, inside the LSTM, and at 0.5 in the
                      classifier -- at a comparable parameter count (788k vs
                      899k). If regularisation is the cause, ICRNNA should
                      diverge visibly later, or less.

  2. protocol         OneCycleLR anneals the learning rate to ~0 over a fixed
                      number of epochs. The tail of that schedule is where a
                      model settles hard into whatever it has, memorisation
                      included. ReduceLROnPlateau with early stopping never
                      runs that tail.

  3. data volume      512 frames/cell of the 4096 available. Not varied here;
                      held fixed precisely so the other two are readable, and
                      addressed separately.

So: two architectures x two protocols, everything else identical, on the exact
data configuration check_overfit.py used -- so its number is the anchor that
says whether this harness reproduces what was already measured.

What "protocol" bundles. `fixed` is the original recipe verbatim -- OneCycleLR,
30 epochs, no gradient clipping, final model kept -- so the anchor is exact.
`plateau` is the recipe we would actually adopt: ReduceLROnPlateau(0.2,
patience 7), early stopping (patience 20), gradient clipping at 5.0, and the
best-validation checkpoint kept rather than the last. Three changes travelling
together, deliberately: the question on that axis is "old recipe vs new
recipe", not which of the three does the work. The architecture axis is clean.

Split. 70/15/15 stratified over (class, SNR) cells. Early stopping, the LR
schedule and the checkpoint choice read **validation only**. Test accuracy is
recorded every epoch because the overfitting curve is the object of study here,
but nothing selects on it -- the reported test number is always the one at the
epoch validation chose.

Resumable: results are written to overfit_2x2.json after every configuration.
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
import torch.nn.functional as F

import cnn
import domains
import model_zoo

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
CKPT = ROOT / "overfit_2x2.json"

DEVICE = cnn.DEVICE
FRAMES = 512                       # same as check_overfit.py
SNRS = list(range(-20, 31, 2))
BS = 256
LR = 1e-3
EPOCHS_FIXED = 30                  # the original measurement's length
EPOCHS_MAX = 60                    # cap for the early-stopping protocol
PATIENCE = 20
SEED = 0

ARCHS = ["IQNet", "ICRNNA"]
PROTOCOLS = ["fixed", "plateau"]


def build(name, n):
    return cnn.IQNet(n) if name == "IQNet" else model_zoo.ICRNNA(n)


@torch.no_grad()
def accuracy(model, X, y, bs=512):
    model.eval()
    c = 0
    for i in range(0, len(X), bs):
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            p = model(X[i:i + bs]).argmax(1)
        c += (p == y[i:i + bs]).sum().item()
    return c / len(X)


def run(arch, protocol, data, max_epochs):
    Xtr_t, ytr_t, Xtr_g, ytr_g, Xva_t, yva_t, Xte_t, yte_t = data
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = build(arch, 5).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    early = protocol == "plateau"
    if early:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.2, patience=7, min_lr=1e-6)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR, total_steps=max_epochs * (len(Xtr_t) // BS + 1))

    print(f"\n{'=' * 74}\n {arch} | {protocol}  ({n_params:,} params, "
          f"max {max_epochs} epochs)\n{'=' * 74}")
    print(f"{'ep':>4} {'loss':>8} {'train':>7} {'val':>7} {'test':>7} "
          f"{'gap':>7} {'lr':>9} {'s':>5}")

    curve = []
    best_val, best_ep, best_test, stale = -1.0, 0, 0.0, 0
    t0 = time.perf_counter()

    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), BS):
            idx = perm[i:i + BS]
            xb = Xtr_t[idx].to(DEVICE, non_blocking=True)
            yb = ytr_t[idx].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = F.cross_entropy(model(xb), yb)
            scaler.scale(loss).backward()
            if early:                                  # the new recipe clips
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if not early:
                sched.step()                           # OneCycleLR: per batch
            tot += loss.item() * len(idx)

        tra = accuracy(model, Xtr_g, ytr_g)
        vaa = accuracy(model, Xva_t, yva_t)
        tea = accuracy(model, Xte_t, yte_t)
        lr_now = opt.param_groups[0]["lr"]
        curve.append(dict(epoch=ep, loss=tot / len(perm), train=tra,
                          val=vaa, test=tea, lr=lr_now))

        if early:
            sched.step(vaa)                            # plateau: per epoch
            if vaa > best_val:
                best_val, best_ep, best_test, stale = vaa, ep, tea, 0
            else:
                stale += 1
        elif vaa > best_val:
            best_val, best_ep, best_test = vaa, ep, tea

        if ep % 3 == 0 or ep == 1:
            print(f"{ep:>4} {tot / len(perm):>8.4f} {tra:>7.4f} {vaa:>7.4f} "
                  f"{tea:>7.4f} {tra - tea:>+7.3f} {lr_now:>9.2e} "
                  f"{time.perf_counter() - t0:>5.0f}")

        if early and stale >= PATIENCE:
            print(f"  early stop at epoch {ep} "
                  f"(best val {best_val:.4f} at epoch {best_ep})")
            break

    final = curve[-1]
    # First epoch after which the train-test gap never returns below 0.15 --
    # a stated rule, rather than a number read off the plot by eye.
    onset = None
    for k, c in enumerate(curve):
        if c["train"] - c["test"] >= 0.15 and all(
                d["train"] - d["test"] >= 0.15 for d in curve[k:]):
            onset = c["epoch"]
            break

    return dict(arch=arch, protocol=protocol, n_params=n_params,
                epochs_run=len(curve), minutes=(time.perf_counter() - t0) / 60,
                best_val=best_val, best_epoch=best_ep,
                test_at_best_val=best_test,
                final_train=final["train"], final_test=final["test"],
                final_gap=final["train"] - final["test"],
                overfit_onset_epoch=onset, curve=curve)


def main() -> None:
    probe = "--probe" in sys.argv

    print(f"loading RadioML, 5 classes x {len(SNRS)} SNRs x {FRAMES} frames ...")
    t0 = time.perf_counter()
    src = domains.RadioMLDomain()
    a = src.load(list(domains.SHARED_CLASSES), SNRS, frames_per_cell=FRAMES,
                 seed=SEED)
    src.close()
    tr, va, te = cnn.split(a["X"], a["y"], a["z"],
                           test_fraction=0.15, val_fraction=0.15, seed=SEED)
    print(f"  {len(tr)} train / {len(va)} val / {len(te)} test "
          f"({time.perf_counter() - t0:.0f}s)")

    Xtr_t = torch.from_numpy(a["X"][tr])
    ytr_t = torch.from_numpy(a["y"][tr])
    data = (Xtr_t, ytr_t,
            Xtr_t.to(DEVICE), ytr_t.to(DEVICE),
            torch.from_numpy(a["X"][va]).to(DEVICE),
            torch.from_numpy(a["y"][va]).to(DEVICE),
            torch.from_numpy(a["X"][te]).to(DEVICE),
            torch.from_numpy(a["y"][te]).to(DEVICE))

    if probe:
        print("\nPROBE: 2 epochs of each architecture, for a time estimate.\n")
        for arch in ARCHS:
            r = run(arch, "fixed", data, 2)
            per_ep = r["minutes"] / 2
            print(f"\n  {arch}: {per_ep * 60:.0f}s/epoch  ->  "
                  f"fixed({EPOCHS_FIXED}) {per_ep * EPOCHS_FIXED:.0f} min, "
                  f"plateau(<={EPOCHS_MAX}) <= {per_ep * EPOCHS_MAX:.0f} min")
        return

    results = json.loads(CKPT.read_text()) if CKPT.exists() else {}
    for arch in ARCHS:
        for protocol in PROTOCOLS:
            key = f"{arch}|{protocol}"
            if key in results:
                print(f"skipping {key} (already done)")
                continue
            results[key] = run(arch, protocol, data,
                               EPOCHS_FIXED if protocol == "fixed"
                               else EPOCHS_MAX)
            CKPT.write_text(json.dumps(results, indent=2))

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 92)
    print(f"{'config':<20}{'epochs':>7}{'best val':>10}{'test@best':>11}"
          f"{'final train':>13}{'final test':>12}{'final gap':>11}{'onset':>7}")
    print("-" * 92)
    for arch in ARCHS:
        for protocol in PROTOCOLS:
            r = results[f"{arch}|{protocol}"]
            onset = r["overfit_onset_epoch"]
            print(f"{arch + ' | ' + protocol:<20}{r['epochs_run']:>7}"
                  f"{r['best_val']:>10.4f}{r['test_at_best_val']:>11.4f}"
                  f"{r['final_train']:>13.4f}{r['final_test']:>12.4f}"
                  f"{r['final_gap']:>+11.3f}"
                  f"{(str(onset) if onset else '--'):>7}")
    print("=" * 92)
    print("onset = first epoch after which the train-test gap stays >= 0.15")
    print("anchor: check_overfit.py measured IQNet/fixed at train 1.000 / "
          "test 0.676, gap +0.324 by epoch 30")

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    colors = {"IQNet": "#2980b9", "ICRNNA": "#c0392b"}
    for ax, protocol in zip(axes, PROTOCOLS):
        for arch in ARCHS:
            c = results[f"{arch}|{protocol}"]["curve"]
            ep = [d["epoch"] for d in c]
            ax.plot(ep, [d["train"] for d in c], "-", color=colors[arch],
                    lw=2, label=f"{arch} train")
            ax.plot(ep, [d["test"] for d in c], "--", color=colors[arch],
                    lw=2, alpha=0.75, label=f"{arch} test")
        ax.set_title(f"protocol: {protocol}", fontsize=12, fontweight="bold")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1.02)
    fig.suptitle("Where does the train-test gap come from: architecture or "
                 "training recipe?\nRadioML, 5 classes, 512 frames/cell, "
                 "70/15/15", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "28_overfit_2x2.png", dpi=140)
    plt.close(fig)
    print("\nwrote figures/28_overfit_2x2.png")


if __name__ == "__main__":
    main()
