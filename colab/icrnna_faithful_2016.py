"""
icrnna_faithful_2016.py -- ICRNNA built from the paper, not from a reproduction.

Purpose: calibration. Every architecture in this project so far was written
here, which makes "is this finding an artifact of your own network?" a question
with no good answer. Reproducing a published architecture and hitting its
published number gives one.

Source of the specification:
  N. El-Haryqy, A. Kharbouche, H. Ouamna et al., "Improved automatic modulation
  recognition using deep learning with additive attention", Results in
  Engineering 26 (2025) 104783.
  Architecture from Fig. 2 and Section 3; hyperparameters from Table 2.
  The paper ships no code -- its availability statement reads "Data will be
  made available on request" -- so this is built from the description.

Target to hit: **63.24%** average accuracy on RML2016.10a (Table 3).
Same table for context: ResNet 57.10, MCNet 56.20, IC-AMCNet 56.30,
MCLDNN 61.42, CNN-BiLSTM-DNN 62.73.

How this differs from the reproduction already in model_zoo.py, which was
transcribed from a peer's code. Five real differences were found against the
paper, all corrected here:

  1. conv1 kernel      peer 5,  paper 3
  2. max-pooling       peer one per conv block (2 total), paper one
  3. BatchNorm in RNN  peer one after the LSTM stack, paper one after each layer
  4. attention         peer none, paper dropout 0.1 + LayerNorm on the context
  5. dense head        peer one layer of 128 with dropout 0.5,
                       paper two layers of 128 and 64, both dropout 0.3

Two places where the paper is genuinely ambiguous, flagged rather than papered
over:

  - Pooling count. Section 3 says "two convolutional layers ... followed by
    batch normalization, max pooling, and a dropout layer", singular, and Fig. 2
    lists one pooling box. Read as one pool, so the LSTM sees 64 timesteps.
    N_POOLS below flips it to the other reading (32 timesteps) in one line; if
    the target accuracy is missed, that is the first thing to try.
  - Attention dropout placement. The text says the attention "includes dropout
    with a rate of 0.1" without saying on what. Applied to the attention
    weights here, which is the usual convention.

Usage (Colab):
  CELL 1:  from google.colab import drive; drive.mount('/content/drive')
  CELL 2:  paste this file, run. Pick a GPU runtime.

Prerequisite: RML2016.10a_dict.pkl in Drive. It is the standard DeepSig file
(~600 MB), the same one the 2016 trainer already uses.

Output: per-seed checkpoints and a summary JSON in SAVE_DIR.
"""

import copy
import gc
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ==========================================================================
# Config -- Table 2 of the paper, RML2016.10a column
# ==========================================================================
CFG = {
    "data_path": "/content/drive/MyDrive/RML2016.10a_dict.pkl",
    "save_dir": "/content/drive/MyDrive/RadioML/ICRNNA_faithful",
    "seeds": [42, 43, 44, 45, 46],
    "epochs": 58,            # Table 2: "Number of Epochs 58"
    "batch_size": 32,        # Table 2: 32 for 10a (the peer's code used 256)
    "lr": 1e-3,              # Table 2
    "weight_decay": 1e-4,    # Table 2: "L2 Regularization 1e-4"
    "sched_factor": 0.2,     # Table 2: "decreases by 0.2x"
    "sched_patience": 5,     # Table 2: "if validation loss does not improve
                             #           for 5 epochs" -- note: LOSS, not acc
    "min_lr": 1e-6,          # Table 2
    "early_stop_patience": 15,
    "num_classes": 11,
    "n_pools": 1,            # see the ambiguity note in the docstring
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CFG["save_dir"], exist_ok=True)
print(f"device {DEVICE} | target to reproduce: 63.24% on RML2016.10a")


# ==========================================================================
# Data
# ==========================================================================
def load_radioml(path):
    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in raw})
    idx = {m: i for i, m in enumerate(mods)}
    X, Ym, Ys = [], [], []
    for (mod, snr), samples in raw.items():
        for s in samples:
            X.append(s)
            Ym.append(idx[mod])
            Ys.append(snr)
    return (np.asarray(X, np.float32), np.asarray(Ym, np.int64),
            np.asarray(Ys, np.float32), mods)


print("loading RML2016.10a ...")
X, Y_MOD, Y_SNR, MODS = load_radioml(CFG["data_path"])
print(f"  {X.shape[0]:,} frames, shape {X.shape}, {len(MODS)} classes: {MODS}")


class RMLDataset(Dataset):
    """Per-sample RMS normalisation; the paper normalises raw I/Q, no more."""

    def __init__(self, X, Y, Z):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.Z = torch.from_numpy(Z)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        iq = self.X[i]
        return iq / (torch.sqrt(torch.mean(iq ** 2)) + 1e-8), self.Y[i], self.Z[i]


def build_loaders(seed):
    """70/15/15, stratified over (modulation, SNR) cells rather than
    modulation alone -- otherwise the per-SNR curve rests on uneven support."""
    groups = Y_MOD * 100 + ((Y_SNR + 20) / 2).astype(int)
    idx = np.arange(len(Y_MOD))
    itr, irest = train_test_split(idx, test_size=0.30, stratify=groups,
                                  random_state=seed)
    iva, ite = train_test_split(irest, test_size=0.50,
                                stratify=groups[irest], random_state=seed)
    kw = dict(batch_size=CFG["batch_size"], num_workers=2, pin_memory=True)
    make = lambda s: RMLDataset(X[s], Y_MOD[s], Y_SNR[s])
    return (DataLoader(make(itr), shuffle=True, **kw),
            DataLoader(make(iva), shuffle=False, **kw),
            DataLoader(make(ite), shuffle=False, **kw))


# ==========================================================================
# Model -- Fig. 2 of the paper, layer for layer
# ==========================================================================
class ImprovedAdditiveAttention(nn.Module):
    """
    Additive (Bahdanau) attention over time, plus the two things the paper
    specifies and the peer reproduction omits: dropout 0.1 and LayerNorm on
    the resulting context vector.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, x):                       # (B, T, D)
        w = torch.softmax(self.v(torch.tanh(self.W(x))), dim=1)
        return self.norm((x * self.drop(w)).sum(dim=1))


class ICRNNAFaithful(nn.Module):
    """
    Part-A CNN     conv(64, k3) + BN + ReLU, conv(128, k3) + BN + ReLU,
                   max-pool 2, dropout 0.3
    Part-B RNN     BiLSTM(128) + BN + dropout 0.3, twice
    Part-C attn    improved additive attention, dropout 0.1, LayerNorm
    Part-D DNN     dense 128 + BN + ReLU + dropout 0.3,
                   dense 64 + BN + ReLU + dropout 0.3, softmax
    """

    def __init__(self, num_classes=11, n_pools=1, p=0.3):
        super().__init__()
        conv = [nn.Conv1d(2, 64, 3, padding=1), nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)]
        if n_pools == 2:
            conv.append(nn.MaxPool1d(2))
        conv += [nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128),
                 nn.ReLU(inplace=True), nn.MaxPool1d(2), nn.Dropout(p)]
        self.cnn = nn.Sequential(*conv)

        # Two separate LSTM layers, so BatchNorm can sit after each one as the
        # paper states. A single nn.LSTM(num_layers=2) cannot express that.
        self.lstm1 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p)

        self.attention = ImprovedAdditiveAttention(256, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(64, num_classes))

    def _rnn(self, x, lstm, bn):
        out, _ = lstm(x)
        return self.drop(bn(out.permute(0, 2, 1)).permute(0, 2, 1))

    def forward(self, x):                       # (B, 2, 128)
        z = self.cnn(x).permute(0, 2, 1)        # (B, T, 128)
        z = self._rnn(z, self.lstm1, self.bn1)
        z = self._rnn(z, self.lstm2, self.bn2)
        return self.head(self.attention(z))


# ==========================================================================
# Training
# ==========================================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total, loss_sum / total


@torch.no_grad()
def per_snr_accuracy(model, loader):
    model.eval()
    p, y, s = [], [], []
    for xb, yb, sb in loader:
        p.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
        y.append(yb.numpy())
        s.append(sb.numpy())
    p, y, s = np.concatenate(p), np.concatenate(y), np.concatenate(s)
    return {int(v): round(float((p[s == v] == y[s == v]).mean()) * 100, 2)
            for v in sorted(np.unique(s))}


def run_seed(seed):
    print(f"\n{'=' * 66}\n SEED {seed}\n{'=' * 66}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    tr, va, te = build_loaders(seed)
    model = ICRNNAFaithful(CFG["num_classes"], CFG["n_pools"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_params:,} parameters "
          f"(peer reproduction: 786,379; paper states 0.79M)")

    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                           weight_decay=CFG["weight_decay"])
    # The paper schedules on validation LOSS, so mode="min".
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=CFG["sched_factor"],
        patience=CFG["sched_patience"], min_lr=CFG["min_lr"])

    best_acc, best_state, best_ep, stale = 0.0, None, 0, 0
    t0 = time.time()
    for ep in range(1, CFG["epochs"] + 1):
        model.train()
        for x, y, _ in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            opt.step()

        acc, loss = evaluate(model, va)
        sched.step(loss)
        if acc > best_acc:
            best_acc, best_ep, stale = acc, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:3d} | val {acc:.2f}% loss {loss:.4f} | "
                  f"best {best_acc:.2f}% (ep {best_ep}) | "
                  f"{(time.time() - t0) / 60:.1f} min")
        if stale >= CFG["early_stop_patience"]:
            print(f"  early stop at epoch {ep}")
            break

    model.load_state_dict(best_state)
    test_acc, _ = evaluate(model, te)
    snr_acc = per_snr_accuracy(model, te)
    print(f"  TEST {test_acc:.2f}%   (paper reports 63.24%)")

    torch.save({"model_state_dict": best_state, "best_val_acc": best_acc,
                "best_epoch": best_ep, "seed": seed},
               os.path.join(CFG["save_dir"], f"icrnna_faithful_seed{seed}.pt"))
    del model, opt, sched
    torch.cuda.empty_cache()
    gc.collect()
    return {"seed": seed, "val_acc": round(best_acc, 4),
            "test_acc": round(test_acc, 4), "best_epoch": best_ep,
            "per_snr_acc": snr_acc, "n_params": n_params}


# ==========================================================================
results = []
for sd in CFG["seeds"]:
    results.append(run_seed(sd))
    accs = [r["test_acc"] for r in results]
    out = {"config": CFG, "per_seed": results, "paper_target": 63.24,
           "summary": {"mean_test_acc": round(float(np.mean(accs)), 4),
                       "std_test_acc": round(float(np.std(accs)), 4),
                       "n_seeds": len(accs)}}
    with open(os.path.join(CFG["save_dir"], "icrnna_faithful_results.json"),
              "w") as f:
        json.dump(out, f, indent=2)

m, s = out["summary"]["mean_test_acc"], out["summary"]["std_test_acc"]
print(f"\n{'=' * 66}")
print(f" {len(results)} seeds: {m:.2f}% +- {s:.2f}%   |   paper: 63.24%")
print(f" difference: {m - 63.24:+.2f} points")
if abs(m - 63.24) < 1.5:
    print(" -> within ~1.5 points: the reproduction stands.")
else:
    print(" -> outside 1.5 points. First thing to try: CFG['n_pools'] = 2")
    print("    (the pooling count is the one genuinely ambiguous item).")
print(f" saved to {CFG['save_dir']}")
