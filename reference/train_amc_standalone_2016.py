"""
train_amc_standalone.py — Modulasyon Siniflandirma Egitimi (TEK DOSYA)
=======================================================================
final_full_cell1-4'un birlestirilmis, kendi basina calisan hali.
Kalibrasyon YOK — sadece normal egitim: RadioML 2016.10a uzerinde
modulasyon tipini siniflandirmayi ogrenir.

ISAIA paper protokolunun birebir aynisi:
  Adam lr=1e-3, weight decay 1e-4, ReduceLROnPlateau(0.2, patience 7),
  batch 256, max 200 epoch, early stopping patience 20,
  70/15/15 (modulasyon x SNR) stratified split, per-seed,
  per-sample RMS normalizasyonu.

Kullanim (Colab):
  HUCRE 1: from google.colab import drive; drive.mount('/content/drive')
  HUCRE 2: bu dosyayi yapistir, calistir.
  MODEL_NAME = 'icrnna' (786K param, ~63%) veya 'simplecnn' (43K, ~57%).

Cikti: her seed icin en iyi checkpoint (.pt) + ozet JSON (Drive'a).
"""

import copy, json, math, os, pickle, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================
# 1. CONFIG  (cell1)
# ============================================================
CFG = {
    'data_path':  '/content/drive/MyDrive/RML2016.10a_dict.pkl',
    'save_dir':   '/content/drive/MyDrive/RadioML/Standalone',
    'model':      'icrnna',          # 'icrnna' | 'simplecnn'
    'seeds':      [42, 43, 44, 45, 46],
    'epochs':     200,
    'batch_size': 256,
    'lr':         1e-3,
    'weight_decay': 1e-4,
    'grad_clip':  5.0,
    'patience':   20,                # early stopping
    'sched_factor': 0.2,
    'sched_patience': 7,
    'min_lr':     1e-6,
    'split_ratio': (0.70, 0.15, 0.15),
    'normalize_rms': True,
    'num_classes': 11,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(CFG['save_dir'], exist_ok=True)
print(f"Device: {device}  |  Model: {CFG['model']}")

# ============================================================
# 2. VERI  (cell2)
# ============================================================
def load_radioml(path):
    with open(path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
    mod_classes = sorted(set(k[0] for k in raw.keys()))
    mod_to_idx = {m: i for i, m in enumerate(mod_classes)}
    X, Ym, Ys = [], [], []
    for (mod, snr), samples in raw.items():
        for s in samples:
            X.append(s); Ym.append(mod_to_idx[mod]); Ys.append(snr)
    return (np.array(X, dtype=np.float32), np.array(Ym, dtype=np.int64),
            np.array(Ys, dtype=np.float32), mod_classes)

print("Loading RadioML 2016.10a ...")
X, Y_mod, Y_snr, MOD_CLASSES = load_radioml(CFG['data_path'])
print(f"  {X.shape[0]:,} samples  {X.shape}  classes: {MOD_CLASSES}")

def stratified_split(Y_mod, Y_snr, split_ratio, seed):
    """70/15/15, (modulasyon x SNR) gruplariyla stratified."""
    groups = Y_mod * 100 + ((Y_snr + 20) / 2).astype(int)
    train_r, val_r, test_r = split_ratio
    idx = np.arange(len(Y_mod))
    idx_train, idx_rest = train_test_split(
        idx, test_size=(val_r + test_r), stratify=groups, random_state=seed)
    val_frac = val_r / (val_r + test_r)
    idx_val, idx_test = train_test_split(
        idx_rest, test_size=(1.0 - val_frac),
        stratify=groups[idx_rest], random_state=seed)
    return idx_train, idx_val, idx_test

class RadioMLDataset(Dataset):
    """Per-sample RMS normalizasyonu; augmentation yok (ISAIA final protokolu)."""
    def __init__(self, X, Y_mod, Y_snr, normalize_rms=True):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y_mod)
        self.S = torch.from_numpy(Y_snr)
        self.normalize_rms = normalize_rms
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        iq = self.X[i].clone()
        if self.normalize_rms:
            rms = torch.sqrt(torch.mean(iq ** 2))
            iq = iq / (rms + 1e-8)
        return iq, self.Y[i], self.S[i]

def build_loaders(seed):
    itr, iva, ite = stratified_split(Y_mod, Y_snr, CFG['split_ratio'], seed)
    kw = dict(batch_size=CFG['batch_size'], num_workers=2, pin_memory=True)
    return (
        DataLoader(RadioMLDataset(X[itr], Y_mod[itr], Y_snr[itr],
                                  CFG['normalize_rms']), shuffle=True, **kw),
        DataLoader(RadioMLDataset(X[iva], Y_mod[iva], Y_snr[iva],
                                  CFG['normalize_rms']), shuffle=False, **kw),
        DataLoader(RadioMLDataset(X[ite], Y_mod[ite], Y_snr[ite],
                                  CFG['normalize_rms']), shuffle=False, **kw),
    )

# ============================================================
# 3. MODELLER  (cell3)
# ============================================================
class ImprovedAdditiveAttention(nn.Module):
    """Bahdanau tarzi additive attention, Xavier init."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.v.weight)
    def forward(self, x):                       # [B, L, H]
        w = F.softmax(self.v(torch.tanh(self.W(x))), dim=1)
        return (x * w).sum(dim=1)               # [B, H]

class BiLSTMWithBN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.bn = nn.BatchNorm1d(hidden_size * 2)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.permute(0, 2, 1)
        out = self.bn(out)
        return out.permute(0, 2, 1)

class ICRNNAFaithful(nn.Module):
    """
    El-Haryqy 2025 ICRNNA'nin sadik reproduksiyonu (KARAR-016 eklemeleriyle:
    Conv sonrasi Dropout, BiLSTM sonrasi BN, FC iceri BN).
    786,379 parametre, RadioML 2016.10a'da ~63.0%.
    """
    def __init__(self, num_classes=11):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 64, 5, padding=2), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.MaxPool1d(2), nn.Dropout(0.3))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.MaxPool1d(2), nn.Dropout(0.3))
        self.bilstm = BiLSTMWithBN(128, 128, num_layers=2, dropout=0.3)
        self.attention = ImprovedAdditiveAttention(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(128, num_classes))
    def forward(self, x):                       # [B, 2, 128]
        x = self.conv1(x)                       # [B, 64, 64]
        x = self.conv2(x)                       # [B, 128, 32]
        x = x.permute(0, 2, 1)                  # [B, 32, 128]
        x = self.bilstm(x)                      # [B, 32, 256]
        x = self.attention(x)                   # [B, 256]
        return self.classifier(x)               # [B, 11] logits

class SimpleCNN(nn.Module):
    """2-Conv + GAP + FC kontrol modeli. 43K parametre, ~56.9%."""
    def __init__(self, num_classes=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, 5, padding=2), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.classifier = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(128, num_classes))
    def forward(self, x):
        return self.classifier(self.features(x).mean(dim=2))

MODELS = {'icrnna': ICRNNAFaithful, 'simplecnn': SimpleCNN}

# ============================================================
# 4. EGITIM MOTORU  (cell4)
# ============================================================
def evaluate(model, loader):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            c += model(x).max(1)[1].eq(y).sum().item()
            t += x.size(0)
    return 100.0 * c / t

def per_snr_accuracy(model, loader):
    model.eval()
    preds, ys, snrs = [], [], []
    with torch.no_grad():
        for x, y, s in loader:
            preds.append(model(x.to(device)).max(1)[1].cpu().numpy())
            ys.append(y.numpy()); snrs.append(s.numpy())
    p = np.concatenate(preds); y = np.concatenate(ys); s = np.concatenate(snrs)
    return {int(snr): round(float((p[s == snr] == y[s == snr]).mean()) * 100, 2)
            for snr in sorted(np.unique(s).astype(int))}

def run_single_seed(seed):
    print(f"\n{'='*66}\n SEED {seed} ({CFG['model']})\n{'='*66}")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    train_loader, val_loader, test_loader = build_loaders(seed)
    model = MODELS[CFG['model']](num_classes=CFG['num_classes']).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametre: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=CFG['lr'],
                           weight_decay=CFG['weight_decay'])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=CFG['sched_factor'],
        patience=CFG['sched_patience'], min_lr=CFG['min_lr'])
    ce = nn.CrossEntropyLoss()

    best_val, best_state, best_epoch, pc = 0.0, None, 0, 0
    t0 = time.time()
    for ep in range(1, CFG['epochs'] + 1):
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            opt.step()

        va = evaluate(model, val_loader)
        sch.step(va)
        if va > best_val:
            best_val, best_state, best_epoch, pc = va, copy.deepcopy(model.state_dict()), ep, 0
        else:
            pc += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d} | val {va:.2f}% | best {best_val:.2f}% "
                  f"(ep {best_epoch}) | {(time.time()-t0)/60:.1f} dk")
        if pc >= CFG['patience']:
            print(f"  early stop @ ep {ep}")
            break

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader)
    snr_acc = per_snr_accuracy(model, test_loader)
    print(f"  TEST: {test_acc:.2f}%  (val {best_val:.2f}%, ep {best_epoch})")

    ckpt_path = os.path.join(CFG['save_dir'],
                             f"{CFG['model']}_seed{seed}_best.pt")
    torch.save({'model_state_dict': best_state, 'best_val_acc': best_val,
                'best_epoch': best_epoch, 'seed': seed}, ckpt_path)

    del model, opt, sch
    torch.cuda.empty_cache(); gc.collect()
    return {'seed': seed, 'val_acc': round(best_val, 4),
            'test_acc': round(test_acc, 4), 'best_epoch': best_epoch,
            'per_snr_acc': snr_acc, 'n_params': n_params}

# ============================================================
# 5. 5-SEED KOSUSU + OZET
# ============================================================
results = []
for sd in CFG['seeds']:
    results.append(run_single_seed(sd))
    out = {'config': {k: v for k, v in CFG.items()}, 'per_seed': results}
    accs = [r['test_acc'] for r in results]
    out['summary'] = {'mean_test_acc': round(float(np.mean(accs)), 4),
                      'std_test_acc': round(float(np.std(accs)), 4),
                      'n_seeds': len(accs)}
    with open(os.path.join(CFG['save_dir'],
                           f"{CFG['model']}_training_results.json"), 'w') as f:
        json.dump(out, f, indent=2)

print(f"\n{'='*66}")
print(f" SONUC ({CFG['model']}, {len(results)} seed): "
      f"{out['summary']['mean_test_acc']:.2f}% +- {out['summary']['std_test_acc']:.2f}%")
print(f" Checkpoint + JSON: {CFG['save_dir']}")
