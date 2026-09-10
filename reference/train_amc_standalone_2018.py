"""
train_amc2018_standalone.py — RadioML 2018.01a Egitimi (TEK DOSYA)
===================================================================
2016 standalone trainer'in 2018.01a karsiligi. Kalibrasyon yok, saf
modulasyon siniflandirma egitimi: 24 sinif, GOLD_XYZ hdf5.

Protokol (calibration_2018_multiseed.py / pilot ile ayni):
  - 200K stratified subsample (secim rng'si SABIT 42 -> tum seed'ler
    ayni alt kumeyi kullanir, karsilastirilabilirlik icin)
  - Native 1024-sample sinyal; egitimde RASTGELE 128'lik pencere kirpma
    (augmentation etkisi), val/test'te MERKEZ kirpma
  - ICRNNA mimarisi (num_classes=24), per-sample RMS norm
  - Adam 1e-3, wd 1e-4, ReduceLROnPlateau(0.2, patience 7), batch 256,
    max 100 epoch, early stopping patience 20, split 70/15/15 per-seed

Beklenen dogruluk: ~%44 val (128-crop nedeniyle; tam-1024 modellerle
karistirma — bu bilincli olarak 2016 mimarisiyle uyumlu kirpilmis kurulum).

Kullanim (Colab):
  HUCRE 1: from google.colab import drive; drive.mount('/content/drive')
  HUCRE 2: bu dosyayi yapistir, calistir. GPU runtime (L4) sec.
  Sure: seed basina ~10-15 dk (L4), 5 seed ~1 saat.

Onkosul: /content/drive/MyDrive/RadioML2018/GOLD_XYZ_OSC.0001_1024.hdf5
Cikti: her seed icin checkpoint (.pt) + ozet JSON (Drive'a).
"""

import copy, json, os, time, gc
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================
# 1. CONFIG
# ============================================================
CFG = {
    'h5_path':    '/content/drive/MyDrive/RadioML2018/GOLD_XYZ_OSC.0001_1024.hdf5',
    'save_dir':   '/content/drive/MyDrive/RadioML/Standalone2018',
    'n_subsample': 200_000,
    'crop':       128,
    'seeds':      [42, 43, 44, 45, 46],
    'epochs':     100,
    'batch_size': 256,
    'lr':         1e-3,
    'weight_decay': 1e-4,
    'grad_clip':  5.0,
    'patience':   20,
    'sched_factor': 0.2,
    'sched_patience': 7,
    'min_lr':     1e-6,
    'num_classes': 24,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(CFG['save_dir'], exist_ok=True)
print(f"Device: {device}")

# ============================================================
# 2. VERI — hdf5'ten 200K stratified subsample
# ============================================================
print(f"Loading 2018 subsample ({CFG['n_subsample']:,}) ...")
t0 = time.time()
with h5py.File(CFG['h5_path'], 'r') as f:
    total = f['X'].shape[0]
    rng = np.random.RandomState(42)              # subsample secimi SABIT
    idx = np.sort(rng.choice(total, size=CFG['n_subsample'], replace=False))
    Y_data = f['Y'][idx].argmax(axis=1).astype(np.int64)     # one-hot -> indeks
    Z_data = f['Z'][idx].squeeze(-1).astype(np.float32)      # SNR (dB)
    X_data = np.zeros((CFG['n_subsample'], 1024, 2), dtype=np.float32)
    chunk = 5000
    for start in range(0, CFG['n_subsample'], chunk):
        end = min(start + chunk, CFG['n_subsample'])
        X_data[start:end] = f['X'][idx[start:end]]
X_data = X_data.transpose(0, 2, 1).astype(np.float32)        # [N, 2, 1024]
print(f"  Loaded in {(time.time()-t0)/60:.1f} dk, shape {X_data.shape}, "
      f"{len(np.unique(Y_data))} classes, SNR {Z_data.min():.0f}..{Z_data.max():.0f} dB")

class RML2018Dataset(Dataset):
    """
    1024 -> 128 kirpma: egitimde rastgele pencere (augmentation),
    degerlendirmede merkez pencere. Sonra per-sample RMS norm.
    """
    def __init__(self, X, Y, Z, training=True, crop=128):
        self.X, self.Y, self.Z = X, torch.from_numpy(Y), torch.from_numpy(Z)
        self.training, self.crop, self.full = training, crop, X.shape[-1]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        full = self.X[i]
        if self.training:
            s = np.random.randint(0, self.full - self.crop + 1)
        else:
            s = (self.full - self.crop) // 2
        seg = torch.from_numpy(full[:, s:s + self.crop].copy())
        rms = torch.sqrt(torch.mean(seg ** 2))
        return seg / (rms + 1e-8), self.Y[i], self.Z[i]

def build_loaders(seed):
    idx_all = np.arange(len(X_data))
    itr, ir = train_test_split(idx_all, test_size=0.30,
                               stratify=Y_data, random_state=seed)
    iva, ite = train_test_split(ir, test_size=0.50,
                                stratify=Y_data[ir], random_state=seed)
    kw = dict(batch_size=CFG['batch_size'], num_workers=2, pin_memory=True)
    return (
        DataLoader(RML2018Dataset(X_data[itr], Y_data[itr], Z_data[itr],
                                  training=True, crop=CFG['crop']),
                   shuffle=True, **kw),
        DataLoader(RML2018Dataset(X_data[iva], Y_data[iva], Z_data[iva],
                                  training=False, crop=CFG['crop']),
                   shuffle=False, **kw),
        DataLoader(RML2018Dataset(X_data[ite], Y_data[ite], Z_data[ite],
                                  training=False, crop=CFG['crop']),
                   shuffle=False, **kw),
    )

# ============================================================
# 3. MODEL — ICRNNA (24 sinif)
# ============================================================
class ImprovedAdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.v.weight)
    def forward(self, x):
        w = F.softmax(self.v(torch.tanh(self.W(x))), dim=1)
        return (x * w).sum(dim=1)

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
    def __init__(self, num_classes=24):
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
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.bilstm(x)
        x = self.attention(x)
        return self.classifier(x)

# ============================================================
# 4. EGITIM MOTORU
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
    print(f"\n{'='*66}\n SEED {seed} (2018.01a, 24 sinif, 128-crop)\n{'='*66}")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

    train_loader, val_loader, test_loader = build_loaders(seed)
    model = ICRNNAFaithful(num_classes=CFG['num_classes']).to(device)
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

    torch.save({'model_state_dict': best_state, 'best_val_acc': best_val,
                'best_epoch': best_epoch, 'seed': seed},
               os.path.join(CFG['save_dir'], f'icrnna2018_seed{seed}_best.pt'))

    del model, opt, sch
    torch.cuda.empty_cache(); gc.collect()
    return {'seed': seed, 'val_acc': round(best_val, 4),
            'test_acc': round(test_acc, 4), 'best_epoch': best_epoch,
            'per_snr_acc': snr_acc}

# ============================================================
# 5. 5-SEED KOSUSU + OZET
# ============================================================
results = []
for sd in CFG['seeds']:
    results.append(run_single_seed(sd))
    accs = [r['test_acc'] for r in results]
    out = {'config': {k: v for k, v in CFG.items()},
           'per_seed': results,
           'summary': {'mean_test_acc': round(float(np.mean(accs)), 4),
                       'std_test_acc': round(float(np.std(accs)), 4),
                       'n_seeds': len(accs)}}
    with open(os.path.join(CFG['save_dir'],
                           'icrnna2018_training_results.json'), 'w') as f:
        json.dump(out, f, indent=2)

print(f"\n{'='*66}")
print(f" SONUC (2018.01a, {len(results)} seed): "
      f"{out['summary']['mean_test_acc']:.2f}% +- {out['summary']['std_test_acc']:.2f}%")
print(f" Checkpoint + JSON: {CFG['save_dir']}")
