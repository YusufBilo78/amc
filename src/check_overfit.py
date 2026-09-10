"""Track train vs test accuracy per epoch to check for overfitting."""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, torch, torch.nn.functional as F
import cnn, domains

DEVICE = cnn.DEVICE
np.random.seed(0); torch.manual_seed(0)

# Real RadioML, the setting that matters. 5 shared classes, full SNR range.
src = domains.RadioMLDomain()
a = src.load(list(domains.SHARED_CLASSES), list(range(-20,31,2)),
             frames_per_cell=512, seed=0)
src.close()
tr, te = cnn.split(a["X"], a["y"], a["z"], test_fraction=0.3, seed=0)
Xtr, ytr = a["X"][tr], a["y"][tr]
Xte, yte = a["X"][te], a["y"][te]
print(f"{len(tr)} train / {len(te)} test, 5 classes\n")

model = cnn.IQNet(5).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
EPOCHS, BS = 30, 256
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
    total_steps=EPOCHS*(len(Xtr)//BS+1))
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type=="cuda")

Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
Xte_t = torch.from_numpy(Xte).to(DEVICE); yte_t = torch.from_numpy(yte).to(DEVICE)

@torch.no_grad()
def acc(X, y, bs=512):
    model.eval(); c=0
    for i in range(0,len(X),bs):
        with torch.amp.autocast("cuda", enabled=DEVICE.type=="cuda"):
            p = model(X[i:i+bs]).argmax(1)
        c += (p==y[i:i+bs]).sum().item()
    return c/len(X)

print(f"{'epoch':>5} {'train_loss':>11} {'train_acc':>10} {'test_acc':>9} {'gap':>7}")
Xtr_g = Xtr_t.to(DEVICE)
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(len(Xtr_t)); tot=0
    for i in range(0,len(perm),BS):
        idx = perm[i:i+BS]
        xb = Xtr_t[idx].to(DEVICE); yb = ytr_t[idx].to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type=="cuda"):
            loss = F.cross_entropy(model(xb), yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item()*len(idx)
    if (ep+1)%3==0 or ep==0:
        tra = acc(Xtr_g, ytr_t.to(DEVICE)); tea = acc(Xte_t, yte_t)
        print(f"{ep+1:>5} {tot/len(perm):>11.4f} {tra:>10.4f} {tea:>9.4f} {tra-tea:>+7.3f}")
