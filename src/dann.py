"""
dann.py -- domain-adversarial training as a comparison baseline.

Ganin et al., "Domain-Adversarial Training of Neural Networks" (JMLR 2016), and
the line of AMC papers built on it (e.g. arXiv:2508.06829). A shared feature
extractor feeds two heads: a label classifier, and a domain discriminator
reached through a **gradient reversal layer**. The discriminator learns to tell
source from target; the reversal makes the extractor learn features that defeat
it. What survives is meant to be domain-invariant.

The asymmetry that matters for this project
-------------------------------------------
DANN needs **target-domain samples at training time** -- unlabelled, but
present. Spectral whitening needs none: it is a fixed transform applied to
whatever arrives. So this is not a like-for-like comparison and should not be
presented as one. DANN is given strictly more information. If whitening merely
matches it, that is already the more deployable result; a receiver meeting a
transmitter it has never seen cannot collect target data first.

On implementing a baseline honestly
-----------------------------------
A baseline implemented badly flatters whatever it is compared against. This
follows the standard recipe rather than a minimal one: the published lambda
schedule 2/(1+exp(-10p)) - 1 ramping over training, a two-layer discriminator,
and batches split evenly between domains. It is still a generic DANN and not a
version tuned for modulation classification, which is stated as a limitation
rather than glossed over.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cnn


class GradientReversal(torch.autograd.Function):
    """Identity forwards; negated and scaled gradient backwards."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd: float):
    return GradientReversal.apply(x, lambd)


class DANNNet(nn.Module):
    """
    IQNet's convolutional stack, with a label head and a domain head.

    The feature extractor is deliberately identical to the one used everywhere
    else in this project, so any difference in the results comes from the
    training objective and not from a different network.
    """

    def __init__(self, n_classes: int, width: int = 64):
        super().__init__()
        base = cnn.IQNet(n_classes, width=width)
        self.features = base.features
        feat_dim = width * 4

        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.label_head = nn.Sequential(nn.Dropout(0.3),
                                        nn.Linear(feat_dim, n_classes))
        self.domain_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, 1),
        )

    def embed(self, x):
        return self.pool(self.features(x))

    def forward(self, x):
        return self.label_head(self.embed(x))

    def domain_logits(self, z, lambd: float):
        return self.domain_head(grad_reverse(z, lambd)).squeeze(1)


def lambda_schedule(progress: float) -> float:
    """Published ramp: 0 at the start, ->1 as training proceeds."""
    return 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0


def train_dann(model, Xs, ys, Xt, Xs_val, ys_val, epochs=15, batch_size=256,
               lr=1e-3, verbose_every=5):
    """
    Xs, ys : labelled source frames
    Xt     : target frames, labels never used
    """
    device = cnn.DEVICE
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # A full batch is drawn from each domain, as in the original formulation,
    # so one step consumes batch_size source frames. Deriving the schedule
    # length from anything else silently overruns it.
    steps_per_epoch = (len(Xs) + batch_size - 1) // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps_per_epoch)

    Xs_t = torch.from_numpy(Xs)
    ys_t = torch.from_numpy(ys)
    Xt_t = torch.from_numpy(Xt)
    Xv = torch.from_numpy(Xs_val).to(device)
    yv = torch.from_numpy(ys_val).to(device)

    total_steps = epochs * steps_per_epoch
    step = 0

    for epoch in range(epochs):
        model.train()
        perm_s = torch.randperm(len(Xs_t))
        lab_sum = dom_sum = 0.0
        for i in range(0, len(perm_s), batch_size):
            idx_s = perm_s[i : i + batch_size]
            if len(idx_s) < 2 or step >= total_steps:
                continue
            idx_t = torch.randint(0, len(Xt_t), (len(idx_s),))

            xs = Xs_t[idx_s].to(device, non_blocking=True)
            yb = ys_t[idx_s].to(device, non_blocking=True)
            xt = Xt_t[idx_t].to(device, non_blocking=True)

            lambd = lambda_schedule(step / max(total_steps - 1, 1))
            opt.zero_grad(set_to_none=True)

            zs = model.embed(xs)
            zt = model.embed(xt)
            label_loss = F.cross_entropy(model.label_head(zs), yb)

            d_s = model.domain_logits(zs, lambd)
            d_t = model.domain_logits(zt, lambd)
            domain_loss = (
                F.binary_cross_entropy_with_logits(d_s, torch.zeros_like(d_s))
                + F.binary_cross_entropy_with_logits(d_t, torch.ones_like(d_t))
            ) * 0.5

            (label_loss + domain_loss).backward()
            opt.step()
            sched.step()
            step += 1
            lab_sum += label_loss.item()
            dom_sum += domain_loss.item()

        if (epoch + 1) % verbose_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xv).argmax(1) == yv).float().mean().item()
            n = max(step, 1)
            print(f"  epoch {epoch+1:>3}/{epochs}  label {lab_sum/n:.4f}  "
                  f"domain {dom_sum/n:.4f}  lambda {lambd:.2f}  "
                  f"src-val acc {acc:.4f}")
    return model
