"""
sink_vs_geometry.py -- does the sink land on the geometrically nearest class,
with the similarity defined *before* looking at the results?

The 24-class sweep showed an unseen modulation lands in the same hand-drawn
family 18/24 -- well above chance (3.5), but the six misses were all cases where
my taxonomy and the physics disagreed (FM->GMSK, BPSK->OQPSK, 16APSK->32QAM).
The obvious next move -- redraw the families to fit -- would invalidate the
test.

So instead the similarity structure is defined from the signals themselves,
with no hand-built taxonomy at all:

  1. For each modulation, take its clean symbol alphabet (the ideal
     constellation) and, for the CPM / analog schemes that have no discrete
     alphabet, a dense sampling of the noise-free waveform in the IQ plane.
  2. Reduce each to a rotation-invariant, scale-invariant descriptor: the
     sorted histogram of pairwise point distances (a shape signature that does
     not care about absolute orientation or power).
  3. Distance between two modulations = L1 distance between descriptors.

This gives a 24x24 similarity matrix computed only from modulation definitions.
Then, for each held-out class, ask a single yes/no question fixed in advance:
is the observed sink among the k geometrically nearest classes?

k is reported for k = 1, 2, 3. Nothing about the model's behaviour touches the
distance matrix, so this cannot be tuned to the answer.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(line_buffering=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import modem
import radioml

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

# RadioML's 24 classes mapped to how we synthesise a clean reference for each.
# Where our generator has the exact scheme we use it; otherwise the nearest
# structural equivalent, noted honestly.
REFERENCE = {
    "OOK": ("ask", 2), "4ASK": ("ask", 4), "8ASK": ("ask", 8),
    "BPSK": ("psk", 2), "QPSK": ("psk", 4), "8PSK": ("psk", 8),
    "16PSK": ("psk", 16), "32PSK": ("psk", 32),
    "16APSK": ("apsk", 16), "32APSK": ("apsk", 32),
    "64APSK": ("apsk", 64), "128APSK": ("apsk", 128),
    "16QAM": ("qam", 16), "32QAM": ("qam", 32), "64QAM": ("qam", 64),
    "128QAM": ("qam", 128), "256QAM": ("qam", 256),
    "AM-SSB-WC": ("analog_ssb", 0), "AM-SSB-SC": ("analog_ssb", 0),
    "AM-DSB-WC": ("analog_dsb", 0), "AM-DSB-SC": ("analog_dsb", 0),
    "FM": ("fm", 0), "GMSK": ("cpm", 0), "OQPSK": ("psk_offset", 4),
}


def constellation(kind: str, order: int) -> np.ndarray:
    """A cloud of points in the IQ plane representing the modulation's shape."""
    if kind == "ask":
        levels = np.arange(order)
        pts = (levels - levels.mean()).astype(complex)
    elif kind in ("psk", "psk_offset"):
        pts = np.exp(1j * 2 * np.pi * np.arange(order) / order)
    elif kind == "qam":
        side = int(round(np.sqrt(order)))
        if side * side == order:
            g = np.arange(-(side - 1), side, 2)
            re, im = np.meshgrid(g, g)
            pts = (re + 1j * im).ravel()
        else:  # cross QAM (32,128): square grid with corners removed
            side = int(np.ceil(np.sqrt(order)))
            g = np.arange(-(side - 1), side, 2)
            re, im = np.meshgrid(g, g)
            p = (re + 1j * im).ravel()
            keep = np.argsort(np.abs(p))[:order]
            pts = p[keep]
    elif kind == "apsk":
        # concentric rings, point counts roughly following DVB-S2 style
        rings = {16: [4, 12], 32: [4, 12, 16], 64: [4, 12, 20, 28],
                 128: [8, 16, 24, 36, 44]}[order]
        pts = []
        for r, n in enumerate(rings, 1):
            pts.extend(r * np.exp(1j * 2 * np.pi * np.arange(n) / n))
        pts = np.array(pts)
    elif kind == "analog_dsb":
        # real-valued line (AM double sideband): points on the real axis
        pts = np.linspace(-1, 1, 64).astype(complex)
    elif kind == "analog_ssb":
        # single sideband: analytic, a spiral-ish cloud off the real axis
        t = np.linspace(0, 1, 64)
        pts = np.exp(1j * 2 * np.pi * t) * (0.3 + 0.7 * t)
    elif kind == "fm":
        # constant envelope, phase sweeps: a ring
        pts = np.exp(1j * 2 * np.pi * np.linspace(0, 1, 64))
    elif kind == "cpm":
        # continuous phase (GMSK): also a ring, but denser near transitions
        ph = np.cumsum(np.sin(2 * np.pi * np.linspace(0, 4, 128)))
        pts = np.exp(1j * ph)
    else:
        raise ValueError(kind)
    p = np.asarray(pts, dtype=complex)
    return p / (np.sqrt(np.mean(np.abs(p) ** 2)) + 1e-12)  # unit power


def descriptor(pts: np.ndarray, bins: int = 40) -> np.ndarray:
    """
    Rotation- and scale-invariant shape signature: histogram of pairwise
    distances between constellation points. Rotation cannot change pairwise
    distances, and unit-power normalisation fixes scale, so this describes the
    *shape* of the constellation and nothing about its orientation.
    """
    d = np.abs(pts[:, None] - pts[None, :])
    d = d[np.triu_indices(len(pts), k=1)]
    if len(d) == 0:
        d = np.array([0.0])
    hist, _ = np.histogram(d, bins=bins, range=(0, 3), density=True)
    return hist


def main() -> None:
    classes = list(radioml.CLASSES)
    desc = {c: descriptor(constellation(*REFERENCE[c])) for c in classes}

    n = len(classes)
    D = np.zeros((n, n))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            D[i, j] = np.abs(desc[a] - desc[b]).sum()

    # Load the observed sinks from the completed sweep.
    sinks = json.loads((ROOT / "sink_class_24_partial.json").read_text())

    idx = {c: i for i, c in enumerate(classes)}
    print(f"{'held out':<12} {'observed sink':<13} {'geom. rank':>10}  "
          f"{'nearest 3 by geometry':<34}")
    print("-" * 74)

    ranks = []
    rows = []
    for held in classes:
        if held not in sinks:
            continue
        i = idx[held]
        order = np.argsort(D[i])
        order = [classes[j] for j in order if classes[j] != held]
        sink = sinks[held]["sink"]
        rank = order.index(sink) + 1 if sink in order else 99
        ranks.append(rank)
        near3 = ", ".join(order[:3])
        flag = "" if rank <= 3 else "  <-- far"
        print(f"{held:<12} {sink:<13} {rank:>10}  {near3:<34}{flag}")
        rows.append((held, sink, rank))

    ranks = np.array(ranks)
    print("\n" + "=" * 60)
    for k in (1, 2, 3):
        print(f"sink within geometric top-{k}: "
              f"{int((ranks <= k).sum())}/{len(ranks)}")
    # Chance: with 23 candidates, top-k by chance is k/23.
    for k in (1, 2, 3):
        exp = len(ranks) * k / 23
        print(f"  (chance for top-{k}: {exp:.1f})")
    print(f"median rank: {np.median(ranks):.0f}  mean rank: {ranks.mean():.1f}")
    print("=" * 60)

    # ---------------------------------------------------------------- figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    im = ax1.imshow(D, cmap="viridis_r")
    ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
    ax1.set_xticklabels(classes, rotation=90, fontsize=7)
    ax1.set_yticklabels(classes, fontsize=7)
    ax1.set_title("Geometric distance between modulations\n"
                  "(from constellation shape, no taxonomy)", fontsize=11,
                  fontweight="bold")
    # mark the observed sink in each row
    for held, sink, rank in rows:
        ax1.scatter(idx[sink], idx[held], marker="s", s=40,
                    edgecolor="red", facecolor="none", linewidth=1.5)
    fig.colorbar(im, ax=ax1, fraction=0.046)

    order_r = np.arange(1, len(ranks) + 1)
    ax2.hist(ranks[ranks < 99], bins=np.arange(0.5, 12, 1),
             color="#2980b9", edgecolor="white")
    ax2.axvline(1.5, color="#27ae60", lw=2, ls="--", label="top-1 boundary")
    ax2.axvline(3.5, color="#f39c12", lw=2, ls="--", label="top-3 boundary")
    ax2.set_xlabel("geometric rank of the observed sink")
    ax2.set_ylabel("number of held-out classes")
    ax2.set_title(f"Where the sink falls in the geometric ordering\n"
                  f"top-1: {int((ranks<=1).sum())}/{len(ranks)}, "
                  f"top-3: {int((ranks<=3).sum())}/{len(ranks)} "
                  f"(chance top-3 ≈ {len(ranks)*3/23:.0f})",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Does the unseen-class sink match geometric nearest-neighbour?",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "24_sink_vs_geometry.png", dpi=140)
    plt.close(fig)

    np.savez(ROOT / "sink_vs_geometry.npz", classes=np.array(classes),
             distance=D, ranks=ranks)
    print("\nwrote figures/24_sink_vs_geometry.png")


if __name__ == "__main__":
    main()
