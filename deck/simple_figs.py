"""Pictures for the simple deck. Illustrations, not results: symbols drawn
from the four constellations as RML2016.10a's transmitter sends them
(unscaled grid, root-raised cosine, 8 samples per symbol, roll-off 0.35).
No dataset frame is shown, and no number on a slide comes from here.

    python deck/simple_figs.py     -> deck/simple/*.png, *.gif
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT = pathlib.Path(__file__).parent / "simple"
NAVY, ORANGE, INK, MUTED, CARD = "#0F1B2D", "#F08A24", "#1E293B", "#5B6B7F", "#EEF2F6"
GRIDC = "#3A4A60"
SPS, BETA = 8, 0.35
plt.rcParams.update({"font.family": "DejaVu Sans"})


def points(name):
    if name == "BPSK":
        return np.array([-1, 1], complex)
    if name == "QPSK":
        return np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4))) * np.sqrt(2)
    m = {"QAM16": 4, "QAM64": 8}[name]
    lv = np.arange(-(m - 1), m, 2)
    return (lv[:, None] + 1j * lv[None, :]).ravel()


def rrc(span=8):
    t = np.arange(-span * SPS, span * SPS + 1) / SPS
    h = np.zeros_like(t)
    for i, x in enumerate(t):
        if abs(x) < 1e-9:
            h[i] = 1 - BETA + 4 * BETA / np.pi
        elif abs(abs(x) - 1 / (4 * BETA)) < 1e-9:
            h[i] = BETA / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * BETA))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * BETA)))
        else:
            h[i] = (np.sin(np.pi * x * (1 - BETA)) + 4 * BETA * x * np.cos(np.pi * x * (1 + BETA))) \
                   / (np.pi * x * (1 - (4 * BETA * x) ** 2))
    return h / np.sqrt((h ** 2).sum())


def dark_axes(ax):
    ax.set_facecolor(NAVY)
    for s in ax.spines.values():
        s.set_color(GRIDC)
    ax.tick_params(colors="#9AA8BA", labelsize=8)


def constellation_panel(ax, name, sym, title, lim=None):
    dark_axes(ax)
    p = points(name)
    lim = lim or (np.abs(p.real).max() + 1.2)
    ax.scatter(p.real, p.imag, s=60, facecolors="none", edgecolors="#6B7C93", linewidths=1)
    ax.scatter(sym.real, sym.imag, s=55, color=ORANGE, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color="white", fontsize=12, pad=6)


def frame_figure(rng):
    """One 16-symbol QAM16 frame as the receiver gets it: I and Q over 128 samples."""
    p = points("QAM16")
    sym = p[rng.integers(len(p), size=16 + 16)]
    up = np.zeros(len(sym) * SPS, complex); up[::SPS] = sym
    h = rrc()
    x = np.convolve(np.convolve(up, h), h)          # tx and rx filter: symbols at the peaks
    d = 2 * (len(h) // 2) + 8 * SPS                  # skip the filter start
    x = x[d:d + 128]
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=NAVY)
    dark_axes(ax)
    t = np.arange(128)
    ax.plot(t, x.real, color=ORANGE, lw=1.6, label="I")
    ax.plot(t, x.imag, color="#7FB2E5", lw=1.6, label="Q")
    for k in range(0, 128, SPS):
        ax.axvline(k, color=GRIDC, lw=0.6, zorder=0)
    k = np.arange(0, 128, SPS)
    ax.scatter(k, x.real[k], color=ORANGE, s=28, zorder=3)
    ax.scatter(k, x.imag[k], color="#7FB2E5", s=28, zorder=3)
    ax.set_xlim(0, 127)
    ax.set_xlabel("sample (128 in one frame)  ·  dots: the 16 symbols, one every 8 samples", color="#C9D3E0", fontsize=10)
    ax.legend(loc="upper right", frameon=False, labelcolor="white", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "frame.png", dpi=170, facecolor=NAVY)
    plt.close(fig)


def symbols_figure(rng):
    """The 16 symbols of one frame, for each class, over the class's full grid."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), facecolor=NAVY)
    for ax, name in zip(axes, ("BPSK", "QPSK", "QAM16", "QAM64")):
        p = points(name)
        sym = p[rng.integers(len(p), size=16)]
        constellation_panel(ax, name, sym, f"{name}: {len(p)} possible")
    fig.tight_layout()
    fig.savefig(OUT / "symbols.png", dpi=170, facecolor=NAVY)
    plt.close(fig)


def symbols_grid_figure(rng):
    """The same as symbols_figure, two by two, for a narrow column."""
    fig, axes = plt.subplots(2, 2, figsize=(6, 6.3), facecolor=NAVY)
    for ax, name in zip(axes.ravel(), ("BPSK", "QPSK", "QAM16", "QAM64")):
        p = points(name)
        sym = p[rng.integers(len(p), size=16)]
        constellation_panel(ax, name, sym, f"{name}: {len(p)} possible")
    fig.tight_layout()
    fig.savefig(OUT / "symbols_2x2.png", dpi=170, facecolor=NAVY)
    plt.close(fig)


def fewer_figure(rng):
    """QAM16 against QAM64 when the frame holds 16, 8 and 4 symbols."""
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.8), facecolor=NAVY)
    for r, name in enumerate(("QAM16", "QAM64")):
        p = points(name)
        sym = p[rng.integers(len(p), size=16)]
        for c, n in enumerate((16, 8, 4)):
            constellation_panel(axes[r, c], name, sym[:n], f"{name}, {n} symbols")
    fig.tight_layout()
    fig.savefig(OUT / "fewer.png", dpi=170, facecolor=NAVY)
    plt.close(fig)


def noise_gif(rng):
    """The same 16 QAM16 symbols as the noise grows."""
    p = points("QAM16")
    sym = p[rng.integers(len(p), size=16)]
    z = (rng.standard_normal(16) + 1j * rng.standard_normal(16)) / np.sqrt(2)
    frames = []
    for k, sigma in enumerate((0.0, 0.3, 0.6, 1.0, 1.6, 2.5, 4.0)):
        fig, ax = plt.subplots(figsize=(4, 4.3), facecolor=NAVY)
        constellation_panel(ax, "QAM16", sym + sigma * z, "more noise →" if k else "no noise", lim=8)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    frames[0].save(OUT / "noise.gif", save_all=True, append_images=frames[1:] + frames[::-1][1:-1],
                   duration=700, loop=0)
    frames[0].save(OUT / "noise.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    frame_figure(rng); symbols_figure(rng); symbols_grid_figure(rng); fewer_figure(rng); noise_gif(rng)
    print(sorted(p.name for p in OUT.iterdir()))
