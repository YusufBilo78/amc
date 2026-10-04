"""Do RML2016.10a frames share noise? A check on the pickle itself.

GNU Radio's dynamic_channel_model (3.7.10 source, dynamic_channel_model_impl.cc)
seeds every one of its parts -- clock offset, carrier offset, fading, and the
noise source -- with the same `noise_seed`, and generate_RML2016.10a.py passes
the fixed value 0x1337 and builds a fresh channel for every run. Rebuilt in
GNU Radio 3.10, two runs through that channel are bit-identical with the
noise off, and the noise sequences are identical with it on
(tools/rml2016_channel_audit.py). Each run is cut into frames at random
offsets, so two frames from different runs whose offsets lie within 128
samples of each other carry the *same noise samples*, shifted, and scaled by
the SNR. Across classes too, since every class runs through the same channel.

At a low label the frame is almost all noise, so a shared noise segment is a
shared frame segment, which this finds without knowing the noise: every
16-sample window of every frame is scaled to unit norm and coarsely
quantised into a hash key; frames sharing a key are candidates, and a
candidate pair counts only if the two frames agree (correlation > 0.95) over
the whole overlap the shift implies.

    python tools/rml2016_noise_reuse.py --pkl RML2016.10a_dict.pkl
    python tools/rml2016_noise_reuse.py --selftest     # generator's sampling, simulated

--selftest runs the generator's own framing (start 50..500, then steps of
128..5% of the run) over a shared noise sequence at -20 dB, to show what the
check finds when reuse is known to be there, and over independent noise per
run, to show it finds nothing when it is not.
"""
import argparse
import json
import pathlib
import pickle
from collections import defaultdict

import numpy as np

N, W, Q = 128, 16, 8          # frame length, window, quantisation steps per unit


def keys(frame):
    """Hash keys of every W-sample window, scale-invariant (frames differ by a
    positive real scale only, from the generator's per-frame normalisation)."""
    out = []
    for k in range(N - W + 1):
        v = frame[k:k + W]
        v = v / (np.linalg.norm(v) + 1e-12)
        q = np.round(np.concatenate([v.real, v.imag]) * Q).astype(np.int8)
        out.append((k, q.tobytes()))
    return out


def find_pairs(frames, min_corr=0.95):
    """frames: (n, 128) complex. Returns {(i, j): (shift, overlap, corr)}."""
    table = defaultdict(list)
    for i, f in enumerate(frames):
        for k, key in keys(f):
            table[key].append((i, k))
    cand = {}
    for hits in table.values():
        if len(hits) < 2 or len(hits) > 50:      # >50: degenerate key
            continue
        for a in range(len(hits)):
            for b in range(a + 1, len(hits)):
                (i, ki), (j, kj) = hits[a], hits[b]
                if i != j:
                    p = (i, j) if i < j else (j, i)
                    d = (ki - kj) if i < j else (kj - ki)
                    cand.setdefault(p, d)
    found = {}
    for (i, j), d in cand.items():
        # frame i at position t matches frame j at position t - d
        lo, hi = max(0, d), min(N, N + d)
        a, b = frames[i][lo:hi], frames[j][lo - d:hi - d]
        c = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        if c > min_corr:
            found[(i, j)] = (int(d), int(hi - lo), float(c))
    return found


def summarise(found, labels, n):
    partnered = set()
    same = diff = 0
    for (i, j) in found:
        partnered.update((i, j))
        same += labels[i] == labels[j]
        diff += labels[i] != labels[j]
    ov = [v[1] for v in found.values()]
    return {"frames": n, "frames_with_a_partner": len(partnered),
            "share": round(len(partnered) / n, 4), "pairs": len(found),
            "pairs_same_class": int(same), "pairs_different_class": int(diff),
            "median_overlap": int(np.median(ov)) if ov else 0}


def simulate(shared, rng, runs=30, run_len=40_000, snr_amp=10.0):
    """The generator's framing at label -20 (noise_amp 10, signal ~1.3)."""
    noise_master = (rng.standard_normal(run_len) + 1j * rng.standard_normal(run_len)) / np.sqrt(2)
    frames = []
    for _ in range(runs):
        noise = noise_master if shared else \
            (rng.standard_normal(run_len) + 1j * rng.standard_normal(run_len)) / np.sqrt(2)
        sig = 1.3 * np.exp(1j * np.pi / 2 * rng.integers(4, size=run_len))
        y = sig + snr_amp * noise
        t = int(rng.integers(50, 501))
        while t + N < run_len and len(frames) < 1000:
            x = y[t:t + N]
            frames.append(x / np.abs(x).sum())
            t += int(rng.integers(N, round(run_len * 0.05) + 1))
    return np.array(frames)


def selftest():
    rng = np.random.default_rng(0)
    for shared in (True, False):
        fr = simulate(shared, rng)
        s = summarise(find_pairs(fr), np.zeros(len(fr), int), len(fr))
        print(f"{'shared' if shared else 'independent'} noise: {s}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl")
    p.add_argument("--labels", default="-20,-18,-16")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--out", default="rml2016_noise_reuse.json")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    with open(a.pkl, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    classes = sorted({c for c, _ in d})
    result = {}
    for s in (int(v) for v in a.labels.split(",")):
        X = np.concatenate([d[(c, s)][:, 0] + 1j * d[(c, s)][:, 1] for c in classes])
        lab = np.repeat(np.arange(len(classes)), [len(d[(c, s)]) for c in classes])
        found = find_pairs(X)
        summ = summarise(found, lab, len(X))
        per = {}
        for (i, j) in found:
            for k in (i, j):
                per[classes[lab[k]]] = per.get(classes[lab[k]], 0) + 1
        summ["partner_count_by_class"] = per
        result[s] = summ
        print(f"label {s:+d} dB: {summ}")
    pathlib.Path(a.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
