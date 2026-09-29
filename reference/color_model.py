#!/usr/bin/env python3
"""Reference implementation of spec section 3 (pigment model) + acceptance tests T-C1..T-C6.

Everything here is the *definition*; the GPU code must reproduce these numbers.
Run:  python3 color_model.py        (expects tables.json next to this file)
"""
import json, os, sys
import numpy as np

T = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables.json')))
BASE = {k: np.array(v) for k, v in T['base_spectra'].items()}
CMF = np.array(T['cmf'])                 # 3 x 38
M_XYZ_RGB = np.array(T['xyz_to_rgb'])
C = T['constants']
R_FLOOR, R_CEIL, KAPPA, Y_MIN, A_SCALE = C['R_FLOOR'], C['R_CEIL'], C['KAPPA'], C['Y_MIN'], C['A_SCALE']

# ---------------------------------------------------------------- C1: sRGB companding
def uncompand(x):
    x = np.asarray(x, float)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def compand(x):
    x = np.clip(np.asarray(x, float), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)

# ---------------------------------------------------------------- C2: lRGB -> reflectance (spectral.js lRGB_to_R)
def lrgb_to_R(lrgb, tables=None):
    base = BASE if tables is None else {k: np.array(v) for k, v in tables['base_spectra'].items()}
    r_, g_, b_ = [float(v) for v in lrgb]
    w = min(r_, g_, b_); r_, g_, b_ = r_ - w, g_ - w, b_ - w
    c = min(g_, b_); m = min(r_, b_); y = min(r_, g_)
    r = max(0.0, min(r_ - b_, r_ - g_)); g = max(0.0, min(g_ - b_, g_ - r_)); b = max(0.0, min(b_ - g_, b_ - r_))
    R = (w * base['W'] + c * base['C'] + m * base['M'] + y * base['Y'] + r * base['R'] + g * base['G'] + b * base['B'])
    return np.clip(R, R_FLOOR, R_CEIL)

# ---------------------------------------------------------------- C3/C4: K/S and the paint descriptor
def KS(R): return (1.0 - R) ** 2 / (2.0 * R)
def KM(ks): return 1.0 + ks - np.sqrt(ks * ks + 2.0 * ks)

def make_paint(srgb255, tint=1.0):
    """User colour (sRGB 0..255, any triple) + tinting strength -> paint descriptor (a[38], beta)."""
    R = lrgb_to_R(uncompand(np.asarray(srgb255, float) / 255.0))
    Y = float(CMF[1] @ R)
    sigma = tint * max(Y, Y_MIN) ** KAPPA
    a = A_SCALE * sigma * KS(R)          # stored per unit pigment volume, fp16 on GPU
    return a, sigma                       # (a, beta)

def mix_paints(parts):
    """parts: list of (p, a, beta) with p = pigment volume. Linear -> exact and associative."""
    P = sum(p for p, _, _ in parts)
    a = sum(p * a for p, a, _ in parts) / P
    beta = sum(p * b for p, _, b in parts) / P
    return a, beta

def masstone_R(a, beta):               # C5: infinitely thick layer
    return KM(a / (A_SCALE * beta))

# ---------------------------------------------------------------- C6: finite layer (Kubelka), stable form
def layer_RT(a, beta, p, S0):
    """Layer with pigment thickness p [m]. Returns (R, T) spectra."""
    Sx = S0 * beta * p
    ks = a / (A_SCALE * beta)
    ak = 1.0 + ks
    bk = np.sqrt(np.maximum(ak * ak - 1.0, 0.0))
    xi = bk * Sx
    E = np.exp(-2.0 * xi)
    den = ak * (1.0 - E) + bk * (1.0 + E)
    R_big = (1.0 - E) / np.maximum(den, 1e-30)
    T_big = 2.0 * bk * np.exp(-xi) / np.maximum(den, 1e-30)
    R_small = Sx / (1.0 + ak * Sx)            # series for xi -> 0 (covers K/S -> 0 and thin layers)
    T_small = (1.0 - xi) / (1.0 + ak * Sx)
    use_small = xi < 1e-3
    return np.where(use_small, R_small, R_big), np.where(use_small, T_small, T_big)

def composite(R_layer, T_layer, R_below):  # C7
    return R_layer + T_layer * T_layer * R_below / (1.0 - R_layer * R_below)

def R_to_srgb255(R):                         # C8
    xyz = CMF @ R
    lrgb = np.clip(M_XYZ_RGB @ xyz, 0.0, 1.0)
    return np.round(compand(lrgb) * 255.0)

def dE76(R1, R2, wp=np.array([0.95047, 1.0, 1.08883])):
    def lab(R):
        t = (CMF @ R) / wp
        f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
        return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])
    return float(np.linalg.norm(lab(R1) - lab(R2)))

# ---------------------------------------------------------------- acceptance tests
def fp16_roundtrip(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float64)

def run_tests(S0=49.0 / 0.5e-3, h_stroke=0.5e-3):
    rng = np.random.default_rng(0)
    ok = True
    # T-C1 masstone reproduction of ANY sRGB colour (thick layer, over any ground)
    worst = 0.0
    for _ in range(1000):
        c = rng.integers(0, 256, 3)
        a, beta = make_paint(c)
        R_ref = lrgb_to_R(uncompand(c / 255.0))
        for ground in (0.995, 0.005):
            R, Tl = layer_RT(a, beta, 10 * h_stroke, S0)
            worst = max(worst, dE76(R_ref, composite(R, Tl, np.full(38, ground))))
    print(f'T-C1 masstone (10*h_stroke) reproduces input colour: worst dE76 {worst:.4f}  {"PASS" if worst < 0.05 else "FAIL"}'); ok &= worst < 0.05
    # T-C2 black + white 50:50 by volume -> L* = 31.0 +- 0.5
    aw, bw = make_paint([255, 255, 255]); ak_, bk_ = make_paint([0, 0, 0])
    a, beta = mix_paints([(1, aw, bw), (1, ak_, bk_)])
    xyz = CMF @ masstone_R(a, beta); L = 116 * np.cbrt(xyz[1]) - 16
    print(f'T-C2 black+white 50/50: L* = {L:.2f}  {"PASS" if abs(L - 31.0) < 0.5 else "FAIL"}'); ok &= abs(L - 31.0) < 0.5
    # T-C3 fp16 storage error of the descriptor
    worst = 0.0
    for _ in range(1000):
        parts = []
        for _ in range(rng.integers(2, 5)):
            a, beta = make_paint(rng.integers(0, 256, 3)); parts.append((rng.random() + 0.05, a, beta))
        a_ref, b_ref = mix_paints(parts)
        a16, b16 = mix_paints([(p, fp16_roundtrip(a), float(fp16_roundtrip(b))) for p, a, b in parts])
        worst = max(worst, dE76(masstone_R(a_ref, b_ref), masstone_R(fp16_roundtrip(a16), float(fp16_roundtrip(b16)))))
    print(f'T-C3 fp16 storage: worst dE76 {worst:.4f}  {"PASS" if worst < 0.05 else "FAIL"}'); ok &= worst < 0.05
    # T-C4 associativity
    worst = 0.0
    for _ in range(500):
        P = [(rng.random() + 0.05,) + make_paint(rng.integers(0, 256, 3)) for _ in range(3)]
        ab = mix_paints(P[:2]); abc = mix_paints([(P[0][0] + P[1][0],) + ab, P[2]])
        worst = max(worst, dE76(masstone_R(*abc), masstone_R(*mix_paints(P))))
    print(f'T-C4 associativity: worst dE76 {worst:.2e}  {"PASS" if worst < 1e-6 else "FAIL"}'); ok &= worst < 1e-6
    # T-C5 glazing: thin dark-red layer over white vs black ground must differ; thick must not
    a, beta = make_paint([140, 20, 20])
    for p, must_differ in ((0.01 * h_stroke, True), (10 * h_stroke, False)):
        R, Tl = layer_RT(a, beta, p, S0)
        d = dE76(composite(R, Tl, np.full(38, 0.995)), composite(R, Tl, np.full(38, 0.005)))
        cond = d > 10 if must_differ else d < 0.05
        print(f'T-C5 glaze p={p/h_stroke:.2f}*h_stroke: white-vs-black ground dE76 {d:.2f}  {"PASS" if cond else "FAIL"}'); ok &= cond
    # T-C6 white hiding: full-strength white at h_stroke has contrast ratio >= 0.98
    aw, bw = make_paint([255, 255, 255]); R, Tl = layer_RT(aw, bw, h_stroke, S0)
    cr = (CMF[1] @ composite(R, Tl, np.full(38, 0.005))) / (CMF[1] @ composite(R, Tl, np.full(38, 0.995)))
    print(f'T-C6 white contrast ratio at h_stroke: {cr:.4f}  {"PASS" if cr >= 0.98 else "FAIL"}'); ok &= cr >= 0.98
    print('ALL PASS' if ok else 'SOME FAILED')
    return ok

if __name__ == '__main__':
    sys.exit(0 if run_tests() else 1)
