#!/usr/bin/env python3
"""Reproduce three historical symptoms against the spec, by measurement (no inference):
  S1  noise-like colour left by brush strokes
  S2  darkening at boundaries
  S3  pigment flowing endlessly and covering the canvas
Implements spec 5.3 (stamp B1-B5, no fluid), 7 (render) on top of color_model / fluid_core.
Run: python3 test_symptoms.py
"""
import numpy as np, sys
import color_model as cm
from fluid_core import Params, Canvas

rng = np.random.default_rng(0)
f16 = lambda x: np.asarray(x, np.float32).astype(np.float16).astype(np.float64)

# ------------------------------------------------------------------ brush model (spec 5.3), N_L layers, f16 emulation of a/beta
class BrushCanvas:
    def __init__(self, nx, ny, NL, P):
        self.P = P; self.nx, self.ny, self.NL = nx, ny, NL
        self.d = np.zeros((NL, ny, nx)); self.p = np.zeros((NL, ny, nx)); self.s = np.zeros((NL, ny, nx))
        self.id = np.zeros((NL, ny, nx), np.uint32)
        self.a = np.zeros((NL, ny, nx, 38)); self.beta = np.zeros((NL, ny, nx))
        self.b = np.zeros((ny, nx)); self.Rdry = np.full((ny, nx, 38), 0.95)

    def store_f16(self):
        self.a = f16(self.a); self.beta = f16(self.beta)

    def stamp(self, br, cx, cy, r, pressure, stroke_id, x_dep=0.1, x_pick=0.1, x_mix=0.2, r_rep=0.05, h_max=3e-3):
        P = self.P; NL = self.NL
        ic, jc = int(round(cx)), int(round(cy)); R = br['rmax']
        for j in range(max(0, jc - R), min(self.ny, jc + R + 1)):
            for i in range(max(0, ic - R), min(self.nx, ic + R + 1)):
                rho = np.hypot(i - cx, j - cy)
                M = 1.0 if rho <= 0.8 * r else (max(0.0, (r - rho) / (0.2 * r)) if rho <= r else 0.0)
                m = M * pressure
                if m <= 0: continue
                ti, tj = i - ic + R, j - jc + R                      # brush texel (1:1)
                d = self.d[:, j, i]; occ = d >= P.eps_d; n = int(occ.sum())
                # occupancy is contiguous from the bottom by construction
                dtot = d[:n].sum()
                # B1 engaged layers
                ztop = np.cumsum(d[:n]) if n else np.zeros(0)
                engaged = [k for k in range(n) if ztop[k] > (1 - pressure) * dtot]
                # B2 layer mixing, top-down over adjacent engaged pairs
                for k in reversed(range(n - 1)):
                    if k in engaged and (k + 1) in engaged:
                        Vx = x_mix * m * (1 - max(self.s[k, j, i], self.s[k + 1, j, i])) * min(d[k], d[k + 1])
                        if Vx <= 0: continue
                        fk, fk1 = Vx / d[k], Vx / d[k + 1]
                        pk, pk1 = self.p[k, j, i], self.p[k + 1, j, i]
                        ak, ak1 = self.a[k, j, i].copy(), self.a[k + 1, j, i].copy()
                        bk, bk1 = self.beta[k, j, i], self.beta[k + 1, j, i]
                        sk, sk1 = self.s[k, j, i], self.s[k + 1, j, i]
                        # extensive exchange
                        p_new_k = pk * (1 - fk) + pk1 * fk1; p_new_k1 = pk1 * (1 - fk1) + pk * fk
                        if p_new_k > P.eps_p:
                            self.a[k, j, i] = (ak * pk * (1 - fk) + ak1 * pk1 * fk1) / p_new_k
                            self.beta[k, j, i] = (bk * pk * (1 - fk) + bk1 * pk1 * fk1) / p_new_k
                        if p_new_k1 > P.eps_p:
                            self.a[k + 1, j, i] = (ak1 * pk1 * (1 - fk1) + ak * pk * fk) / p_new_k1
                            self.beta[k + 1, j, i] = (bk1 * pk1 * (1 - fk1) + bk * pk * fk) / p_new_k1
                        self.p[k, j, i], self.p[k + 1, j, i] = p_new_k, p_new_k1
                        self.s[k, j, i] = sk * (1 - fk) + sk1 * fk1; self.s[k + 1, j, i] = sk1 * (1 - fk1) + sk * fk
                # B3 pickup from top layer into brush texel
                hb = br['h'][tj, ti]
                if n > 0:
                    t = n - 1
                    Vp = min(x_pick * m * (1 - self.s[t, j, i]) * d[t], br['hmax'] - hb)
                    if Vp > 0:
                        f = Vp / d[t]; pp = self.p[t, j, i] * f
                        pb = br['p'][tj, ti]; p_new = pb + pp
                        if p_new > P.eps_p:
                            br['a'][tj, ti] = (br['a'][tj, ti] * pb + self.a[t, j, i] * pp) / p_new
                            br['beta'][tj, ti] = (br['beta'][tj, ti] * pb + self.beta[t, j, i] * pp) / p_new
                        br['p'][tj, ti] = p_new; br['h'][tj, ti] = hb + Vp
                        self.d[t, j, i] -= Vp; self.p[t, j, i] -= pp
                        d = self.d[:, j, i]; occ = d >= P.eps_d; n = int(occ.sum()); dtot = d[:n].sum()
                # B4 deposit
                hb = br['h'][tj, ti]
                if n >= 1 and self.id[n - 1, j, i] == stroke_id: target = n - 1
                elif n < NL: target = n
                else:
                    # merge 0 and 1 into 0, shift down
                    p0, p1 = self.p[0, j, i], self.p[1, j, i]; ps = p0 + p1
                    if ps > P.eps_p:
                        self.a[0, j, i] = (self.a[0, j, i] * p0 + self.a[1, j, i] * p1) / ps
                        self.beta[0, j, i] = (self.beta[0, j, i] * p0 + self.beta[1, j, i] * p1) / ps
                    d0, d1 = self.d[0, j, i], self.d[1, j, i]
                    self.s[0, j, i] = (self.s[0, j, i] * d0 + self.s[1, j, i] * d1) / (d0 + d1)
                    self.d[0, j, i] = d0 + d1; self.p[0, j, i] = ps; self.id[0, j, i] = self.id[1, j, i]
                    for k in range(1, NL - 1):
                        for arr in (self.d, self.p, self.s, self.id, self.a, self.beta): arr[k, j, i] = arr[k + 1, j, i]
                    k = NL - 1
                    self.d[k, j, i] = 0; self.p[k, j, i] = 0; self.s[k, j, i] = 0; self.a[k, j, i] = 0; self.beta[k, j, i] = 0
                    target = NL - 1; n = NL - 1; d = self.d[:, j, i]; dtot = d[:n].sum()
                Vd = min(x_dep * m * hb, h_max - dtot, hb)
                if Vd > 0:
                    f = Vd / hb; pp = br['p'][tj, ti] * f
                    pt = self.p[target, j, i]; p_new = pt + pp
                    if p_new > P.eps_p:
                        self.a[target, j, i] = (self.a[target, j, i] * pt + br['a'][tj, ti] * pp) / p_new
                        self.beta[target, j, i] = (self.beta[target, j, i] * pt + br['beta'][tj, ti] * pp) / p_new
                    dt_ = self.d[target, j, i]
                    self.s[target, j, i] = (self.s[target, j, i] * dt_) / (dt_ + Vd)
                    self.d[target, j, i] = dt_ + Vd; self.p[target, j, i] = p_new; self.id[target, j, i] = stroke_id
                    br['h'][tj, ti] = hb - Vd; br['p'][tj, ti] -= pp
                # B5 replenish
                hb = br['h'][tj, ti]; add = r_rep * (br['hmax'] - hb)
                if add > 0:
                    pp = br['phi_u'] * add; pb = br['p'][tj, ti]; p_new = pb + pp
                    br['a'][tj, ti] = (br['a'][tj, ti] * pb + br['a0'] * pp) / p_new
                    br['beta'][tj, ti] = (br['beta'][tj, ti] * pb + br['beta0'] * pp) / p_new
                    br['p'][tj, ti] = p_new; br['h'][tj, ti] = hb + add

    def render(self, S0, light=np.array([-0.4, 0.5, 0.77]), z_s=1.0, dx=0.5 / 1024):
        # 7.1 albedo
        R = self.Rdry.copy()
        for k in range(self.NL):
            occ = self.d[k] >= self.P.eps_d
            Rl, Tl = cm.layer_RT(self.a[k], np.maximum(self.beta[k], 1e-9)[..., None], self.p[k][..., None], S0)
            Rc = cm.composite(Rl, Tl, R)
            R = np.where(occ[..., None], Rc, R)
        xyz = R @ cm.CMF.T
        alb = np.clip(xyz @ cm.M_XYZ_RGB.T, 0, 1)
        # 7.2 normal
        H = self.b + self.d.sum(0)
        gx = np.gradient(H, dx, axis=1); gy = np.gradient(H, dx, axis=0)
        n = np.stack([-z_s * gx, -z_s * gy, np.ones_like(H)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
        l = light / np.linalg.norm(light); v = np.array([0, 0, 1.0]); h = (l + v) / np.linalg.norm(l + v)
        nl = np.clip(n @ l, 0, 1); nv = np.clip(n @ v, 1e-4, 1); nh = np.clip(n @ h, 0, 1); vh = float(v @ h)
        top = np.argmax(self.d >= self.P.eps_d, axis=0)
        s_top = np.take_along_axis(self.s, top[None], 0)[0]
        occ_any = (self.d >= self.P.eps_d).any(0)
        alpha = np.where(occ_any, 0.15 + 0.45 * s_top, 0.7)
        D = alpha ** 2 / (np.pi * (nh ** 2 * (alpha ** 2 - 1) + 1) ** 2)
        F = 0.04 + 0.96 * (1 - vh) ** 5
        G1 = lambda x: 2 * x / (x + np.sqrt(alpha ** 2 + (1 - alpha ** 2) * x ** 2))
        spec = D * F * G1(nl) * G1(nv) / (4 * np.maximum(nl, 1e-4) * nv) * nl
        L = alb * (0.25 + 0.75 * nl)[..., None] + spec[..., None]
        return np.clip(L, 0, 1), alb, nl, H

def lum(rgb): return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

def make_brush(rmax, srgb, phi_u, h_stroke):
    a0, b0 = cm.make_paint(srgb)
    n = 2 * rmax + 1
    br = dict(rmax=rmax, hmax=1.5 * h_stroke, phi_u=phi_u, a0=a0, beta0=b0,
              h=np.full((n, n), 1.5 * h_stroke), p=np.full((n, n), phi_u * 1.5 * h_stroke),
              a=np.tile(a0, (n, n, 1)), beta=np.full((n, n), b0))
    return br

def stroke(c, br, x0, x1, y, r, pressure, sid, s_sp=0.25, f16_store=True):
    n = int(abs(x1 - x0) / (s_sp * r)) + 1
    for x in np.linspace(x0, x1, n):
        c.stamp(br, x, y, r, pressure, sid)
        if f16_store: c.store_f16()

# ------------------------------------------------------------------ S1 / S2
def test_S1_S2():
    P = Params(); P.nx, P.ny = 160, 64; NL = 2; S0 = 49 / P.h_stroke
    c = BrushCanvas(P.nx, P.ny, NL, P)
    # pre-painted blue band (stroke 1), then red stroke across it (stroke 2), then back-and-forth smudging (stroke 3)
    blue = make_brush(12, [30, 60, 200], 1.0, P.h_stroke); red = make_brush(12, [200, 30, 30], 1.0, P.h_stroke)
    stroke(c, blue, 20, 140, 32, 12, 1.0, 1)
    stroke(c, red, 20, 140, 32, 12, 1.0, 2)
    img, alb, nl, H = c.render(S0)
    # --- S1: noise along the stroke centre line and across a homogeneous interior patch
    ys, xs = 32, slice(40, 120)
    patch = alb[26:39, 40:120]                    # interior of the stroke (|dy|<=6 -> M=1 region)
    hp = H[26:39, 40:120]
    lap = patch[1:-1, 1:-1] - 0.25 * (patch[:-2, 1:-1] + patch[2:, 1:-1] + patch[1:-1, :-2] + patch[1:-1, 2:])
    noise_alb = np.sqrt((lap ** 2).mean()) * 255
    lap_h = hp[1:-1, 1:-1] - 0.25 * (hp[:-2, 1:-1] + hp[2:, 1:-1] + hp[1:-1, :-2] + hp[1:-1, 2:])
    noise_h = np.sqrt((lap_h ** 2).mean())
    print(f'S1  interior albedo high-frequency residual: {noise_alb:.3f} /255   height residual: {noise_h*1e6:.3f} um (h_stroke=500 um)')
    print(f'S1  interior albedo std (8-bit): {(patch.std(axis=(0,1))*255).round(3)}   mean sRGB: {np.round(cm.compand(patch.mean((0,1)))*255)}')
    print(f'S1  interior thickness: mean {hp.mean()*1e6:.1f} um, std {hp.std()*1e6:.1f} um, min {hp.min()*1e6:.1f}, max {hp.max()*1e6:.1f}')
    # heavy smudging: 40 passes back and forth with a clean dry-ish brush (r_rep=0 -> finite load, empties, then only mixes)
    grey = make_brush(12, [128, 128, 128], 1.0, P.h_stroke); grey['h'][:] = 0; grey['p'][:] = 0
    for k in range(20):
        for x in np.linspace(40, 120, 27) if k % 2 == 0 else np.linspace(120, 40, 27):
            c.stamp(grey, x, 32, 12, 1.0, 3 + k, r_rep=0.0); c.store_f16()
    img2, alb2, nl2, H2 = c.render(S0)
    patch2 = alb2[26:39, 40:120]
    lap2 = patch2[1:-1, 1:-1] - 0.25 * (patch2[:-2, 1:-1] + patch2[2:, 1:-1] + patch2[1:-1, :-2] + patch2[1:-1, 2:])
    print(f'S1  after 20x2 smudge passes (540 stamps): albedo HF residual {np.sqrt((lap2**2).mean())*255:.3f} /255, std {(patch2.std(axis=(0,1))*255).round(3)}')
    # f16 accumulation: repeat the identical smudge sequence in f64 and compare per-cell colour
    c64 = BrushCanvas(P.nx, P.ny, NL, P)
    stroke(c64, make_brush(12, [30, 60, 200], 1.0, P.h_stroke), 20, 140, 32, 12, 1.0, 1, f16_store=False)
    stroke(c64, make_brush(12, [200, 30, 30], 1.0, P.h_stroke), 20, 140, 32, 12, 1.0, 2, f16_store=False)
    grey = make_brush(12, [128, 128, 128], 1.0, P.h_stroke); grey['h'][:] = 0; grey['p'][:] = 0
    for k in range(20):
        for x in np.linspace(40, 120, 27) if k % 2 == 0 else np.linspace(120, 40, 27):
            c64.stamp(grey, x, 32, 12, 1.0, 3 + k, r_rep=0.0)
    img64, alb64, _, _ = c64.render(S0)
    dE = []
    for j in range(26, 39):
        for i in range(40, 120, 4):
            R16 = alb2[j, i]; R64 = alb64[j, i]
            # compare in Lab via XYZ
            x16 = cm.M_XYZ_RGB @ R16; x64 = cm.M_XYZ_RGB @ R64  # not exact inverse; use direct XYZ instead:
    # exact: recompute XYZ from spectra of both canvases
    def xyz_of(cv):
        R = cv.Rdry.copy()
        for k in range(cv.NL):
            occ = cv.d[k] >= P.eps_d
            Rl, Tl = cm.layer_RT(cv.a[k], np.maximum(cv.beta[k], 1e-9)[..., None], cv.p[k][..., None], S0)
            R = np.where(occ[..., None], cm.composite(Rl, Tl, R), R)
        return R @ cm.CMF.T
    X16, X64 = xyz_of(c), xyz_of(c64)
    def lab(x):
        t = x / np.array([0.95047, 1, 1.08883]); f = np.where(t > (6/29)**3, np.cbrt(t), t/(3*(6/29)**2) + 4/29)
        return np.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])], -1)
    dE = np.linalg.norm(lab(X16) - lab(X64), axis=-1)[26:39, 40:120]
    print(f'S1  f16 vs f64 after 540 stamps: dE76 mean {dE.mean():.4f} max {dE.max():.4f}')
    # --- S2: boundary darkening. Profile across the stroke at x=80: luminance of shaded image and of albedo
    col = 80
    Lsh = lum(img[:, col]); Lal = lum(alb[:, col]); Hc = H[:, col] * 1e6
    print('S2  cross-section at x=80 (y: shaded L, albedo L, height um, n.l):')
    for y in list(range(16, 49, 2)):
        print(f'      y={y:2d}  shaded {Lsh[y]:.3f}  albedo {Lal[y]:.3f}  H {Hc[y]:7.1f}  n.l {nl[y,col]:.3f}')
    ground = lum(alb[5, col]); edge_alb = Lal[19:46].min(); interior = Lal[32]
    print(f'S2  albedo: ground {ground:.3f}, stroke interior {interior:.3f}, min across edge {edge_alb:.3f} -> edge darker than both? {edge_alb < min(ground, interior) - 0.01}')
    print(f'S2  shaded: min across edge {Lsh[19:46].min():.3f} vs interior {Lsh[32]:.3f} (relief shading; light from -x,+y)')
    # S2b: does a thin translucent edge (mask ramp) composite darker than either colour? albedo at ramp cells
    ramp = [(y, Lal[y], c.p[:, y, col].sum() * 1e6) for y in (20, 21, 22, 42, 43, 44)]
    print('S2  ramp cells (y, albedo L, pigment um):', [(y, round(L, 3), round(pg, 1)) for y, L, pg in ramp])

# ------------------------------------------------------------------ S3: does flow stop?
def test_S3():
    P = Params(); P.nx, P.ny = 160, 96   # cell = 0.488 mm -> canvas 78 x 47 mm
    def run(phi0, gvec, seconds, label, depth=1e-3, r=12):
        c = Canvas(P); c.gvec = np.array(gvec, float)
        yy, xx = np.mgrid[0:P.ny, 0:P.nx]; m = ((xx - 48) ** 2 + (yy - 48) ** 2) < r * r
        c.d[0][m] = depth; c.p[0] = phi0 * c.d[0]
        V0 = c.d.sum(); area0 = (c.d.sum(0) > P.eps_d).sum()
        hist = []
        for t in range(int(seconds / P.dt)):
            c.step()
            if t % int(5 / P.dt) == 0 or t == int(seconds / P.dt) - 1:
                D = c.d.sum(0); wet = D > P.eps_d
                hist.append((t * P.dt, wet.sum(), np.abs(c.u).max() * P.dt / P.dx, D.max() * 1e6, (D * xx).sum() / D.sum()))
        print(f'S3  {label}: volume drift {abs(c.d.sum()-V0)/V0:.1e}; wet cells {area0} -> {hist[-1][1]} (canvas {P.nx*P.ny}); '
              f'max|u| at end {hist[-1][2]:.2e} cells/step; max depth {hist[-1][3]:.1f} um; CoM x {hist[-1][4]:.1f}')
        for t, w, mu, dm, cx in hist:
            print(f'       t={t:5.1f}s wet={w:6d} max|u|={mu:.2e} maxdepth={dm:7.1f}um CoMx={cx:6.1f}')
    run(0.1, (0, 0, 9.81), 60, 'dilute phi=0.1, flat canvas, 60 s')
    run(0.1, (9.81 * np.sin(np.radians(30)), 0, 9.81 * np.cos(np.radians(30))), 60, 'dilute phi=0.1, 30 deg tilt, 60 s')
    run(0.5, (9.81 * np.sin(np.radians(30)), 0, 9.81 * np.cos(np.radians(30))), 30, 'phi=0.5, 30 deg tilt, 30 s')
    run(0.1, (9.81 * np.sin(np.radians(30)), 0, 9.81 * np.cos(np.radians(30))), 30, 'dilute, 3 mm blob, 30 deg tilt, 30 s', depth=3e-3)

if __name__ == '__main__':
    test_S1_S2()
    test_S3()
