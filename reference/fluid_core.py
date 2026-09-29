#!/usr/bin/env python3
"""Reference implementation of spec section 4 (fluid step) with N_L pigment layers,
plus acceptance tests T-F1..T-F6.  NumPy, float64, closed-box canvas.

The GPU implementation must reproduce these formulas pass by pass (see spec 4.x numbering).
Run:  python3 fluid_core.py
"""
import numpy as np

class Params:
    # units: SI
    nx, ny = 128, 96        # sim grid (test size)
    cell = 0.5 / 1024       # cell size [m] (= W/N_x for a 0.5 m canvas at 1024 cells); tests keep the cell size, not W
    dt = 1.0 / 60.0
    g = 9.81
    h_max = 3.0e-3          # fluid cap per cell [m]
    h_stroke = 0.5e-3       # nominal one-pass deposit [m]
    eps_d = 1e-3 * 0.5e-3   # layer "empty" threshold [m] = 5e-7
    eps_p = 0.01 * 1e-3 * 0.5e-3  # colour-reset threshold for pigment thickness
    phi_a, phi_b = 0.2, 0.8
    nu_min, nu_max = 1e-4, 1e-1
    d_grav_min = 0.01 * 3.0e-3
    g_z_min_frac = 0.05
    u_clamp = 6.0e-4        # yield velocity [m/s] (resolution independent). = 0.02 cells/step at 0.488 mm, 60 Hz
    N_GS = 32
    N_sub = 4
    N_L = 3
    @property
    def dx(self): return self.cell
    @property
    def W(self): return self.cell * self.nx

def S3(t):
    t = np.clip(t, 0.0, 1.0); return t * t * (3.0 - 2.0 * t)

def bilerp(F, x, y):
    ny, nx = F.shape
    x = np.clip(x, 0, nx - 1); y = np.clip(y, 0, ny - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, nx - 1); y1 = np.minimum(y0 + 1, ny - 1)
    fx = x - x0; fy = y - y0
    return F[y0, x0] * (1 - fx) * (1 - fy) + F[y0, x1] * fx * (1 - fy) + F[y1, x0] * (1 - fx) * fy + F[y1, x1] * fx * fy

class Canvas:
    def __init__(self, P):
        self.P = P; nx, ny, L = P.nx, P.ny, P.N_L
        self.b = np.zeros((ny, nx))                 # bed height (paper + dried paint)
        self.d = np.zeros((L, ny, nx))              # layer fluid thickness
        self.p = np.zeros((L, ny, nx))              # layer pigment thickness (<= d)
        self.s = np.zeros((L, ny, nx))              # layer dryness 0..1 (intensive)
        self.u = np.zeros((ny, nx + 1)); self.v = np.zeros((ny + 1, nx))
        self.gvec = np.array([0.0, 0.0, P.g])
        # colour descriptors (a[38], beta) are transported identically to s (intensive per pigment);
        # they are omitted here; see color_model.py and spec 4.11 for the extensive-flux rule.

    # -------- 4.1 cell auxiliaries
    def aux(self):
        P = self.P
        d = self.d.sum(0)
        pt = self.p.sum(0)
        phi = np.where(d > P.eps_d, pt / np.maximum(d, P.eps_d), 0.0)
        sbar = np.where(d > P.eps_d, (self.s * self.d).sum(0) / np.maximum(d, P.eps_d), 0.0)
        phi_eff = phi + sbar * (1.0 - phi)
        t = (phi_eff - P.phi_a) / (P.phi_b - P.phi_a)
        Phi = S3(t)
        nu = P.nu_min + np.clip(t, 0, 1) * (P.nu_max - P.nu_min)
        wet = d >= P.d_grav_min
        gz = max(self.gvec[2], P.g_z_min_frac * P.g)
        g_eff = np.where(wet, (1 - Phi) * gz, 0.0)
        gl_x = np.where(wet, (1 - Phi) * self.gvec[0], 0.0)
        gl_y = np.where(wet, (1 - Phi) * self.gvec[1], 0.0)
        return d, nu, g_eff, gl_x, gl_y

    def fx(self, c):  # cell -> x faces (ny, nx+1), boundary 0
        f = np.zeros((self.P.ny, self.P.nx + 1)); f[:, 1:-1] = 0.5 * (c[:, :-1] + c[:, 1:]); return f
    def fy(self, c):
        f = np.zeros((self.P.ny + 1, self.P.nx)); f[1:-1, :] = 0.5 * (c[:-1, :] + c[1:, :]); return f

    def step(self, brush=None):
        """brush: None or (m_cells (ny,nx) in [0,1], vbx, vby) [m/s] for the Dirichlet BC 4.9"""
        P = self.P; dx, dt = P.dx, P.dt; ny, nx = P.ny, P.nx
        d, nu, g_eff, gl_x, gl_y = self.aux()
        eta = self.b + d
        # -------- 4.2 face averages and friction factor
        d_fx, d_fy = self.fx(d), self.fy(d)
        nu_fx, nu_fy = self.fx(nu), self.fy(nu)
        f_x = 1.0 / (1.0 + 3.0 * nu_fx * dt / np.maximum(d_fx, P.eps_d) ** 2)
        f_y = 1.0 / (1.0 + 3.0 * nu_fy * dt / np.maximum(d_fy, P.eps_d) ** 2)
        # -------- 4.3 semi-Lagrangian advection of face velocities
        jx, ix = np.mgrid[0:ny, 0:nx + 1]
        v_c = 0.5 * (self.v[:-1, :] + self.v[1:, :]); v_xf = np.zeros_like(self.u); v_xf[:, 1:-1] = 0.5 * (v_c[:, :-1] + v_c[:, 1:])
        u_t = bilerp(self.u, ix - self.u * dt / dx, jx - v_xf * dt / dx)
        jy, iy = np.mgrid[0:ny + 1, 0:nx]
        u_c = 0.5 * (self.u[:, :-1] + self.u[:, 1:]); u_yf = np.zeros_like(self.v); u_yf[1:-1, :] = 0.5 * (u_c[:-1, :] + u_c[1:, :])
        v_t = bilerp(self.v, iy - u_yf * dt / dx, jy - self.v * dt / dx)
        # -------- 4.4 lateral gravity
        u_t += dt * self.fx(gl_x); v_t += dt * self.fy(gl_y)
        # -------- 4.5 viscous relaxation (exponential, unconditionally stable)
        for F, nuf in ((u_t, nu_fx), (v_t, nu_fy)):
            Fp = np.pad(F, 1, mode='edge')
            m4 = 0.25 * (Fp[1:-1, :-2] + Fp[1:-1, 2:] + Fp[:-2, 1:-1] + Fp[2:, 1:-1])
            F[:] = m4 + (F - m4) * np.exp(-4.0 * nuf * dt / dx ** 2)
        # -------- 4.6 bed friction (implicit) + closed boundary
        u_t *= f_x; v_t *= f_y
        u_t[:, 0] = u_t[:, -1] = 0.0; v_t[0, :] = v_t[-1, :] = 0.0
        # -------- 4.7 semi-implicit surface, red-black Gauss-Seidel
        gx_f = self.fx(g_eff) * f_x; gy_f = self.fy(g_eff) * f_y
        cx = (dt / dx) ** 2 * gx_f * d_fx; cy = (dt / dx) ** 2 * gy_f * d_fy
        rhs = eta - (dt / dx) * ((d_fx * u_t)[:, 1:] - (d_fx * u_t)[:, :-1] + (d_fy * v_t)[1:, :] - (d_fy * v_t)[:-1, :])
        diag = 1.0 + cx[:, 1:] + cx[:, :-1] + cy[1:, :] + cy[:-1, :]
        def nb(e):
            ep = np.pad(e, 1, mode='edge')
            return cx[:, 1:] * ep[1:-1, 2:] + cx[:, :-1] * ep[1:-1, :-2] + cy[1:, :] * ep[2:, 1:-1] + cy[:-1, :] * ep[:-2, 1:-1]
        e = eta.copy(); r0 = np.abs(diag * e - nb(e) - rhs).max()
        red = (np.indices(e.shape).sum(0) % 2 == 0)
        for _ in range(P.N_GS):
            for m in (red, ~red):
                e = np.where(m, (rhs + nb(e)) / diag, e)
        self.resid_ratio = np.abs(diag * e - nb(e) - rhs).max() / max(r0, 1e-30)
        # -------- 4.8 velocity correction
        u_n = u_t.copy(); v_n = v_t.copy()
        u_n[:, 1:-1] -= dt * gx_f[:, 1:-1] * (e[:, 1:] - e[:, :-1]) / dx
        v_n[1:-1, :] -= dt * gy_f[1:-1, :] * (e[1:, :] - e[:-1, :]) / dx
        # -------- 4.9 brush Dirichlet BC (surface velocity = brush velocity; depth mean = /1.5)
        if brush is not None:
            m, vbx, vby = brush
            mx, my = self.fx(m), self.fy(m)
            u_n = (1 - mx) * u_n + mx * (vbx / 1.5); v_n = (1 - my) * v_n + my * (vby / 1.5)
        # -------- 4.10 clamp (yield) and cap
        uc = P.u_clamp; ucap = P.N_sub * dx / dt
        u_c = 0.5 * (u_n[:, :-1] + u_n[:, 1:]); v_c = 0.5 * (v_n[:-1, :] + v_n[1:, :])
        keep = (np.hypot(u_c, v_c) >= uc).astype(float)
        u_n[:, 1:-1] *= np.maximum(keep[:, :-1], keep[:, 1:]); v_n[1:-1, :] *= np.maximum(keep[:-1, :], keep[1:, :])
        u_n = np.clip(u_n, -ucap, ucap); v_n = np.clip(v_n, -ucap, ucap)
        u_n[:, 0] = u_n[:, -1] = 0.0; v_n[0, :] = v_n[-1, :] = 0.0
        self.u, self.v = u_n, v_n
        # -------- 4.11 conservative layer transport, N_sub substeps, alternating sweep order
        for k in range(P.N_sub):
            order = (1, 0) if k % 2 == 0 else (0, 1)
            for axis in order:
                self.sweep(axis)

    def profile_factor(self, z1, z2, d):
        """mean of the Poiseuille profile 3(z - z^2/2) over [z1,z2] (zeta = z/d); sum_k d_k pi_k = d exactly."""
        with np.errstate(divide='ignore', invalid='ignore'):
            z1 = z1 / d; z2 = z2 / d
            pi = 3.0 * ((z2 ** 2 - z1 ** 2) / 2.0 - (z2 ** 3 - z1 ** 3) / 6.0) / (z2 - z1)
        return np.where((d > 0) & (z2 > z1), pi, 0.0)

    def sweep(self, axis):
        P = self.P; dts = P.dt / P.N_sub; dx = P.dx
        d, p, s = self.d, self.p, self.s
        uf = self.u if axis == 1 else self.v
        if axis == 0:
            d = d.transpose(0, 2, 1); p = p.transpose(0, 2, 1); s = s.transpose(0, 2, 1); uf = uf.T
        L, ny, nx = d.shape
        dtot = d.sum(0)
        zb = np.cumsum(d, 0) - d          # layer bottoms
        zt = np.cumsum(d, 0)              # layer tops
        pi = np.stack([self.profile_factor(zb[k], zt[k], dtot) for k in range(L)])
        pos = uf > 0
        def donor(q):                     # (L,ny,nx) -> donor value on faces (L,ny,nx+1)
            out = np.zeros((L, ny, nx + 1))
            out[:, :, 1:-1] = np.where(pos[None, :, 1:-1], q[:, :, :-1], q[:, :, 1:])
            return out
        d_D, pi_D = donor(d), donor(pi)
        F = d_D * pi_D * uf[None] * dts / dx                       # unlimited layer flux (+x)
        out = np.maximum(F[:, :, 1:], 0) - np.minimum(F[:, :, :-1], 0)
        K = np.where(out > 0, np.minimum(1.0, d / np.maximum(out, 1e-300)), 1.0)
        F *= donor(K)
        phi_D = donor(np.where(d > 0, p / np.maximum(d, 1e-300), 0.0))
        s_D = donor(s)
        Fp = F * phi_D; Fs = F * s_D                                # pigment and (extensive) dryness flux
        # (colour descriptors: F_a = Fp * a_D, F_beta = Fp * beta_D, same donor rule)
        d_new = np.maximum(d - (F[:, :, 1:] - F[:, :, :-1]), 0.0)
        p_new = np.clip(p - (Fp[:, :, 1:] - Fp[:, :, :-1]), 0.0, d_new)
        sd_new = s * d - (Fs[:, :, 1:] - Fs[:, :, :-1])
        s_new = np.where(d_new > P.eps_d, np.clip(sd_new / np.maximum(d_new, 1e-300), 0, 1), 0.0)
        if axis == 0:
            d_new = d_new.transpose(0, 2, 1); p_new = p_new.transpose(0, 2, 1); s_new = s_new.transpose(0, 2, 1)
        self.d, self.p, self.s = d_new, p_new, s_new

# ------------------------------------------------------------------ tests
def blob(P, phi0, depth, gvec, r=12, cx=48, cy=32, layers=1):
    c = Canvas(P); c.gvec = np.array(gvec, float)
    yy, xx = np.mgrid[0:P.ny, 0:P.nx]; m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
    for k in range(layers):
        c.d[k][m] = depth / layers; c.p[k] = phi0 * c.d[k]
    return c, xx

def run_tests():
    P = Params(); ok = True
    steps = int(10 / P.dt)  # 10 s
    # T-F1/F2/F3 thin paint runs downhill, exact conservation, non-negative
    c, xx = blob(P, 0.1, 1e-3, (P.g * np.sin(np.radians(30)), 0, P.g * np.cos(np.radians(30))), layers=3)
    V0, Pg0 = c.d.sum(), c.p.sum(); com0 = (c.d.sum(0) * xx).sum() / c.d.sum(); mind = 0; worst = 0; maxu = 0
    for _ in range(steps):
        c.step(); mind = min(mind, c.d.min()); worst = max(worst, c.resid_ratio); maxu = max(maxu, np.abs(c.u).max() * P.dt / P.dx)
    com = (c.d.sum(0) * xx).sum() / c.d.sum()
    dv = abs(c.d.sum() - V0) / V0; dp = abs(c.p.sum() - Pg0) / Pg0; disp = (com - com0) * P.dx * 1e3
    t1 = dv < 1e-9 and dp < 1e-9; t2 = mind >= 0; t3 = disp > 10; t5 = worst <= 0.1
    print(f'T-F1 conservation (3 layers, 10 s): vol {dv:.1e} pig {dp:.1e}  {"PASS" if t1 else "FAIL"}')
    print(f'T-F2 non-negativity: min d {mind:.1e}  {"PASS" if t2 else "FAIL"}')
    print(f'T-F3 thin paint (phi=0.1) 1 mm blob, 30 deg tilt: CoM moved {disp:.1f} mm downhill in 10 s  {"PASS" if t3 else "FAIL"}')
    print(f'T-F5 RB-GS N_GS={P.N_GS}: worst residual ratio {worst:.3f}  {"PASS" if t5 else "FAIL"}   (max |u| {maxu:.2f} cells/step)')
    ok &= t1 and t2 and t3 and t5
    # T-F4 tube paint does not move at all
    c, xx = blob(P, 1.0, 1e-3, (P.g * np.sin(np.radians(30)), 0, P.g * np.cos(np.radians(30))))
    d0 = c.d.copy()
    for _ in range(steps): c.step()
    t4 = np.array_equal(d0, c.d) and np.abs(c.u).max() == 0
    print(f'T-F4 tube paint (phi=1): bit-identical after 10 s, u==0  {"PASS" if t4 else "FAIL"}'); ok &= t4
    # T-F6 layer profile: sum_k d_k pi_k == d (transport of the total is the depth-mean velocity)
    rng = np.random.default_rng(0); dk = rng.random((3, 4, 4)) * 1e-3; dtot = dk.sum(0)
    zb = np.cumsum(dk, 0) - dk; zt = np.cumsum(dk, 0)
    cc = Canvas(P); tot = sum(dk[k] * cc.profile_factor(zb[k], zt[k], dtot) for k in range(3))
    t6 = np.allclose(tot, dtot, rtol=1e-12)
    print(f'T-F6 Poiseuille layer profile identity sum d_k*pi_k = d: max err {np.abs(tot - dtot).max():.1e}  {"PASS" if t6 else "FAIL"}'); ok &= t6
    # T-F7 drying stiffens: dilute blob with s=1 must not move
    c, xx = blob(P, 0.1, 1e-3, (P.g * np.sin(np.radians(30)), 0, P.g * np.cos(np.radians(30))))
    c.s[:] = np.where(c.d > 0, 1.0, 0.0); d0 = c.d.copy()
    for _ in range(60): c.step()
    t7 = np.array_equal(d0, c.d)
    print(f'T-F7 dry (s=1) dilute paint is frozen  {"PASS" if t7 else "FAIL"}'); ok &= t7
    print('ALL PASS' if ok else 'SOME FAILED')
    return ok

if __name__ == '__main__':
    import sys; sys.exit(0 if run_tests() else 1)
