import numpy as np, proto_fluid as pf
class P2(pf.Params):
    dx = 0.5/1024; dt = 1/45; g = 9.81; h_max = 3e-3; d_grav_min = 0.01
    nu_min, nu_max = 1e-4, 1e-1; u_clamp_cells = 0.02; n_iter = 16; omega = 1.0; friction = True
class Sim2(pf.Sim):
    def step(self, brush_bc=None):
        P = self.P; dx, dt = P.dx, P.dt
        d, b = self.d, self.b; eta = b + d; phi = self.phi(); Ph = pf.Phi(phi, P)
        wet = d >= P.d_grav_min * P.h_max
        g_eff = np.where(wet, (1 - Ph) * self.gvec[2], 0.0)
        gl_x = np.where(wet, (1 - Ph) * self.gvec[0], 0.0); gl_y = np.where(wet, (1 - Ph) * self.gvec[1], 0.0)
        ny, nx = self.ny, self.nx
        jx, ix = np.mgrid[0:ny, 0:nx + 1]; v_at_xf = np.zeros_like(self.u)
        v_c = 0.5 * (self.v[:-1, :] + self.v[1:, :]); v_at_xf[:, 1:-1] = 0.5 * (v_c[:, :-1] + v_c[:, 1:])
        u_t = pf.bilerp(self.u, ix - self.u * dt / dx, jx - v_at_xf * dt / dx)
        jy, iy = np.mgrid[0:ny + 1, 0:nx]; u_at_yf = np.zeros_like(self.v)
        u_c = 0.5 * (self.u[:, :-1] + self.u[:, 1:]); u_at_yf[1:-1, :] = 0.5 * (u_c[:-1, :] + u_c[1:, :])
        v_t = pf.bilerp(self.v, iy - u_at_yf * dt / dx, jy - self.v * dt / dx)
        u_t += dt * self.face_avg_x(gl_x); v_t += dt * self.face_avg_y(gl_y)
        nu = pf.nu_of(phi, P); nu_x = self.face_avg_x(nu); nu_y = self.face_avg_y(nu)
        for F, nuf in ((u_t, nu_x), (v_t, nu_y)):
            Fp = np.pad(F, 1, mode='edge')
            mean4 = 0.25 * (Fp[1:-1, :-2] + Fp[1:-1, 2:] + Fp[:-2, 1:-1] + Fp[2:, 1:-1])
            F[:] = mean4 + (F - mean4) * np.exp(-4 * nuf * dt / dx**2)
        u_t[:, 0] = u_t[:, -1] = 0; v_t[0, :] = v_t[-1, :] = 0
        d_fx = self.face_avg_x(d); d_fy = self.face_avg_y(d)
        fx = 1 / (1 + 3 * nu_x * dt / np.maximum(d_fx, 1e-9) ** 2); fy = 1 / (1 + 3 * nu_y * dt / np.maximum(d_fy, 1e-9) ** 2)
        if not P.friction: fx[:] = 1; fy[:] = 1
        u_t *= fx; v_t *= fy
        g_fx = self.face_avg_x(g_eff) * fx; g_fy = self.face_avg_y(g_eff) * fy
        cx = (dt / dx) ** 2 * g_fx * d_fx; cy = (dt / dx) ** 2 * g_fy * d_fy
        rhs = eta - (dt / dx) * ((d_fx * u_t)[:, 1:] - (d_fx * u_t)[:, :-1] + (d_fy * v_t)[1:, :] - (d_fy * v_t)[:-1, :])
        diag = 1 + cx[:, 1:] + cx[:, :-1] + cy[1:, :] + cy[:-1, :]
        def nbsum(e):
            ep = np.pad(e, 1, mode='edge')
            return cx[:, 1:] * ep[1:-1, 2:] + cx[:, :-1] * ep[1:-1, :-2] + cy[1:, :] * ep[2:, 1:-1] + cy[:-1, :] * ep[:-2, 1:-1]
        e = eta.copy(); r0 = np.abs(diag * e - nbsum(e) - rhs).max()
        mask = ((np.indices(e.shape).sum(0)) % 2 == 0)
        for _ in range(P.n_iter):
            for m in (mask, ~mask):
                e = np.where(m, (1 - P.omega) * e + P.omega * (rhs + nbsum(e)) / diag, e)
        self.last_resid = (r0, np.abs(diag * e - nbsum(e) - rhs).max())
        u_n = u_t.copy(); v_n = v_t.copy()
        u_n[:, 1:-1] -= dt * g_fx[:, 1:-1] * (e[:, 1:] - e[:, :-1]) / dx
        v_n[1:-1, :] -= dt * g_fy[1:-1, :] * (e[1:, :] - e[:-1, :]) / dx
        uc = P.u_clamp_cells * dx / dt
        u_c = 0.5 * (u_n[:, :-1] + u_n[:, 1:]); v_c = 0.5 * (v_n[:-1, :] + v_n[1:, :])
        keep = (np.sqrt(u_c**2 + v_c**2) >= uc).astype(float)
        u_n[:, 1:-1] *= np.maximum(keep[:, :-1], keep[:, 1:]); v_n[1:-1, :] *= np.maximum(keep[:-1, :], keep[1:, :])
        u_n[:, 0] = u_n[:, -1] = 0; v_n[0, :] = v_n[-1, :] = 0
        self.u, self.v = u_n, v_n
        self._sweep(self.u, axis=1); self._sweep(self.v, axis=0)

def run(name, phi0, gvec, steps, n_iter=16, omega=1.0, depth=1e-3, nx=128, ny=96, blob=(48, 32, 12), quiet=False):
    P = P2(); P.n_iter = n_iter; P.omega = omega
    s = Sim2(nx, ny, P); s.gvec = np.array(gvec, float)
    yy, xx = np.mgrid[0:ny, 0:nx]; cx, cy, r = blob
    s.d[((xx - cx) ** 2 + (yy - cy) ** 2) < r * r] = depth; s.p[:] = phi0 * s.d
    V0 = s.d.sum(); maxu = 0; worst = 0
    for i in range(steps):
        s.step(); maxu = max(maxu, np.abs(s.u).max() * P.dt / P.dx); worst = max(worst, s.last_resid[1] / max(s.last_resid[0], 1e-30))
        if not np.isfinite(s.d).all(): print(name, 'NaN at', i); return
    com = (s.d * xx).sum() / s.d.sum()
    if not quiet: print(f"{name:40s} vol drift {abs(s.d.sum()-V0)/V0:.1e} max|u| {maxu:7.3f} c/step  CoM {cx}->{com:6.1f}  maxdepth {s.d.max()*1e3:.3f}mm  worst resid ratio {worst:.2e}")
    return s
if __name__ == '__main__':
    T = 450
    for n_iter, om in ((16, 1.0), (32, 1.0), (64, 1.0), (16, 1.5), (32, 1.5), (32, 1.7)):
        run(f'3mm 60deg thin RBGS{n_iter} w={om}', 0.1, (8.5, 0, 4.9), T, n_iter=n_iter, omega=om, depth=3e-3)
    run('3mm 60deg thin RBGS32 w1.5 256x192 blob r24', 0.1, (8.5, 0, 4.9), T, n_iter=32, omega=1.5, depth=3e-3, nx=256, ny=192, blob=(96, 64, 24))
