"""Reference prototype of the paint fluid step (spec section 4).
Arakawa-C grid, Casulli-type semi-implicit SWE (Jacobi), exponential viscosity,
Stuyck density-modulated gravity, velocity clamp, conservative limited layer transport.
Single layer here (transport is per-layer identical). Units: Stuyck (L=1 canvas width, m, s).
"""
import numpy as np

class Params:
    dx = 1 / 1024; dt = 1 / 45
    g = 9.81
    h_max = 0.05                    # fluid cap (Stuyck maxMedium)
    phi_a, phi_b = 0.2, 0.8         # Stuyck gamma, zeta on normalised pigment ratio
    nu_min, nu_max = 1e-6, 1e-2     # Stuyck table 1
    u_clamp_cells = 0.2             # velocity clamp in cells/step (IMPaSTo gate analogue)
    d_grav_min = 0.01               # fraction of h_max below which gravity is ignored (Stuyck 1%)
    n_jacobi = 4

def smoothstep(t): t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)

def Phi(phi, P):  return smoothstep((phi - P.phi_a) / (P.phi_b - P.phi_a))
def nu_of(phi, P): return P.nu_min + smoothstep((phi - P.phi_a) / (P.phi_b - P.phi_a)) * (P.nu_max - P.nu_min)

def bilerp(F, x, y):
    """sample array F (ny,nx) at fractional index coords (x,y), clamped."""
    ny, nx = F.shape
    x = np.clip(x, 0, nx - 1); y = np.clip(y, 0, ny - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, nx - 1); y1 = np.minimum(y0 + 1, ny - 1)
    fx = x - x0; fy = y - y0
    return (F[y0, x0] * (1 - fx) * (1 - fy) + F[y0, x1] * fx * (1 - fy) + F[y1, x0] * (1 - fx) * fy + F[y1, x1] * fx * fy)

class Sim:
    def __init__(self, nx, ny, P=Params()):
        self.P = P; self.nx, self.ny = nx, ny
        self.b = np.zeros((ny, nx))            # bed (paper + dried)
        self.d = np.zeros((ny, nx))            # fluid depth (single layer)
        self.p = np.zeros((ny, nx))            # pigment thickness (<= d)
        self.u = np.zeros((ny, nx + 1))        # x-face velocities (depth-averaged)
        self.v = np.zeros((ny + 1, nx))        # y-face velocities
        self.gvec = np.array([0.0, 0.0, P.g])  # (gx, gy, gz>0 down into canvas)

    # ---- helpers ------------------------------------------------------------
    def phi(self):
        return np.where(self.d > 1e-12, self.p / np.maximum(self.d, 1e-12), 0.0)

    def face_avg_x(self, c):  # cell -> x-faces (ny, nx+1), zero at boundary
        f = np.zeros((self.ny, self.nx + 1)); f[:, 1:-1] = 0.5 * (c[:, :-1] + c[:, 1:]); return f
    def face_avg_y(self, c):
        f = np.zeros((self.ny + 1, self.nx)); f[1:-1, :] = 0.5 * (c[:-1, :] + c[1:, :]); return f

    def step(self, brush_bc=None):
        P = self.P; dx, dt = P.dx, P.dt
        d, b = self.d, self.b
        eta = b + d
        phi = self.phi()
        Ph = Phi(phi, P)
        g_eff = (1 - Ph) * self.gvec[2]                       # Stuyck modulated gravity (per cell)
        g_eff = np.where(d < P.d_grav_min * P.h_max, 0.0, g_eff)
        gl_x = (1 - Ph) * self.gvec[0]; gl_y = (1 - Ph) * self.gvec[1]
        gl_x = np.where(d < P.d_grav_min * P.h_max, 0.0, gl_x); gl_y = np.where(d < P.d_grav_min * P.h_max, 0.0, gl_y)

        # ---- 1. momentum predictor: semi-Lagrangian advection of face velocities --------
        ny, nx = self.ny, self.nx
        # velocity at x-faces: (u, v interpolated)
        jx, ix = np.mgrid[0:ny, 0:nx + 1]
        v_at_xf = np.zeros_like(self.u)
        v_c = 0.5 * (self.v[:-1, :] + self.v[1:, :])          # v at cell centres
        v_at_xf[:, 1:-1] = 0.5 * (v_c[:, :-1] + v_c[:, 1:])
        xb = ix - self.u * dt / dx; yb = jx - v_at_xf * dt / dx
        u_t = bilerp(self.u, xb, yb)
        jy, iy = np.mgrid[0:ny + 1, 0:nx]
        u_at_yf = np.zeros_like(self.v)
        u_c = 0.5 * (self.u[:, :-1] + self.u[:, 1:])
        u_at_yf[1:-1, :] = 0.5 * (u_c[:-1, :] + u_c[1:, :])
        xb = iy - u_at_yf * dt / dx; yb = jy - self.v * dt / dx
        v_t = bilerp(self.v, xb, yb)
        # lateral gravity (body force), face-averaged
        u_t += dt * self.face_avg_x(gl_x); v_t += dt * self.face_avg_y(gl_y)
        # viscosity: exponential relaxation towards 4-neighbour mean (unconditionally stable)
        nu_x = self.face_avg_x(nu_of(phi, P)); nu_y = self.face_avg_y(nu_of(phi, P))
        for F, nu in ((u_t, nu_x), (v_t, nu_y)):
            Fp = np.pad(F, 1, mode='edge')
            mean4 = 0.25 * (Fp[1:-1, :-2] + Fp[1:-1, 2:] + Fp[:-2, 1:-1] + Fp[2:, 1:-1])
            k = np.exp(-4 * nu * dt / dx**2)
            F[:] = mean4 + (F - mean4) * k
        u_t[:, 0] = u_t[:, -1] = 0; v_t[0, :] = v_t[-1, :] = 0   # closed box

        # ---- 2. semi-implicit surface (Casulli): eta - (dt^2/dx^2) div(g d grad eta) = eta^n - dt div(d u~)
        d_fx = self.face_avg_x(d); d_fy = self.face_avg_y(d)
        g_fx = self.face_avg_x(g_eff); g_fy = self.face_avg_y(g_eff)
        cx = (dt / dx) ** 2 * g_fx * d_fx                    # (ny, nx+1)
        cy = (dt / dx) ** 2 * g_fy * d_fy                    # (ny+1, nx)
        rhs = eta - (dt / dx) * ((d_fx * u_t)[:, 1:] - (d_fx * u_t)[:, :-1] + (d_fy * v_t)[1:, :] - (d_fy * v_t)[:-1, :])
        diag = 1 + cx[:, 1:] + cx[:, :-1] + cy[1:, :] + cy[:-1, :]
        e = eta.copy()
        for _ in range(P.n_jacobi):
            ep = np.pad(e, 1, mode='edge')
            nb = cx[:, 1:] * ep[1:-1, 2:] + cx[:, :-1] * ep[1:-1, :-2] + cy[1:, :] * ep[2:, 1:-1] + cy[:-1, :] * ep[:-2, 1:-1]
            e = (rhs + nb) / diag
        # velocities from the (approximate) new surface
        u_n = u_t.copy(); v_n = v_t.copy()
        u_n[:, 1:-1] -= dt * g_fx[:, 1:-1] * (e[:, 1:] - e[:, :-1]) / dx
        v_n[1:-1, :] -= dt * g_fy[1:-1, :] * (e[1:, :] - e[:-1, :]) / dx
        # brush Dirichlet BC (mask on faces, target depth-averaged velocity)
        if brush_bc is not None:
            mx, my, ub, vb = brush_bc
            u_n = (1 - mx) * u_n + mx * ub; v_n = (1 - my) * v_n + my * vb
        # velocity clamp (Stuyck surface tension): |U| below threshold -> 0
        uc = P.u_clamp_cells * dx / dt
        u_c = 0.5 * (u_n[:, :-1] + u_n[:, 1:]); v_c = 0.5 * (v_n[:-1, :] + v_n[1:, :])
        speed = np.sqrt(u_c**2 + v_c**2)
        keep = (speed >= uc).astype(float)
        u_n[:, 1:-1] *= np.maximum(keep[:, :-1], keep[:, 1:]); v_n[1:-1, :] *= np.maximum(keep[:-1, :], keep[1:, :])
        u_n[:, 0] = u_n[:, -1] = 0; v_n[0, :] = v_n[-1, :] = 0
        self.u, self.v = u_n, v_n

        # ---- 3. conservative, limited transport (x sweep then y sweep) -------------------
        for axis in (0, 1):
            if axis == 0:
                self._sweep(self.u, axis=1)
            else:
                self._sweep(self.v, axis=0)

    def _sweep(self, uf, axis):
        """upwind flux of (d, p) across faces along `axis` with per-cell outflow limiter."""
        P = self.P; dt, dx = P.dt, P.dx
        d, p = self.d, self.p
        if axis == 0:  # move along y: transpose
            d = d.T; p = p.T; uf = uf.T
        # uf: (ny, nx+1); upwind donor
        pos = uf > 0
        d_up = np.zeros_like(uf); p_up = np.zeros_like(uf)
        d_up[:, 1:-1] = np.where(pos[:, 1:-1], d[:, :-1], d[:, 1:])
        p_up[:, 1:-1] = np.where(pos[:, 1:-1], p[:, :-1], p[:, 1:])
        F = d_up * uf * dt / dx                     # unlimited depth flux (positive = +x)
        # per-cell outflow = sum of positive flux out east + negative flux out west
        out = np.maximum(F[:, 1:], 0) - np.minimum(F[:, :-1], 0)
        K = np.where(out > 1e-15, np.minimum(1.0, d / np.maximum(out, 1e-15)), 1.0)
        # scale each face flux by the donor cell's K
        K_face = np.ones_like(uf)
        K_face[:, 1:-1] = np.where(pos[:, 1:-1], K[:, :-1], K[:, 1:])
        F *= K_face
        Fp = F * np.where(d_up > 0, p_up / np.maximum(d_up, 1e-15), 0)
        d_new = d - (F[:, 1:] - F[:, :-1]); p_new = p - (Fp[:, 1:] - Fp[:, :-1])
        d_new = np.maximum(d_new, 0); p_new = np.clip(p_new, 0, d_new)
        if axis == 0: d_new = d_new.T; p_new = p_new.T
        self.d, self.p = d_new, p_new


def run(name, phi0, gvec, steps, n_jacobi=4, blob=(48, 32, 12), nx=128, ny=96):
    P = Params(); P.n_jacobi = n_jacobi
    s = Sim(nx, ny, P); s.gvec = np.array(gvec, float)
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy, r = blob
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
    s.d[mask] = 0.02; s.p[:] = phi0 * s.d
    V0 = s.d.sum(); Pg0 = s.p.sum()
    maxu, minD = 0, 0
    for i in range(steps):
        s.step()
        maxu = max(maxu, np.abs(s.u).max() * P.dt / P.dx); minD = min(minD, s.d.min())
        if not np.isfinite(s.d).all(): print(name, 'NaN at step', i); return
    com_x = (s.d * xx).sum() / s.d.sum()
    print(f"{name:34s} vol drift {abs(s.d.sum()-V0)/V0:.2e} pig drift {abs(s.p.sum()-Pg0)/max(Pg0,1e-30):.2e} "
          f"min d {minD:.1e} max|u| {maxu:.3f} cell/step  blob CoM x {cx}->{com_x:.1f}  max depth {s.d.max():.4f}")
    return s

if __name__ == '__main__':
    run('dilute (phi=0.1), tilt gx=5, J=4', 0.1, (5, 0, 8.4), 600)
    run('dilute (phi=0.1), tilt gx=5, J=20', 0.1, (5, 0, 8.4), 600, n_jacobi=20)
    run('dilute (phi=0.1), flat, J=4 (level)', 0.1, (0, 0, 9.81), 600)
    run('tube paint (phi=1.0), tilt gx=5', 1.0, (5, 0, 8.4), 600)
    run('mid (phi=0.5), tilt gx=9', 0.5, (9, 0, 4), 600)
    run('dilute, tilt gx=9.8 (edge-on), J=4', 0.1, (9.8, 0, 0.5), 1500)
