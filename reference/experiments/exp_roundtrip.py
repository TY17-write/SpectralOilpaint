"""Storage as linear RGB (re-projected after every KS mix) vs exact 38-sample KS storage.
How much error does repeated re-projection (smudging) introduce?"""
import numpy as np
from exp_color import BASE, CMF, XYZ_RGB, lrgb_to_R, KS, KM, xyz_to_lab, dE76, uncompand, R_FLOOR, R_CEIL
rng = np.random.default_rng(3)
KAPPA = 0.5
Y = lambda R: R @ CMF[1]
sigma = lambda R: max(Y(R), 1e-4) ** KAPPA

def mix_exact(Rs, vols):
    w = np.array([v * sigma(R) for R, v in zip(Rs, vols)])
    return KM((w[:, None] * KS(np.array(Rs))).sum(0) / w.sum())

def to_lrgb(R):
    return XYZ_RGB @ (CMF @ R)

def reproject(R):
    """spectrum -> XYZ -> linear RGB (clamped to [0,1]) -> spectral.js reconstruction"""
    return lrgb_to_R(np.clip(to_lrgb(R), 0, 1))

# 1) single re-projection error of a mixture (metamer swap), and gamut clipping incidence
e1, clipped = [], 0
for _ in range(3000):
    k = rng.integers(2, 4)
    Rs = [lrgb_to_R(uncompand(rng.random(3))) for _ in range(k)]
    Rm = mix_exact(Rs, rng.random(k) + 0.05)
    rgb = to_lrgb(Rm)
    if (rgb < -1e-6).any() or (rgb > 1 + 1e-6).any(): clipped += 1
    e1.append(dE76(CMF @ Rm, CMF @ reproject(Rm)))
print(f'1) single reprojection: mean {np.mean(e1):.3f} p99 {np.percentile(e1,99):.3f} max {np.max(e1):.3f}; out-of-gamut {clipped/3000:.1%}')

# 2) smudge chain: c <- mix(c : A) with volumes (1-f : f) repeated n times, compare stored-RGB pipeline
#    against exact KS accumulation. Exact reference: after n steps the cell holds volumes
#    (1-f)^n of the original and 1-(1-f)^n of A -> a two-component mix.
for f, n in ((0.1, 20), (0.05, 50), (0.3, 10)):
    errs = []
    for _ in range(1000):
        R0 = lrgb_to_R(uncompand(rng.random(3))); RA = lrgb_to_R(uncompand(rng.random(3)))
        # exact
        ref = mix_exact([R0, RA], [(1 - f) ** n, 1 - (1 - f) ** n])
        # RGB-storage pipeline
        Rc = R0
        for i in range(n):
            Rc = reproject(mix_exact([Rc, RA], [1 - f, f]))
        errs.append(dE76(CMF @ ref, CMF @ Rc))
    print(f'2) smudge f={f} n={n}: mean {np.mean(errs):.3f} p99 {np.percentile(errs,99):.3f} max {np.max(errs):.3f}')

# 3) associativity: (A+B)+C vs exact A+B+C with RGB storage
errs = []
for _ in range(2000):
    Rs = [lrgb_to_R(uncompand(rng.random(3))) for _ in range(3)]
    v = rng.random(3) + 0.05
    ref = mix_exact(Rs, v)
    ab = reproject(mix_exact(Rs[:2], v[:2]))
    abc = reproject(mix_exact([ab, Rs[2]], [v[0] + v[1], v[2]]))
    errs.append(dE76(CMF @ ref, CMF @ abc))
print(f'3) associativity 3-way: mean {np.mean(errs):.3f} p99 {np.percentile(errs,99):.3f} max {np.max(errs):.3f}')

# 4) is sigma (=Y^kappa) preserved through reprojection? (Y is exactly preserved by construction)
d = [abs(Y(Rm := mix_exact([lrgb_to_R(uncompand(rng.random(3))) for _ in range(2)], rng.random(2)+0.05)) - Y(reproject(Rm))) for _ in range(500)]
print(f'4) |Y - Y(reproject)| max {max(d):.2e}')
