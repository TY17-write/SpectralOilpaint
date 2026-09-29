"""Same as exp_roundtrip but the state carries (RGB, B) where B = sum c_i sigma_i (S per unit volume) explicitly."""
import numpy as np
from exp_color import CMF, XYZ_RGB, lrgb_to_R, KS, KM, dE76, uncompand
rng = np.random.default_rng(3)
KAPPA = 0.5
Y = lambda R: R @ CMF[1]
sigma = lambda R: max(Y(R), 1e-4) ** KAPPA
def mix_ks(KSs, Bs, vols):
    w = np.array(Bs) * np.array(vols)
    return (w[:, None] * np.array(KSs)).sum(0) / w.sum(), w.sum() / np.sum(vols)
def reproject(R): return lrgb_to_R(np.clip(XYZ_RGB @ (CMF @ R), 0, 1))
def paint(rgb):  # user colour -> (R, B)
    R = lrgb_to_R(uncompand(rgb)); return R, sigma(R)
for f, n in ((0.1, 20), (0.05, 50), (0.3, 10), (0.5, 4)):
    errs, errs38 = [], []
    for _ in range(800):
        R0, B0 = paint(rng.random(3)); RA, BA = paint(rng.random(3))
        ks_ref, _ = mix_ks([KS(R0), KS(RA)], [B0, BA], [(1 - f) ** n, 1 - (1 - f) ** n])
        Rc, Bc = R0, B0
        for i in range(n):
            ks, Bc = mix_ks([KS(Rc), KS(RA)], [Bc, BA], [1 - f, f])
            Rc = reproject(KM(ks))            # RGB storage: metamer swap each step
        errs.append(dE76(CMF @ KM(ks_ref), CMF @ Rc))
    print(f'smudge f={f} n={n} (RGB+B storage): mean {np.mean(errs):.3f} p99 {np.percentile(errs,99):.3f} max {np.max(errs):.3f}')
errs = []
for _ in range(2000):
    P = [paint(rng.random(3)) for _ in range(3)]; v = rng.random(3) + 0.05
    ks_ref, _ = mix_ks([KS(p[0]) for p in P], [p[1] for p in P], v)
    ks_ab, B_ab = mix_ks([KS(P[0][0]), KS(P[1][0])], [P[0][1], P[1][1]], v[:2]); R_ab = reproject(KM(ks_ab))
    ks_abc, _ = mix_ks([KS(R_ab), KS(P[2][0])], [B_ab, P[2][1]], [v[0] + v[1], v[2]])
    errs.append(dE76(CMF @ KM(ks_ref), CMF @ reproject(KM(ks_abc))))
print(f'associativity 3-way (RGB+B): mean {np.mean(errs):.3f} p99 {np.percentile(errs,99):.3f} max {np.max(errs):.3f}')
