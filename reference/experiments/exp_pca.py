"""Can the 38-sample KS spectrum be stored as d linear coefficients (PCA) so that
mixing (linear in KS) stays exact and only display reconstruction is approximate?"""
import numpy as np, json
from exp_color import BASE, CMF, LAM, lrgb_to_R, KS, KM, xyz_to_lab, dE76, uncompand, mixR
rng = np.random.default_rng(7)

def rand_lrgb(n):
    return uncompand(rng.random((n, 3)))

# training set: dense sRGB grid (including near-black/near-white), KS spectra
g = np.linspace(0, 1, 17)
grid = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
train_R = np.array([lrgb_to_R(uncompand(c)) for c in grid])
train_KS = KS(train_R)
# also include mixtures so the basis spans mixed KS (mixtures are in the span anyway, linear)
print('KS range', train_KS.min(), train_KS.max())

# weighting: darker colours have huge KS; PCA on raw KS is dominated by them. Try both raw and
# "row-normalised" (each spectrum scaled by its max) PCA; reconstruction is still linear.
def pca_basis(X, d):
    mu = np.zeros(X.shape[1])  # no mean removal: keep linear (mixing must stay linear)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:d]  # d x 38 orthonormal

def evaluate(V, ntest=3000, d=None):
    errs, errs_single = [], []
    P = V.T @ V  # projector
    for _ in range(ntest):
        k = rng.integers(2, 5)
        cols = [lrgb_to_R(c) for c in rand_lrgb(k)]
        ws = rng.random(k) + 0.05
        ref = mixR(cols, ws) @ CMF.T
        coeff = np.array([V @ KS(c) for c in cols])          # store d coeffs per colour
        mixed_coeff = (ws[:, None] * coeff).sum(0) / ws.sum()  # linear mixing in coefficient space
        ks_rec = np.maximum(V.T @ mixed_coeff, 0)              # reconstruct, clamp >=0
        errs.append(dE76(ref, KM(ks_rec) @ CMF.T))
    for c in rand_lrgb(1000):
        R = lrgb_to_R(c); ks_rec = np.maximum(P @ KS(R), 0)
        errs_single.append(dE76(R @ CMF.T, KM(ks_rec) @ CMF.T))
    return np.mean(errs), np.percentile(errs, 99), np.max(errs), np.mean(errs_single), np.max(errs_single)

out = {}
for norm in ('raw', 'rownorm', 'sqrt'):
    if norm == 'raw': X = train_KS
    elif norm == 'rownorm': X = train_KS / train_KS.max(1, keepdims=True)
    else: X = np.sqrt(train_KS)  # emphasise mid tones; basis still applied linearly to raw KS
    for d in (6, 8, 10, 12, 14, 16, 20):
        V = pca_basis(X, d)
        r = evaluate(V)
        out[f'{norm}-{d}'] = r
        print(f'{norm:8s} d={d:2d}  mix: mean {r[0]:.3f} p99 {r[1]:.3f} max {r[2]:.3f} | single: mean {r[3]:.3f} max {r[4]:.3f}')
