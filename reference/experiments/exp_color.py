"""Numerical experiments for the oil-paint spec.
A) how many wavelength nodes are needed for KS-space mixing + display
B) black/white 50:50 result vs. scattering exponent kappa
C) sanity of finite-thickness Kubelka formulas
Tables are parsed from spectral.js (MIT) source verbatim.
"""
import re, json, numpy as np
np.set_printoptions(precision=5, suppress=True, linewidth=150)
SRC = open('spectral.js/spectral.js').read()

def arr(name):
    m = re.search(name + r':\s*\[(.*?)\]', SRC, re.S)
    return np.array([float(x) for x in m.group(1).replace('\n', ' ').split(',') if x.strip()])

BASE = {k: arr(r'\b' + k) for k in 'WCMYRGB'}
cmf_block = re.search(r'CMF:\s*\[(.*?)\],\s*\}\)', SRC, re.S).group(1)
rows = re.findall(r'\[(.*?)\]', cmf_block, re.S)
CMF = np.array([[float(x) for x in r.replace('\n', ' ').split(',') if x.strip()] for r in rows])  # 3x38, D65-weighted, sum(Y)=1
assert CMF.shape == (3, 38) and all(BASE[k].shape == (38,) for k in BASE)
LAM = 380 + 10 * np.arange(38)
RGB_XYZ = np.array([[0.41239079926595934, 0.357584339383878, 0.1804807884018343],
                    [0.21263900587151027, 0.715168678767756, 0.07219231536073371],
                    [0.01933081871559182, 0.11919477979462598, 0.9505321522496607]])
XYZ_RGB = np.linalg.inv(RGB_XYZ)
print('sum CMF rows (X,Y,Z):', CMF.sum(1))

R_FLOOR, R_CEIL = 0.005, 0.995

def lrgb_to_R(lrgb, base=BASE):
    r_, g_, b_ = lrgb
    w = min(lrgb); r_, g_, b_ = r_ - w, g_ - w, b_ - w
    c = min(g_, b_); m = min(r_, b_); y = min(r_, g_)
    r = max(0, min(r_ - b_, r_ - g_)); g = max(0, min(g_ - b_, g_ - r_)); b = max(0, min(b_ - g_, b_ - r_))
    R = w * base['W'] + c * base['C'] + m * base['M'] + y * base['Y'] + r * base['R'] + g * base['G'] + b * base['B']
    return np.clip(R, R_FLOOR, R_CEIL)

KS = lambda R: (1 - R) ** 2 / (2 * R)
KM = lambda ks: 1 + ks - np.sqrt(ks * ks + 2 * ks)

def xyz_to_lab(xyz, wp=np.array([0.95047, 1.0, 1.08883])):
    t = xyz / wp
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])

def dE76(x1, x2): return np.linalg.norm(xyz_to_lab(x1) - xyz_to_lab(x2))

# ---- test set of reflectances ----
rng = np.random.default_rng(1)
grid = np.linspace(0, 1, 6)
srgb_grid = np.array(np.meshgrid(grid, grid, grid)).reshape(3, -1).T
uncompand = lambda x: np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
single = [lrgb_to_R(uncompand(c)) for c in srgb_grid]
# KS mixtures of 2..4 random colours with random volume weights (sigma = Y^0.5 weighting, see B)
def mixR(Rs, ws):
    ws = np.asarray(ws)[:, None]
    return KM((ws * KS(np.array(Rs))).sum(0) / ws.sum())
mixes = []
for _ in range(3000):
    k = rng.integers(2, 5)
    cols = [lrgb_to_R(uncompand(rng.random(3))) for _ in range(k)]
    mixes.append(mixR(cols, rng.random(k) + 0.05))
TEST = np.array(single + mixes)
print('test spectra:', TEST.shape)

# ---- A) node count -----------------------------------------------------------
def gl_nodes(n, a=380, b=750):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w

def interp(tab, nodes):  # linear interpolation of a 38-sample table at nodes
    return np.interp(nodes, LAM, tab)

results = {}
for n in (6, 8, 10, 12, 16, 20):
    nodes, w = gl_nodes(n)
    # (1) plain Gauss-Legendre weights: XYZ ~ sum_k w_k * cmf(lam_k)/10 * R(lam_k)
    Wgl = np.array([w * interp(CMF[i], nodes) / 10 for i in range(3)])
    # (2) fitted weights: least squares on the training set (node-sampled R -> XYZ38)
    A = np.array([interp(R, nodes) for R in TEST])          # T x n
    Bref = TEST @ CMF.T                                     # T x 3
    Wfit = np.linalg.lstsq(A, Bref, rcond=None)[0].T        # 3 x n
    # error of *mixing at nodes* (the real pipeline): mix in node space, display with W
    errs = {'gl': [], 'fit': []}
    for _ in range(2000):
        k = rng.integers(2, 5)
        cols = [lrgb_to_R(uncompand(rng.random(3))) for _ in range(k)]
        ws = rng.random(k) + 0.05
        ref = mixR(cols, ws) @ CMF.T
        node_cols = [interp(c, nodes) for c in cols]
        Rn = mixR(node_cols, ws)
        errs['gl'].append(dE76(ref, Wgl @ Rn)); errs['fit'].append(dE76(ref, Wfit @ Rn))
    # error for single (unmixed) colours through the node path
    e_single = [dE76(R @ CMF.T, Wfit @ interp(R, nodes)) for R in single]
    results[n] = dict(gl_mean=np.mean(errs['gl']), gl_max=np.max(errs['gl']), gl_p99=np.percentile(errs['gl'], 99),
                      fit_mean=np.mean(errs['fit']), fit_max=np.max(errs['fit']), fit_p99=np.percentile(errs['fit'], 99),
                      single_fit_max=np.max(e_single), nodes=nodes.round(2).tolist(), Wfit=Wfit.tolist(), Wgl=Wgl.tolist())
    print(f"N={n:2d}  GL: mean {results[n]['gl_mean']:.3f} p99 {results[n]['gl_p99']:.3f} max {results[n]['gl_max']:.3f} | "
          f"FIT: mean {results[n]['fit_mean']:.3f} p99 {results[n]['fit_p99']:.3f} max {results[n]['fit_max']:.3f} | single max {results[n]['single_fit_max']:.3f}")

# ---- B) black/white 50:50 vs kappa ------------------------------------------
Rw = lrgb_to_R([1, 1, 1]); Rk = lrgb_to_R([0, 0, 0])
Yw, Yk = (Rw @ CMF[1]), (Rk @ CMF[1])
print(f"\nY_white={Yw:.4f} Y_black={Yk:.4f}")
print('kappa   Y_mix   L*_mix   (weights sigma=Y^kappa, equal volumes)')
for kap in (0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
    sw, sk = Yw ** kap, Yk ** kap
    Rm = KM((sw * KS(Rw) + sk * KS(Rk)) / (sw + sk))
    xyz = Rm @ CMF.T
    print(f"{kap:5.2f}  {xyz[1]:.4f}   {xyz_to_lab(xyz)[0]:6.2f}")

# ---- C) finite-thickness Kubelka sanity ---------------------------------------
def layer_RT(K_x, S_x):
    """Kubelka finite layer with K*x, S*x (dimensionless). Returns R,T. S_x -> 0 handled by limit."""
    S_x = np.maximum(S_x, 1e-9)
    a = 1 + K_x / S_x
    b = np.sqrt(np.maximum(a * a - 1, 0))
    sh, ch = np.sinh(b * S_x), np.cosh(b * S_x)
    denom = a * sh + b * ch
    R = sh / denom
    T = b / denom
    # pure absorber limit (b->0): R->0, T->exp(-K x)
    return np.where(b > 1e-6, R, 0.0), np.where(b > 1e-6, T, np.exp(-K_x))

def composite(R1, T1, R2):
    return R1 + T1 * T1 * R2 / (1 - R1 * R2)

ks = KS(lrgb_to_R(uncompand(np.array([0.9, 0.2, 0.1]))))  # a red
for Sx in (0.01, 0.1, 1, 5, 50):
    R, T = layer_RT(ks * Sx, Sx)
    print(f"S*x={Sx:6.2f}: R/Rinf max dev {np.max(np.abs(R - KM(ks))):.4f}  T mean {T.mean():.4f}  overWhite Y {composite(R,T,0.995*np.ones(38))@CMF[1]:.4f}  overBlack Y {composite(R,T,0.005*np.ones(38))@CMF[1]:.4f}")

json.dump({'nodes_results': {str(k): v for k, v in results.items()}}, open('quad_results.json', 'w'))
