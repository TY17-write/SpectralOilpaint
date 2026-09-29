#!/usr/bin/env python3
"""Generate tables.json for the paint engine from the spectral.js source (MIT).

Usage:  python3 gen_tables.py path/to/spectral.js  [--pca]

Output tables.json contains
  lambda_nm[38]          380..750 nm, 10 nm step
  base_spectra{W,C,M,Y,R,G,B}[38]   spectral.js BASE_SPECTRA, verbatim
  cmf[3][38]             spectral.js CIE.CMF (CIE 1931 2deg x D65, normalised so sum(cmf[1]) == 1), verbatim
  rgb_to_xyz[3][3], xyz_to_rgb[3][3]  spectral.js CONVERSION matrices, verbatim
  constants{...}         the colour constants fixed by the spec (R_FLOOR, R_CEIL, KAPPA, A_SCALE)
  pca20 (optional)       20x38 orthonormal basis for memory mode B, fitted on the sRGB 17^3 grid
"""
import re, sys, json
import numpy as np

def parse_array(src, name):
    m = re.search(r'\b' + name + r':\s*\[(.*?)\]', src, re.S)
    if not m:
        raise SystemExit('table %s not found' % name)
    return [float(x) for x in m.group(1).replace('\n', ' ').split(',') if x.strip()]

def parse_matrix(src, name, rows):
    start = re.search(r'\b' + name + r':\s*\[', src).end()
    found = re.findall(r'\[(.*?)\]', src[start:], re.S)[:rows]
    out = [[float(x) for x in r.replace('\n', ' ').split(',') if x.strip()] for r in found]
    assert len(out) == rows and all(len(r) == 3 for r in out), name
    return out

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'spectral.js'
    src = open(path, encoding='utf-8').read()
    base = {k: parse_array(src, k) for k in 'WCMYRGB'}
    for k, v in base.items():
        assert len(v) == 38, (k, len(v))
    cmf_blk = re.search(r'CMF:\s*\[(.*?)\],\s*\}\)', src, re.S).group(1)
    cmf = [[float(x) for x in r.replace('\n', ' ').split(',') if x.strip()] for r in re.findall(r'\[(.*?)\]', cmf_blk, re.S)]
    assert len(cmf) == 3 and all(len(r) == 38 for r in cmf)
    assert abs(sum(cmf[1]) - 1.0) < 1e-9, 'CMF Y row must be normalised to 1'
    rgb_to_xyz = parse_matrix(src, 'RGB_XYZ', 3)
    xyz_to_rgb = parse_matrix(src, 'XYZ_RGB', 3)
    tables = {
        'source': 'spectral.js (MIT, https://github.com/rvanwijnen/spectral.js), tables copied verbatim',
        'lambda_nm': [380 + 10 * i for i in range(38)],
        'base_spectra': base,
        'cmf': cmf,
        'rgb_to_xyz': rgb_to_xyz,
        'xyz_to_rgb': xyz_to_rgb,
        'constants': {'R_FLOOR': 0.005, 'R_CEIL': 0.995, 'KAPPA': 0.5, 'Y_MIN': 0.005, 'A_SCALE': 256.0},
    }
    if '--pca' in sys.argv:
        from color_model import lrgb_to_R, KS, uncompand
        g = np.linspace(0, 1, 17)
        grid = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
        X = np.array([KS(lrgb_to_R(uncompand(c), tables)) for c in grid])
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        tables['pca20'] = Vt[:20].tolist()
    json.dump(tables, open('tables.json', 'w'), indent=1)
    print('wrote tables.json', {k: (len(v) if hasattr(v, '__len__') else v) for k, v in tables.items()})

if __name__ == '__main__':
    main()
