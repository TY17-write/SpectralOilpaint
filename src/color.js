export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const linear = (x) =>
  x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
export const compand = (x) => {
  x = clamp(x);
  return x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055;
};
export function reflectance(rgb, t) {
  let [r, g, b] = rgb.map((x) => linear(x / 255));
  const w = Math.min(r, g, b);
  r -= w;
  g -= w;
  b -= w;
  const weights = [
    w,
    Math.min(g, b),
    Math.min(r, b),
    Math.min(r, g),
    Math.max(0, Math.min(r - b, r - g)),
    Math.max(0, Math.min(g - b, g - r)),
    Math.max(0, Math.min(b - g, b - r)),
  ];
  return Array.from({ length: 38 }, (_, i) =>
    clamp(
      ["W", "C", "M", "Y", "R", "G", "B"].reduce(
        (v, k, j) => v + weights[j] * t.base_spectra[k][i],
        0,
      ),
      0.005,
      0.995,
    ),
  );
}
export function paint(rgb, t, tint = 1, kappa = 0.5) {
  const r = reflectance(rgb, t),
    beta =
      tint *
      Math.max(
        0.005,
        r.reduce((s, x, i) => s + x * t.cmf[1][i], 0),
      ) **
        kappa;
  return { a: r.map((x) => (256 * beta * (1 - x) ** 2) / (2 * x)), beta };
}
export function mix(parts) {
  const p = parts.reduce((s, x) => s + x.p, 0);
  return {
    a: Array.from(
      { length: 38 },
      (_, i) =>
        parts.reduce((s, x) => s + x.p * x.a[i], 0) / Math.max(p, 1e-30),
    ),
    beta: parts.reduce((s, x) => s + x.p * x.beta, 0) / Math.max(p, 1e-30),
  };
}
export function composite(paint, p, ground, S0 = 98000) {
  if (p <= 0 || paint.beta <= 0) return [...ground];
  const sx = S0 * paint.beta * p;
  return paint.a.map((a, i) => {
    const ak = 1 + a / (256 * paint.beta),
      bk = Math.sqrt(Math.max(0, ak * ak - 1)),
      xi = bk * sx,
      e = Math.exp(-2 * xi),
      den = ak * (1 - e) + bk * (1 + e);
    const r = xi < 0.001 ? sx / (1 + ak * sx) : (1 - e) / Math.max(den, 1e-30);
    const tr =
      xi < 0.001
        ? (1 - xi) / (1 + ak * sx)
        : (2 * bk * Math.exp(-xi)) / Math.max(den, 1e-30);
    return r + (tr * tr * ground[i]) / Math.max(1 - r * ground[i], 1e-30);
  });
}
export function rgb(r, t) {
  const xyz = t.cmf.map((row) => row.reduce((s, c, i) => s + c * r[i], 0));
  return t.xyz_to_rgb.map(
    (row) => 255 * compand(row.reduce((s, c, i) => s + c * xyz[i], 0)),
  );
}
export function lab(r, t) {
  const v = t.cmf.map((row, j) => {
    const x =
      row.reduce((s, c, i) => s + c * r[i], 0) / [0.95047, 1, 1.08883][j];
    return x > (6 / 29) ** 3 ? Math.cbrt(x) : x / (3 * (6 / 29) ** 2) + 4 / 29;
  });
  return [116 * v[1] - 16, 500 * (v[0] - v[1]), 200 * (v[1] - v[2])];
}
export const deltaE = (a, b, t) =>
  Math.hypot(...lab(a, t).map((x, i) => x - lab(b, t)[i]));
// IEEE-754 binary16, round to nearest/even; independent of shader-f16.
const scratch = new DataView(new ArrayBuffer(4));
export function half(x) {
  scratch.setFloat32(0, x, false);
  const b = scratch.getUint32(0, false),
    sign = (b >>> 16) & 32768,
    exp = (b >>> 23) & 255,
    m = b & 8388607;
  if (exp === 255) return sign | 31744 | (m ? 512 : 0);
  let e = exp - 127 + 15;
  if (e >= 31) return sign | 31744;
  if (e <= 0) {
    if (e < -10) return sign;
    const f = m | 8388608,
      shift = 14 - e;
    let q = f >>> shift;
    const rem = f & ((1 << shift) - 1),
      mid = 1 << (shift - 1);
    if (rem > mid || (rem === mid && q & 1)) q++;
    return sign | q;
  }
  let q = m >>> 13;
  const rem = m & 8191;
  if (rem > 4096 || (rem === 4096 && q & 1)) {
    q++;
    if (q === 1024) {
      q = 0;
      e++;
    }
  }
  return sign | (e << 10) | q;
}
export function unhalf(h) {
  const sign = h & 32768 ? -1 : 1,
    e = (h >>> 10) & 31,
    m = h & 1023;
  return (
    sign *
    (e === 31
      ? m
        ? NaN
        : Infinity
      : e
        ? 2 ** (e - 15) * (1 + m / 1024)
        : 2 ** -14 * (m / 1024))
  );
}
export const pack = (a, b) => (half(a) | (half(b) << 16)) >>> 0;
