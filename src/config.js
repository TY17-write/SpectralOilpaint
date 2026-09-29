export const defaults = {
  nx: 1024,
  ny: 768,
  nl: 2,
  width: 0.5,
  dt: 1 / 60,
  g: 9.81,
  gzMin: 0.05,
  hMax: 0.003,
  hStroke: 0.0005,
  epsD: 5e-7,
  phiA: 0.2,
  phiB: 0.8,
  nuMin: 0.0001,
  nuMax: 0.1,
  dGravMin: 0.00003,
  uClamp: 0.0006,
  gs: 32,
  sub: 4,
  kappa: 0.5,
  tint: 1,
  S0: 12000,
  spacing: 0.125,
  deposit: 0.05,
  pickup: 0.05,
  mixing: 0.2,
  replenish: 0.05,
  capacity: 0.00075,
  dryTime: Infinity,
  F0: 0.04,
  wetRough: 0.38,
  dryRough: 0.6,
  canvasRough: 0.7,
  zScale: 0.35,
  normalRadius: 3,
  referenceMode: false,
  paintPickup: 0.05,
  paintDrag: 0.15,
  maxStampSpacing: 2,
  contactDepthMin: 0.000015,
  contactDepthPressure: 0.000085,
  bristleStrength: 0.45,
  contactExchange: 0.18,
  ambient: 0.6,
  diffuse: 0.3,
  tiltX: 0,
  tiltY: 0,
  phi: 1,
  radius: 18,
  pauseFluidDuringStroke: false,
  shading: true,
};
export function validate(c) {
  for (const k of ["bristleStrength", "contactExchange"])
    if (!Number.isFinite(c[k]) || c[k] < 0 || c[k] > 1)
      throw Error("Invalid " + k);
  if (
    !(c.contactDepthMin > 0) ||
    !Number.isFinite(c.contactDepthMin) ||
    !(c.contactDepthPressure >= 0) ||
    !Number.isFinite(c.contactDepthPressure)
  )
    throw Error("Invalid brush contact depth");
  if (
    !Number.isInteger(c.normalRadius) ||
    c.normalRadius < 0 ||
    c.normalRadius > 8
  )
    throw Error("normalRadius must be an integer in 0..8");
  for (const k of ["nx", "ny", "nl", "gs", "sub"])
    if (!Number.isInteger(c[k]) || c[k] < 1)
      throw Error(k + " must be a positive integer");
  if (c.nl < 2 || c.nl > 3 || c.nx > 2048 || c.ny > 1536)
    throw Error("Supported: 2–3 layers, up to 2048 × 1536");
  for (const k of ["width", "dt", "epsD", "capacity", "hMax", "spacing", "S0"])
    if (!(c[k] > 0 && Number.isFinite(c[k]))) throw Error("Invalid " + k);
  if (c.phiA >= c.phiB) throw Error("phiA must be less than phiB");
  return c;
}
