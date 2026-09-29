import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  paint,
  mix,
  composite,
  reflectance,
  deltaE,
  lab,
  half,
  unhalf,
} from "../src/color.js";
const t = JSON.parse(
  readFileSync(new URL("../reference/tables.json", import.meta.url)),
);
let seed = 73214;
const random = () => {
  seed = (1664525 * seed + 1013904223) >>> 0;
  return seed / 4294967296;
};
const color = () => Array.from({ length: 3 }, () => Math.floor(random() * 256));
const ground = (v) => Array(38).fill(v);
const masstone = (p) =>
  p.a.map((a) => {
    const x = a / (256 * p.beta);
    return 1 / (1 + x + Math.sqrt(x * x + 2 * x));
  });
test("T-C1 1000 arbitrary sRGB colors, 10 h_stroke, two grounds", () => {
  let worst = 0;
  for (let i = 0; i < 1000; i++) {
    const c = color(),
      p = paint(c, t),
      r = reflectance(c, t);
    for (const g of [0.005, 0.995])
      worst = Math.max(worst, deltaE(r, composite(p, 0.005, ground(g)), t));
  }
  assert.ok(worst < 0.05, `Delta E ${worst}`);
  console.log("T-C1 max ΔE", worst);
});
test("T-C2 equal black and white pigment volumes", () => {
  const p = mix([
    { p: 1, ...paint([0, 0, 0], t) },
    { p: 1, ...paint([255, 255, 255], t) },
  ]);
  const L = lab(masstone(p), t)[0];
  assert.ok(Math.abs(L - 31) < 0.5);
  console.log("T-C2 L*", L);
});
test("T-C3 half precision packed descriptors", () => {
  let worst = 0;
  const quant = (p) => ({
    a: p.a.map((x) => unhalf(half(x))),
    beta: unhalf(half(p.beta)),
  });
  for (let i = 0; i < 1000; i++) {
    const parts = Array.from({ length: 2 + Math.floor(random() * 3) }, () => ({
      p: random() + 0.05,
      ...paint(color(), t),
    }));
    const exact = mix(parts);
    const rounded = quant(mix(parts.map((p) => ({ p: p.p, ...quant(p) }))));
    worst = Math.max(worst, deltaE(masstone(exact), masstone(rounded), t));
  }
  assert.ok(worst < 0.05, `Delta E ${worst}`);
  console.log("T-C3 max ΔE", worst);
});
test("T-C4 pigment mixing is associative before storage quantization", () => {
  for (let i = 0; i < 500; i++) {
    const parts = Array.from({ length: 3 }, () => ({
      p: random() + 0.05,
      ...paint(color(), t),
    }));
    const ab = mix(parts.slice(0, 2)),
      abc = mix([{ p: parts[0].p + parts[1].p, ...ab }, parts[2]]);
    assert.ok(deltaE(masstone(abc), masstone(mix(parts)), t) < 1e-6);
  }
});
test("T-C5 finite thickness transmits ground color", () => {
  const p = paint([140, 20, 20], t);
  for (const thickness of [0.000005, 0.005]) {
    const e = deltaE(
      composite(p, thickness, ground(0.005)),
      composite(p, thickness, ground(0.995)),
      t,
    );
    assert.ok(thickness < 0.001 ? e > 10 : e < 0.05);
  }
});
test("T-C6 white hiding ratio", () => {
  const p = paint([255, 255, 255], t),
    a = composite(p, 0.0005, ground(0.005)),
    b = composite(p, 0.0005, ground(0.995));
  const y = (r) => r.reduce((s, x, i) => s + x * t.cmf[1][i], 0);
  assert.ok(y(a) / y(b) >= 0.98);
});
test("half float known values and nearest/even", () => {
  assert.equal(half(1), 0x3c00);
  assert.equal(half(-2), 0xc000);
  assert.equal(unhalf(1), 2 ** -24);
  assert.equal(half(1 + 2 ** -11), half(1));
  assert.equal(half(65504), 0x7bff);
});
