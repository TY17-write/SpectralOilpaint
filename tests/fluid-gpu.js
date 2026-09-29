import { OilEngine } from "../src/engine.js";
import {
  paint,
  pack,
  unhalf,
  reflectance,
  composite,
  deltaE,
} from "../src/color.js";
export async function acceptanceGPU(canvas, tables) {
  const e = await OilEngine.create(canvas, tables, {
      nx: 128,
      ny: 96,
      nl: 3,
      width: (128 * 0.5) / 1024,
      shading: false,
      referenceMode: true,
    }),
    c = e.config,
    N = c.nx * c.ny;
  const logs = [];
  const assert = (x, m) => {
    if (!x) throw Error(m);
  };
  const reset = async (phi, dry = 0) => {
    const enc = e.device.createCommandEncoder();
    for (const name of [
      "l0",
      "l1",
      "l2",
      "s0",
      "s1",
      "s2",
      "vel",
      "adv",
      "pred",
      "coef",
      "aux",
      "aux2",
      "mask",
      "solve",
      "keep",
      "profile",
      "flux",
      "limit",
    ])
      enc.clearBuffer(e.buffers[name]);
    e.dispatch(enc, "init", "full");
    e.endCompute();
    e.device.queue.submit([enc.finish()]);
    const p = paint([140, 40, 25], tables);
    for (let k = 0; k < 3; k++) {
      const a = new ArrayBuffer(N * 16),
        f = new Float32Array(a),
        u = new Uint32Array(a),
        s = new Uint32Array(N * 20);
      for (let y = 0; y < c.ny; y++)
        for (let x = 0; x < c.nx; x++) {
          if ((x - 48) ** 2 + (y - 32) ** 2 < 144) {
            const i = y * c.nx + x;
            f[i * 4] = 0.001 / 3;
            f[i * 4 + 1] = (0.001 / 3) * phi;
            f[i * 4 + 2] = dry;
            u[i * 4 + 3] = 1;
            for (let j = 0; j < 19; j++)
              s[i * 20 + j] = pack(p.a[2 * j], p.a[2 * j + 1]);
            s[i * 20 + 19] = pack(p.beta, 0);
          }
        }
      e.device.queue.writeBuffer(e.buffers["l" + k], 0, a);
      e.device.queue.writeBuffer(e.buffers["s" + k], 0, s);
    }
    await e.device.queue.onSubmittedWorkDone();
  };
  const measure = async () => {
    let d = 0,
      p = 0,
      cx = 0,
      min = Infinity;
    const arrays = [];
    for (let k = 0; k < 3; k++) {
      const a = new Float32Array(await e.read("l" + k));
      arrays.push(a);
      for (let i = 0; i < N; i++) {
        d += a[4 * i];
        p += a[4 * i + 1];
        cx += a[4 * i] * (i % c.nx);
        min = Math.min(min, a[4 * i]);
        assert(Number.isFinite(a[4 * i]), "Nonfinite fluid depth");
      }
    }
    return { d, p, cx: cx / d, min, arrays };
  };
  const steps = async (n) => {
    for (let i = 0; i < n; i++) {
      e.step({ present: false });
      if (i % 10 === 9) await e.device.queue.onSubmittedWorkDone();
    }
    await e.device.queue.onSubmittedWorkDone();
  };
  try {
    c.tiltX = 60;
    await reset(0.1);
    for (let k = 0; k < 3; k++) {
      const f = new Float32Array(await e.read("l" + k));
      for (let i = 0; i < N; i++) {
        f[i * 4] *= 3;
        f[i * 4 + 1] *= 3;
      }
      e.device.queue.writeBuffer(e.buffers["l" + k], 0, f);
    }
    e.writeParams();
    let enc = e.device.createCommandEncoder();
    e.compact(enc, "active");
    for (const name of [
      "aux",
      "coef",
      "advect",
      "visc",
      "solveInit",
      "profile",
    ])
      e.dispatch(enc, name);
    e.endCompute();
    e.device.queue.submit([enc.finish()]);
    const aux = new Float32Array(await e.read("aux")),
      coef = new Float32Array(await e.read("coef")),
      startSolve = new Float32Array(await e.read("solve")),
      prof = new Float32Array(await e.read("profile"));
    const residual = (a) => {
      let max = 0;
      for (let y = 0; y < c.ny; y++)
        for (let x = 0; x < c.nx; x++) {
          const i = y * c.nx + x,
            fx = y * (c.nx + 1) + x,
            fy = (c.nx + 1) * c.ny + y * c.nx + x;
          let nb = 0;
          if (x > 0) nb += coef[fx * 4] * a[(i - 1) * 4];
          if (x + 1 < c.nx) nb += coef[(fx + 1) * 4] * a[(i + 1) * 4];
          if (y > 0) nb += coef[fy * 4] * a[(i - c.nx) * 4];
          if (y + 1 < c.ny) nb += coef[(fy + c.nx) * 4] * a[(i + c.nx) * 4];
          max = Math.max(
            max,
            Math.abs(a[i * 4 + 2] * a[i * 4] - nb - a[i * 4 + 1]),
          );
        }
      return max;
    };
    const r0 = residual(startSolve);
    enc = e.device.createCommandEncoder();
    for (let n = 0; n < c.gs; n++) {
      e.dispatch(enc, "gs", "active", 0);
      e.dispatch(enc, "gs", "active", 2);
    }
    e.endCompute();
    e.device.queue.submit([enc.finish()]);
    const ratio = residual(new Float32Array(await e.read("solve"))) / r0;
    assert(ratio <= 0.1, "T-F5 residual " + ratio);
    logs.push({ id: "T-F5 3mm 60 degrees", residualRatio: ratio });
    let profileError = 0;
    const data = await measure();
    for (let i = 0; i < N; i++) {
      let total = 0,
        weighted = 0;
      for (let k = 0; k < 3; k++) {
        const h = data.arrays[k][i * 4];
        total += h;
        weighted += h * prof[i * 4 + k];
      }
      if (total > 0)
        profileError = Math.max(
          profileError,
          Math.abs(weighted - total) / total,
        );
    }
    assert(profileError < 1e-6, "T-F6 f32 profile " + profileError);
    logs.push({
      id: "T-F6 GPU f32",
      relativeError: profileError,
      tolerance: 1e-6,
    });
    c.tiltX = 30;
    await reset(0.1);
    const initial = await measure();
    await steps(600);
    const end = await measure(),
      drift = Math.abs(end.d - initial.d) / initial.d,
      pigment = Math.abs(end.p - initial.p) / initial.p,
      displacement = (end.cx - initial.cx) * (c.width / c.nx) * 1000;
    assert(drift < 1e-5 && pigment < 1e-5, `T-F1 drift ${drift}, ${pigment}`);
    assert(end.min >= 0, "T-F2 negative thickness");
    assert(displacement > 10, `T-F3 displacement ${displacement}`);
    logs.push({
      id: "T-F1/F2/F3",
      volumeDrift: drift,
      pigmentDrift: pigment,
      minThickness: end.min,
      displacementMM: displacement,
    });
    await reset(1);
    const tube = await measure();
    await steps(600);
    const tubeEnd = await measure();
    for (let k = 0; k < 3; k++) {
      const before = new Uint32Array(tube.arrays[k].buffer),
        after = new Uint32Array(tubeEnd.arrays[k].buffer);
      assert(
        before.every((x, i) => x === after[i]),
        "T-F4 tube changed",
      );
    }
    assert(
      new Float32Array(await e.read("vel")).every((x) => x === 0),
      "T-F4 nonzero velocity",
    );
    logs.push({ id: "T-F4", bitIdentical: true });
    for (const tilt of [0, 30]) {
      c.tiltX = tilt;
      await reset(0.1);
      await steps(3300);
      const at55 = await measure();
      await steps(300);
      const at60 = await measure();
      const wet = (m) =>
        Array.from({ length: N }, (_, i) =>
          m.arrays.reduce((sum, a) => sum + a[i * 4], 0),
        ).filter((x) => x > c.epsD).length;
      const velocity = new Float32Array(await e.read("vel"));
      const maxVelocity = velocity.reduce(
        (m, x) => Math.max(m, Math.abs(x)),
        0,
      );
      assert(
        maxVelocity === 0 && wet(at55) === wet(at60),
        "T-S3 flow did not settle at " + tilt,
      );
      logs.push({
        id: "T-S3",
        tilt,
        seconds: 60,
        maxVelocity,
        wetCells: wet(at60),
      });
    }
    // Fully dry paint is baked by the application. Assert geometry is preserved and velocity is zero.
    await reset(0.1, 1);
    e.dryRequested = true;
    const frozen = await measure();
    await steps(60);
    const dried = await measure(),
      bed = new Float32Array(await e.read("bed"));
    let baked = 0;
    for (let i = 0; i < N; i++) baked += bed[20 * i];
    assert(
      Math.abs(baked - frozen.d) / frozen.d < 1e-6 && dried.d === 0,
      "T-F7 dry geometry changed",
    );
    assert(
      new Float32Array(await e.read("vel")).every((x) => x === 0),
      "T-F7 dry paint flowed",
    );
    logs.push({
      id: "T-F7 application bake",
      volumeDrift: Math.abs(baked - frozen.d) / frozen.d,
    });
    assert(e.errors.length === 0, e.errors.join("\n"));
    return logs;
  } finally {
    e.destroy();
  }
}
