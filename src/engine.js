import { defaults, validate } from "./config.js";
import { paint, pack } from "./color.js";
import { shaders, displayShader } from "./shaders.js";
export class OilEngine {
  static async create(canvas, tables, options = {}) {
    const e = new OilEngine(canvas, tables, options);
    try {
      await e.init();
      return e;
    } catch (error) {
      e.destroy();
      throw error;
    }
  }
  constructor(canvas, tables, options) {
    this.canvas = canvas;
    this.tables = tables;
    this.config = validate({
      ...defaults,
      ...(options.referenceMode
        ? { S0: 98000, wetRough: 0.15, zScale: 1, ambient: 0.25, diffuse: 0.75 }
        : {}),
      ...options,
    });
    this.buffers = {};
    this.resources = [];
    this.pending = [];
    this.strokeId = 0;
    this.errors = [];
    this.frame = 0;
    this.stroking = false;
    this.destroyed = false;
    this.color = "#c83228";
  }
  buffer(
    name,
    size,
    usage = GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  ) {
    const b = this.device.createBuffer({
      label: name,
      size: Math.ceil(size / 4) * 4,
      usage,
    });
    this.buffers[name] = b;
    this.resources.push(b);
    return b;
  }
  async init() {
    if (!navigator.gpu)
      throw Error(
        "WebGPU が利用できません。最新版の Chrome / Edge で localhost を開いてください。",
      );
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) throw Error("WebGPU アダプターを取得できません。");
    const c = this.config,
      N = c.nx * c.ny,
      F = (c.nx + 1) * c.ny + c.nx * (c.ny + 1),
      FM = Math.max((c.nx + 1) * c.ny, c.nx * (c.ny + 1));
    const need = Math.max(80 * N, 88 * FM);
    if (
      need > adapter.limits.maxStorageBufferBindingSize ||
      need > adapter.limits.maxBufferSize
    )
      throw Error(
        `このGPUでは ${c.nx}×${c.ny} を一括確保できません。低い解像度を選択してください（行帯分割は未実装）。`,
      );
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.max(134217728, need),
        maxBufferSize: Math.max(268435456, need),
        maxUniformBufferBindingSize: 65536,
      },
    });
    const d = this.device;
    d.addEventListener("uncapturederror", (e) => {
      this.errors.push(e.error.message);
      this.onError?.(e.error.message);
    });
    d.lost.then((info) => {
      if (!this.destroyed) {
        this.errors.push(info.message);
        this.onError?.(
          "GPU 接続が失われました。再起動してください。" + info.message,
        );
      }
    });
    d.pushErrorScope("validation");
    this.buffer("bed", 80 * N);
    this.buffer("surface", 4 * N);
    for (let k = 0; k < c.nl; k++) {
      this.buffer("l" + k, 16 * N);
      this.buffer("s" + k, 80 * N);
    }
    for (const name of ["aux", "aux2", "solve", "mask", "profile"])
      this.buffer(name, 16 * N);
    for (const name of ["vel", "adv", "pred"]) this.buffer(name, 4 * F);
    this.buffer("coef", 16 * F);
    this.buffer("keep", 4 * N);
    this.buffer("limit", 4 * N);
    this.buffer("flux", 88 * FM);
    this.buffer("fluxEpoch", 4 * FM);
    this.buffer("brush", 88 * 257 * 257);
    const T = Math.ceil(c.nx / 32) * Math.ceil(c.ny / 32);
    this.tileCount = T;
    this.buffer("flags", 4 * T);
    this.buffer("list", 65536);
    this.buffer(
      "indirect",
      12,
      GPUBufferUsage.STORAGE |
        GPUBufferUsage.INDIRECT |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    );
    this.buffer(
      "activeIndirect",
      12,
      GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    );
    this.buffer(
      "flowIndirect",
      12,
      GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    );
    for (const name of ["activeTiles", "flowTiles", "allTiles"])
      this.buffer(
        name,
        65536,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      );
    this.buffer(
      "params",
      256,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    this.buffer(
      "ops",
      256 * 16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    this.buffer(
      "stamps",
      256 * 2048,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    const tileData = new Uint32Array(16384);
    let at = 0;
    for (let y = 0; y < Math.ceil(c.ny / 32); y++)
      for (let x = 0; x < Math.ceil(c.nx / 32); x++)
        tileData[at++] = (x << 16) | y;
    d.queue.writeBuffer(this.buffers.allTiles, 0, tileData);
    d.queue.writeBuffer(this.buffers.activeTiles, 0, tileData);
    d.queue.writeBuffer(
      this.buffers.activeIndirect,
      0,
      new Uint32Array([T * 16, 1, 1]),
    );
    this.opData = new Uint32Array(16 * 64);
    for (let i = 0; i < 16; i++) {
      this.opData[i * 64] = i & 1;
      this.opData[i * 64 + 2] = (i >>> 1) & 1;
      this.opData[i * 64 + 3] = i >>> 2;
    }
    d.queue.writeBuffer(this.buffers.ops, 0, this.opData);
    this.globalLayout = d.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform",
            hasDynamicOffset: true,
            minBindingSize: 16,
          },
        },
      ],
    });
    this.globals = {};
    for (const kind of ["active", "flow", "all"])
      this.globals[kind] = d.createBindGroup({
        layout: this.globalLayout,
        entries: [
          { binding: 0, resource: { buffer: this.buffers.params } },
          { binding: 1, resource: { buffer: this.buffers[kind + "Tiles"] } },
          { binding: 2, resource: { buffer: this.buffers.ops, size: 16 } },
        ],
      });
    this.stampLayout = d.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform",
            hasDynamicOffset: true,
            minBindingSize: 144,
          },
        },
      ],
    });
    this.stampGroup = d.createBindGroup({
      layout: this.stampLayout,
      entries: [
        { binding: 0, resource: { buffer: this.buffers.stamps, size: 144 } },
      ],
    });
    this.out = d.createTexture({
      size: [c.nx, c.ny],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
    this.resources.push(this.out);
    this.outLayout = d.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: "rgba8unorm" },
        },
      ],
    });
    this.outGroup = d.createBindGroup({
      layout: this.outLayout,
      entries: [{ binding: 0, resource: this.out.createView() }],
    });
    this.pipelines = {};
    const sources = shaders(c.nl, this.tables);
    for (const [name, value] of Object.entries(sources)) {
      if (Array.isArray(value)) {
        this.pipelines[name] = [];
        for (let k = 0; k < value.length; k++)
          this.pipelines[name].push(await this.compile(name + k, value[k]));
      } else this.pipelines[name] = await this.compile(name, value);
    }
    this.canvas.width = c.nx;
    this.canvas.height = c.ny;
    this.context = this.canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: d,
      format: this.format,
      alphaMode: "opaque",
    });
    const displayModule = d.createShaderModule({ code: displayShader });
    this.display = d.createRenderPipeline({
      layout: "auto",
      vertex: { module: displayModule, entryPoint: "vs" },
      fragment: {
        module: displayModule,
        entryPoint: "fs",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });
    this.displayGroup = d.createBindGroup({
      layout: this.display.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.out.createView() },
        {
          binding: 1,
          resource: d.createSampler({
            magFilter: "linear",
            minFilter: "linear",
          }),
        },
      ],
    });
    this.writeParams();
    const enc = d.createCommandEncoder();
    this.dispatch(enc, "init", "full");
    this.dispatch(enc, "surface", "full");
    this.dispatch(enc, "render", "full");
    this.present(enc);
    d.queue.submit([enc.finish()]);
    const err = await d.popErrorScope();
    if (err) throw Error(err.message);
    await d.queue.onSubmittedWorkDone();
    this.memoryMB =
      this.resources.reduce((s, b) => s + (b.size || 0), 0) / 1048576;
  }
  async compile(name, def) {
    const d = this.device;
    const module = d.createShaderModule({ label: name, code: def.source });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((x) => x.type === "error");
    if (errors.length)
      throw Error(
        name +
          ": " +
          errors
            .map((x) => `${x.lineNum}:${x.linePos} ${x.message}`)
            .join("\n"),
      );
    const entries = def.bindings.map((key, binding) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: def.source.includes(`var<storage,read_write> ${key}:`)
          ? "storage"
          : "read-only-storage",
      },
    }));
    if (entries.length > 8) throw Error(name + ": storage binding overflow");
    const storage = d.createBindGroupLayout({ entries });
    const layouts = [this.globalLayout, storage];
    if (name === "brush") layouts.push(this.stampLayout);
    if (name === "render") layouts.push(this.outLayout);
    const pipeline = await d.createComputePipelineAsync({
      label: name,
      layout: d.createPipelineLayout({ bindGroupLayouts: layouts }),
      compute: { module, entryPoint: "main" },
    });
    const group = d.createBindGroup({
      layout: storage,
      entries: def.bindings.map((key, binding) => ({
        binding,
        resource: { buffer: this.buffers[key] },
      })),
    });
    return { pipeline, group, name };
  }
  writeParams() {
    const c = this.config,
      a = new Float32Array(64),
      dx = c.width / c.nx;
    const values = [
      [c.nx, c.ny, dx, c.dt],
      [
        c.g * Math.sin((c.tiltX * Math.PI) / 180),
        c.g * Math.sin((c.tiltY * Math.PI) / 180),
        c.g *
          Math.cos((c.tiltX * Math.PI) / 180) *
          Math.cos((c.tiltY * Math.PI) / 180),
        c.g * c.gzMin,
      ],
      [c.hMax, c.epsD, c.uClamp, c.dGravMin],
      [c.phiA, c.phiB, c.nuMin, c.nuMax],
      [c.S0, (c.sub * dx) / c.dt, c.sub, 256],
      [c.deposit, c.pickup, c.mixing, c.capacity],
      [
        Number.isFinite(c.dryTime) ? 1 / c.dryTime : 0,
        c.wetRough,
        c.dryRough,
        c.canvasRough,
      ],
      [c.zScale, c.ambient, c.diffuse, c.shading ? 1 : 0],
      [c.F0, 0, (this.frame % 1000000) + 1, c.normalRadius],
      [c.referenceMode ? 1 : 0, c.paintPickup, c.paintDrag, 0],
      [
        c.contactDepthMin,
        c.contactDepthPressure,
        c.bristleStrength,
        c.contactExchange,
      ],
    ];
    values.forEach((v, i) => a.set(v, i * 4));
    this.device.queue.writeBuffer(this.buffers.params, 0, a);
  }
  dispatch(enc, name, kind = "active", op = 0, layer = 0, stampIndex = 0) {
    const item = Array.isArray(this.pipelines[name])
      ? this.pipelines[name][layer]
      : this.pipelines[name];
    if (this.computeEncoder !== enc) {
      this.endCompute();
      this.computeEncoder = enc;
    }
    const pass = (this.computePass ??= enc.beginComputePass({
      label: "oilpaint compute",
    }));
    pass.setPipeline(item.pipeline);
    pass.setBindGroup(
      0,
      this.globals[
        this.fullGridForTest
          ? "all"
          : kind === "flow"
            ? "flow"
            : kind === "active"
              ? "active"
              : "all"
      ],
      [op * 256],
    );
    pass.setBindGroup(1, item.group);
    if (name === "render") pass.setBindGroup(2, this.outGroup);
    if (name === "brush") {
      pass.setBindGroup(2, this.stampGroup, [stampIndex * 256]);
      const side = 2 * Math.ceil(this.batch[stampIndex].r) + 1;
      pass.dispatchWorkgroups(Math.ceil(side / 8), Math.ceil(side / 8));
    } else if (name === "compact") pass.dispatchWorkgroups(1);
    else if (name === "scan") pass.dispatchWorkgroups(this.tileCount);
    else if (kind === "full")
      pass.dispatchWorkgroups(
        Math.ceil(this.config.nx / 8),
        Math.ceil(this.config.ny / 8),
      );
    else if (this.fullGridForTest) pass.dispatchWorkgroups(this.tileCount * 16);
    else pass.dispatchWorkgroupsIndirect(this.buffers[kind + "Indirect"], 0);
  }
  endCompute() {
    this.computePass?.end();
    this.computePass = null;
    this.computeEncoder = null;
  }
  compact(enc, kind) {
    this.dispatch(enc, "scan", "full");
    this.dispatch(enc, "compact", "full", kind === "flow" ? 8 : 0);
    this.endCompute();
    enc.copyBufferToBuffer(
      this.buffers.list,
      0,
      this.buffers[kind + "Tiles"],
      0,
      65536,
    );
    enc.copyBufferToBuffer(
      this.buffers.indirect,
      0,
      this.buffers[kind + "Indirect"],
      0,
      12,
    );
  }
  encodeStamps() {
    this.batch = this.pending.splice(0, 2048);
    if (!this.batch.length) return;
    const data = new ArrayBuffer(this.batch.length * 256),
      f = new Float32Array(data),
      u = new Uint32Array(data);
    this.batch.forEach((s, i) => {
      const b = i * 64;
      f.set(
        [
          s.x,
          s.y,
          s.r,
          s.pressure,
          s.vx,
          s.vy,
          s.phi,
          s.id,
          s.paint.beta,
          s.first ? 1 : 0,
          s.empty ? 1 : 0,
          s.refill,
        ],
        b,
      );
      u[b + 7] = s.id >>> 0;
      f[b + 32] = s.scale ?? 1;
      f[b + 33] = s.dirX ?? 0;
      f[b + 34] = s.dirY ?? 0;
      for (let j = 0; j < 19; j++)
        u[b + 12 + j] = pack(s.paint.a[j * 2], s.paint.a[j * 2 + 1]);
    });
    this.device.queue.writeBuffer(this.buffers.stamps, 0, data);
  }
  step({ fluid = true, present = true } = {}) {
    if (this.destroyed) return;
    this.writeParams();
    this.encodeStamps();
    const e = this.device.createCommandEncoder();
    if (this.frame % 1000000 === 0) e.clearBuffer(this.buffers.fluxEpoch);
    for (let i = 0; i < this.batch.length; i++)
      this.dispatch(e, "brush", "full", 0, 0, i);
    if (this.dryRequested || Number.isFinite(this.config.dryTime))
      this.dispatch(e, "dry", "full", this.dryRequested ? 4 : 0);
    this.dryRequested = false;
    this.compact(e, "active");
    if (fluid && !(this.config.pauseFluidDuringStroke && this.stroking)) {
      for (const pass of ["aux", "coef", "advect", "visc", "solveInit"])
        this.dispatch(e, pass);
      for (let i = 0; i < this.config.gs; i++) {
        this.dispatch(e, "gs", "active", 0);
        this.dispatch(e, "gs", "active", 2);
      }
      for (const pass of ["correct", "keep", "clamp"]) this.dispatch(e, pass);
      this.compact(e, "flow");
      for (let sub = 0; sub < this.config.sub; sub++)
        for (const axis of sub % 2 ? [1, 0] : [0, 1]) {
          this.dispatch(e, "profile", "flow", axis);
          for (let k = 0; k < this.config.nl; k++)
            for (const pass of ["flux", "limiter", "apply"])
              this.dispatch(e, pass, "flow", axis + sub * 4, k);
        }
    }
    this.dispatch(e, "surface", "full");
    this.dispatch(e, "render", "full");
    this.dispatch(e, "clearMask");
    if (present) this.present(e);
    this.endCompute();
    this.device.queue.submit([e.finish()]);
    this.frame++;
  }
  present(e) {
    this.endCompute();
    const p = e.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: [1, 1, 1, 1],
          storeOp: "store",
        },
      ],
    });
    p.setPipeline(this.display);
    p.setBindGroup(0, this.displayGroup);
    p.draw(3);
    p.end();
  }
  beginStroke(point, { empty = false } = {}) {
    this.stroking = true;
    this.strokeId++;
    this.stroke = {
      id: this.strokeId,
      r: this.config.radius,
      phi: this.config.phi,
      paint: paint(
        this.color.match(/\w\w/g).map((x) => parseInt(x, 16)),
        this.tables,
        this.config.tint,
        this.config.kappa,
      ),
      empty,
      refill: empty ? 0 : this.config.replenish,
      scale: this.config.referenceMode
        ? 1
        : Math.min(
            1,
            this.config.maxStampSpacing /
              (this.config.spacing * this.config.radius),
          ),
    };
    this.last = point;
    this.remaining = 0;
    this.stamp(point, 0, 0, true);
  }
  stamp(p, vx, vy, first = false) {
    this.pending.push({
      ...this.stroke,
      ...p,
      vx,
      vy,
      first,
      scale: first && !this.config.referenceMode ? 1 : this.stroke.scale,
    });
  }
  moveStroke(point) {
    if (!this.stroking) return;
    const a = this.last,
      dist = Math.hypot(point.x - a.x, point.y - a.y);
    if (dist === 0) {
      this.last = point;
      return;
    }
    if (this.stroke.dirX === undefined) {
      this.stroke.dirX = (point.x - a.x) / dist;
      this.stroke.dirY = (point.y - a.y) / dist;
    }
    const spacing = this.config.spacing * this.stroke.r * this.stroke.scale,
      delta = Math.max(0.001, (point.time - a.time) / 1000);
    let vx = ((point.x - a.x) * (this.config.width / this.config.nx)) / delta,
      vy = ((point.y - a.y) * (this.config.width / this.config.nx)) / delta;
    const cap =
      (1.5 * this.config.sub * (this.config.width / this.config.nx)) /
      this.config.dt;
    const factor = Math.min(1, cap / Math.max(Math.hypot(vx, vy), 1e-30));
    vx *= factor;
    vy *= factor;
    let pos = spacing - this.remaining;
    for (; pos <= dist; pos += spacing) {
      const t = pos / dist;
      this.stamp(
        {
          x: a.x + (point.x - a.x) * t,
          y: a.y + (point.y - a.y) * t,
          pressure: a.pressure + (point.pressure - a.pressure) * t,
          time: a.time + (point.time - a.time) * t,
        },
        vx,
        vy,
      );
    }
    this.remaining = (this.remaining + dist) % spacing;
    this.last = point;
  }
  endStroke() {
    this.stroking = false;
  }
  async read(name) {
    const source = this.buffers[name],
      b = this.device.createBuffer({
        size: source.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    try {
      const e = this.device.createCommandEncoder();
      e.copyBufferToBuffer(source, 0, b, 0, source.size);
      this.device.queue.submit([e.finish()]);
      await b.mapAsync(GPUMapMode.READ);
      return b.getMappedRange().slice(0);
    } finally {
      b.destroy();
    }
  }
  async exportPNG() {
    await this.device.queue.onSubmittedWorkDone();
    const { nx, ny } = this.config,
      row = Math.ceil((nx * 4) / 256) * 256,
      b = this.device.createBuffer({
        size: row * ny,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    try {
      const e = this.device.createCommandEncoder();
      e.copyTextureToBuffer(
        { texture: this.out },
        { buffer: b, bytesPerRow: row },
        [nx, ny],
      );
      this.device.queue.submit([e.finish()]);
      await b.mapAsync(GPUMapMode.READ);
      const source = new Uint8Array(b.getMappedRange()),
        data = new Uint8ClampedArray(nx * ny * 4);
      for (let y = 0; y < ny; y++)
        data.set(source.subarray(y * row, y * row + nx * 4), y * nx * 4);
      const canvas = document.createElement("canvas");
      canvas.width = nx;
      canvas.height = ny;
      canvas.getContext("2d").putImageData(new ImageData(data, nx, ny), 0, 0);
      return await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/png"),
      );
    } finally {
      b.destroy();
    }
  }
  destroy() {
    this.destroyed = true;
    for (const r of this.resources) r.destroy();
    this.device?.destroy();
  }
}
