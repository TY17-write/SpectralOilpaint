import { commonWGSL } from "../src/shaders.js";
import { paint, pack, reflectance, composite, deltaE } from "../src/color.js";
export async function colorGPU(e, t) {
  const d = e.device,
    n = 1000;
  const source =
    commonWGSL +
    `struct Input { sp:Spec,p:f32,ground:f32,pad:vec2<f32> };@group(1) @binding(0) var<storage,read> input:array<Input>;@group(1) @binding(1) var<storage,read_write> output:array<f32>;@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x;if(i>=1000u){return;}let q=input[i];for(var j=0u;j<19u;j++){let a=pair(q.sp.a[j]);output[i*38u+j*2u]=over(a.x,beta(q.sp),q.p,q.ground);output[i*38u+j*2u+1u]=over(a.y,beta(q.sp),q.p,q.ground);}}`;
  const module = d.createShaderModule({ code: source });
  const pipeline = await d.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const input = d.createBuffer({
      size: n * 96,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    output = d.createBuffer({
      size: n * 38 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    }),
    read = d.createBuffer({
      size: n * 38 * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  const raw = new ArrayBuffer(n * 96),
    u = new Uint32Array(raw),
    f = new Float32Array(raw),
    colors = [];
  let seed = 31214;
  const random = () => {
    seed = (1664525 * seed + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let i = 0; i < n; i++) {
    const rgb = Array.from({ length: 3 }, () => Math.floor(random() * 256));
    colors.push(rgb);
    const p = paint(rgb, t);
    for (let j = 0; j < 19; j++)
      u[i * 24 + j] = pack(p.a[j * 2], p.a[j * 2 + 1]);
    u[i * 24 + 19] = pack(p.beta, 0);
    f[i * 24 + 20] = 0.005;
    f[i * 24 + 21] = i % 2 ? 0.995 : 0.005;
  }
  d.queue.writeBuffer(input, 0, raw);
  const g0 = d.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: e.buffers.params } }],
    }),
    g1 = d.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
      ],
    });
  const enc = d.createCommandEncoder(),
    p = enc.beginComputePass();
  p.setPipeline(pipeline);
  p.setBindGroup(0, g0);
  p.setBindGroup(1, g1);
  p.dispatchWorkgroups(Math.ceil(n / 64));
  p.end();
  enc.copyBufferToBuffer(output, 0, read, 0, read.size);
  d.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const results = new Float32Array(read.getMappedRange());
  let worst = 0;
  for (let i = 0; i < n; i++)
    worst = Math.max(
      worst,
      deltaE(
        reflectance(colors[i], t),
        Array.from(results.subarray(i * 38, (i + 1) * 38)),
        t,
      ),
    );
  input.destroy();
  output.destroy();
  read.destroy();
  if (!(worst < 0.05)) throw Error("GPU T-C1 Delta E " + worst);
  return { id: "T-C1 GPU f16 + finite KM", colors: n, maxDeltaE: worst };
}
