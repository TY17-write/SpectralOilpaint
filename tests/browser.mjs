// Local, isolated Chrome profile. No user browser session is read or modified.
import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { setTimeout as sleep } from "node:timers/promises";
import path from "node:path";
const root = path.resolve(import.meta.dirname, ".."),
  profile = path.join(root, ".gpu-test-profile");
await mkdir(profile, { recursive: true });
const server = spawn(process.execPath, ["server.mjs"], {
  cwd: root,
  windowsHide: true,
  stdio: "ignore",
  env: { ...process.env, PORT: "8083" },
});
const chrome =
  process.env.CHROME_PATH ||
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const browser = spawn(
  chrome,
  [
    "--headless=new",
    "--no-first-run",
    "--no-default-browser-check",
    "--enable-unsafe-webgpu",
    "--remote-debugging-port=9227",
    "--remote-allow-origins=*",
    "--user-data-dir=" + profile,
    "about:blank",
  ],
  { windowsHide: true, stdio: "ignore" },
);
let socket;
try {
  let targets;
  for (let n = 0; n < 60; n++) {
    try {
      targets = await fetch("http://127.0.0.1:9227/json").then((r) => r.json());
      if (targets.length) break;
    } catch {}
    await sleep(500);
  }
  if (!targets) throw Error("Chrome debugging endpoint unavailable");
  socket = new WebSocket(
    targets.find((t) => t.type === "page").webSocketDebuggerUrl,
  );
  await new Promise((r, j) => {
    socket.onopen = r;
    socket.onerror = j;
  });
  let seq = 0;
  const pending = new Map();
  socket.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.id) {
      const p = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error
        ? p.reject(Error(JSON.stringify(msg.error)))
        : p.resolve(msg.result);
    }
  };
  const call = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = ++seq;
      pending.set(id, { resolve, reject });
      socket.send(JSON.stringify({ id, method, params }));
    });
  await call("Page.navigate", {
    url:
      ["ui", "ui-layout"].includes(process.argv[2])
        ? "http://localhost:8083/"
        : "http://localhost:8083/tests/" + (process.argv[2] || "gpu.html"),
  });
  if (["ui", "ui-layout"].includes(process.argv[2])) {
    await call("Emulation.setDeviceMetricsOverride", {
      width: 1440,
      height: 1100,
      deviceScaleFactor: 1,
      mobile: false,
    });
    for (let i = 0; i < 100; i++) {
      const r = await call("Runtime.evaluate", {
        expression: "Boolean(window.oilpaint)",
        returnByValue: true,
      });
      if (r.result?.value) break;
      await sleep(100);
    }
    await call("Runtime.evaluate", {
      expression: `(async()=>{const e=window.oilpaint;for(const [color,y] of [['#c83228',300],['#285ac8',340],['#e5b536',380]]){e.color=color;e.beginStroke({x:170,y,pressure:1,time:0});e.moveStroke({x:820,y:y-50,pressure:.8,time:600});e.endStroke();}await new Promise(r=>setTimeout(r,1200));window.testResult={ok:e.errors.length===0,errors:e.errors,status:document.getElementById('status').textContent};})()`,
      awaitPromise: true,
      returnByValue: true,
    });
    const png = await call("Page.captureScreenshot", { format: "png" });
    await writeFile(
      path.join(root, "tests", "ui-preview.png"),
      Buffer.from(png.data, "base64"),
    );
  }
  if (process.argv[2] === "ui-layout") {
    const checks = [];
    const evaluate = async expression => {
      const r = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
      if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));
      return r.result.value;
    };
    checks.push(await evaluate(`({name:'defaults',ok:oilpaint.config.tiltY===10&&!oilpaint.config.shading&&document.querySelectorAll('#swatches button').length===24&&!document.querySelector('header')})`));
    for (const [width,height] of [[1920,1080],[1366,768],[800,600],[390,844]]) {
      await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
      await sleep(250);
      checks.push(await evaluate(`(()=>{const c=document.querySelector('canvas').getBoundingClientRect(),s=document.querySelector('#canvasstage').getBoundingClientRect();return {name:'fit ${width}x${height}',ok:c.width>0&&c.left>=s.left-1&&c.top>=s.top-1&&c.right<=s.right+1&&c.bottom<=s.bottom+1&&document.documentElement.scrollWidth<=innerWidth&&document.documentElement.scrollHeight<=innerHeight,ratio:c.width/c.height}})()`));
      const shot=await call('Page.captureScreenshot',{format:'png'});
      await writeFile(path.join(root,'tests',`ui-${width}.png`),Buffer.from(shot.data,'base64'));
    }
    checks.push(await evaluate(`(()=>{const c=document.querySelector('#color');c.value='#123abc';c.dispatchEvent(new Event('input'));c.dispatchEvent(new Event('change'));const t=document.querySelector('#theme');t.value='light';t.dispatchEvent(new Event('change'));return {name:'palette/theme',ok:document.querySelector('#swatches [aria-label="絵具 #123abc"]')!==null&&document.documentElement.dataset.theme==='light'}})()`));
    await call('Page.reload');await sleep(1800);
    checks.push(await evaluate(`({name:'persistence',ok:document.querySelector('#swatches [aria-label="絵具 #123abc"]')!==null&&document.documentElement.dataset.theme==='light'})`));
    checks.push(await evaluate(`(async()=>{document.querySelector('#format').value='M';document.querySelector('#orientation').value='portrait';document.querySelector('#resolution').value='512';window.confirm=()=>true;document.querySelector('#restart').click();for(let i=0;i<100;i++){await new Promise(r=>setTimeout(r,50));if(window.oilpaint?.config.ny===512)break;}return {name:'new portrait canvas',ok:oilpaint.config.nx===322&&oilpaint.config.ny===512&&oilpaint.errors.length===0};})()`));
    await evaluate(`window.testResult=${JSON.stringify({ok:checks.every(c=>c.ok),checks})}`);
  }
  let result;
  for (let n = 0; n < 1200; n++) {
    await sleep(500);
    const value = await call("Runtime.evaluate", {
      expression: "window.testResult",
      returnByValue: true,
    });
    result = value.result?.value;
    if (result) break;
  }
  if (!result) throw Error("GPU tests timed out");
  if (process.argv[3]) {
    await call("Emulation.setDeviceMetricsOverride", {
      width: 1200,
      height: 950,
      deviceScaleFactor: 1,
      mobile: false,
    });
    await sleep(200);
    const png = await call("Page.captureScreenshot", { format: "png" });
    await writeFile(
      path.join(root, "tests", process.argv[3]),
      Buffer.from(png.data, "base64"),
    );
  }
  console.log(JSON.stringify(result, null, 2));
  if (!result.ok) process.exitCode = 1;
  await call("Browser.close");
} finally {
  socket?.close();
  browser.kill();
  server.kill();
}
