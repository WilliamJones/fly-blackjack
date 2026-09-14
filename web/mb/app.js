import { MushroomBody } from "./mb.js";

const T = window.THREE;
const $ = s => document.querySelector(s);
const COL = {
  pn: new T.Color(0x6e8494), kc: new T.Color(0xE2893A),
  mbon: new T.Color(0x2FA39A), dan: new T.Color(0xA971D6),
  quiet: new T.Color(0x2b3641), dark: new T.Color(0x7a2f28),
};

(async function main() {
  const data = await fetch("mb.json").then(r => r.json());
  const mb = new MushroomBody(data);
  const A = data.odours.geraniol, B = data.odours.octanol;

  $("#s-kc").textContent = mb.nKc.toLocaleString();
  $("#s-mbon").textContent = mb.nMbon;
  $("#s-dan").textContent = mb.nDan;

  // ---------- 3D: every cell at its measured soma position ----------
  const groups = ["pn", "kc", "mbon", "dan"];
  const all = [];
  for (const g of groups) for (const p of data.soma[g]) all.push(p);
  const centre = [0, 1, 2].map(k => all.reduce((a, p) => a + p[k], 0) / all.length);
  const spread = Math.max(...all.map(p => Math.hypot(...p.map((v, k) => v - centre[k]))));

  const offsets = {}, positions = [], colors = [], sizes = [];
  let cursor = 0;
  for (const g of groups) {
    offsets[g] = cursor;
    for (const p of data.soma[g]) {
      positions.push((p[0] - centre[0]) / spread,
                     -(p[1] - centre[1]) / spread,   // anatomical y is ventral-down
                     (p[2] - centre[2]) / spread);
      colors.push(COL[g].r, COL[g].g, COL[g].b);
      sizes.push(g === "kc" ? 0.030 : g === "mbon" ? 0.055 : g === "dan" ? 0.042 : 0.024);
      cursor++;
    }
  }
  const geom = new T.BufferGeometry();
  geom.setAttribute("position", new T.BufferAttribute(new Float32Array(positions), 3));
  const colorAttr = new T.BufferAttribute(new Float32Array(colors), 3);
  geom.setAttribute("color", colorAttr);
  geom.setAttribute("size", new T.BufferAttribute(new Float32Array(sizes), 1));

  const dot = document.createElement("canvas"); dot.width = dot.height = 64;
  const dc = dot.getContext("2d");
  const grad = dc.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.5, "rgba(255,255,255,0.9)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  dc.fillStyle = grad; dc.fillRect(0, 0, 64, 64);

  const material = new T.ShaderMaterial({
    uniforms: { map: { value: new T.CanvasTexture(dot) }, scale: { value: 620 } },
    vertexShader: `attribute float size; varying vec3 vColor;
      uniform float scale;
      void main(){ vColor = color;
        vec4 mv = modelViewMatrix * vec4(position,1.0);
        gl_PointSize = size * scale / -mv.z;
        gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform sampler2D map; varying vec3 vColor;
      void main(){ vec4 t = texture2D(map, gl_PointCoord);
        if (t.a < 0.05) discard;
        gl_FragColor = vec4(vColor, t.a); }`,
    transparent: true, depthWrite: false, vertexColors: true,
  });

  const scene = new T.Scene();
  const cloud = new T.Points(geom, material);
  scene.add(cloud);

  // faint KC -> MBON wiring, subsampled so it reads as structure not fog
  const sp = data.w_kc_mbon, step = Math.max(1, Math.floor(sp.v.length / 2200));
  const lineArr = [];
  for (let i = 0; i < sp.v.length; i += step) {
    const kc = offsets.kc + sp.r[i], mo = offsets.mbon + sp.c[i];
    lineArr.push(positions[kc * 3], positions[kc * 3 + 1], positions[kc * 3 + 2],
                 positions[mo * 3], positions[mo * 3 + 1], positions[mo * 3 + 2]);
  }
  const lg = new T.BufferGeometry();
  lg.setAttribute("position", new T.BufferAttribute(new Float32Array(lineArr), 3));
  scene.add(new T.LineSegments(lg, new T.LineBasicMaterial({
    color: 0x18222c, transparent: true, opacity: 0.34 })));

  const canvas = $("#view");
  const renderer = new T.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(0x070a0e, 1);
  const camera = new T.PerspectiveCamera(40, 1.25, 0.01, 40);
  let theta = 0.6, phi = 1.32, radius = 2.45, dragging = false, lx = 0, ly = 0, spin = true;
  canvas.addEventListener("pointerdown", e => { dragging = true; spin = false; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", e => {
    if (!dragging) return;
    theta -= (e.clientX - lx) * 0.007; phi = Math.min(3.0, Math.max(0.15, phi - (e.clientY - ly) * 0.007));
    lx = e.clientX; ly = e.clientY;
  });
  const stop = e => { dragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };
  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointercancel", stop);
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    radius = Math.min(8, Math.max(1.2, radius * (1 + Math.sign(e.deltaY) * 0.1)));
  }, { passive: false });

  // ---------- live state ----------
  let flash = 0, danFlash = 0, trials = 0;
  const history = [];           // {a, b} per training trial
  let baseA = mb.valenceOf(A), baseB = mb.valenceOf(B);

  function paint() {
    const c = colorAttr.array;
    for (let i = 0; i < mb.nKc; i++) {
      const o = (offsets.kc + i) * 3;
      const active = mb.kc[i] * flash;
      const dep = mb.kcDepression[i];
      const base = COL.quiet.clone().lerp(COL.dark, Math.min(1, dep * 3.2));
      const lit = base.clone().lerp(COL.kc, active);
      c[o] = lit.r; c[o + 1] = lit.g; c[o + 2] = lit.b;
    }
    for (let i = 0; i < mb.nDan; i++) {
      const o = (offsets.dan + i) * 3;
      const on = mb.punish[i] ? danFlash : 0;
      const col = COL.quiet.clone().lerp(COL.dan, 0.35 + 0.65 * on);
      c[o] = col.r; c[o + 1] = col.g; c[o + 2] = col.b;
    }
    for (let m = 0; m < mb.nMbon; m++) {
      const o = (offsets.mbon + m) * 3;
      const drive = Math.min(1, Math.abs(mb.mbon[m]) * 26);
      const target = mb.valence[m] > 0 ? COL.kc : COL.mbon;
      const col = COL.quiet.clone().lerp(target, 0.3 + 0.7 * drive);
      c[o] = col.r; c[o + 1] = col.g; c[o + 2] = col.b;
    }
    colorAttr.needsUpdate = true;
  }

  function meter(el, span, value) {
    const v = Math.max(-1, Math.min(1, value / 1.6));
    el.style.background = v >= 0 ? "var(--excite)" : "var(--inhibit)";
    el.style.left = v >= 0 ? "50%" : `${50 + v * 50}%`;
    el.style.width = `${Math.abs(v) * 50}%`;
    span.textContent = value.toFixed(2);
  }

  const curve = $("#curve"), cx = curve.getContext("2d");
  function drawCurve() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = curve.clientWidth, h = 150;
    curve.width = w * dpr; curve.height = h * dpr;
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const style = getComputedStyle(document.body);
    const ink = style.getPropertyValue("--ink").trim();
    const rule = style.getPropertyValue("--rule").trim();
    const muted = style.getPropertyValue("--muted").trim();
    cx.clearRect(0, 0, w, h);
    const pad = 22, span = Math.max(12, history.length);
    const lo = -2.2, hi = 1.0;
    const y = v => pad + (hi - v) / (hi - lo) * (h - pad * 2);
    const x = i => pad + (i / Math.max(1, span - 1)) * (w - pad * 1.4);

    cx.strokeStyle = rule; cx.lineWidth = 1;
    cx.beginPath(); cx.moveTo(pad, y(0)); cx.lineTo(w - pad * 0.4, y(0)); cx.stroke();
    cx.fillStyle = muted; cx.font = '400 9px "IBM Plex Mono", monospace';
    cx.fillText("0", 6, y(0) + 3);
    cx.fillText("-2", 2, y(-2) + 3);
    if (!history.length) {
      cx.fillStyle = muted;
      cx.fillText("shock an odour to begin", pad + 6, h / 2);
      return;
    }
    for (const [key, colour] of [["b", "#2FA39A"], ["a", "#E2893A"]]) {
      cx.strokeStyle = colour; cx.lineWidth = 1.8; cx.beginPath();
      history.forEach((p, i) => i ? cx.lineTo(x(i), y(p[key])) : cx.moveTo(x(i), y(p[key])));
      cx.stroke();
      const last = history[history.length - 1];
      cx.fillStyle = colour;
      cx.beginPath(); cx.arc(x(history.length - 1), y(last[key]), 2.6, 0, 7); cx.fill();
    }
    cx.fillStyle = ink; cx.font = '500 9px "IBM Plex Mono", monospace';
    cx.fillText("geraniol", w - 58, y(history[history.length - 1].a) - 7);
    cx.fillText("octanol", w - 56, y(history[history.length - 1].b) - 7);
  }

  function refresh(note) {
    meter($("#m-a"), $("#v-a"), mb.valenceOf(A) - baseA);
    meter($("#m-b"), $("#v-b"), mb.valenceOf(B) - baseB);
    $("#st-sparse").textContent = `${mb.activeCount()} of ${mb.nKc}`;
    $("#st-dep").textContent = `${(mb.depression * 100).toFixed(1)}%`;
    $("#st-trials").textContent = trials;
    if (note) $("#tag").textContent = note;
    drawCurve();
  }

  function present(odour, punish, label) {
    mb.present(odour, { punish: punish ? 1 : 0 });
    flash = 1; if (punish) danFlash = 1;
    refresh(label);
  }

  async function train(times) {
    for (let i = 0; i < times; i++) {
      present(A, true, "shock paired with geraniol");
      await new Promise(r => setTimeout(r, 210));
      mb.present(B, {});
      trials++;
      history.push({ a: mb.valenceOf(A) - baseA, b: mb.valenceOf(B) - baseB });
      refresh();
      await new Promise(r => setTimeout(r, 120));
    }
    $("#tag").textContent = mb.dopamineOn && mb.plasticityOn
      ? "memory formed - geraniol now repels" : "no memory formed";
  }

  $("#train").addEventListener("click", () => train(5));
  $("#present-a").addEventListener("click", () => present(A, false, "geraniol, no shock"));
  $("#present-b").addEventListener("click", () => present(B, false, "octanol, no shock"));
  $("#reset").addEventListener("click", () => {
    mb.reset(); trials = 0; history.length = 0;
    baseA = mb.valenceOf(A); baseB = mb.valenceOf(B);
    refresh("naive fly");
  });
  const toggle = (id, key, onText, offText) => {
    const b = $(id);
    b.addEventListener("click", () => {
      mb[key] = !mb[key];
      b.setAttribute("aria-pressed", String(!mb[key]));
      b.textContent = mb[key] ? offText : onText;
      $("#stage").style.outline = (!mb.dopamineOn || !mb.plasticityOn)
        ? "2px solid var(--alert)" : "none";
      $("#stage").style.outlineOffset = "-2px";
      refresh(mb[key] ? "restored" : onText.toLowerCase());
    });
  };
  toggle("#nodopamine", "dopamineOn", "Dopamine disconnected", "Disconnect dopamine");
  toggle("#noplastic", "plasticityOn", "Synapses frozen", "Freeze synapses");

  $("#loading").remove();
  refresh("naive fly - present an odour");

  function frame() {
    requestAnimationFrame(frame);
    const r = canvas.getBoundingClientRect();
    if (canvas.width !== Math.round(r.width * renderer.getPixelRatio())) {
      renderer.setSize(r.width, r.height, false);
      camera.aspect = r.width / r.height; camera.updateProjectionMatrix();
    }
    if (spin) theta += 0.0016;
    flash *= 0.94; danFlash *= 0.9;
    camera.position.set(radius * Math.sin(phi) * Math.cos(theta),
                        radius * Math.cos(phi),
                        radius * Math.sin(phi) * Math.sin(theta));
    camera.lookAt(0, 0, 0);
    paint();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
  window.addEventListener("resize", drawCurve);
})();
