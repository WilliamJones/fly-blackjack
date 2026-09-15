// The mushroom body on its own: every measured soma, the sampled wiring, and
// the activity of a decision flowing through it -- projection neurons, then
// the sparse Kenyon-cell code, then the output neurons, then dopamine.
// `BrainActivity` is the shared state; the fly scene reads it too.
const T = window.THREE;
export const COL = {
  pn: 0x8fa6b6, kc: 0xE2893A, mbon: 0x2FA39A, dan: 0xA971D6,
  quiet: 0x33414d, dark: 0x7a2f28, wire: 0x0f171f,
};

const clamp01 = x => x < 0 ? 0 : x > 1 ? 1 : x;
const ramp = (t, w = 0.18) => clamp01(t / w);

/** What the brain is doing right now, as smoothed per-neuron levels. */
export class BrainActivity {
  constructor(mush) {
    this.mush = mush;
    this.pn = new Float32Array(mush.nPn); this.pnT = new Float32Array(mush.nPn);
    this.kc = new Float32Array(mush.nKc); this.kcT = new Float32Array(mush.nKc);
    this.mbon = new Float32Array(mush.nMbon); this.mbonT = new Float32Array(mush.nMbon);
    this.since = -1e9;           // seconds since the current situation arrived
    this.hold = 2.6;             // how long a situation stays lit before fading
    this.danFlash = 0; this.danKind = null; this.danAge = 1e9;
    this.level = 0;              // overall excitement 0..1, for the fly's head glow
    this.listeners = new Set();
  }
  onChange(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit() { for (const fn of this.listeners) fn(this); }

  /** A situation arrives: PN pattern, Kenyon-cell code, MBON readout. */
  present({ pn, kc, mbon }) {
    if (pn) this.pnT.set(pn); else this.pnT.fill(0);
    let top = 0; for (let i = 0; i < kc.length; i++) if (kc[i] > top) top = kc[i];
    for (let i = 0; i < kc.length; i++) this.kcT[i] = top ? clamp01(kc[i] / top) : 0;
    let m = 0; for (let i = 0; i < mbon.length; i++) m = Math.max(m, Math.abs(mbon[i]));
    for (let i = 0; i < mbon.length; i++) this.mbonT[i] = m ? mbon[i] / m : 0;
    this.since = 0;
    this.emit();
  }
  flash(kind) { this.danKind = kind; this.danFlash = 1; this.danAge = 0; this.emit(); }
  clear() { this.since = 1e9; }

  update(dt) {
    this.since += dt; this.danAge += dt;
    const t = this.since;
    // arrive in stages, hold, then fade together
    const fade = t < this.hold ? 1 : Math.exp(-(t - this.hold) / 1.6);
    const gPn = ramp(t) * fade, gKc = ramp(t - 0.10) * fade, gMb = ramp(t - 0.22) * fade;
    const ease = 1 - Math.exp(-dt * 18);
    for (let i = 0; i < this.pn.length; i++) this.pn[i] += (this.pnT[i] * gPn - this.pn[i]) * ease;
    for (let i = 0; i < this.kc.length; i++) this.kc[i] += (this.kcT[i] * gKc - this.kc[i]) * ease;
    for (let i = 0; i < this.mbon.length; i++) this.mbon[i] += (this.mbonT[i] * gMb - this.mbon[i]) * ease;
    this.danFlash *= Math.exp(-dt * 2.2);
    this.level = Math.max(gKc, this.danFlash);
  }
  kcFiring() { let n = 0; for (let i = 0; i < this.kc.length; i++) if (this.kc[i] > 0.05) n++; return n; }
}

function sampleEdges(sp, max) {
  const n = sp.v.length, step = Math.max(1, Math.floor(n / max)), out = [];
  for (let k = 0; k < n; k += step) out.push(k);
  return out;
}

export class BrainView {
  constructor(canvas, { mb, mush, activity, readout }) {
    this.canvas = canvas; this.mush = mush; this.act = activity; this.readout = readout || {};
    this.renderer = new T.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(0x000000, 0);
    this.scene = new T.Scene();
    this.group = new T.Group(); this.scene.add(this.group);
    this.camera = new T.PerspectiveCamera(34, 1.4, 0.05, 20);

    // --- somata, centred and scaled into the unit sphere
    const groups = ["pn", "kc", "mbon", "dan"], all = [];
    for (const g of groups) for (const p of mb.soma[g]) all.push(p);
    const c = [0, 1, 2].map(k => all.reduce((a, p) => a + p[k], 0) / all.length);
    const spread = Math.max(...all.map(p => Math.hypot(...p.map((v, k) => v - c[k]))));
    this.offsets = {}; this.n = {};
    const pos = [], col = [], size = [];
    let i = 0;
    for (const g of groups) {
      this.offsets[g] = i; this.n[g] = mb.soma[g].length;
      for (const p of mb.soma[g]) {
        pos.push((p[0] - c[0]) / spread, -(p[1] - c[1]) / spread, (p[2] - c[2]) / spread);
        const cc = new T.Color(COL.quiet); col.push(cc.r, cc.g, cc.b);
        size.push(g === "kc" ? 0.085 : g === "mbon" ? 0.17 : g === "dan" ? 0.12 : 0.07);
        i++;
      }
    }
    this.pos = pos;
    // fit on the bulk of the cloud, not the farthest outlier
    const radii = []; for (let k = 0; k < pos.length; k += 3) radii.push(Math.hypot(pos[k], pos[k + 1], pos[k + 2]));
    radii.sort((a, b) => a - b); this.fitRadius = radii[Math.floor(radii.length * 0.93)] * 1.06;
    const geom = new T.BufferGeometry();
    geom.setAttribute("position", new T.BufferAttribute(new Float32Array(pos), 3));
    this.colorAttr = new T.BufferAttribute(new Float32Array(col), 3);
    this.sizeAttr = new T.BufferAttribute(new Float32Array(size), 1);
    this.baseSize = Float32Array.from(size);
    geom.setAttribute("color", this.colorAttr);
    geom.setAttribute("size", this.sizeAttr);
    const dot = document.createElement("canvas"); dot.width = dot.height = 64;
    const dc = dot.getContext("2d"), grad = dc.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, "rgba(255,255,255,1)"); grad.addColorStop(0.45, "rgba(255,255,255,.85)"); grad.addColorStop(1, "rgba(255,255,255,0)");
    dc.fillStyle = grad; dc.fillRect(0, 0, 64, 64);
    this.pointMat = new T.ShaderMaterial({
      uniforms: { map: { value: new T.CanvasTexture(dot) }, scale: { value: 160 } },
      vertexShader: `attribute float size; varying vec3 vColor; uniform float scale;
        void main(){ vColor=color; vec4 mv=modelViewMatrix*vec4(position,1.0);
          gl_PointSize=size*scale/-mv.z; gl_Position=projectionMatrix*mv; }`,
      fragmentShader: `uniform sampler2D map; varying vec3 vColor;
        void main(){ vec4 t=texture2D(map,gl_PointCoord); if(t.a<0.05) discard; gl_FragColor=vec4(vColor,t.a); }`,
      transparent: true, depthWrite: false, vertexColors: true,
    });
    this.points = new T.Points(geom, this.pointMat); this.points.renderOrder = 2; this.group.add(this.points);

    // --- wiring: PN->KC (input) and KC->MBON (output), sampled, per-vertex colour
    this.wires = [];
    const addWires = (sp, fromOff, toOff, max, srcOf) => {
      const idx = sampleEdges(sp, max), lp = [], lc = [], src = [];
      for (const k of idx) {
        const a = fromOff + sp.r[k], b = toOff + sp.c[k];
        lp.push(pos[a * 3], pos[a * 3 + 1], pos[a * 3 + 2], pos[b * 3], pos[b * 3 + 1], pos[b * 3 + 2]);
        lc.push(0, 0, 0, 0, 0, 0);
        src.push(srcOf(k));
      }
      const g = new T.BufferGeometry();
      g.setAttribute("position", new T.BufferAttribute(new Float32Array(lp), 3));
      const cattr = new T.BufferAttribute(new Float32Array(lc), 3);
      g.setAttribute("color", cattr);
      const mesh = new T.LineSegments(g, new T.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.6,
        depthWrite: false }));
      mesh.renderOrder = 1; this.group.add(mesh);
      this.wires.push({ cattr, src });
      return mesh;
    };
    this.inWires = addWires(mb.w_pn_kc, this.offsets.pn, this.offsets.kc, 900, k => ({ pn: mb.w_pn_kc.r[k], kc: mb.w_pn_kc.c[k] }));
    this.outWires = addWires(mb.w_kc_mbon, this.offsets.kc, this.offsets.mbon, 1500, k => ({ kc: mb.w_kc_mbon.r[k], mbon: mb.w_kc_mbon.c[k] }));

    // --- camera: orbit by drag, drift otherwise
    this.yaw = 0.25; this.pitch = 0.22; this.targetYaw = this.yaw; this.targetPitch = this.pitch;
    this.idleSince = 0; this.drag = null;
    canvas.style.touchAction = "none";
    canvas.addEventListener("pointerdown", e => { this.drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener("pointermove", e => {
      if (!this.drag) return;
      this.targetYaw += (e.clientX - this.drag.x) * 0.008;
      this.targetPitch = Math.max(-1.2, Math.min(1.2, this.targetPitch + (e.clientY - this.drag.y) * 0.006));
      this.drag = { x: e.clientX, y: e.clientY }; this.idleSince = 0;
    });
    const end = () => { this.drag = null; };
    canvas.addEventListener("pointerup", end); canvas.addEventListener("pointercancel", end);

    this.col = { quiet: new T.Color(COL.quiet), dark: new T.Color(COL.dark), pn: new T.Color(COL.pn),
      kc: new T.Color(COL.kc), mbon: new T.Color(COL.mbon), dan: new T.Color(COL.dan), wire: new T.Color(COL.wire),
      black: new T.Color(0x000000), red: new T.Color(0xC8553D) };
    this.tmp = new T.Color();
    this.last = performance.now(); this.readoutAt = 0;
    requestAnimationFrame(t => this.frame(t));
  }

  fit() {
    const r = this.canvas.getBoundingClientRect();
    if (!r.width || !r.height) return false;
    const pr = this.renderer.getPixelRatio();
    if (this.canvas.width !== Math.round(r.width * pr) || this.canvas.height !== Math.round(r.height * pr)) {
      this.renderer.setSize(r.width, r.height, false);
      this.camera.aspect = r.width / r.height; this.camera.updateProjectionMatrix();
      this.pointMat.uniforms.scale.value = r.height * pr * 0.55;
    }
    const fovV = this.camera.fov * Math.PI / 180;
    const fovH = 2 * Math.atan(Math.tan(fovV / 2) * this.camera.aspect);
    this.dist = this.fitRadius / Math.sin(Math.min(fovV, fovH) / 2);
    return true;
  }

  frame(now) {
    requestAnimationFrame(t => this.frame(t));
    const dt = Math.min(0.05, (now - this.last) / 1000); this.last = now;
    if (!this.fit()) return;
    const act = this.act;
    act.update(dt);

    // camera
    this.idleSince += dt;
    if (!this.drag && this.idleSince > 4) this.targetYaw += dt * 0.12;
    this.yaw += (this.targetYaw - this.yaw) * 0.12; this.pitch += (this.targetPitch - this.pitch) * 0.12;
    const cp = Math.cos(this.pitch);
    this.camera.position.set(Math.sin(this.yaw) * cp * this.dist, Math.sin(this.pitch) * this.dist, Math.cos(this.yaw) * cp * this.dist);
    this.camera.lookAt(0, 0, 0);

    // somata
    const c = this.colorAttr.array, sz = this.sizeAttr.array, o = this.offsets, mush = this.mush, K = this.col, tmp = this.tmp;
    const paint = (idx, colour, boost) => {
      const k = idx * 3; c[k] = colour.r; c[k + 1] = colour.g; c[k + 2] = colour.b;
      sz[idx] = this.baseSize[idx] * (1 + boost);
    };
    for (let i = 0; i < this.n.pn; i++) {
      const a = act.pn[i];
      tmp.copy(K.quiet).lerp(K.pn, 0.25 + 0.75 * a); paint(o.pn + i, tmp, a * 0.9);
    }
    for (let i = 0; i < this.n.kc; i++) {
      const a = act.kc[i];
      tmp.copy(K.quiet).lerp(K.dark, Math.min(1, (mush.kcDepression[i] || 0) * 2.5)).lerp(K.kc, a);
      paint(o.kc + i, tmp, a * 1.8);
    }
    for (let i = 0; i < this.n.mbon; i++) {
      const d = act.mbon[i], a = Math.min(1, Math.abs(d) * 1.2);
      tmp.copy(K.quiet).lerp(mush.valence[i] > 0 ? K.kc : K.mbon, 0.3 + 0.7 * a); paint(o.mbon + i, tmp, a * 0.8);
    }
    const punishing = act.danKind === "punish", flash = act.danFlash;
    for (let i = 0; i < this.n.dan; i++) {
      const on = (mush.punish[i] === 1) === punishing ? flash : 0;
      tmp.copy(K.quiet).lerp(K.dan, 0.3 + 0.7 * on); paint(o.dan + i, tmp, on * 1.2);
    }
    this.colorAttr.needsUpdate = true; this.sizeAttr.needsUpdate = true;

    // wires: lit by their source neuron; the output wires flush red when punished
    const [win, wout] = this.wires;
    let a = win.cattr.array;
    for (let i = 0; i < win.src.length; i++) {
      const s = win.src[i], lit = act.pn[s.pn] * (0.35 + 0.65 * act.kc[s.kc]);
      tmp.copy(K.wire).lerp(K.pn, lit * 1.0);
      const k = i * 6; a[k] = tmp.r; a[k + 1] = tmp.g; a[k + 2] = tmp.b; a[k + 3] = tmp.r; a[k + 4] = tmp.g; a[k + 5] = tmp.b;
    }
    win.cattr.needsUpdate = true;
    a = wout.cattr.array;
    for (let i = 0; i < wout.src.length; i++) {
      const s = wout.src[i], lit = act.kc[s.kc];
      tmp.copy(K.wire).lerp(K.kc, lit);
      if (lit > 0.05 && flash > 0.05) tmp.lerp(punishing ? K.red : K.dan, flash * lit);
      const k = i * 6; a[k] = tmp.r * 0.7; a[k + 1] = tmp.g * 0.7; a[k + 2] = tmp.b * 0.7; a[k + 3] = tmp.r; a[k + 4] = tmp.g; a[k + 5] = tmp.b;
    }
    wout.cattr.needsUpdate = true;

    this.renderer.render(this.scene, this.camera);
    if ((this.readoutAt += dt) > 0.12) { this.readoutAt = 0; this.updateReadout(); }
  }

  updateReadout() {
    const r = this.readout, act = this.act;
    if (r.kc) r.kc.textContent = act.kcFiring();
    if (r.pn) r.pn.textContent = act.pn.reduce((n, v) => n + (v > 0.05 ? 1 : 0), 0);
    if (r.danBar) {
      r.danBar.style.opacity = (0.15 + act.danFlash * 0.85).toFixed(2);
      r.danBar.classList.toggle("punish", act.danKind === "punish");
    }
    if (r.danLabel) r.danLabel.textContent = act.danAge > 12 ? "quiet" : act.danKind === "punish" ? "punishment" : "reward";
  }
}
