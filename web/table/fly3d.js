// The fly at the table: the anatomical body idling with real grooming gestures,
// and its mushroom body beside it, lighting up as it decides.
// Body kinematics are the same as the Connectome Fly page (verified against
// MuJoCo); gestures are keyframes solved offline against the real skeleton.
const T = window.THREE;
const COL = {
  pn: 0x6e8494, kc: 0xE2893A, mbon: 0x2FA39A, dan: 0xA971D6, quiet: 0x2b3641, dark: 0x7a2f28,
};

function b64(s, Type) {
  const bin = atob(s), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}
function parseMeshes(data) {
  const out = new Map();
  for (const [name, m] of Object.entries(data.bodies)) {
    const g = new T.BufferGeometry();
    g.setAttribute("position", new T.BufferAttribute(b64(m.positions, Float32Array), 3));
    g.setIndex(new T.BufferAttribute(b64(m.indices, Uint32Array), 1));
    g.computeVertexNormals();
    out.set(name, g);
  }
  return out;
}
function buildBody(skeleton, geoms) {
  const material = new T.MeshStandardMaterial({ color: 0xd9c3a4, roughness: 0.62, metalness: 0.08, side: T.DoubleSide });
  const attach = new Map(), joints = new Map();
  let root = null;
  for (const b of skeleton.bodies) {
    const node = new T.Group();
    node.position.fromArray(b.pos);
    node.quaternion.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]);
    if (b.parent === null) root = node; else attach.get(b.parent).add(node);
    let tip = node;
    for (const j of b.joints) {
      const pivot = new T.Group(); pivot.position.fromArray(j.anchor);
      const inner = new T.Group(); inner.position.set(-j.anchor[0], -j.anchor[1], -j.anchor[2]);
      pivot.add(inner); tip.add(pivot); tip = inner;
      joints.set(j.name, { node: pivot, axis: new T.Vector3().fromArray(j.axis).normalize() });
    }
    if (b.mesh && geoms.has(b.name)) tip.add(new T.Mesh(geoms.get(b.name), material));
    attach.set(b.name, tip);
  }
  return { root, joints };
}

function buildBrain(mb) {
  const groups = ["pn", "kc", "mbon", "dan"], all = [];
  for (const g of groups) for (const p of mb.soma[g]) all.push(p);
  const c = [0, 1, 2].map(k => all.reduce((a, p) => a + p[k], 0) / all.length);
  const spread = Math.max(...all.map(p => Math.hypot(...p.map((v, k) => v - c[k]))));
  const offsets = {}, pos = [], col = [], size = [];
  let i = 0;
  for (const g of groups) {
    offsets[g] = i;
    for (const p of mb.soma[g]) {
      pos.push((p[0] - c[0]) / spread, -(p[1] - c[1]) / spread, (p[2] - c[2]) / spread);
      const cc = new T.Color(COL[g]); col.push(cc.r, cc.g, cc.b);
      size.push(g === "kc" ? 0.034 : g === "mbon" ? 0.06 : g === "dan" ? 0.046 : 0.026);
      i++;
    }
  }
  const geom = new T.BufferGeometry();
  geom.setAttribute("position", new T.BufferAttribute(new Float32Array(pos), 3));
  const colorAttr = new T.BufferAttribute(new Float32Array(col), 3);
  geom.setAttribute("color", colorAttr);
  geom.setAttribute("size", new T.BufferAttribute(new Float32Array(size), 1));
  const dot = document.createElement("canvas"); dot.width = dot.height = 64;
  const dc = dot.getContext("2d"), grad = dc.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)"); grad.addColorStop(0.5, "rgba(255,255,255,.9)"); grad.addColorStop(1, "rgba(255,255,255,0)");
  dc.fillStyle = grad; dc.fillRect(0, 0, 64, 64);
  const mat = new T.ShaderMaterial({
    uniforms: { map: { value: new T.CanvasTexture(dot) }, scale: { value: 140 } },
    vertexShader: `attribute float size; varying vec3 vColor; uniform float scale;
      void main(){ vColor=color; vec4 mv=modelViewMatrix*vec4(position,1.0);
        gl_PointSize=size*scale/-mv.z; gl_Position=projectionMatrix*mv; }`,
    fragmentShader: `uniform sampler2D map; varying vec3 vColor;
      void main(){ vec4 t=texture2D(map,gl_PointCoord); if(t.a<0.05) discard; gl_FragColor=vec4(vColor,t.a); }`,
    transparent: true, depthWrite: false, vertexColors: true,
  });
  const group = new T.Group();
  group.add(new T.Points(geom, mat));
  const sp = mb.w_kc_mbon, step = Math.max(1, Math.floor(sp.v.length / 1500)), lines = [];
  for (let k = 0; k < sp.v.length; k += step) {
    const a = offsets.kc + sp.r[k], b = offsets.mbon + sp.c[k];
    lines.push(pos[a * 3], pos[a * 3 + 1], pos[a * 3 + 2], pos[b * 3], pos[b * 3 + 1], pos[b * 3 + 2]);
  }
  const lg = new T.BufferGeometry(); lg.setAttribute("position", new T.BufferAttribute(new Float32Array(lines), 3));
  group.add(new T.LineSegments(lg, new T.LineBasicMaterial({ color: 0x18222c, transparent: true, opacity: 0.35 })));
  return { group, colorAttr, offsets, n: { kc: mb.soma.kc.length, mbon: mb.soma.mbon.length, dan: mb.soma.dan.length } };
}

// -------------------------------------------------------------- gestures
const MOODS = {
  idle:     [["breathe", 3], ["antenna_twitch_left", 1], ["antenna_twitch_right", 1], ["head_tilt_left", 1],
             ["head_tilt_right", 1], ["tap_left", 1.5], ["chin_left", 0.6]],
  thinking: [["chin_left", 3], ["chin_right", 1.5], ["scratch_left", 2], ["scratch_right", 1.2],
             ["head_tilt_left", 1], ["head_tilt_right", 1], ["head_nod", 0.8]],
  won:      [["wing_shrug", 1], ["head_up", 1]],
  lost:     [["flinch", 1]],
};

export class FlyAtTable {
  constructor(canvas, { fly, skeleton, gestures, mb, mush }) {
    this.canvas = canvas; this.gestures = gestures; this.mush = mush;
    this.renderer = new T.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(0x000000, 0);
    this.scene = new T.Scene();
    this.scene.add(new T.HemisphereLight(0x9fb6c9, 0x0a1016, 0.9));
    const key = new T.DirectionalLight(0xffe6c4, 1.2); key.position.set(0.5, -0.6, 1.0); this.scene.add(key);
    const rim = new T.DirectionalLight(0x6fd7cd, 0.4); rim.position.set(-0.6, 0.5, 0.3); this.scene.add(rim);

    const { root, joints } = buildBody(skeleton, parseMeshes(fly));
    this.joints = joints;
    this.body = new T.Group(); this.body.add(root);
    this.body.position.set(0, 0, 0.125);          // feet on the table
    this.scene.add(this.body);
    const table = new T.Mesh(new T.CircleGeometry(0.6, 48),
      new T.MeshStandardMaterial({ color: 0x0f1a16, roughness: 1 }));
    this.scene.add(table);

    this.brain = buildBrain(mb);
    this.brain.group.scale.setScalar(0.115);
    this.brain.group.position.set(-0.04, -0.27, 0.27);   // above and beside the wing
    this.scene.add(this.brain.group);

    this.camera = new T.PerspectiveCamera(30, 2, 0.02, 10);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.60, -0.30, 0.30);
    this.camera.lookAt(0.02, -0.08, 0.12);

    this.current = new Map(); this.target = new Map(); this.rate = new Map();
    this.mood = "idle"; this.nextGesture = 0; this.phase = 0; this.t = 0;
    this.kcLit = new Float32Array(this.brain.n.kc); this.danFlash = 0; this.danKind = null;
    this.mbonDrive = new Float32Array(this.brain.n.mbon);
    this.last = performance.now();
    requestAnimationFrame(t => this.frame(t));
  }

  setMood(m) {
    if (m === this.mood) return;
    this.mood = m; this.nextGesture = 0;
    if (m === "won" || m === "lost") this.play(m === "won" ? "wing_shrug" : "flinch", 0.22, 900);
  }
  /** Which Kenyon cells the current situation activates, and the MBON readout. */
  showSituation(kc, mbon) {
    for (let i = 0; i < this.kcLit.length; i++) this.kcLit[i] = kc ? Math.min(1, kc[i] * 1.2) : 0;
    if (mbon) for (let i = 0; i < this.mbonDrive.length; i++) this.mbonDrive[i] = mbon[i];
  }
  flash(kind) { this.danKind = kind; this.danFlash = 1; }

  play(name, rate = 0.1, holdMs = 1400) {
    const pose = this.gestures[name];
    if (!pose) return;
    for (const [j, v] of Object.entries(pose)) { this.target.set(j, v); this.rate.set(j, rate); }
    clearTimeout(this.holdTimer);
    this.holdTimer = setTimeout(() => { for (const j of Object.keys(pose)) { this.target.set(j, 0); this.rate.set(j, rate * 0.7); } }, holdMs);
  }
  playSweep(side) {
    const a = `scratch_${side}_a`, b = `scratch_${side}_b`;
    let n = 0; const step = () => {
      const pose = this.gestures[n % 2 ? b : a];
      for (const [j, v] of Object.entries(pose)) { this.target.set(j, v); this.rate.set(j, 0.2); }
      if (++n < 5) this.sweepTimer = setTimeout(step, 260);
      else this.sweepTimer = setTimeout(() => { for (const j of Object.keys(pose)) { this.target.set(j, 0); this.rate.set(j, 0.08); } }, 300);
    };
    clearTimeout(this.sweepTimer); step();
  }
  pickGesture() {
    const menu = MOODS[this.mood] || MOODS.idle;
    const total = menu.reduce((a, [, w]) => a + w, 0);
    let r = Math.random() * total;
    for (const [g, w] of menu) { r -= w; if (r <= 0) return g; }
    return menu[0][0];
  }

  frame(now) {
    requestAnimationFrame(t => this.frame(t));
    const dt = Math.min(0.05, (now - this.last) / 1000); this.last = now; this.t += dt;
    const r = this.canvas.getBoundingClientRect();
    if (r.width && this.canvas.width !== Math.round(r.width * this.renderer.getPixelRatio())) {
      this.renderer.setSize(r.width, r.height, false);
      this.camera.aspect = r.width / r.height; this.camera.updateProjectionMatrix();
    }
    if (this.t > this.nextGesture) {
      const g = this.pickGesture();
      const wait = this.mood === "thinking" ? 1.4 + Math.random() * 1.6 : 2.5 + Math.random() * 4;
      if (g.startsWith("scratch")) this.playSweep(g.split("_")[1]);
      else if (g === "chin_left" || g === "chin_right") this.play(g, 0.07, this.mood === "thinking" ? 2600 : 1600);
      else if (g === "breathe") { /* continuous, handled below */ }
      else this.play(g, 0.12, 700 + Math.random() * 600);
      this.nextGesture = this.t + wait;
    }
    // ease every joint toward its target, breathing on top
    const angles = {};
    for (const [j] of this.joints) {
      const tgt = this.target.get(j) ?? 0, k = this.rate.get(j) ?? 0.1;
      const cur = (this.current.get(j) ?? 0) + (tgt - (this.current.get(j) ?? 0)) * k;
      this.current.set(j, cur); angles[j] = cur;
    }
    const breath = Math.sin(this.t * 1.9) * 0.025;
    for (const j of ["abdomen", "abdomen_2", "abdomen_3"]) angles[j] = (angles[j] || 0) + breath;
    for (const [name, jt] of this.joints) jt.node.quaternion.setFromAxisAngle(jt.axis, angles[name] ?? 0);
    // lean in while thinking, a faint sway otherwise
    const lean = this.mood === "thinking" ? 0.06 : 0;
    this.body.rotation.y += ((lean + Math.sin(this.t * 0.7) * 0.01) - this.body.rotation.y) * 0.05;
    this.body.rotation.z = Math.sin(this.t * 0.45) * 0.015;

    // brain: Kenyon cells lit by the situation, darkened by depression, DANs by outcome
    this.danFlash *= 0.93;
    const c = this.brain.colorAttr.array, o = this.brain.offsets, mb = this.mush;
    const quiet = new T.Color(COL.quiet), dark = new T.Color(COL.dark), amber = new T.Color(COL.kc);
    const teal = new T.Color(COL.mbon), violet = new T.Color(COL.dan);
    for (let i = 0; i < this.brain.n.kc; i++) {
      const base = quiet.clone().lerp(dark, Math.min(1, (mb.kcDepression[i] || 0) * 2.5));
      const lit = base.lerp(amber, this.kcLit[i]);
      const k = (o.kc + i) * 3; c[k] = lit.r; c[k + 1] = lit.g; c[k + 2] = lit.b;
    }
    for (let i = 0; i < this.brain.n.mbon; i++) {
      const d = Math.min(1, Math.abs(this.mbonDrive[i]) * 20);
      const col = quiet.clone().lerp(mb.valence[i] > 0 ? amber : teal, 0.3 + 0.7 * d);
      const k = (o.mbon + i) * 3; c[k] = col.r; c[k + 1] = col.g; c[k + 2] = col.b;
    }
    for (let i = 0; i < this.brain.n.dan; i++) {
      const isPun = mb.punish[i] === 1;
      const on = this.danKind === (isPun ? "punish" : "reward") ? this.danFlash : 0;
      const col = quiet.clone().lerp(violet, 0.3 + 0.7 * on);
      const k = (o.dan + i) * 3; c[k] = col.r; c[k + 1] = col.g; c[k + 2] = col.b;
    }
    this.brain.colorAttr.needsUpdate = true;
    this.brain.group.rotation.z += dt * 0.25;
    this.renderer.render(this.scene, this.camera);
  }
}
