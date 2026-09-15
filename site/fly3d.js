// The fly at the table: the anatomical body idling with real grooming gestures,
// facing its cards, its head glowing faintly as its mushroom body works.
// Body kinematics are the same as the Connectome Fly page (verified against
// MuJoCo); gestures are keyframes solved offline against the real skeleton.
const T = window.THREE;

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
  let root = null, head = null;
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
    if (b.name === "head") head = tip;
    attach.set(b.name, tip);
  }
  return { root, joints, head };
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
  constructor(canvas, { fly, skeleton, gestures, activity }) {
    this.canvas = canvas; this.gestures = gestures; this.act = activity;
    this.renderer = new T.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(0x000000, 0);
    this.scene = new T.Scene();
    this.scene.add(new T.HemisphereLight(0x9fb6c9, 0x0a1016, 0.9));
    const key = new T.DirectionalLight(0xffe6c4, 1.2); key.position.set(0.3, -0.7, 1.0); this.scene.add(key);
    const rim = new T.DirectionalLight(0x6fd7cd, 0.45); rim.position.set(-0.6, 0.6, 0.3); this.scene.add(rim);

    const { root, joints, head } = buildBody(skeleton, parseMeshes(fly));
    this.joints = joints;
    this.body = new T.Group(); this.body.add(root);
    this.body.position.set(0, 0, 0.125);          // feet on the table; the fly faces +x
    this.scene.add(this.body);
    const table = new T.Mesh(new T.CircleGeometry(1.2, 64),
      new T.MeshStandardMaterial({ color: 0x0f1a16, roughness: 1 }));
    this.scene.add(table);

    // the mushroom body sits in the head: a glow that follows brain activity
    this.glow = new T.PointLight(0xE2893A, 0, 0.35, 2);
    this.glow.position.set(0.012, 0, 0.012);
    head.add(this.glow);
    const spr = document.createElement("canvas"); spr.width = spr.height = 64;
    const sc = spr.getContext("2d"), g = sc.createRadialGradient(32, 32, 0, 32, 32, 32);
    g.addColorStop(0, "rgba(255,190,120,.9)"); g.addColorStop(0.35, "rgba(226,137,58,.45)"); g.addColorStop(1, "rgba(226,137,58,0)");
    sc.fillStyle = g; sc.fillRect(0, 0, 64, 64);
    this.halo = new T.Sprite(new T.SpriteMaterial({ map: new T.CanvasTexture(spr), transparent: true, depthWrite: false,
      blending: T.AdditiveBlending, opacity: 0 }));
    this.halo.scale.setScalar(0.07); this.halo.position.copy(this.glow.position);
    head.add(this.halo);

    // seen from its left-front, so it faces right -- toward its cards
    this.camera = new T.PerspectiveCamera(28, 2, 0.02, 10);
    this.camera.up.set(0, 0, 1);
    this.frameCamera(2);

    this.current = new Map(); this.target = new Map(); this.rate = new Map();
    this.mood = "idle"; this.nextGesture = 0; this.phase = 0; this.t = 0;
    this.last = performance.now();
    requestAnimationFrame(t => this.frame(t));
  }

  /** Wide frames put the fly left of its cards; tall ones put it above them. */
  frameCamera(aspect) {
    const f = Math.min(1.9, Math.max(1, 2.2 / aspect));       // pull back as the frame narrows
    this.camera.position.set(0.08 * f, -0.74 * f, 0.36 * f + 0.05 * (f - 1));
    this.camera.lookAt(0.15 - 0.10 * (f - 1), 0.0, 0.07 - 0.12 * (f - 1));
  }
  setMood(m) {
    if (m === this.mood) return;
    this.mood = m; this.nextGesture = 0;
    if (m === "won" || m === "lost") this.play(m === "won" ? "wing_shrug" : "flinch", 0.22, 900);
  }

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
    if (!r.width) return;
    if (this.canvas.width !== Math.round(r.width * this.renderer.getPixelRatio())) {
      this.renderer.setSize(r.width, r.height, false);
      this.camera.aspect = r.width / r.height; this.camera.updateProjectionMatrix();
      this.frameCamera(this.camera.aspect);
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
    // lean in toward the cards while thinking, a faint sway otherwise
    const lean = this.mood === "thinking" ? 0.06 : 0;
    this.body.rotation.y += ((lean + Math.sin(this.t * 0.7) * 0.01) - this.body.rotation.y) * 0.05;
    this.body.rotation.z = Math.sin(this.t * 0.45) * 0.015;

    // head glow follows the brain
    const lvl = this.act ? this.act.level : 0;
    this.glow.intensity = lvl * 2.4;
    this.halo.material.opacity = lvl * 0.9;
    this.halo.scale.setScalar(0.05 + lvl * 0.05);
    this.renderer.render(this.scene, this.camera);
  }
}
