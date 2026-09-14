import { Circuit, act, ARENAS, ARENA, Walker, poseFrom, Forage } from "./sim.js";
import { MushroomBody } from "./mb.js";

const T = window.THREE;
const SPECIMEN = 0x070a0e, EXCITE = 0xe2893a, INHIBIT = 0x2fa39a, ALERT = 0xc8553d;
const FLY_SCALE = 114.5;         // body length 0.2619 -> ~30 arena units
const FOOT_Z = -0.1256;          // measured stance height, MuJoCo units
const DECISION_HZ = 30;

const $ = s => document.querySelector(s);

// ------------------------------------------------------------------ assets
async function loadAll() {
  const [meshes, skeleton, legmap, brain, policies] = await Promise.all([
    fetch("fly.json").then(r => r.json()),
    fetch("skeleton.json").then(r => r.json()),
    fetch("legmap.json").then(r => r.json()),
    fetch("brain.json").then(r => r.json()),
    fetch("policies.json").then(r => r.json()),
  ]);
  const clips = await fetch("clips.json").then(r => r.json()).catch(() => ({}));
  const mb = await fetch("mb.json").then(r => r.json()).catch(() => null);
  return { geoms: parseMeshes(meshes), skeleton, legmap, brain, policies, clips, mb };
}

/** Rebuild BufferGeometry per body from the base64 buffers that
 *  flybrain/body/export_body.py writes. Positions are in MuJoCo length units,
 *  in each body's own frame; the skeleton does the rest. */
function b64(s, Type) {
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
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

// ---------------------------------------------------------------- skeleton
/** Rebuild MuJoCo's body tree. Each body is a node at (pos, quat); each hinge
 *  joint becomes a pivot at its anchor plus a counter-translation, which is
 *  exactly rotate-about-a-point. Verified against mj_kinematics to 1e-16. */
function buildBody(skeleton, geoms) {
  const material = new T.MeshStandardMaterial({
    color: 0xd9c3a4, roughness: 0.62, metalness: 0.08, side: T.DoubleSide,
  });
  const attach = new Map();          // body name -> node children hang from
  const joints = new Map();          // joint name -> { node, axis }
  let root = null;

  for (const b of skeleton.bodies) {
    let node = new T.Group();
    node.position.fromArray(b.pos);
    const q = b.quat;                                   // MuJoCo w,x,y,z
    node.quaternion.set(q[1], q[2], q[3], q[0]);

    if (b.parent === null) { root = node; }
    else { attach.get(b.parent).add(node); }

    let tip = node;
    for (const j of b.joints) {
      const pivot = new T.Group();
      pivot.position.fromArray(j.anchor);
      const inner = new T.Group();
      inner.position.set(-j.anchor[0], -j.anchor[1], -j.anchor[2]);
      pivot.add(inner); tip.add(pivot); tip = inner;
      joints.set(j.name, { node: pivot, axis: new T.Vector3().fromArray(j.axis).normalize() });
    }
    if (b.mesh && geoms.has(b.name)) tip.add(new T.Mesh(geoms.get(b.name), material));
    attach.set(b.name, tip);
  }
  return { root, joints };
}

const _q = new T.Quaternion();
function applyPose(joints, angles) {
  for (const [name, j] of joints) {
    j.node.quaternion.setFromAxisAngle(j.axis, angles[name] ?? 0);
  }
}

// ------------------------------------------------------------------- brain
function buildBrain(graph) {
  const n = graph.nodes.length;
  const raw = graph.nodes.map(nd => nd.position);
  const centre = [0, 1, 2].map(k => raw.reduce((a, p) => a + p[k], 0) / n);
  const spread = Math.max(...raw.map(p => Math.hypot(...p.map((v, k) => v - centre[k]))));
  // real soma coordinates, centred and normalised. anatomy, not a layout.
  const pos = new Float32Array(n * 3);
  raw.forEach((p, i) => {
    pos[i * 3] = (p[0] - centre[0]) / spread;
    pos[i * 3 + 1] = (p[2] - centre[2]) / spread;   // MuJoCo-ish z up -> view y
    pos[i * 3 + 2] = (p[1] - centre[1]) / spread;
  });

  const colors = new Float32Array(n * 3);
  const points = new T.BufferGeometry();
  points.setAttribute("position", new T.BufferAttribute(pos, 3));
  points.setAttribute("color", new T.BufferAttribute(colors, 3));
  const dot = document.createElement("canvas"); dot.width = dot.height = 64;
  const dc = dot.getContext("2d");
  const grad = dc.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)"); grad.addColorStop(0.55, "rgba(255,255,255,0.92)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  dc.fillStyle = grad; dc.fillRect(0, 0, 64, 64);
  const cloud = new T.Points(points, new T.PointsMaterial({
    size: 0.085, vertexColors: true, sizeAttenuation: true, map: new T.CanvasTexture(dot),
    transparent: true, opacity: 0.98, depthWrite: false,
  }));

  const strongest = [...graph.edges].sort((a, b) => b[2] - a[2]).slice(0, 420);
  const linePos = new Float32Array(strongest.length * 6);
  strongest.forEach(([a, b], i) => {
    for (let k = 0; k < 3; k++) {
      linePos[i * 6 + k] = pos[a * 3 + k];
      linePos[i * 6 + 3 + k] = pos[b * 3 + k];
    }
  });
  const lineGeom = new T.BufferGeometry();
  lineGeom.setAttribute("position", new T.BufferAttribute(linePos, 3));
  const lines = new T.LineSegments(lineGeom, new T.LineBasicMaterial({
    color: 0x2c3946, transparent: true, opacity: 0.45,
  }));

  const group = new T.Group();
  group.add(lines); group.add(cloud);
  return { group, colors, attr: points.getAttribute("color"), n };
}

const EX = new T.Color(EXCITE), IN = new T.Color(INHIBIT), REST = new T.Color(0x30404e);
function paintBrain(brain, h) {
  for (let i = 0; i < brain.n; i++) {
    const v = Math.min(1, Math.abs(h[i]));
    const c = h[i] >= 0 ? EX : IN;
    brain.colors[i * 3] = REST.r + (c.r - REST.r) * v;
    brain.colors[i * 3 + 1] = REST.g + (c.g - REST.g) * v;
    brain.colors[i * 3 + 2] = REST.b + (c.b - REST.b) * v;
  }
  brain.attr.needsUpdate = true;
}

// ------------------------------------------------------------------ arena
function buildArena() {
  const g = new T.Group();

  const floor = new T.Mesh(
    new T.PlaneGeometry(ARENA * 2, ARENA * 2),
    new T.MeshStandardMaterial({ color: 0x0d141b, roughness: 1, metalness: 0 }));
  floor.position.z = -0.1;
  g.add(floor);

  const step = ARENA / 5, lines = [];
  for (let i = -5; i <= 5; i++) {
    lines.push(-ARENA, i * step, 0, ARENA, i * step, 0);
    lines.push(i * step, -ARENA, 0, i * step, ARENA, 0);
  }
  const grid = new T.BufferGeometry();
  grid.setAttribute("position", new T.BufferAttribute(new Float32Array(lines), 3));
  g.add(new T.LineSegments(grid, new T.LineBasicMaterial({
    color: 0x1d2a35, transparent: true, opacity: 0.9 })));

  return g;
}

function ring(radius, colour, segments = 48) {
  const pts = [];
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    pts.push(Math.cos(a) * radius, Math.sin(a) * radius, 0);
  }
  const geo = new T.BufferGeometry();
  geo.setAttribute("position", new T.BufferAttribute(new Float32Array(pts), 3));
  return new T.Line(geo, new T.LineBasicMaterial({ color: colour }));
}

// ---------------------------------------------------------------- controls
function orbit(canvas, camera, target, { radius = 170, phi = 1.06, theta = -0.7 } = {}) {
  const state = { radius, phi, theta, dragging: false, lx: 0, ly: 0 };
  const apply = () => {
    camera.position.set(
      target.x + state.radius * Math.sin(state.phi) * Math.cos(state.theta),
      target.y + state.radius * Math.sin(state.phi) * Math.sin(state.theta),
      target.z + state.radius * Math.cos(state.phi));
    camera.lookAt(target);
  };
  canvas.addEventListener("pointerdown", e => {
    state.dragging = true; state.lx = e.clientX; state.ly = e.clientY;
    state.moved = false; canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", e => {
    if (!state.dragging) return;
    const dx = e.clientX - state.lx, dy = e.clientY - state.ly;
    if (Math.abs(dx) + Math.abs(dy) > 3) state.moved = true;
    state.lx = e.clientX; state.ly = e.clientY;
    state.theta -= dx * 0.006;
    state.phi = Math.min(1.5, Math.max(0.12, state.phi - dy * 0.006));
  });
  const stop = e => { state.dragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch {} };
  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointercancel", stop);
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    state.radius = Math.min(700, Math.max(45, state.radius * (1 + Math.sign(e.deltaY) * 0.11)));
  }, { passive: false });
  return { state, apply };
}

// -------------------------------------------------------------------- main
(async function main() {
  const { geoms, skeleton, legmap, brain: graph, policies, clips, mb: mbData } = await loadAll();

  // part four: the mushroom body, learning inside the episode
  let mushroom = null, naiveMushroom = null, driveBad = null, driveSafe = null;
  if (mbData) {
    mushroom = new MushroomBody(mbData);
    naiveMushroom = new MushroomBody(mbData);
    driveBad = mushroom.pnDrive(mbData.odours.geraniol);
    driveSafe = mushroom.pnDrive(mbData.odours.octanol);
  }

  const canvas = $("#view");
  const renderer = new T.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(SPECIMEN, 1);
  renderer.autoClear = false;

  // ---- arena scene
  const scene = new T.Scene();
  scene.add(new T.HemisphereLight(0x9fb6c9, 0x0a1016, 0.85));
  const key = new T.DirectionalLight(0xffe6c4, 1.25);
  key.position.set(140, -180, 260); scene.add(key);
  const rim = new T.DirectionalLight(0x6fd7cd, 0.35);
  rim.position.set(-200, 160, 90); scene.add(rim);
  scene.add(buildArena());

  const { root: bodyRoot, joints } = buildBody(skeleton, geoms);
  const fly = new T.Group();
  const scaled = new T.Group();
  scaled.scale.setScalar(FLY_SCALE);
  scaled.position.z = -FOOT_Z * FLY_SCALE;
  scaled.add(bodyRoot);
  fly.add(scaled);
  scene.add(fly);

  const goalRing = ring(16, EXCITE); scene.add(goalRing);
  const goalBeam = new T.Mesh(new T.CylinderGeometry(1.1, 1.1, 90, 8),
    new T.MeshBasicMaterial({ color: EXCITE, transparent: true, opacity: 0.28 }));
  goalBeam.rotation.x = Math.PI / 2; scene.add(goalBeam);

  const obstacleGroup = new T.Group(); scene.add(obstacleGroup);

  const swatter = new T.Mesh(new T.CylinderGeometry(34, 40, 6, 32),
    new T.MeshStandardMaterial({ color: ALERT, roughness: 0.8 }));
  swatter.rotation.x = Math.PI / 2; scene.add(swatter);
  const swatterRing = ring(34, ALERT); scene.add(swatterRing);

  const prey = new T.Mesh(new T.SphereGeometry(11, 20, 14),
    new T.MeshStandardMaterial({ color: 0x8fa4b5, roughness: 0.7 }));
  scene.add(prey);

  const plumeCanvas = document.createElement("canvas");
  plumeCanvas.width = plumeCanvas.height = 128;
  const plumeCtx = plumeCanvas.getContext("2d");
  const plumeTex = new T.CanvasTexture(plumeCanvas);
  const plume = new T.Mesh(new T.PlaneGeometry(ARENA * 2, ARENA * 2),
    new T.MeshBasicMaterial({ map: plumeTex, transparent: true, opacity: 0.7 }));
  plume.position.z = 0.6; scene.add(plume);

  const TRAIL = 420;
  const trailPos = new Float32Array(TRAIL * 3);
  const trailGeo = new T.BufferGeometry();
  trailGeo.setAttribute("position", new T.BufferAttribute(trailPos, 3));
  const trail = new T.Line(trailGeo, new T.LineBasicMaterial({
    color: 0x55707f, transparent: true, opacity: 0.75 }));
  scene.add(trail);

  const camera = new T.PerspectiveCamera(42, 16 / 10, 1, 4000);
  camera.up.set(0, 0, 1);
  const focus = new T.Vector3();
  const cam = orbit(canvas, camera, focus);

  // ---- brain scene
  const brainScene = new T.Scene();
  const brain = buildBrain(graph);
  brainScene.add(brain.group);
  const brainCam = new T.PerspectiveCamera(38, 1, 0.05, 40);
  brainCam.up.set(0, 1, 0);

  // ---- readout strip: real descending cell types, from the connectome
  const outputTypes = graph.outputs.map(i => graph.nodes[i].type || "DN");
  const outputSigns = graph.outputs.map(i => graph.nodes[i].sign);
  const cellsHost = $("#cells");
  const bars = outputTypes.map((type, i) => {
    const cell = document.createElement("div"); cell.className = "cell";
    const bar = document.createElement("div"); bar.className = "bar";
    const fill = document.createElement("i");
    const zero = document.createElement("u");
    bar.append(zero, fill);
    const label = document.createElement("span");
    label.textContent = type; label.title = `${type} · ${graph.nodes[graph.outputs[i]].nt ?? "unknown"}`;
    cell.append(bar, label); cellsHost.append(cell);
    return fill;
  });
  $("#legend-count").textContent =
    `${graph.nodes.length} cells · ${graph.edges.length.toLocaleString()} measured connections`;
  $("#s-cells").textContent = graph.nodes.length;
  $("#s-contacts").textContent = graph.edges.reduce((a, e) => a + e[2], 0).toLocaleString();

  // ---- simulation state
  const circuit = new Circuit(graph);
  let taskName = "targets", env = null, theta = null, obs = null, action = 1, activity = [];
  let episodeSeed = 1, paused = false;

  function startEpisode(seed) {
    env = taskName === "forage"
      ? new Forage(1e9, mushroom, naiveMushroom, driveBad, driveSafe)
      : new ARENAS[taskName](180);             // long run; it is a toy, not a benchmark
    theta = policies[taskName]?.theta ?? null;
    obs = env.reset(seed);
    circuit.reset();
    activity = circuit.step(obs);
    const p = env.walker;
    for (let i = 0; i < TRAIL; i++) { trailPos[i * 3] = p.x; trailPos[i * 3 + 1] = p.y; trailPos[i * 3 + 2] = 1.2; }
    trailGeo.getAttribute("position").needsUpdate = true;
    obstacleGroup.clear();
    if (env.obstacles) for (const [ox, oy, r] of env.obstacles) {
      const m = new T.Mesh(new T.CylinderGeometry(r, r, 26, 20),
        new T.MeshStandardMaterial({ color: 0x27333d, roughness: 0.95 }));
      m.rotation.x = Math.PI / 2; m.position.set(ox, oy, 13);
      obstacleGroup.add(m);
    }
    plume.visible = taskName === "plume";
    goalRing.visible = goalBeam.visible = taskName === "targets" || taskName === "forage";
    swatter.visible = taskName === "swat";
    swatterRing.visible = taskName === "swat" || taskName === "forage";
    prey.visible = taskName === "chase";
    if (taskName === "forage") { goalBeam.visible = true; swatterRing.scale.setScalar(1); }
  }

  function decide() {
    activity = circuit.step(obs);
    action = theta ? act(theta, activity, circuit.outputs.length, Walker.N_ACTIONS) : 1;
    const [nextObs, score, done] = env.step(action);
    obs = nextObs;
    if (done) startEpisode(++episodeSeed);
    return score;
  }

  function pushTrail(x, y) {
    trailPos.copyWithin(0, 3);
    trailPos[(TRAIL - 1) * 3] = x; trailPos[(TRAIL - 1) * 3 + 1] = y; trailPos[(TRAIL - 1) * 3 + 2] = 1.2;
    trailGeo.getAttribute("position").needsUpdate = true;
  }

  let plumeAge = 0;
  function drawPlume() {
    if (typeof env?.concentration !== "function") return;
    const img = plumeCtx.createImageData(128, 128);
    for (let j = 0; j < 128; j++) for (let i = 0; i < 128; i++) {
      const x = (i / 127) * 2 * ARENA - ARENA, y = ARENA - (j / 127) * 2 * ARENA;
      const c = env.concentration(x, y);
      const k = (j * 128 + i) * 4;
      img.data[k] = 47 + c * 180; img.data[k + 1] = 163 + c * 40;
      img.data[k + 2] = 154; img.data[k + 3] = c * 210;
    }
    plumeCtx.putImageData(img, 0, 0);
    plumeTex.needsUpdate = true;
  }

  // ---- UI
  const TASKS = [["targets", "Navigate"], ["swat", "Escape"], ["plume", "Track odour"],
                 ["chase", "Pursue"], ["forage", "Learn"]];
  const NOTE = {
    targets: "goal · obstacles · channels 0-5",
    swat: "looming threat on LC4 / LPLC2 · channels 3-5",
    plume: "odour on channel 7 · wind on channels 1-2",
    chase: "moving prey · channels 0-3",
    forage: "two smells, one shocks · the mushroom body learns which",
  };
  const taskHost = $("#tasks");
  TASKS.forEach(([key, label]) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = label; b.id = `task-${key}`;
    b.setAttribute("aria-pressed", String(key === taskName));
    b.addEventListener("click", () => {
      taskName = key; clipMode = false;
      [...taskHost.children].forEach(c => c.setAttribute("aria-pressed", String(c === b)));
      startEpisode(++episodeSeed); syncClipButton();
    });
    taskHost.append(b);
  });

  const silence = $("#silence");
  silence.addEventListener("click", () => {
    circuit.silenced = !circuit.silenced;
    if (mushroom) mushroom.dopamineOn = !circuit.silenced;
    silence.setAttribute("aria-pressed", String(circuit.silenced));
    silence.textContent = circuit.silenced ? "Circuit silenced" : "Silence circuit";
    $("#stage").dataset.silenced = String(circuit.silenced);
  });
  $("#restart").addEventListener("click", () => { clipMode = false; syncClipButton(); startEpisode(++episodeSeed); });

  // ---- recorded MuJoCo physics
  let clipMode = false, clipFrame = 0, clipTime = 0;
  const clipButton = $("#physics");
  function syncClipButton() {
    const available = !!clips[taskName];
    clipButton.disabled = !available;
    clipButton.setAttribute("aria-pressed", String(clipMode));
    clipButton.textContent = clipMode ? "Physics clip" : "Play physics clip";
  }
  clipButton.addEventListener("click", () => {
    if (!clips[taskName]) return;
    clipMode = !clipMode; clipFrame = 0; clipTime = 0; syncClipButton();
  });

  /** Drive the body straight from a recorded MuJoCo frame:
   *  [x, y, z, qw, qx, qy, qz, ...joint angles]. */
  function applyClipFrame(clip, f) {
    const row = clip.frames[f];
    fly.position.set(row[0] * FLY_SCALE, row[1] * FLY_SCALE, row[2] * FLY_SCALE);
    fly.quaternion.set(row[4], row[5], row[6], row[3]);
    scaled.position.z = 0;
    const angles = {};
    for (let i = 0; i < clip.joints.length; i++) angles[clip.joints[i]] = row[7 + i];
    applyPose(joints, angles);
  }

  // click the floor to move whatever the fly is chasing
  const ray = new T.Raycaster(), ndc = new T.Vector2(), plane = new T.Plane(new T.Vector3(0, 0, 1), 0);
  const hit = new T.Vector3();
  canvas.addEventListener("pointerup", e => {
    if (cam.state.moved) return;
    const r = canvas.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    if (!ray.ray.intersectPlane(plane, hit)) return;
    const x = Math.max(-ARENA, Math.min(ARENA, hit.x)), y = Math.max(-ARENA, Math.min(ARENA, hit.y));
    if (taskName === "targets") { env.goal = [x, y]; env.prev = env.goalDistance(); }
    else if (taskName === "chase") { env.px = x; env.py = y; }
    else if (taskName === "plume") { env.ox = x; env.oy = y; env.best = Math.hypot(x - env.walker.x, y - env.walker.y); drawPlume(); }
    else if (taskName === "swat") { env.sx = x; env.sy = y; env.sz = 1; env.wait = 0.25; }
  });

  // ---- loop
  startEpisode(episodeSeed);
  syncClipButton();
  drawPlume();
  let last = performance.now(), accumulator = 0, score = 0;
  $("#loading").remove();

  function resize() {
    const r = canvas.getBoundingClientRect();
    if (canvas.width !== Math.round(r.width * renderer.getPixelRatio())) {
      renderer.setSize(r.width, r.height, false);
      camera.aspect = r.width / r.height; camera.updateProjectionMatrix();
    }
  }

  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    if (!paused) {
      accumulator += dt;
      const stepTime = 1 / DECISION_HZ;
      let steps = 0;
      while (accumulator >= stepTime && steps++ < 4) { score = decide(); accumulator -= stepTime; }
    }
    resize();

    const w = env.walker;
    if (clipMode && clips[taskName]) {
      const clip = clips[taskName];
      clipTime += dt;
      clipFrame = Math.floor(clipTime * clip.fps) % clip.frames.length;
      applyClipFrame(clip, clipFrame);
      pushTrail(fly.position.x, fly.position.y);
      focus.lerp(new T.Vector3(fly.position.x, fly.position.y, 16), 0.07);
    } else {
      fly.position.set(w.x, w.y, 0);
      fly.rotation.set(0, 0, w.heading);
      scaled.position.z = -FOOT_Z * FLY_SCALE;
      applyPose(joints, poseFrom(legmap, w.phase, w.speed, w.turn));
      pushTrail(w.x, w.y);
      focus.lerp(new T.Vector3(w.x, w.y, 16), 0.07);
    }
    cam.apply();

    if (taskName === "targets") {
      goalRing.position.set(env.goal[0], env.goal[1], 0.5);
      goalBeam.position.set(env.goal[0], env.goal[1], 45);
    } else if (taskName === "swat") {
      swatter.position.set(env.sx, env.sy, 12 + env.sz * 150);
      swatterRing.position.set(env.sx, env.sy, 0.5);
      swatterRing.scale.setScalar(1 + (1 - env.sz) * 0.12);
    } else if (taskName === "chase") {
      prey.position.set(env.px, env.py, 11);
    } else if (taskName === "plume") {
      plumeAge += dt; if (plumeAge > 0.1) { plumeAge = 0; drawPlume(); }
    } else if (taskName === "forage") {
      goalRing.position.set(env.safe[0], env.safe[1], 0.5);
      goalBeam.position.set(env.safe[0], env.safe[1], 45);
      swatterRing.position.set(env.bad[0], env.bad[1], 0.5);
      swatterRing.scale.setScalar(1 + env.flash * 0.5);
    }

    // descending-cell bars
    for (let i = 0; i < bars.length; i++) {
      const v = Math.max(-1, Math.min(1, activity[i] ?? 0));
      const h = Math.abs(v) * 26;
      const s = bars[i].style;
      s.height = `${h}px`;
      s.background = v >= 0 ? "var(--excite)" : "var(--inhibit)";
      s.transform = v >= 0 ? "translateY(-100%)" : "none";
    }
    paintBrain(brain, circuit.h);
    $("#s-score").textContent = Math.round(score).toLocaleString();
    const verb = ["hold", "walk", "turn left", "turn right", "back"][action];
    if (clipMode) {
      const clip = clips[taskName];
      $("#tag").innerHTML = `<b>MuJoCo physics</b> &nbsp;·&nbsp; 68 segments, 102 joints &nbsp;·&nbsp; ` +
        `walked ${clip.travelled_body_lengths} body lengths`;
    } else
    if (taskName === "forage" && !clipMode) {
      const v = env.valence ?? 0;
      const verdict = v < -0.25 ? "<b>avoid</b>" : v > 0.25 ? "<b>approach</b>" : "no opinion yet";
      $("#tag").innerHTML = `${NOTE.forage} &nbsp;·&nbsp; fed ${env.food} &nbsp;·&nbsp; ` +
        `shocked ${env.shocks} &nbsp;·&nbsp; verdict on this smell: ${verdict}`;
    } else
    $("#tag").innerHTML = circuit.silenced
      ? `<b style="color:${"#C8553D"}">circuit silenced</b> — readout blind`
      : `${NOTE[taskName]} &nbsp;·&nbsp; <b>${verb}</b>`;

    // main viewport
    const size = new T.Vector2(); renderer.getSize(size);
    renderer.setScissorTest(false);
    renderer.clear();
    renderer.setViewport(0, 0, size.x, size.y);
    renderer.render(scene, camera);

    // brain inset, top right
    const iw = Math.max(140, Math.min(280, size.x * 0.3)), ih = iw * 0.86;
    const ix = size.x - iw - 10, iy = size.y - ih - 10;
    renderer.setScissorTest(true);
    renderer.setScissor(ix, iy, iw, ih);
    renderer.setViewport(ix, iy, iw, ih);
    renderer.clearDepth();
    brain.group.rotation.y += dt * 0.22;
    brainCam.aspect = iw / ih; brainCam.updateProjectionMatrix();
    brainCam.position.set(0, 0.32, 2.45); brainCam.lookAt(0, 0, 0);
    renderer.render(brainScene, brainCam);
    renderer.setScissorTest(false);
  }
  requestAnimationFrame(frame);

  document.addEventListener("visibilitychange", () => { paused = document.hidden; last = performance.now(); });
})();
