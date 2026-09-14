// Port of flybrain/circuit.py, readout.py, body/gait.py and body/arenas.py.
// Verified numerically against the Python originals by tools/check_parity.mjs --
// if you edit this file, run that check before you trust what you see.

// ---------------------------------------------------------------- circuit
export class Circuit {
  constructor(graph, { gain = 1.4, leak = 0.3, steps = 3 } = {}) {
    this.nodes = graph.nodes;
    this.channels = graph.channels;
    this.inputs = graph.inputs;
    this.outputs = graph.outputs;
    this.n = graph.nodes.length;
    this.gain = gain; this.leak = leak; this.steps = steps;

    const sign = graph.nodes.map(nd => nd.sign);
    const signed = new Float64Array(this.n * this.n);   // [pre * n + post]
    const abs = new Float64Array(this.n * this.n);
    for (const [pre, post, contacts] of graph.edges) {
      signed[pre * this.n + post] += contacts * sign[pre];
      abs[pre * this.n + post] += contacts * Math.abs(sign[pre]);
    }
    const denom = new Float64Array(this.n);
    for (let pre = 0; pre < this.n; pre++)
      for (let post = 0; post < this.n; post++) denom[post] += abs[pre * this.n + post];
    this.W = signed;
    for (let post = 0; post < this.n; post++) {
      const d = denom[post] === 0 ? 1 : denom[post];
      for (let pre = 0; pre < this.n; pre++) this.W[pre * this.n + post] /= d;
    }
    this.h = new Float64Array(this.n);
    this.u = new Float64Array(this.n);
    this.silenced = false;
  }

  reset() { this.h.fill(0); }

  step(obs) {
    if (this.silenced) { this.h.fill(0); return this.outputs.map(() => 0); }
    this.u.fill(0);
    for (const [node, channel] of this.inputs) this.u[node] = 2 * (obs[channel] - 0.5);
    for (let s = 0; s < this.steps; s++) {
      const next = new Float64Array(this.n);
      for (let post = 0; post < this.n; post++) {
        let drive = 0;
        for (let pre = 0; pre < this.n; pre++) {
          const h = this.h[pre];
          if (h !== 0) drive += h * this.W[pre * this.n + post];
        }
        next[post] = this.leak * this.h[post] +
                     (1 - this.leak) * Math.tanh(this.u[post] + this.gain * drive);
      }
      this.h = next;
    }
    return this.outputs.map(i => this.h[i]);
  }
}

// ---------------------------------------------------------------- readout
export const HIDDEN = 12, INPUT_GAIN = 4;

export function act(theta, activity, nIn, nOut) {
  let i = 0;
  let best = -Infinity, bestIndex = 0;
  const hidden = new Float64Array(HIDDEN);
  for (let h = 0; h < HIDDEN; h++) {
    let s = 0;
    for (let k = 0; k < nIn; k++) s += INPUT_GAIN * activity[k] * theta[k * HIDDEN + h];
    hidden[h] = s;
  }
  i = nIn * HIDDEN;
  for (let h = 0; h < HIDDEN; h++) hidden[h] = Math.tanh(hidden[h] + theta[i + h]);
  i += HIDDEN;
  for (let o = 0; o < nOut; o++) {
    let s = theta[i + HIDDEN * nOut + o];
    for (let h = 0; h < HIDDEN; h++) s += hidden[h] * theta[i + h * nOut + o];
    if (s > best) { best = s; bestIndex = o; }
  }
  return bestIndex;
}

// ------------------------------------------------------------------- gait
// Mirrors flybrain/body/gait.py. Foot targets are in MuJoCo length units and
// are turned into joint angles by the measured inverse matrices in legmap.json.
export const LEGS = ["T1_left", "T2_left", "T3_left", "T1_right", "T2_right", "T3_right"];
const TRIPOD_B = new Set(["T1_right", "T2_left", "T3_right"]);
const DUTY = 0.62, STRIDE_RATE = 7.5, IDLE_RATE = 0.6;
const STRIDE = 0.055, LIFT = 0.030, LATERAL_TURN = 0.028;
export const LEG_JOINTS = ["coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur", "tibia"];
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export function legPhase(phase, leg) {
  let p = (phase + (TRIPOD_B.has(leg) ? Math.PI : 0)) % (2 * Math.PI);
  return p < 0 ? p + 2 * Math.PI : p;
}

export function footOffset(phase, leg, speed, turn) {
  const side = leg.endsWith("left") ? 1 : -1;
  const p = legPhase(phase, leg) / (2 * Math.PI);
  let stride = clamp(1 + 0.55 * turn * side, 0.2, 1.8) * clamp(Math.abs(speed), 0.12, 1);
  const direction = speed >= 0 ? 1 : -1;
  let along, lift;
  if (p < DUTY) { const s = p / DUTY; along = (0.5 - s) * STRIDE; lift = 0; }
  else { const s = (p - DUTY) / (1 - DUTY); along = (s - 0.5) * STRIDE; lift = Math.sin(Math.PI * s) * LIFT; }
  return [along * stride * direction,
          -side * turn * LATERAL_TURN * clamp(Math.abs(speed), 0, 1),
          lift];
}

// legmap: { joints: [...], legs: { "T1_left": 6x3 matrix, ... } }
export function poseFrom(legmap, phase, speed, turn) {
  const angles = {};
  for (const leg of LEGS) {
    const d = footOffset(phase, leg, speed, turn);
    const M = legmap.legs[leg];
    for (let k = 0; k < legmap.joints.length; k++) {
      angles[`${legmap.joints[k]}_${leg}`] = M[k][0] * d[0] + M[k][1] * d[1] + M[k][2] * d[2];
    }
  }
  return angles;
}

export class Walker {
  static ACTIONS = [[0, 0], [1, 0], [0.75, 1], [0.75, -1], [-0.6, 0]];
  static N_ACTIONS = 5;
  static MAX_SPEED = 22; static MAX_TURN = 4.2; static RESPONSE = 0.28;

  constructor() { this.reset(0, 0, 0); }
  reset(x, y, heading) {
    this.x = x; this.y = y; this.heading = heading;
    this.speed = 0; this.turn = 0; this.phase = 0;
  }
  step(action, dt) {
    const [ts, tt] = Walker.ACTIONS[action];
    this.speed += (ts - this.speed) * Walker.RESPONSE;
    this.turn += (tt - this.turn) * Walker.RESPONSE;
    this.heading += this.turn * Walker.MAX_TURN * dt;
    this.heading = ((this.heading + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    this.x += this.speed * Walker.MAX_SPEED * Math.cos(this.heading) * dt;
    this.y += this.speed * Walker.MAX_SPEED * Math.sin(this.heading) * dt;
    const rate = IDLE_RATE + STRIDE_RATE * Math.abs(this.speed);
    this.phase = (this.phase + 2 * Math.PI * rate * dt) % (2 * Math.PI);
  }
  jointPose(legmap) { return poseFrom(legmap, this.phase, this.speed, this.turn); }
}

// ----------------------------------------------------------------- arenas
export const ARENA = 220, TAU = 2 * Math.PI;

function bearingChannels(dx, dy, heading) {
  const s = Math.sin(Math.atan2(dy, dx) - heading);
  return [Math.max(0, s), Math.max(0, -s)];
}

// Deterministic RNG matching numpy's default_rng closely enough for the browser.
// The browser does not need to reproduce Python's exact episodes -- it needs to
// be reproducible for a given seed so a shared link shows the same run.
class RNG {
  constructor(seed) { this.s = (seed >>> 0) || 1; }
  next() { let x = this.s; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; this.s = x >>> 0; return this.s / 4294967296; }
  uniform(a, b) { return a + (b - a) * this.next(); }
  normal(mu = 0, sd = 1) {
    const u = Math.max(this.next(), 1e-12), v = this.next();
    return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
  }
  int(n) { return Math.floor(this.next() * n) % n; }
}

class ArenaBase {
  static N_OBS = 8;
  constructor(seconds = 20) { this.seconds = seconds; this.walker = new Walker(); this.DT = 1 / 60; this.CONTROL_EVERY = 2; }
  get N_ACTIONS() { return Walker.N_ACTIONS; }
  reset(seed = 0) {
    this.rng = new RNG(seed); this.t = 0; this.score = 0; this.trail = [];
    this.walker.reset(0, 0, this.rng.uniform(-Math.PI, Math.PI));
    this.setup();
    return this.sense();
  }
  step(action) {
    for (let i = 0; i < this.CONTROL_EVERY; i++) {
      this.walker.step(action, this.DT);
      this.walker.x = clamp(this.walker.x, -ARENA, ARENA);
      this.walker.y = clamp(this.walker.y, -ARENA, ARENA);
      this.t += this.DT;
      this.advance();
      this.score += this.reward() * this.DT;
    }
    this.trail.push([this.walker.x, this.walker.y]);
    if (this.trail.length > 600) this.trail.shift();
    return [this.sense(), this.score, this.terminal() || this.t >= this.seconds];
  }
  setup() {} advance() {} reward() { return 0; } terminal() { return false; }
  speedChannel() { return clamp(0.5 + 0.5 * this.walker.speed, 0, 1); }
}

export class Targets extends ArenaBase {
  static NAME = "targets"; static LABEL = "Navigate";
  setup() {
    this.obstacles = Array.from({ length: 6 }, () =>
      [this.rng.uniform(-160, 160), this.rng.uniform(-160, 160), this.rng.uniform(16, 30)]);
    this.reached = 0; this.newGoal(); this.prev = this.goalDistance();
  }
  newGoal() {
    for (let i = 0; i < 60; i++) {
      const gx = this.rng.uniform(-180, 180), gy = this.rng.uniform(-180, 180);
      if (this.obstacles.every(([ox, oy, r]) => Math.hypot(gx - ox, gy - oy) > r + 28)) { this.goal = [gx, gy]; return; }
    }
    this.goal = [this.rng.uniform(-180, 180), this.rng.uniform(-180, 180)];
  }
  goalDistance() { return Math.hypot(this.goal[0] - this.walker.x, this.goal[1] - this.walker.y); }
  nearestObstacle() {
    let best = null, bd = 1e9;
    for (const [ox, oy, r] of this.obstacles) {
      const d = Math.hypot(ox - this.walker.x, oy - this.walker.y) - r;
      if (d < bd) { best = [ox, oy, r]; bd = d; }
    }
    return [best, bd];
  }
  sense() {
    const o = new Array(8).fill(0);
    const dx = this.goal[0] - this.walker.x, dy = this.goal[1] - this.walker.y;
    o[0] = 1 - Math.min(1, Math.hypot(dx, dy) / 320);
    [o[1], o[2]] = bearingChannels(dx, dy, this.walker.heading);
    const [[ox, oy], d] = this.nearestObstacle();
    o[3] = 1 - Math.min(1, Math.max(0, d) / 120);
    [o[4], o[5]] = bearingChannels(ox - this.walker.x, oy - this.walker.y, this.walker.heading);
    o[6] = this.speedChannel();
    return o;
  }
  reward() {
    const d = this.goalDistance();
    const progress = (this.prev - d) * 60; this.prev = d;
    const [, od] = this.nearestObstacle();
    const penalty = od < 0 ? 40 : 0;
    if (d < 16) { this.reached++; this.newGoal(); this.prev = this.goalDistance(); return 300 - penalty; }
    return progress - penalty;
  }
}

export class Swat extends ArenaBase {
  static NAME = "swat"; static LABEL = "Escape";
  setup() { this.arm(true); this.hits = 0; this.dodges = 0; }
  arm(first = false) {
    const angle = this.rng.uniform(-Math.PI, Math.PI), reach = this.rng.uniform(90, 150);
    this.sx = this.walker.x + reach * Math.cos(angle);
    this.sy = this.walker.y + reach * Math.sin(angle);
    this.sz = 1; this.descent = this.rng.uniform(0.42, 0.72);
    this.wait = first ? 1.2 : this.rng.uniform(0.4, 1.1);
  }
  advance() {
    if (this.wait > 0) { this.wait -= this.DT; return; }
    this.sx += (this.walker.x - this.sx) * 0.9 * this.DT;
    this.sy += (this.walker.y - this.sy) * 0.9 * this.DT;
    this.sz -= this.descent * this.DT;
    if (this.sz <= 0) {
      const d = Math.hypot(this.walker.x - this.sx, this.walker.y - this.sy);
      if (d < 34) this.hits++; else this.dodges++;
      this.arm();
    }
  }
  sense() {
    const o = new Array(8).fill(0);
    const dx = this.sx - this.walker.x, dy = this.sy - this.walker.y, d = Math.hypot(dx, dy);
    o[3] = clamp((1 - this.sz) * (1 - Math.min(1, d / 180)), 0, 1);
    [o[4], o[5]] = bearingChannels(dx, dy, this.walker.heading);
    o[0] = 1 - Math.min(1, d / 260);
    o[6] = this.speedChannel();
    return o;
  }
  reward() {
    const d = Math.hypot(this.walker.x - this.sx, this.walker.y - this.sy);
    return Math.min(d, 120) * Math.pow(1 - this.sz, 2) * 0.6;
  }
  terminal() { return this.hits > 0; }
}

export class Plume extends ArenaBase {
  static NAME = "plume"; static LABEL = "Track odour";
  setup() {
    this.wind = this.rng.uniform(-Math.PI, Math.PI);
    const reach = this.rng.uniform(180, 300);
    this.ox = this.walker.x - reach * Math.cos(this.wind);
    this.oy = this.walker.y - reach * Math.sin(this.wind);
    this.found = 0;
    this.best = Math.hypot(this.ox - this.walker.x, this.oy - this.walker.y);
  }
  concentration(x, y) {
    const dx = x - this.ox, dy = y - this.oy;
    const downwind = dx * Math.cos(this.wind) + dy * Math.sin(this.wind);
    const cross = -dx * Math.sin(this.wind) + dy * Math.cos(this.wind);
    if (downwind < 0) return 0;
    const width = 12 + 0.3 * downwind;
    const c = Math.exp(-0.5 * Math.pow(cross / width, 2)) * Math.exp(-downwind / 420);
    const packet = 0.55 + 0.45 * Math.sin(downwind * 0.08 - this.t * 3.1 + cross * 0.02);
    return clamp(c * packet, 0, 1);
  }
  sense() {
    const o = new Array(8).fill(0);
    o[7] = this.concentration(this.walker.x, this.walker.y);
    [o[1], o[2]] = bearingChannels(Math.cos(this.wind + Math.PI), Math.sin(this.wind + Math.PI), this.walker.heading);
    const d = Math.hypot(this.ox - this.walker.x, this.oy - this.walker.y);
    o[0] = 1 - Math.min(1, d / 420);
    o[6] = this.speedChannel();
    return o;
  }
  reward() {
    const d = Math.hypot(this.ox - this.walker.x, this.oy - this.walker.y);
    const gain = Math.max(0, this.best - d) * 60; this.best = Math.min(this.best, d);
    if (d < 22 && this.found === 0) { this.found = 1; return 600; }
    return gain + 18 * this.concentration(this.walker.x, this.walker.y);
  }
  terminal() { return this.found > 0; }
}

export class Chase extends ArenaBase {
  static NAME = "chase"; static LABEL = "Pursue";
  setup() {
    const angle = this.rng.uniform(-Math.PI, Math.PI);
    this.px = this.walker.x + 150 * Math.cos(angle);
    this.py = this.walker.y + 150 * Math.sin(angle);
    this.pheading = this.rng.uniform(-Math.PI, Math.PI);
    this.contact = 0;
  }
  advance() {
    const dx = this.px - this.walker.x, dy = this.py - this.walker.y, d = Math.hypot(dx, dy);
    this.pheading += this.rng.normal(0, 1.4) * this.DT;
    if (d < 110) {
      const away = Math.atan2(dy, dx);
      let delta = ((away - this.pheading + Math.PI) % TAU + TAU) % TAU - Math.PI;
      this.pheading += clamp(delta, -3, 3) * 2.2 * this.DT;
    }
    const speed = d < 110 ? 16 : 9;
    this.px = clamp(this.px + speed * Math.cos(this.pheading) * this.DT, -ARENA, ARENA);
    this.py = clamp(this.py + speed * Math.sin(this.pheading) * this.DT, -ARENA, ARENA);
  }
  sense() {
    const o = new Array(8).fill(0);
    const dx = this.px - this.walker.x, dy = this.py - this.walker.y, d = Math.hypot(dx, dy);
    o[0] = 1 - Math.min(1, d / 340);
    [o[1], o[2]] = bearingChannels(dx, dy, this.walker.heading);
    o[3] = clamp((d - 40) / 200, 0, 1);
    o[6] = this.speedChannel();
    return o;
  }
  reward() {
    const d = Math.hypot(this.px - this.walker.x, this.py - this.walker.y);
    if (d < 26) { this.contact += this.DT; return 200; }
    return Math.max(0, 1 - d / 340) * 60;
  }
}

export const ARENAS = { targets: Targets, swat: Swat, plume: Plume, chase: Chase };

// ------------------------------------------------------- forage (part four)
// Port of flybrain/mb/embodied.py. The mushroom body learns inside the episode;
// the steering weights never change. Channels 3 and 4 are the only route by
// which odour identity reaches the steering circuit.
export const ANTENNA = 14, SOURCE_SIGMA = 95, REACH = 30,
             MB_EVERY = 5, SHOCK_PAIRINGS = 4, MIN_SEPARATION = 230;

export class Forage extends ArenaBase {
  static NAME = "forage"; static LABEL = "Learn";
  constructor(seconds = 120, mb = null, naive = null, driveBad = null, driveSafe = null) {
    super(seconds);
    this.mb = mb; this.naive = naive;
    this.driveBad = driveBad; this.driveSafe = driveSafe;
    this.tick = 0;
  }
  setup() {
    if (this.mb) { this.mb.reset(); this.naive.reset(); }
    this.shocks = 0; this.food = 0; this.valence = 0; this.intensity = 0;
    this.lastSmell = 0; this.shockCooldown = 0; this.tick = 0; this.flash = 0;
    this.bad = this.spot(null); this.safe = this.spot(this.bad);
  }
  spot(awayFrom) {
    let x = 0, y = 0;
    for (let i = 0; i < 40; i++) {
      const a = this.rng.uniform(-Math.PI, Math.PI), r = this.rng.uniform(120, 190);
      x = clamp(this.walker.x + r * Math.cos(a), -ARENA, ARENA);
      y = clamp(this.walker.y + r * Math.sin(a), -ARENA, ARENA);
      if (!awayFrom || Math.hypot(x - awayFrom[0], y - awayFrom[1]) >= MIN_SEPARATION) break;
    }
    return [x, y];
  }
  concentration(x, y, s) {
    return Math.exp(-0.5 * Math.pow(Math.hypot(x - s[0], y - s[1]) / SOURCE_SIGMA, 2));
  }
  smellAt(x, y) {
    return [this.concentration(x, y, this.bad), this.concentration(x, y, this.safe)];
  }
  runMushroomBody(punish) {
    if (!this.mb) return;
    const [cb, cs] = this.smellAt(this.walker.x, this.walker.y);
    const drive = new Float64Array(this.mb.nKc);
    for (let i = 0; i < drive.length; i++) drive[i] = cb * this.driveBad[i] + cs * this.driveSafe[i];
    this.intensity = cb + cs;
    const learned = this.mb.presentDrive(drive, punish);
    const naive = this.naive.presentDrive(drive, 0, false);
    this.valence = learned - naive;
  }
  advance() {
    this.tick++;
    this.shockCooldown = Math.max(0, this.shockCooldown - this.DT);
    this.flash *= 0.9;
    const w = this.walker;
    if (Math.abs(w.x) >= ARENA - 1 || Math.abs(w.y) >= ARENA - 1) {
      w.heading = Math.atan2(-w.y, -w.x);
      w.x = clamp(w.x, -ARENA + 6, ARENA - 6); w.y = clamp(w.y, -ARENA + 6, ARENA - 6);
    }
    const dBad = Math.hypot(w.x - this.bad[0], w.y - this.bad[1]);
    const dSafe = Math.hypot(w.x - this.safe[0], w.y - this.safe[1]);
    if (dBad < REACH && this.shockCooldown <= 0) {
      for (let i = 0; i < SHOCK_PAIRINGS; i++) this.runMushroomBody(1);
      this.shocks++; this.shockCooldown = 1.5; this.flash = 1;
      this.bad = this.spot(this.safe);
      this.lastSmell = this.smellAt(w.x, w.y).reduce((a, b) => a + b, 0);
    } else if (this.tick % MB_EVERY === 0) {
      this.runMushroomBody(0);
    }
    if (dSafe < REACH) {
      this.food++;
      this.safe = this.spot(this.bad);
      this.lastSmell = this.smellAt(w.x, w.y).reduce((a, b) => a + b, 0);
    }
  }
  sense() {
    const o = new Array(8).fill(0), w = this.walker, h = w.heading;
    const lx = w.x - ANTENNA * Math.sin(h), ly = w.y + ANTENNA * Math.cos(h);
    const rx = w.x + ANTENNA * Math.sin(h), ry = w.y - ANTENNA * Math.cos(h);
    const sum = p => p[0] + p[1];
    const here = sum(this.smellAt(w.x, w.y));
    const d = sum(this.smellAt(lx, ly)) - sum(this.smellAt(rx, ry));
    o[0] = Math.min(1, here);
    o[1] = clamp(0.5 + 12 * d, 0, 1);
    o[2] = clamp(0.5 + 60 * d, 0, 1);
    o[3] = clamp(0.5 + 0.5 * this.valence, 0, 1);
    o[4] = clamp(Math.abs(this.valence) * Math.min(1, here * 2), 0, 1);
    o[5] = this.speedChannel();
    const dB = Math.hypot(w.x - this.bad[0], w.y - this.bad[1]);
    const dS = Math.hypot(w.x - this.safe[0], w.y - this.safe[1]);
    const near = dB < dS ? this.bad : this.safe;
    [o[6], o[7]] = bearingChannels(near[0] - w.x, near[1] - w.y, h);
    return o;
  }
  reward() { return 0; }
}
ARENAS.forage = Forage;
