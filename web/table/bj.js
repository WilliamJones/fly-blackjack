// Port of flybrain/mb/blackjack.py. Checked against Python in tools/check_bj_parity.
import { MushroomBody } from "./mb.js";

export const HIT = 0, STAND = 1, DOUBLE = 2;
export const ACTION_NAMES = ["hit", "stand", "double"];
const CARDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11];
const FACES = { 11: "A", 10: ["10", "J", "Q", "K"] };

export function total(cards) {
  let t = cards.reduce((a, b) => a + b, 0);
  let aces = cards.filter(c => c === 11).length;
  while (t > 21 && aces) { t -= 10; aces--; }
  return [t, aces > 0];
}

export function basicStrategy(pt, soft, up, canDouble) {
  if (soft) {
    if (pt >= 19) return STAND;
    if (pt === 18) {
      if ([3, 4, 5, 6].includes(up)) return canDouble ? DOUBLE : STAND;
      return [2, 7, 8].includes(up) ? STAND : HIT;
    }
    if (pt === 17) return canDouble && [3, 4, 5, 6].includes(up) ? DOUBLE : HIT;
    if (pt === 15 || pt === 16) return canDouble && [4, 5, 6].includes(up) ? DOUBLE : HIT;
    return canDouble && [5, 6].includes(up) ? DOUBLE : HIT;
  }
  if (pt >= 17) return STAND;
  if (pt >= 13) return up <= 6 ? STAND : HIT;
  if (pt === 12) return [4, 5, 6].includes(up) ? STAND : HIT;
  if (pt === 11) return canDouble && up <= 10 ? DOUBLE : HIT;
  if (pt === 10) return canDouble && up <= 9 ? DOUBLE : HIT;
  if (pt === 9) return canDouble && [3, 4, 5, 6].includes(up) ? DOUBLE : HIT;
  return HIT;
}

export function chartCells() {
  const cells = [];
  for (let up = 2; up <= 11; up++) {
    for (let t = 5; t <= 21; t++) cells.push([t, false, up]);
    for (let t = 13; t <= 21; t++) cells.push([t, true, up]);
  }
  return cells;
}

/** A card for display: value plus a face and suit chosen at deal time. */
export function drawCard(rng) {
  const v = CARDS[Math.floor(rng() * CARDS.length)];
  const face = v === 11 ? "A" : v === 10 ? FACES[10][Math.floor(rng() * 4)] : String(v);
  const suit = "♠♥♦♣"[Math.floor(rng() * 4)];
  return { v, face, suit };
}

export class Situation {
  constructor(mb, patterns) {
    this.mb = mb;
    this.drive = {};
    for (const [feature, levels] of Object.entries(patterns)) {
      this.drive[feature] = levels.map(cells => {
        const pn = new Float64Array(mb.nPn);
        for (const [i, v] of cells) pn[i] = v;
        return mb.pnDrive(pn);
      });
    }
  }
  encode(pt, soft, up, first, action) {
    const parts = [
      this.drive.total[Math.min(17, Math.max(0, pt - 4))],
      this.drive.upcard[up - 2],
      this.drive.action[action],
      this.drive.soft[soft ? 1 : 0],
      this.drive.first[first ? 1 : 0],
    ];
    const d = new Float64Array(this.mb.nKc);
    for (const p of parts) for (let i = 0; i < d.length; i++) d[i] += p[i];
    return d;
  }
}

export class FlyPlayer {
  constructor(mbData, patterns, { lr = 0.05, decay = 0.00005, explore = 0.35,
                                  exploreFloor = 0.03, halflife = 3000 } = {}) {
    this.mb = new MushroomBody(mbData);
    this.mb.lr = lr; this.mb.decay = decay;
    this.sit = new Situation(this.mb, patterns);
    this.explore0 = explore; this.exploreFloor = exploreFloor; this.halflife = halflife;
    this.hands = 0;
  }
  exploration() {
    return this.exploreFloor + (this.explore0 - this.exploreFloor) * Math.pow(0.5, this.hands / this.halflife);
  }
  value(pt, soft, up, first, a) {
    return this.mb.presentDrive(this.sit.encode(pt, soft, up, first, a), 0, false);
  }
  values(pt, soft, up, first, legal) {
    const out = {};
    for (const a of legal) out[a] = this.value(pt, soft, up, first, a);
    return out;
  }
  greedy(pt, soft, up, first, legal) {
    const v = this.values(pt, soft, up, first, legal);
    return legal.reduce((best, a) => (v[a] > v[best] ? a : best), legal[0]);
  }
  choose(pt, soft, up, first, legal, rng) {
    if (rng() < this.exploration()) return legal[Math.floor(rng() * legal.length)];
    return this.greedy(pt, soft, up, first, legal);
  }
  outcome(net, decisions) {
    this.hands++;
    if (net === 0 || !decisions.length) return;
    const strength = Math.min(1, Math.max(0.15, Math.abs(net) / 2));
    const reward = net > 0 ? strength : 0, punish = net < 0 ? strength : 0;
    for (const [[pt, soft, up, first], a] of decisions) {
      this.mb.presentDriveLearn(this.sit.encode(pt, soft, up, first, a), punish, reward);
    }
  }
  agreement() {
    let hits = 0; const cells = chartCells();
    for (const [t, soft, up] of cells)
      if (basicStrategy(t, soft, up, true) === this.greedy(t, soft, up, true, [HIT, STAND, DOUBLE])) hits++;
    return hits / cells.length;
  }
  /** Compact brain for the shared store: gains quantised to bytes. */
  exportBrain() {
    const g = this.mb.gain, bytes = new Uint8Array(g.length);
    for (let i = 0; i < g.length; i++) bytes[i] = Math.round(Math.min(1, Math.max(0, g[i])) * 255);
    let s = ""; for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
    return { gain: btoa(s), hands: this.hands, n: g.length };
  }
  importBrain(b) {
    if (!b || !b.gain) return false;
    const s = atob(b.gain);
    if (s.length !== this.mb.gain.length) return false;
    for (let i = 0; i < s.length; i++) this.mb.gain[i] = s.charCodeAt(i) / 255;
    this.hands = b.hands || 0;
    this.mb.refreshDepression();
    return true;
  }
}

/** Seeded RNG so a hand can be replayed. */
export function makeRng(seed) {
  let s = (seed >>> 0) || 1;
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}
