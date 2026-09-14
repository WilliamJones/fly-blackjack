import { FlyPlayer, total, drawCard, makeRng, basicStrategy, chartCells,
         HIT, STAND, DOUBLE, ACTION_NAMES } from "./bj.js";
import { FlyAtTable } from "./fly3d.js";

const $ = s => document.querySelector(s);
const ROUNDS_PER_SEAT = 10;
const HEARTBEAT_MS = 5000, STALE_MS = 40000;
const now = () => Date.now();
const sleep = ms => new Promise(r => setTimeout(r, ms));
class Abandoned extends Error {}
const pause = async ms => { await sleep(ms); if (!seated) throw new Abandoned(); };

// ------------------------------------------------------------ identity
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};
const clientId = store.get("fly-table-id") || ("c" + Math.random().toString(36).slice(2, 10));
store.set("fly-table-id", clientId);
const nameInput = $("#name");
nameInput.value = store.get("fly-table-name") || "";
nameInput.addEventListener("input", () => store.set("fly-table-name", nameInput.value.trim()));
const myName = () => nameInput.value.trim() || "Someone";

// ------------------------------------------------------------- the brain
const [mbData, patterns] = await Promise.all([
  fetch("mb.json").then(r => r.json()), fetch("situation.json").then(r => r.json())]);
const fly = new FlyPlayer(mbData, patterns);
let lifetime = { hands: 0, units: 0 };

let scene = null;
(async () => {
  if (!window.THREE) return;
  try {
    const [flyMesh, skeleton, gestures] = await Promise.all([
      fetch("fly.json").then(r => r.json()), fetch("skeleton.json").then(r => r.json()),
      fetch("gestures.json").then(r => r.json())]);
    scene = new FlyAtTable($("#scene"), { fly: flyMesh, skeleton, gestures, mb: mbData, mush: fly.mb });
  } catch (e) { console.warn("scene unavailable", e); }
})();

/** Light the brain for a spot: recomputed locally so spectators see it too. */
function showSpot(spot) {
  if (!scene || !spot) return;
  const [pt, soft, up, first, action] = spot;
  fly.mb.presentDrive(fly.sit.encode(pt, soft, up, first, action), 0, false);
  scene.showSituation(Float32Array.from(fly.mb.kc), Float32Array.from(fly.mb.mbon));
}
function moodFor(status) {
  if (!status) return "idle";
  if (status.startsWith("the fly is thinking")) return "thinking";
  if (status.startsWith("the fly takes")) return "won";
  if (status.startsWith("you take")) return "lost";
  return "idle";
}

// ----------------------------------------------------- store adapters
// Two implementations of one small interface, so the public version is an
// adapter swap: this one talks to the artifact's shared store (Firestore-
// shaped), the other keeps everything in this tab.
class SharedStore {
  constructor(db) { this.db = db; }
  watchSeat(cb) { return this.db.doc("table/seat").onSnapshot(s => cb(s.exists ? s.data() : null), () => cb(null)); }
  watchQueue(cb) {
    return this.db.collection("queue").orderBy("joinedAt").limit(50)
      .onSnapshot(q => cb(q.docs.map(d => ({ id: d.id, ...d.data() }))), () => cb([]));
  }
  watchLive(cb) { return this.db.doc("table/live").onSnapshot(s => cb(s.exists ? s.data() : null), () => {}); }
  watchBrain(cb) { return this.db.doc("fly/brain").onSnapshot(s => cb(s.exists ? s.data() : null), () => cb(null)); }
  watchHands(cb) {
    return this.db.collection("hands").orderBy("ts", "desc").limit(12)
      .onSnapshot(q => cb(q.docs.map(d => d.data())), () => cb([]));
  }
  async claimSeat() {
    const ref = this.db.doc("table/seat");
    const res = await ref.acquire({ holder: clientId, ttlMs: 15000 });
    if (!res.acquired) return false;
    const snap = await ref.get();
    const d = snap.exists ? snap.data() : null;
    const free = !d || !d.owner || d.owner === clientId || now() - (d.heartbeat || 0) > STALE_MS;
    if (!free) return false;
    await ref.set({ owner: clientId, name: myName(), since: now(), heartbeat: now() });
    return true;
  }
  async heartbeat() {
    try { await this.db.doc("table/seat").update({ heartbeat: now() }); } catch {}
    try { await this.db.doc("table/seat").acquire({ holder: clientId, ttlMs: 15000 }); } catch {}
  }
  async releaseSeat() {
    const ref = this.db.doc("table/seat");
    try { await ref.set({ owner: null, name: null, since: null, heartbeat: now() }); } catch {}
    // Leases have no release verb -- they only expire. A holder may renew its
    // own lease with a shorter ttl, so renewing at the 1 s minimum is how we
    // hand the seat over without the next player waiting out our 15 s.
    try { await ref.acquire({ holder: clientId, ttlMs: 1000 }); } catch {}
  }
  async joinQueue() { await this.db.doc("queue/" + clientId).set({ name: myName(), joinedAt: now() }); }
  async leaveQueue() { try { await this.db.doc("queue/" + clientId).delete(); } catch {} }
  async publishLive(state) { try { await this.db.doc("table/live").set(state); } catch {} }
  async saveBrain(brain) { try { await this.db.doc("fly/brain").set(brain); } catch (e) { console.warn("brain save", e); } }
  async addHand(rec) { try { await this.db.collection("hands").add(rec); } catch {} }
}

class LocalStore {
  constructor() { this.seatCb = null; this.brainCb = null; this.handsCb = null; this.hands = []; }
  watchSeat(cb) { this.seatCb = cb; cb(this.seat || null); return () => {}; }
  watchQueue(cb) { cb([]); return () => {}; }
  watchLive(cb) { return () => {}; }
  watchBrain(cb) {
    this.brainCb = cb;
    let saved = null; try { saved = JSON.parse(store.get("fly-table-brain") || "null"); } catch {}
    cb(saved); return () => {};
  }
  watchHands(cb) { this.handsCb = cb; cb(this.hands); return () => {}; }
  async claimSeat() { this.seat = { owner: clientId, name: myName(), since: now(), heartbeat: now() }; this.seatCb?.(this.seat); return true; }
  async heartbeat() {}
  async releaseSeat() { this.seat = null; this.seatCb?.(null); }
  async joinQueue() {} async leaveQueue() {}
  async publishLive() {}
  async saveBrain(b) { store.set("fly-table-brain", JSON.stringify(b)); }
  async addHand(r) { this.hands = [r, ...this.hands].slice(0, 12); this.handsCb?.(this.hands); }
}

// --------------------------------------------------------------- render
const el = {
  dealerHand: $("#dealer-hand"), youHand: $("#you-hand"), flyHand: $("#fly-hand"),
  dealerTotal: $("#dealer-total"), youTotal: $("#you-total"), flyTotal: $("#fly-total"),
  youVerdict: $("#you-verdict"), flyVerdict: $("#fly-verdict"), youName: $("#you-name"),
  think: $("#think"), status: $("#status"), score: $("#score"), brainrow: $("#brainrow"),
  hit: $("#hit"), stand: $("#stand"), double: $("#double"),
};

function cardEl(c, hidden = false, fresh = false) {
  const d = document.createElement("div");
  d.className = "card" + (hidden ? " back" : "") + (fresh ? " new" : "") +
    (!hidden && (c.suit === "♥" || c.suit === "♦") ? " red" : "");
  if (!hidden) d.innerHTML = `<span>${c.face}</span><span class="s">${c.suit}</span>`;
  return d;
}

function renderHand(host, cards, { hideSecond = false, freshFrom = 0 } = {}) {
  host.replaceChildren(...cards.map((c, i) => cardEl(c, hideSecond && i === 1, i >= freshFrom)));
}

function totalText(cards, hideSecond = false) {
  if (!cards.length) return "";
  const shown = hideSecond ? cards.slice(0, 1) : cards;
  const [t, soft] = total(shown.map(c => c.v));
  if (t > 21) return `<b>${t}</b> bust`;
  return `<b>${t}</b>${soft && t <= 21 ? " soft" : ""}`;
}

function verdictEl(net) {
  if (net == null) return "";
  const cls = net > 0 ? "win" : net < 0 ? "lose" : "push";
  const txt = net > 0 ? `+${net}` : net < 0 ? `${net}` : "push";
  return `<span class="verdict ${cls}">${txt}</span>`;
}

function renderBrainRow(values) {
  el.brainrow.replaceChildren();
  if (!values) return;
  for (const a of [HIT, STAND, DOUBLE]) {
    const i = document.createElement("i");
    const v = values[a];
    const h = v == null ? 2 : Math.max(2, Math.min(26, 13 + v * 30));
    i.style.height = h + "px";
    i.style.background = v == null ? "#2B3A44" : ["#E2893A", "#2FA39A", "#A971D6"][a];
    i.title = `${ACTION_NAMES[a]} ${v == null ? "n/a" : v.toFixed(3)}`;
    el.brainrow.append(i);
  }
}

/** One function renders any table state -- ours or a spectator's copy. */
function renderTable(s) {
  renderHand(el.dealerHand, s.dealer.cards, { hideSecond: s.dealer.hidden, freshFrom: s.dealer.fresh ?? 99 });
  renderHand(el.youHand, s.you.cards, { freshFrom: s.you.fresh ?? 99 });
  renderHand(el.flyHand, s.fly.cards, { freshFrom: s.fly.fresh ?? 99 });
  el.dealerTotal.innerHTML = totalText(s.dealer.cards, s.dealer.hidden);
  el.youTotal.innerHTML = totalText(s.you.cards);
  el.flyTotal.innerHTML = totalText(s.fly.cards);
  el.youName.textContent = s.you.name || "You";
  el.youVerdict.innerHTML = verdictEl(s.you.net);
  el.flyVerdict.innerHTML = verdictEl(s.fly.net);
  el.think.innerHTML = s.fly.thought || "";
  el.status.textContent = s.status || "";
  el.score.textContent = s.round ? `round ${s.round} of ${ROUNDS_PER_SEAT} · ${s.you.name || "you"} ${s.score.you} – fly ${s.score.fly}` : "";
  renderBrainRow(s.fly.values);
  if (scene) {
    scene.setMood(moodFor(s.status));
    if (s.fly.spot && (s.fly.spotKey !== lastSpotKey)) { lastSpotKey = s.fly.spotKey; showSpot(s.fly.spot); }
    if (s.fly.flash && s.fly.flashKey !== lastFlashKey) { lastFlashKey = s.fly.flashKey; scene.flash(s.fly.flash); }
  }
}
let lastSpotKey = null, lastFlashKey = null;

// ------------------------------------------------------------ the chart
const chart = $("#chart"), cx = chart.getContext("2d");
function drawChart() {
  const W = chart.width, H = chart.height;
  const style = getComputedStyle(document.body);
  const ink = style.getPropertyValue("--ink").trim(), muted = style.getPropertyValue("--muted").trim();
  const bg = style.getPropertyValue("--surface").trim();
  cx.fillStyle = bg; cx.fillRect(0, 0, W, H);
  const cols = 10, left = 46, top = 22, cw = (W - left - 6) / cols;
  const hardRows = 17, softRows = 9, gap = 14;
  const ch = (H - top - gap - 8) / (hardRows + softRows);
  const colour = ["#E2893A", "#2FA39A", "#A971D6"];
  cx.font = '500 10px "IBM Plex Mono", monospace'; cx.textAlign = "center"; cx.fillStyle = muted;
  for (let c = 0; c < cols; c++) cx.fillText(c === 9 ? "A" : String(c + 2), left + cw * (c + 0.5), top - 8);
  cx.textAlign = "right";
  const cell = (t, soft, up, x, y) => {
    const got = fly.greedy(t, soft, up, true, [HIT, STAND, DOUBLE]);
    const want = basicStrategy(t, soft, up, true);
    cx.fillStyle = colour[got]; cx.globalAlpha = 0.85;
    cx.fillRect(x + 1, y + 1, cw - 2, ch - 2); cx.globalAlpha = 1;
    if (got !== want) { cx.fillStyle = "rgba(0,0,0,.55)"; cx.beginPath(); cx.arc(x + cw / 2, y + ch / 2, 2.2, 0, 7); cx.fill(); }
  };
  for (let r = 0; r < hardRows; r++) {
    const t = 21 - r, y = top + r * ch;
    cx.fillStyle = muted; cx.fillText(String(t), left - 6, y + ch * 0.72);
    for (let c = 0; c < cols; c++) cell(t, false, c + 2, left + c * cw, y);
  }
  const y0 = top + hardRows * ch + gap;
  cx.fillStyle = muted; cx.textAlign = "left"; cx.fillText("soft", 4, y0 - 3); cx.textAlign = "right";
  for (let r = 0; r < softRows; r++) {
    const t = 21 - r, y = y0 + r * ch;
    cx.fillStyle = muted; cx.fillText("A" + (t - 11), left - 6, y + ch * 0.72);
    for (let c = 0; c < cols; c++) cell(t, true, c + 2, left + c * cw, y);
  }
  cx.fillStyle = ink;
}

function refreshStats() {
  $("#s-hands").textContent = lifetime.hands.toLocaleString();
  $("#s-win").textContent = lifetime.hands >= 20 ? `${(100 * lifetime.units / lifetime.hands).toFixed(1)}%` : "—";
  $("#s-agree").textContent = `${(fly.agreement() * 100).toFixed(0)}%`;
}

// ---------------------------------------------------------------- game
let seated = false, round = 0, score = { you: 0, fly: 0 }, live = null, heartbeatTimer = null;
let state = null, resolveAction = null, backend = null, roomApi = null;

function blankState(status = "") {
  return { round, score, status, dealer: { cards: [], hidden: true }, you: { name: myName(), cards: [], net: null },
           fly: { cards: [], net: null, values: null, thought: "" } };
}
function publish() { renderTable(state); backend?.publishLive(state); }

function setButtons(legal) {
  el.hit.disabled = !legal.includes(HIT); el.stand.disabled = !legal.includes(STAND);
  el.double.disabled = !legal.includes(DOUBLE);
}
function waitAction(legal) {
  setButtons(legal);
  return new Promise(res => { resolveAction = a => { resolveAction = null; setButtons([]); res(a); }; });
}
el.hit.addEventListener("click", () => resolveAction?.(HIT));
el.stand.addEventListener("click", () => resolveAction?.(STAND));
el.double.addEventListener("click", () => resolveAction?.(DOUBLE));

async function playRound() {
  round++;
  const rng = makeRng((now() ^ (round * 7919)) >>> 0);
  state = blankState("dealing");
  state.round = round; state.score = { ...score };
  const draw = () => drawCard(rng);
  state.dealer.cards = [draw(), draw()];
  state.you.cards = [draw(), draw()];
  state.fly.cards = [draw(), draw()];
  state.dealer.fresh = 0; state.you.fresh = 0; state.fly.fresh = 0;
  publish(); await pause(500);
  delete state.dealer.fresh; delete state.you.fresh; delete state.fly.fresh;

  const up = state.dealer.cards[0].v;
  const vals = c => c.map(x => x.v);
  const [dt] = total(vals(state.dealer.cards));
  let youBet = 1, flyBet = 1;
  const flyDecisions = [];

  // -- you
  let [yt] = total(vals(state.you.cards));
  if (yt !== 21 && dt !== 21) {
    state.status = "your move"; publish();
    let first = true;
    while (true) {
      const legal = [HIT, STAND, ...(first ? [DOUBLE] : [])];
      const a = await waitAction(legal);
      if (!seated) throw new Abandoned();
      if (a === STAND) break;
      state.you.cards.push(draw()); state.you.fresh = state.you.cards.length - 1;
      [yt] = total(vals(state.you.cards));
      if (a === DOUBLE) { youBet = 2; publish(); break; }
      first = false; publish(); delete state.you.fresh;
      if (yt > 21) break;
    }
  }

  // -- the fly: imagine each option, take the one that smells best
  let [ft] = total(vals(state.fly.cards));
  if (ft !== 21 && dt !== 21) {
    let first = true;
    while (true) {
      const [pt, soft] = total(vals(state.fly.cards));
      const legal = [HIT, STAND, ...(first ? [DOUBLE] : [])];
      const values = fly.values(pt, soft, up, first, legal);
      const a = fly.choose(pt, soft, up, first, legal, rng);
      state.fly.values = values;
      state.fly.thought = `smelling ${legal.map(x => `${ACTION_NAMES[x]} ${values[x].toFixed(2)}`).join(" · ")} → <b>${ACTION_NAMES[a]}</b>`;
      state.fly.spot = [pt, soft, up, first, a]; state.fly.spotKey = `${round}-${flyDecisions.length + 1}`;
      state.status = "the fly is thinking"; publish(); await pause(700);
      flyDecisions.push([[pt, soft, up, first], a]);
      if (a === STAND) break;
      state.fly.cards.push(draw()); state.fly.fresh = state.fly.cards.length - 1;
      [ft] = total(vals(state.fly.cards));
      publish(); await pause(450); delete state.fly.fresh;
      if (a === DOUBLE) { flyBet = 2; break; }
      first = false;
      if (ft > 21) break;
    }
  }

  // -- dealer
  state.dealer.hidden = false; state.status = "dealer"; publish(); await pause(500);
  const anyoneAlive = yt <= 21 || ft <= 21;
  if (anyoneAlive) {
    while (total(vals(state.dealer.cards))[0] < 17) {
      state.dealer.cards.push(draw()); state.dealer.fresh = state.dealer.cards.length - 1;
      publish(); await pause(450); delete state.dealer.fresh;
    }
  }
  const [dfinal] = total(vals(state.dealer.cards));
  const settle = (t, bet, natural) => {
    if (natural && dt === 21) return 0;
    if (natural) return 1.5;
    if (dt === 21) return -1;
    if (t > 21) return -bet;
    if (dfinal > 21 || t > dfinal) return bet;
    if (t < dfinal) return -bet;
    return 0;
  };
  const youNet = settle(yt, youBet, state.you.cards.length === 2 && yt === 21);
  const flyNet = settle(ft, flyBet, state.fly.cards.length === 2 && ft === 21);
  state.you.net = youNet; state.fly.net = flyNet;
  if (youNet > flyNet) score.you++; else if (flyNet > youNet) score.fly++;
  state.score = { ...score };
  state.status = youNet > flyNet ? "you take the round" : flyNet > youNet ? "the fly takes the round" : "split";
  state.fly.thought = flyNet < 0 ? "punishment dopamine — those choices smell worse now"
    : flyNet > 0 ? "reward dopamine — those choices smell better now" : "push — nothing learned";
  state.fly.flash = flyNet < 0 ? "punish" : flyNet > 0 ? "reward" : null; state.fly.flashKey = `${round}`;
  publish();

  // -- the lesson
  fly.outcome(flyNet, flyDecisions);
  lifetime.hands++; lifetime.units += flyNet;
  const brain = { ...fly.exportBrain(), units: lifetime.units, agreement: fly.agreement(), updatedAt: now() };
  await backend?.saveBrain(brain);
  await backend?.addHand({
    ts: now(), player: myName(), you: { cards: state.you.cards.map(c => c.face + c.suit), net: youNet },
    fly: { cards: state.fly.cards.map(c => c.face + c.suit), net: flyNet,
           actions: flyDecisions.map(d => ACTION_NAMES[d[1]]) },
    dealer: state.dealer.cards.map(c => c.face + c.suit),
  });
  refreshStats(); drawChart();
  await sleep(1400);
}

async function runSeat() {
  seated = true; round = 0; score = { you: 0, fly: 0 };
  $("#join").hidden = true; $("#leave").hidden = true; $("#stand-up").hidden = false;
  roomApi?.presence({ name: myName(), role: "seat" });
  heartbeatTimer = setInterval(() => backend?.heartbeat(), HEARTBEAT_MS);
  try {
    while (seated && round < ROUNDS_PER_SEAT) await playRound();
  } catch (e) {
    if (!(e instanceof Abandoned)) throw e;
  } finally {
    clearInterval(heartbeatTimer);
    seated = false; resolveAction = null; setButtons([]);
    state = blankState("seat open"); state.round = 0; publish();
    await backend?.releaseSeat();
    $("#stand-up").hidden = true; $("#join").hidden = false;
    roomApi?.presence({ name: myName(), role: "watch" });
  }
}

// ------------------------------------------------------- seat & queue
let seatDoc = null, queue = [], inQueue = false, claiming = false, retryTimer = null;

function renderQueue() {
  const list = $("#queue"); list.replaceChildren();
  if (seatDoc?.owner) {
    const li = document.createElement("li"); li.className = "seat" + (seatDoc.owner === clientId ? " me" : "");
    li.innerHTML = `<span class="n">▶</span><span>${seatDoc.name || "Someone"}</span><span style="margin-left:auto;font-size:11px;opacity:.7">playing</span>`;
    list.append(li);
  }
  queue.forEach((q, i) => {
    const li = document.createElement("li"); li.className = q.id === clientId ? "me" : "";
    li.innerHTML = `<span class="n">${i + 1}</span><span>${q.name || "Someone"}</span>`;
    list.append(li);
  });
  $("#queue-empty").hidden = !!(seatDoc?.owner || queue.length);
  $("#queue-empty").textContent = "Seat is open.";
  inQueue = queue.some(q => q.id === clientId);
  $("#join").hidden = seated || inQueue;
  $("#leave").hidden = seated || !inQueue;
  $("#join").textContent = (seatDoc?.owner && now() - (seatDoc.heartbeat || 0) < STALE_MS) || queue.length ? "Join the queue" : "Sit down";
}

async function maybeTakeSeat() {
  if (seated || claiming || !backend) return;
  const seatFree = !seatDoc?.owner || now() - (seatDoc.heartbeat || 0) > STALE_MS;
  const myTurn = queue.length ? queue[0].id === clientId : false;
  if (!seatFree || !myTurn) { clearTimeout(retryTimer); retryTimer = null; return; }
  claiming = true;
  let got = false;
  try {
    got = await backend.claimSeat();
    if (got) { await backend.leaveQueue(); runSeat(); }
  } finally { claiming = false; }
  // the previous holder's lease may have a second or two left: try again soon
  if (!got && !retryTimer) retryTimer = setTimeout(() => { retryTimer = null; maybeTakeSeat(); }, 2000);
}

$("#join").addEventListener("click", async () => {
  if (!backend) return;
  if (backend instanceof LocalStore) { if (await backend.claimSeat()) runSeat(); return; }
  await backend.joinQueue();
  roomApi?.presence({ name: myName(), role: "queue" });
});
$("#leave").addEventListener("click", async () => { await backend?.leaveQueue(); roomApi?.presence({ name: myName(), role: "watch" }); });
$("#stand-up").addEventListener("click", () => { seated = false; resolveAction?.(STAND); });

// ----------------------------------------------------------------- boot
state = blankState("loading"); renderTable(state); drawChart(); refreshStats();

const dbApi = await (window.claude?.use?.("db") ?? Promise.resolve(null));
backend = dbApi ? new SharedStore(dbApi) : new LocalStore();
$("#conn").textContent = dbApi ? "shared" : "this tab only";
$("#conn").classList.toggle("live", !!dbApi);

let brainLoaded = false;
backend.watchBrain(b => {
  if (seated) return;                      // we are the writer; don't overwrite ourselves
  if (b && fly.importBrain(b)) { lifetime = { hands: b.hands || 0, units: b.units || 0 }; }
  brainLoaded = true; refreshStats(); drawChart();
});
backend.watchSeat(d => { seatDoc = d; renderQueue(); maybeTakeSeat(); });
backend.watchQueue(q => { queue = q; renderQueue(); maybeTakeSeat(); });
backend.watchLive(s => { if (!seated && s) renderTable(s); });
backend.watchHands(hands => {
  const feed = $("#feed"); feed.replaceChildren();
  $("#feed-empty").hidden = hands.length > 0;
  for (const h of hands) {
    const li = document.createElement("li");
    const res = h.fly.net > 0 ? "fly won" : h.fly.net < 0 ? "fly lost" : "push";
    li.innerHTML = `<span>${new Date(h.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false })}</span>` +
      `<span><b>${h.player}</b> ${h.you.net > 0 ? "+" : ""}${h.you.net} · fly ${h.fly.actions.join(",") || "natural"} ${h.fly.net > 0 ? "+" : ""}${h.fly.net} · dealer ${h.dealer.join(" ")}</span>`;
    feed.append(li);
  }
});
if (!dbApi) state = blankState("this tab only — sit down to play");
renderTable(state);

roomApi = await (window.claude?.use?.("room") ?? Promise.resolve(null));
if (roomApi) {
  roomApi.presence({ name: myName(), role: "watch" });
  roomApi.onPeers(({ peers }) => {
    const watching = peers.length;
    const names = peers.map(p => p.presence?.name).filter(Boolean).slice(0, 6).join(", ");
    $("#pres").innerHTML = `<b>${watching}</b> here now${names ? " · " + names : ""}`;
  });
  nameInput.addEventListener("change", () => roomApi.presence({ name: myName() }));
}
