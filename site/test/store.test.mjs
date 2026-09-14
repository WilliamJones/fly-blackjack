// Two clients, one store. Run:  node test/store.test.mjs          (fake)
//                              node test/store.test.mjs --emulator (real Firestore emulator)
import assert from "node:assert/strict";
import { createFirestoreStore, STALE_MS } from "../store-firestore.js";

const useEmulator = process.argv.includes("--emulator");
let db, sdk, teardown = async () => {};

if (useEmulator) {
  const { initializeApp, deleteApp } = await import("firebase/app");
  const fs = await import("firebase/firestore");
  const app = initializeApp({ projectId: "demo-flytable" });
  db = fs.getFirestore(app);
  fs.connectFirestoreEmulator(db, "127.0.0.1", 8085);
  sdk = { doc: fs.doc, collection: fs.collection, getDoc: fs.getDoc, setDoc: fs.setDoc, updateDoc: fs.updateDoc,
          deleteDoc: fs.deleteDoc, addDoc: fs.addDoc, onSnapshot: fs.onSnapshot, query: fs.query,
          orderBy: fs.orderBy, limit: fs.limit, runTransaction: fs.runTransaction };
  // wipe between runs
  await fetch("http://127.0.0.1:8085/emulator/v1/projects/demo-flytable/databases/(default)/documents", { method: "DELETE" });
  teardown = () => deleteApp(app);
} else {
  const { makeFakeFirestore } = await import("./fake-firestore.js");
  ({ db, sdk } = makeFakeFirestore());
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
const alice = createFirestoreStore(sdk, db, "uid-alice", () => "Alice");
const bob   = createFirestoreStore(sdk, db, "uid-bob",   () => "Bob");
let passed = 0;
const test = async (name, fn) => { await fn(); passed++; console.log("  ok  " + name); };

await test("only one of two simultaneous claims wins", async () => {
  const [a, b] = await Promise.all([alice.claimSeat(), bob.claimSeat()]);
  assert.equal(a + b, 1, `expected exactly one winner, got alice=${a} bob=${b}`);
});

const holder = (await alice.claimSeat()) ? alice : bob;      // re-claim by owner is idempotent
const other = holder === alice ? bob : alice;
await test("the holder may re-claim; the other cannot", async () => {
  assert.equal(await holder.claimSeat(), true);
  assert.equal(await other.claimSeat(), false);
});

await test("queue orders by arrival and leaving removes you", async () => {
  await other.joinQueue(); await sleep(5); await holder.joinQueue();
  const seen = await new Promise(res => { const off = other.watchQueue(q => { if (q.length === 2) { off(); res(q); } }); });
  assert.deepEqual(seen.map(q => q.id), [other.uid, holder.uid]);
  await holder.leaveQueue();
  const after = await new Promise(res => { const off = other.watchQueue(q => { if (q.length === 1) { off(); res(q); } }); });
  assert.deepEqual(after.map(q => q.id), [other.uid]);
});

await test("release hands the seat to the next player", async () => {
  await holder.releaseSeat();
  assert.equal(await other.claimSeat(), true);
  await other.leaveQueue();
});

await test("a late release by the previous holder does not evict the new one", async () => {
  await holder.releaseSeat();                                   // holder no longer owns it
  const seat = await new Promise(res => { const off = holder.watchSeat(s => { off(); res(s); }); });
  assert.equal(seat.owner, other.uid);
});

await test("a stale seat (no heartbeat) can be taken over", async () => {
  await sdk.setDoc(sdk.doc(db, "table", "seat"), { owner: other.uid, name: "Ghost", since: 0, heartbeat: Date.now() - STALE_MS - 1000 });
  assert.equal(await holder.claimSeat(), true);
});

await test("heartbeat keeps a live seat", async () => {
  await holder.heartbeat();
  const seat = await new Promise(res => { const off = holder.watchSeat(s => { off(); res(s); }); });
  assert.ok(Date.now() - seat.heartbeat < 2000);
  assert.equal(await other.claimSeat(), false);
});

await test("brain round-trips and spectators see it", async () => {
  const brain = { gain: "QUJD".repeat(50), hands: 30000, n: 18183, units: -5358, agreement: 0.52, updatedAt: Date.now() };
  const got = new Promise(res => { const off = other.watchBrain(b => { if (b && b.hands === 30000) { off(); res(b); } }); });
  await holder.saveBrain(brain);
  assert.deepEqual(await got, brain);
});

await test("hands feed is newest-first and capped at 12", async () => {
  for (let i = 0; i < 15; i++) await holder.addHand({ ts: 1000 + i, player: "Alice", you: { net: 1 }, fly: { net: -1, actions: [] }, dealer: [] });
  const feed = await new Promise(res => { const off = holder.watchHands(h => { if (h.length === 12) { off(); res(h); } }); });
  assert.equal(feed[0].ts, 1014); assert.equal(feed[11].ts, 1003);
});

await test("live state reaches a watcher", async () => {
  const p = new Promise(res => { const off = other.watchLive(s => { if (s && s.status === "your move") { off(); res(s); } }); });
  await holder.publishLive({ status: "your move", round: 3 });
  assert.equal((await p).round, 3);
});

await teardown();
console.log(`\n${passed} passed against ${useEmulator ? "the Firestore EMULATOR" : "the in-memory fake"}`);
