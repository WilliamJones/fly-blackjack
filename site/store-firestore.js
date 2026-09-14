// Shared state for The Fly's Table on Cloud Firestore.
//
// Same interface as SharedStore/LocalStore in app.js. The SDK functions are
// injected (`sdk`) rather than imported, so this exact file runs against the
// real Firebase SDK in the browser and against a faithful fake in the tests.
//
// Documents:
//   table/seat   { owner, name, since, heartbeat }
//   table/live   the last rendered table state (spectators watch this)
//   fly/brain    { gain, hands, n, units, agreement, updatedAt }
//   queue/<uid>  { name, joinedAt }
//   hands/<auto> one record per completed hand
//
// The seat is claimed inside a transaction: read, check it is free or stale,
// write -- so two people clicking at once cannot both sit down.

export const STALE_MS = 40000;

export function createFirestoreStore(sdk, db, uid, getName) {
  const { doc, collection, getDoc, setDoc, updateDoc, deleteDoc, addDoc,
          onSnapshot, query, orderBy, limit, runTransaction } = sdk;
  const now = () => Date.now();
  const seatRef = doc(db, "table", "seat");
  const liveRef = doc(db, "table", "live");
  const brainRef = doc(db, "fly", "brain");
  const myQueueRef = doc(db, "queue", uid);

  const watchDoc = (ref, cb) =>
    onSnapshot(ref, snap => cb(snap.exists() ? snap.data() : null), () => cb(null));

  return {
    uid,
    watchSeat: cb => watchDoc(seatRef, cb),
    watchLive: cb => watchDoc(liveRef, cb),
    watchBrain: cb => watchDoc(brainRef, cb),
    watchQueue: cb => onSnapshot(
      query(collection(db, "queue"), orderBy("joinedAt"), limit(50)),
      q => cb(q.docs.map(d => ({ id: d.id, ...d.data() }))), () => cb([])),
    watchHands: cb => onSnapshot(
      query(collection(db, "hands"), orderBy("ts", "desc"), limit(12)),
      q => cb(q.docs.map(d => d.data())), () => cb([])),

    async claimSeat() {
      try {
        return await runTransaction(db, async tx => {
          const snap = await tx.get(seatRef);
          const d = snap.exists() ? snap.data() : null;
          const free = !d || !d.owner || d.owner === uid || now() - (d.heartbeat || 0) > STALE_MS;
          if (!free) return false;
          tx.set(seatRef, { owner: uid, name: getName(), since: now(), heartbeat: now() });
          return true;
        });
      } catch (e) { console.warn("claimSeat", e); return false; }
    },
    async heartbeat() { try { await updateDoc(seatRef, { heartbeat: now() }); } catch {} },
    async releaseSeat() {
      // only give up a seat we actually hold -- a stale-seat takeover by
      // someone else must not be undone by our late release
      try {
        await runTransaction(db, async tx => {
          const snap = await tx.get(seatRef);
          if (snap.exists() && snap.data().owner === uid)
            tx.set(seatRef, { owner: null, name: null, since: null, heartbeat: now() });
        });
      } catch {}
    },
    async joinQueue() { await setDoc(myQueueRef, { name: getName(), joinedAt: now() }); },
    async leaveQueue() { try { await deleteDoc(myQueueRef); } catch {} },
    async publishLive(state) { try { await setDoc(liveRef, state); } catch {} },
    async saveBrain(brain) { try { await setDoc(brainRef, brain); } catch (e) { console.warn("saveBrain", e); } },
    async addHand(rec) { try { await addDoc(collection(db, "hands"), rec); } catch {} },
  };
}
