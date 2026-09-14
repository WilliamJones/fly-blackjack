// A small in-memory stand-in for the Firebase modular Firestore API subset the
// adapter uses. Semantics that matter here are kept honest: snapshot listeners
// fire on every change, queries order and limit, transactions are serialised
// and commit their writes atomically. Used because the real emulator's engine
// cannot be downloaded in the authoring sandbox; run `npm run test:emulator`
// on a machine that can to check the same tests against the real thing.
export function makeFakeFirestore() {
  const docs = new Map();                 // "col/id" -> data
  const listeners = new Set();
  const notify = () => { for (const fn of [...listeners]) fn(); };
  let chain = Promise.resolve();          // transaction mutex

  const snapOf = (path) => {
    const data = docs.get(path);
    return { id: path.split("/").pop(), exists: () => data !== undefined,
             data: () => data === undefined ? undefined : structuredClone(data) };
  };
  const runQuery = (q) => {
    let rows = [...docs.keys()].filter(k => k.startsWith(q.col + "/") && k.split("/").length === 2)
      .map(k => ({ id: k.split("/").pop(), data: () => structuredClone(docs.get(k)), exists: () => true }));
    for (const [field, dir] of q.order)
      rows.sort((a, b) => (a.data()[field] > b.data()[field] ? 1 : a.data()[field] < b.data()[field] ? -1 : 0) * (dir === "desc" ? -1 : 1));
    if (q.lim) rows = rows.slice(0, q.lim);
    return { docs: rows, size: rows.length, empty: rows.length === 0 };
  };

  const db = { fake: true };
  const sdk = {
    doc: (db, col, id) => ({ kind: "doc", path: `${col}/${id}`, id }),
    collection: (db, col) => ({ kind: "col", col }),
    query: (colref, ...cs) => cs.reduce((q, c) => c(q), { kind: "query", col: colref.col, order: [], lim: 0 }),
    orderBy: (f, dir = "asc") => q => ({ ...q, order: [...q.order, [f, dir]] }),
    limit: n => q => ({ ...q, lim: n }),
    getDoc: async ref => snapOf(ref.path),
    setDoc: async (ref, data) => { docs.set(ref.path, structuredClone(data)); notify(); },
    updateDoc: async (ref, data) => {
      if (!docs.has(ref.path)) throw new Error("not-found");
      docs.set(ref.path, { ...docs.get(ref.path), ...structuredClone(data) }); notify();
    },
    deleteDoc: async ref => { docs.delete(ref.path); notify(); },
    addDoc: async (colref, data) => {
      const id = Math.random().toString(36).slice(2, 10);
      docs.set(`${colref.col}/${id}`, structuredClone(data)); notify();
      return { path: `${colref.col}/${id}`, id };
    },
    onSnapshot: (target, next, onError) => {
      const fire = () => { try { next(target.kind === "doc" ? snapOf(target.path) : runQuery(target)); } catch (e) { onError?.(e); } };
      listeners.add(fire); queueMicrotask(fire);
      return () => listeners.delete(fire);
    },
    runTransaction: (db, fn) => {
      const run = async () => {
        const writes = [];
        const tx = {
          get: async ref => snapOf(ref.path),
          set: (ref, data) => writes.push(["set", ref.path, structuredClone(data)]),
          update: (ref, data) => writes.push(["update", ref.path, structuredClone(data)]),
          delete: ref => writes.push(["delete", ref.path]),
        };
        const result = await fn(tx);
        for (const [op, path, data] of writes) {
          if (op === "set") docs.set(path, data);
          else if (op === "update") docs.set(path, { ...(docs.get(path) || {}), ...data });
          else docs.delete(path);
        }
        if (writes.length) notify();
        return result;
      };
      const p = chain.then(run, run);
      chain = p.catch(() => {});
      return p;
    },
  };
  return { db, sdk, _docs: docs };
}
