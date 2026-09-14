// Browser bootstrap for Firestore: loads the SDK from Google's CDN, signs the
// visitor in anonymously, and hands app.js a ready store. Loaded only when
// firebase-config.js provides a config, so the page still works without it.
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.6.1/firebase-app.js";
import { getAuth, signInAnonymously, onAuthStateChanged }
  from "https://www.gstatic.com/firebasejs/11.6.1/firebase-auth.js";
import * as fs from "https://www.gstatic.com/firebasejs/11.6.1/firebase-firestore.js";
import { createFirestoreStore } from "./store-firestore.js";

export async function connectFirestore(config, getName) {
  const app = initializeApp(config);
  const auth = getAuth(app);
  const user = await new Promise((resolve, reject) => {
    onAuthStateChanged(auth, u => u && resolve(u), reject);
    signInAnonymously(auth).catch(reject);
  });
  const db = fs.getFirestore(app);
  const sdk = {
    doc: fs.doc, collection: fs.collection, getDoc: fs.getDoc, setDoc: fs.setDoc,
    updateDoc: fs.updateDoc, deleteDoc: fs.deleteDoc, addDoc: fs.addDoc,
    onSnapshot: fs.onSnapshot, query: fs.query, orderBy: fs.orderBy, limit: fs.limit,
    runTransaction: fs.runTransaction,
  };
  return createFirestoreStore(sdk, db, user.uid, getName);
}
