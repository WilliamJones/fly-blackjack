# The Fly's Table — deploy

A static site (Vercel) plus Cloud Firestore for the shared fly. Without a
Firestore config the page still runs as a single-player table.

## 1. Firebase (once, ~10 minutes)

1. Create a project at console.firebase.google.com.
2. **Authentication → Sign-in method → Anonymous → Enable.** Players are
   identified by an anonymous sign-in; nobody types a password.
3. **Firestore Database → Create database** (production mode, any region).
4. Deploy the security rules from this folder:
   ```sh
   npm install -g firebase-tools
   firebase login
   firebase use --add            # pick the project
   firebase deploy --only firestore:rules
   ```
   The rules let anyone signed in read the table, let only the seat holder
   write the live state, the brain and hand records, and let each person add
   or remove only themselves from the queue.
5. **Project settings → Your apps → Web app → SDK setup**: copy the config
   object into `firebase-config.js`. These values are public by design;
   `firestore.rules` is what controls access.

## 2. Vercel

```sh
npm install -g vercel
vercel                            # from this folder; accept the defaults
```
It is a plain static site: no build step, no server. Every later `vercel --prod`
(or a push, if you connect the repo) redeploys.

## Testing the shared store

```sh
cd test && npm install
npm test                          # 10 tests against an in-memory fake
npm run test:emulator             # the same 10 against the real Firestore emulator
```
The emulator run needs Java 11+ and downloads the emulator on first use. The
tests cover the things that bite in production: two simultaneous seat claims
with exactly one winner, handoff on release, stale-seat takeover, a late release
not evicting the new holder, queue ordering, and the brain round-trip.

## What starts in the store

Nothing. The first time the page finds no `fly/brain` document it begins from
`brain_seed.json` — the fly that played 30,000 hands alone (−18%, 52% chart
agreement) — and the first hand dealt at the table writes it to Firestore.
To reset the fly, delete `fly/brain` in the Firebase console.

## Cost

Firestore's free tier is 50,000 document reads and 20,000 writes a day. A hand
is roughly 20 writes plus one read per write per spectator. A table with a
handful of watchers and a hundred hands a day stays free; it starts to cost at
the point where the table is genuinely popular. If that day comes, the live
table state alone can move to the Realtime Database, which bills by bandwidth.
