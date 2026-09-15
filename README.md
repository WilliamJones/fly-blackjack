# flybrain-starter

A measured fruit-fly connectome, made to do things: drive a game, walk an anatomical body, learn by its own mushroom body, and play you at blackjack. Every result ships with the control that would have exposed it.

A small, readable kit for doing the thing you've been seeing on X: take a **real
measured fruit-fly connectome**, run it as a network, wire it to something, and
train a tiny readout to control it.

Pure Python + numpy. No GPU. The included demo trains in about **two minutes** on
a laptop CPU.

```
8 game observations
        │
        ▼
  32 visual cells (LC4, LC11, LC9, LC15, LC16, LC17, LC21, LPLC2)
        │       ← 1,296 real measured connections, 26,029 synaptic contacts
        ▼
  32 interneurons
        │
        ▼
  16 descending cells
        │
        ▼
  243-parameter readout  ← the ONLY thing that learns
        │
        ▼
  run / jump / duck
```

## Quickstart

```sh
pip install numpy
python3 -m flybrain.train --generations 40 --population 48 --seconds 30
python3 -m flybrain.play                       # watch it
python3 -m flybrain.play --silenced            # watch the control condition fail
```

## Result from the included run (seed 20260913, 50 held-out courses)

| controller | completed | mean survival | mean score |
|---|---:|---:|---:|
| connectome + trained readout | **45 / 50** | 29.12 s | 210.7 |
| same readout, circuit silenced | 0 / 50 | 3.48 s | 18.7 |
| untrained readout | 0 / 50 | 2.26 s | 11.8 |
| uniform random actions | 0 / 50 | 2.43 s | 12.7 |

The second row is the row that matters. It is the same trained readout with the
circuit's output replaced by zeros. If that row ever looks good, your connectome
is decoration and the demo proves nothing. `flybrain/train.py:benchmark()` runs
it automatically — keep it that way.

## What's real and what isn't

**Real:** the cell identities, their neurotransmitter predictions, and every
directed connection and synaptic contact count between them. That is measured
data from MaleCNS v1.0 (~166,000 neurons, ~125M synapses), released June 2026 and
described in *Cell* in September 2026.

**Engineered by me, not biology:**

- The neuron model. Cells are signed leaky `tanh` rate units settled for 3 steps
  per decision. Real neurons spike; these don't. The activity numbers are
  dimensionless — not firing rates, not membrane voltage.
- The sign rule (acetylcholine `+1`, GABA/glutamate `−1`, unknown `0`).
- The mapping from game features onto cell types. LC4 is a looming detector in a
  real fly; here it just receives "channel 0". That is an arbitrary wiring
  choice with no biological claim behind it.
- Which 80 of the 166,000 cells to include, and the hard boundary around them.

**What the demo shows:** a trainable readout can learn a task when its only
input is activity computed through measured fly wiring, and it fails completely
without that activity. **What it does not show:** that the fly's topology beats
an equally sized random or artificial network. Testing that needs rewired and
artificial controls trained on an equal budget — worth building, and the honest
next experiment if you want one.

Most viral posts about this skip straight past all of the above. Don't.

## Files

| file | what it does |
|---|---|
| `flybrain/circuit.py` | loads the graph, builds the signed normalised weight matrix, runs the dynamics |
| `flybrain/envs.py` | the dodge game. **Replace this file to change what the fly controls.** |
| `flybrain/readout.py` | 16 → 12 tanh → 3 MLP, 243 parameters |
| `flybrain/train.py` | cross-entropy method training + held-out benchmark with controls |
| `flybrain/play.py` | terminal viewer with live cell activity |
| `build_your_own_circuit.py` | cut a **different** circuit from the full connectome — read this one |
| `fetch_circuit.py` | re-download the bundled circuit |
| `data/circuit.json` | the 80-cell graph (22 KB) |

## Making it yours

**Easiest — change the environment.** Anything with the interface in
`envs.py` works: `reset(seed) -> obs`, `step(action) -> obs, reward, done`, plus
`N_OBS` and `N_ACTIONS`. A Gymnasium env wraps in about ten lines. If you change
`N_OBS`, you need that many input cell types in the circuit — see below.

**Better — change the brain.** `build_your_own_circuit.py` cuts a new subgraph
from the raw MaleCNS tables: you choose the input cell types (olfactory instead
of visual, say), how many cells per type, and how big the circuit is. This is
the step that makes the project original rather than a fork. It needs a one-time
1.2 GB download; the script has the exact `curl` commands at the top.

**Ambitious — go bigger.** 80 cells runs fine in numpy. Whole-brain (~130k cells)
needs sparse matrices and probably a GPU; `eonfathom/FastFly` (CUDA) and
`eonsystemspbc/fly-brain` are the references. For a physical body instead of a
2D game, `TuragaLab/flybody` (MuJoCo) and `NeLy-EPFL/flygym` are the real ones.

## Where to look next

- [awesome-fly](https://github.com/cobanov/awesome-fly) — ~60 projects in this genre
- [MaleCNS](https://male-cns.janelia.org/) — the dataset and its downloads
- [neuPrint](https://neuprint.janelia.org) / [Codex](https://codex.flywire.ai) — browse cell types to pick your inputs
- [Fly Dino](https://github.com/cobanov/flyjump) — the project this kit's circuit and method come from, and the clearest write-up of an honest experiment in the genre


---

# Part two: the fly with a body

The Dino demo above is the 20-minute version. This part puts the same 80-cell
circuit in charge of **flybody** — the anatomically detailed *Drosophila* model
from DeepMind and HHMI Janelia — and teaches it four behaviours.

```sh
pip install numpy mujoco fast-simplification
git clone https://github.com/TuragaLab/flybody          # the body model

python3 -m flybrain.body.train_body --task all          # ~17 min, all four
python3 build_web.py --flybody flybody/flybody/fruitfly/assets --clips
python3 -m http.server -d web 8000                      # then open localhost:8000
```

## Results (40 held-out seeds per task)

| behaviour | connectome + trained readout | circuit silenced | untrained readout | random |
|---|---:|---:|---:|---:|
| Navigate to targets | **203** | -166 | -95 | -79 |
| Escape the swatter | **128** | 14 | 17 | 18 |
| Track an odour plume | **328** | 103 | 177 | 115 |
| Pursue another fly | **1485** | 499 | 432 | 485 |

Every task separates cleanly from its silenced-circuit control, which is the
only column that makes the first one mean anything.

## The four behaviours

All four share one 8-channel sensor interface, so the **same circuit drives all
of them** without being rebuilt — only the 243-parameter readout is retrained.

- **Navigate** — walk to a goal, around obstacles.
- **Escape** — a looming swatter descends. This is the one where the cell types
  mean what they say: channels 3–5 feed LC4 and LPLC2, genuinely
  looming-sensitive escape cells in a living fly.
- **Track odour** — cast up a wind-borne plume to its source, the way real flies
  surge upwind on contact and cast crosswind when they lose it.
- **Pursue** — chase another fly that keeps trying to leave.

## What actually walks

The gait is a hand-written tripod central pattern generator, and it is **not**
from the connectome. Real insects generate walking rhythm in the ventral nerve
cord; descending neurons from the brain modulate speed and turning. That is the
split modelled here — the CPG produces the rhythm, the 16 descending cells steer
it. Claiming the connectome learned to walk would be false.

The legs are not animated by hand either. `flybrain/body/legs.py` perturbs each
joint in the real model, measures where the foot goes, and builds a damped
pseudo-inverse — so a step is solved against the actual anatomy, landing feet
within 5.5% of a body length of target.

**It walks under real physics.** `flybrain/body/mujoco_clip.py` closes the loop
through MuJoCo — body pose → sensors → circuit → readout → CPG → leg IK →
position actuators → physics — and the fly covers 5 to 8.5 body lengths per clip
on ground reaction forces alone. Nothing teleports it.

## Browser version

`web/` is a self-contained Three.js page: the live simulation, the anatomical
body posed by MuJoCo's own kinematics, the recorded physics clips, and the 80
cells rendered at their **real soma coordinates**, lighting up as the fly works.

The JavaScript in `web/sim.js` is a port of the Python, and the port is verified
rather than assumed — circuit output, readout decisions, gait and leg IK all
match Python to the last decimal, and `flybrain/body/check_kinematics.py`
confirms the browser's forward kinematics reproduce `mj_kinematics` to 1e-16.

## Files added in part two

| file | what it does |
|---|---|
| `flybrain/body/gait.py` | tripod CPG and the planar walker used for training |
| `flybrain/body/legs.py` | measured leg Jacobians and damped-inverse IK |
| `flybrain/body/arenas.py` | the four behaviours, one 8-channel sensor interface |
| `flybrain/body/train_body.py` | CEM training and the held-out benchmark |
| `flybrain/body/export_body.py` | 151 MB of meshes down to a browser asset |
| `flybrain/body/mujoco_clip.py` | closed-loop physics recording |
| `flybrain/body/check_kinematics.py` | proves the browser body matches MuJoCo |
| `build_web.py` | builds everything `web/` needs |


---

# Part three: the fly that learns

Parts one and two never learn anything — the wiring is fixed and CEM evolves a
readout across generations. This part uses the fly's actual learning organ.

The **mushroom body** is where *Drosophila* forms associative memories. Kenyon
cells encode odour identity sparsely, dopaminergic neurons signal reward or
punishment, and coincidence of the two **depresses** the Kenyon-to-output
synapse. All of those cells are in MaleCNS with measured connectivity, so the
mechanism can be implemented as the animal actually runs it.

```sh
pip install numpy scipy pandas
python3 -m flybrain.mb.fetch
python3 -m flybrain.mb.extract
python3 -m flybrain.mb.experiment
python3 -m flybrain.mb.tmaze
python3 -m flybrain.mb.export_web && python3 -m http.server -d web/mb 8001
```

**You do not need the 1.2 GB Google Storage download.** `fetch.py` pulls the same
data as preprocessed matrices from
[connectome_data_prep](https://github.com/YijieYin/connectome_data_prep) — 107 MB —
and asserts every count in the table below on load, so silent drift is impossible.

## The circuit

| population | cells | role |
|---|---:|---|
| ALPN | 682 | antennal lobe projection neurons — odour input |
| Kenyon cells | 1,200 of 3,811 on the pathway | sparse odour identity code |
| MBON | 97 | output neurons that bias behaviour |
| PAM + PPL1 | 332 | reward and punishment dopamine |

22,586 ALPN→KC connections, 61,210 KC→MBON, 3,123 DAN→MBON — all measured. MBON
valence is derived from the connectome rather than a lookup table: an MBON's
dominant dopaminergic input identifies which side of the approach/avoid axis it
sits on, which splits them 56/41.

## Results (12 seeds, 20 trials)

| condition | learning index | T-maze shift |
|---|---:|---:|
| connectome + plasticity | **−1.30** | **+0.80** |
| plasticity off | 0.00 | +0.00 |
| dopamine disconnected | 0.00 | +0.00 |
| unpaired odour and shock | 0.00 | +0.00 |
| KC→MBON shuffled | −1.23 | +0.83 |

Naive flies are near indifferent (preference index +0.15); trained flies avoid
the punished odour (+0.95). Extinction recovers about a third of the memory over
20 unreinforced trials, and reversal flips the association.

**The last row is a real finding, not a failure.** Shuffling the Kenyon-to-output
wiring barely hurts — and that agrees with the biology. This connectivity is
thought to be largely random in real flies; the learning rule needs *a* sparse
code, not a *particular* one. What the circuit contributes is the dopamine
pathway: which output neurons a punishment signal can reach is anatomy, and
cutting it abolishes learning completely.

## Reproducibility

Cell selection is byte-identical on any machine: most Kenyon cells tie on the
ranking score, so the sort is explicitly stable -- an unstable sort silently
hands different numpy versions a different subset. `python3 -m flybrain.mb.extract`
twice should give you the same md5 for `data/mb_circuit.npz`.

## Honesty

**Measured:** every connection, cell identity, neurotransmitter and soma
coordinate. **Modelled:** rate units rather than spikes; the learning rate, decay
and floor; k-winners-take-all standing in for APL feedback inhibition; which
glomeruli each odour drives. The decay term produces *forgetting* — real
extinction in flies is an active dopaminergic process and is not modelled here.
We did not verify the mapping between MaleCNS MBON numbering and individually
characterised MBONs in the literature; the result does not depend on it.

Reproducing the shape of a published curve is not proof the model is correct.
This is a connectome-constrained model of mushroom body learning — which is both
accurate and more interesting than "the fly learned".

| file | what it does |
|---|---|
| `flybrain/mb/fetch.py` | download and verify the connectome |
| `flybrain/mb/extract.py` | cut the mushroom body, derive MBON valence |
| `flybrain/mb/odour.py` | odours as glomerular activity patterns |
| `flybrain/mb/model.py` | sparse coding and dopamine-gated depression |
| `flybrain/mb/experiment.py` | conditioning, extinction, reversal, controls |
| `flybrain/mb/tmaze.py` | preference-index assay |
| `flybrain/mb/export_web.py` | build `web/mb/` |


---

# Part four: learning and walking, joined

Parts two and three never met. The body walked but never learned; the mushroom
body learned but never moved. This part connects them, and the result is a fly
that **works out which of two smells is dangerous and then walks away from it**.

```sh
python3 -c "from flybrain.mb.embodied import train; train(generations=32, quiet=False)"
python3 -c "
import json, numpy as np
from flybrain.mb.embodied import evaluate
t = np.array(json.load(open('policies/forage.json'))['theta'])
for label, kw in [('learning', {}), ('dopamine cut', dict(dopamine=False))]:
    thirds, food = evaluate(t, seeds=40, seconds=90.0, **kw)
    print(label, 'shocks', thirds.sum(axis=1).mean(), 'food', food.mean())
"
```

## Two kinds of learning, on two timescales

The **mushroom body** learns inside the episode, by the animal's own mechanism.
Nothing about it is trained; it happens while you watch. The **steering readout**
is shaped across generations by CEM before the episode starts, and never learns
what any odour means — only what to do about a valence. That split is the point:
evolution builds the machinery that acts on valence, experience fills in which
smell carries which.

The steering circuit's only access to odour identity is channels 3 and 4,
which carry the mushroom body's verdict. So if the fly avoids the shocked
source, the mushroom body is the only thing that could have told it to.

## Result (40 seeds, 90 s each)

| condition | shocks per 90 s | food per 90 s |
|---|---:|---:|
| connectome + learning | **3.85** | 5.47 |
| dopamine disconnected | 7.50 | 5.25 |
| plasticity off | 7.50 | 5.25 |

**49% fewer shocks** (+3.65 ± 1.14) with food unchanged — so it is not avoiding
odour in general, it is avoiding one specific smell. Cut the dopamine and the
benefit vanishes entirely while foraging continues, which is the shape a real
result should have. The browser port reproduces it independently at 47%.

## Three bugs this took to find

Worth recording, because each one produced a plausible-looking fly that was
learning nothing.

1. **The reward leaked the answer.** Rewarding approach to the safe plume and
   penalising the dangerous one let the steering circuit avoid shocks without
   ever consulting the mushroom body. The shaping term is now total odour in the
   air, identical for both sources.
2. **Food was never paid out.** `_advance()` moved the source before `_reward()`
   ran, so the fly was standing somewhere else by the time the reward was
   computed — and the sudden drop in odour was then charged as a penalty. The
   fly was being punished for finding food, and CEM correctly learned to do
   nothing.
3. **Overlapping plumes taught the wrong lesson.** A quarter of the time the two
   sources spawned within one plume-width, so the fly only ever smelled a blend
   and learned "all odour is dangerous". With a minimum separation enforced,
   discrimination is clean: −1.13 for the shocked odour against −0.04 for the
   safe one.

| file | what it does |
|---|---|
| `flybrain/mb/embodied.py` | the foraging arena, the join, training and evaluation |
| `web/mb.js`, `web/mb.json` | the mushroom body in the browser |


---

# Part five: the table

**Live at [fly-blackjack.vercel.app](https://fly-blackjack.vercel.app/).**

`site/` is a live blackjack table against the fly: one seat, a queue,
spectators watching the hand and the fly's reasoning, and a brain that
persists between players so every hand anyone plays is one more conditioning
trial. The fly sits at the table in 3D with its mushroom body above it, lighting
up as it weighs each option. The chart panel shows the fly's current choice in
all 260 spots against basic strategy, and changes as people play.

The player is a `MushroomBody` from part three with no readout at all: each
option (hit, stand, double) is imagined as an odour, the mushroom body's output
neurons vote on it, and the best-smelling option is taken. Winning fires the
reward dopamine neurons, losing fires the punishment ones, and the synapses that
carried that situation are depressed. The dealer is the environment; nothing
else is trained.

## Result (30,000 solo hands, then the same brain graded)

| condition | winrate | agreement with basic strategy |
|---|---:|---:|
| untrained | -45% | -- |
| dopamine disconnected | -46% | -- |
| connectome + dopamine | **-16%** | 52% of 260 spots |

It learned **don't bust** -- it stands on nearly everything and hits only on 10
and 11, and knows nothing about doubling. That is the first thing a person
learns at a blackjack table, and it is not basic strategy. The table opens with
that 30,000-hand brain (`site/brain_seed.json`) and every hand played on it is
added to it. Poker was tried first and did not work; `flybrain/mb/poker.py`
records why.

## Hosting

The page is a plain static site with one small adapter for shared state.
`site/store-firestore.js` talks to Cloud Firestore; anonymous sign-in
identifies players, `firestore.rules` lets only the seat holder write the live
hand, the brain and the hand records, and a transaction guarantees one seat
holder at a time. Without a Firebase config the same page runs as a
single-player table with the brain kept in the browser. `site/README.md` has
the deploy steps and `site/test/` the store tests, which run against an
in-memory fake or the Firestore emulator.

`web/table/` is the earlier version of the same page built on claude.ai's
artifact runtime; it is organisation-internal and kept for reference.

| file | what it does |
|---|---|
| `flybrain/mb/blackjack.py` | the game, the fly player, basic strategy and the grading |
| `site/bj.js`, `site/mb.js` | the same, in the browser |
| `site/fly3d.js`, `site/gestures.json` | the fly at the table and its idle gestures |
| `site/store-firestore.js`, `site/firestore.rules` | shared state and who may write it |

---

Attribution and licences: see `NOTICE.md`. The connectome is CC BY 4.0 — cite it
if you publish anything.
