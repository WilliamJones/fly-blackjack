"""Cut YOUR OWN circuit out of the full MaleCNS connectome.

This is the step that makes the project yours rather than a fork. You choose
which cell types feed the network and how big it is; the wiring you get back is
measured, not invented.

--- Step 1: download the raw tables (about 1.2 GB, once) ---------------------

    mkdir -p ~/malecns
    base=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
    curl -fL "$base/body-annotations-male-cns-v1.0-minconf-0.5.feather"  -o ~/malecns/annotations.feather
    curl -fL "$base/connectome-weights-male-cns-v1.0-minconf-0.5.feather" -o ~/malecns/edges.feather
    curl -fL "$base/body-neurotransmitters-male-cns-v1.0.feather"        -o ~/malecns/neurotransmitters.feather

--- Step 2: pick your input cell types ---------------------------------------

Browse cell types at https://neuprint.janelia.org or https://codex.flywire.ai .
Some starting points:
    vision (looming/motion) : LC4 LC11 LC9 LC15 LC16 LC17 LC21 LPLC2
    smell (olfactory PNs)   : DA1_lPN DL5_adPN DM1_lPN DC1_adPN
    wind / mechanosensory   : WED PN types, JO-B
    taste                   : Gr64f-related, MN types
You need exactly as many types as your environment has observation channels.

--- Step 3: run ---------------------------------------------------------------

    pip install pyarrow numpy
    python3 build_your_own_circuit.py ~/malecns \
        --types LC4 LC11 LC9 LC15 LC16 LC17 LC21 LPLC2 \
        --per-type 4 --readout 16 --bridges 32 \
        --out data/my_circuit.json

Then point the kit at it:  Circuit(json.load(open("data/my_circuit.json")))

Selection uses ANATOMY ONLY and never looks at task performance. Keep it that
way -- if you pick cells because they scored well, you have quietly trained the
topology and the "measured circuit" claim stops being true.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

SIGNS = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="folder holding the three .feather files")
    ap.add_argument("--types", nargs="+", required=True,
                    help="one input cell type per observation channel")
    ap.add_argument("--per-type", type=int, default=4)
    ap.add_argument("--readout", type=int, default=16)
    ap.add_argument("--bridges", type=int, default=32)
    ap.add_argument("--out", default="data/my_circuit.json")
    a = ap.parse_args()

    import pyarrow.feather as feather
    root = Path(a.data_dir).expanduser()

    rows = feather.read_table(root / "annotations.feather",
                              columns=["bodyId", "type", "superclass", "somaLocation"]).to_pylist()
    ann = {r["bodyId"]: r for r in rows if r["somaLocation"]}
    e = feather.read_table(root / "edges.feather")
    pre, post, weight = e["body_pre"].to_numpy(), e["body_post"].to_numpy(), e["weight"].to_numpy()
    nt = {r["body"]: r["consensus_nt"] for r in feather.read_table(
        root / "neurotransmitters.feather", columns=["body", "consensus_nt"]).to_pylist()}

    # descending neurons carry brain output toward the nerve cord -> our readout
    dn = np.array([i for i, r in ann.items() if r["superclass"] == "descending_neuron"])
    to_dn = np.isin(post, dn)

    inputs, targets = [], []
    for channel, typ in enumerate(a.types):
        ids = np.array([i for i, r in ann.items() if r["type"] == typ])
        if ids.size == 0:
            raise SystemExit(f"no cells of type {typ!r} found - check spelling on neuPrint")
        ix = np.flatnonzero(np.isin(pre, ids) & to_dn)
        strength = {int(i): int(weight[ix[pre[ix] == i]].sum()) for i in np.unique(pre[ix])}
        chosen = sorted(strength, key=lambda i: (-strength[i], i))[:a.per_type]
        inputs.extend((i, channel) for i in chosen)
        jx = ix[np.isin(pre[ix], chosen)]
        ranks = {int(i): int(weight[jx[post[jx] == i]].sum()) for i in np.unique(post[jx])}
        targets.extend(sorted(ranks, key=lambda i: (-ranks[i], i))[:2])

    sel_in = [i for i, _ in inputs]
    ix = np.flatnonzero(np.isin(pre, sel_in) & to_dn)
    strength = {int(i): int(weight[ix[post[ix] == i]].sum()) for i in np.unique(post[ix])}
    targets = list(dict.fromkeys(targets))
    for i in sorted(strength, key=lambda i: (-strength[i], i)):
        if len(targets) >= a.readout:
            break
        if i not in targets:
            targets.append(i)

    # two-hop bridge cells: strong both from our inputs and onto our readout
    im, om = np.isin(pre, sel_in), np.isin(post, targets)
    u, inv = np.unique(post[im], return_inverse=True)
    incoming = dict(zip(u, np.bincount(inv, weights=weight[im])))
    u, inv = np.unique(pre[om], return_inverse=True)
    outgoing = dict(zip(u, np.bincount(inv, weights=weight[om])))
    bridges = sorted((int(i) for i in incoming.keys() & outgoing.keys()
                      if i in ann and i not in sel_in + targets),
                     key=lambda i: (-min(incoming[i], outgoing[i]), i))[:a.bridges]

    ids = sorted(set(sel_in + targets + bridges))
    idx = {i: j for j, i in enumerate(ids)}
    mask = np.isin(pre, ids) & np.isin(post, ids)
    edges = sorted([[idx[int(x)], idx[int(y)], int(w)]
                    for x, y, w in zip(pre[mask], post[mask], weight[mask])])

    graph = dict(
        version="malecns-custom-v1",
        nodes=[dict(id=i, type=ann[i]["type"], position=ann[i]["somaLocation"],
                    nt=nt.get(i), sign=SIGNS.get(nt.get(i), 0),
                    role="input" if i in sel_in else "output" if i in targets else "interneuron")
               for i in ids],
        edges=edges,
        inputs=[[idx[i], c] for i, c in inputs],
        outputs=[idx[i] for i in targets],
        channels=list(a.types),
    )
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph, separators=(",", ":")) + "\n")
    print(f"wrote {out}: {len(ids)} cells, {len(edges)} edges, "
          f"{sum(x[2] for x in edges)} synaptic contacts")
    print("Source: FlyEM / HHMI Janelia MaleCNS v1.0 (CC BY 4.0) - cite it.")


if __name__ == "__main__":
    main()
