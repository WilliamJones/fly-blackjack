"""Cut the mushroom body out of MaleCNS.

Populations are selected by CELL TYPE ONLY, never by how well they perform --
the same discipline as build_your_own_circuit.py. Every measured edge among the
selected cells is retained.

  ALPN  682   antennal lobe projection neurons: odour input
  KC   ~3.8k  Kenyon cells: sparse odour identity code
  MBON   97   mushroom body output neurons: the read-out that biases behaviour
  DAN   332   dopaminergic neurons, PAM (316) + PPL1 (16): reward and punishment

KC->KC edges are NOT retained. The matrix lists ~643k of them, almost all
single-contact, and the real inhibition that sparsifies Kenyon cell activity
comes from the APL neuron, not from KC-to-KC excitation. model.py imposes that
sparsity directly and says so.

MBON valence is derived from the connectome, not from a lookup table. In the
compartmental model of the mushroom body, punishment (PPL1) dopamine depresses
KC synapses onto approach-driving MBONs, and reward (PAM) dopamine depresses
those onto avoidance-driving MBONs. So an MBON's dominant dopaminergic input
identifies which side of that axis it sits on. See the note in model.py about
what this does and does not claim.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from .fetch import load


def parse_soma(s):
    if not isinstance(s, str):
        return None
    try:
        return [float(v) for v in s.strip("[] ").split()][:3]
    except ValueError:
        return None


def extract(kenyon: int | None = 1200, seed: int = 20260913):
    M, meta = load()
    t = meta["type"].fillna("")
    cls = meta["class"].fillna("")
    where = lambda pat: meta.index[t.str.match(pat, na=False)].to_numpy()

    PN = meta.index[cls == "ALPN"].to_numpy()
    KC_all = where(r"^KC")
    MBON = where(r"^MBON")
    PAM, PPL1 = where(r"^PAM"), where(r"^PPL1")

    # keep only Kenyon cells that actually sit on the pathway
    pn_in = np.asarray(M[PN][:, KC_all].sum(axis=0)).ravel()
    mbon_out = np.asarray(M[KC_all][:, MBON].sum(axis=1)).ravel()
    live = (pn_in > 0) & (mbon_out > 0)
    KC = KC_all[live]
    if kenyon and kenyon < len(KC):
        # Rank by total pathway involvement, then take an evenly spaced sample so
        # the subset spans the range rather than only the most connected cells.
        # kind="stable" matters: most Kenyon cells tie on this score, and an
        # unstable sort orders ties differently between numpy versions, which
        # silently gives different machines a different subset of cells.
        order = np.argsort(-(pn_in[live] + mbon_out[live]), kind="stable")
        KC = KC[np.sort(order[np.linspace(0, len(order) - 1, kenyon).astype(int)])]

    DAN = np.concatenate([PAM, PPL1])
    blocks = {
        "pn_kc": M[PN][:, KC],
        "kc_mbon": M[KC][:, MBON],
        "dan_mbon": M[DAN][:, MBON],
        "mbon_mbon": M[MBON][:, MBON],
    }

    pam_drive = np.asarray(M[PAM][:, MBON].sum(axis=0)).ravel()
    ppl_drive = np.asarray(M[PPL1][:, MBON].sum(axis=0)).ravel()
    # +1 approach-driving (punishment compartment), -1 avoidance-driving (reward)
    valence = np.where(ppl_drive > pam_drive, 1, -1)
    valence[(pam_drive == 0) & (ppl_drive == 0)] = 0

    sign_of = lambda ids: meta.loc[ids, "sign"].fillna(0).to_numpy(dtype=float)
    out = dict(
        populations={k: [int(x) for x in v] for k, v in
                     dict(pn=PN, kc=KC, mbon=MBON, dan=DAN).items()},
        types={k: [str(x) for x in meta.loc[v, "type"].fillna("?")] for k, v in
               dict(pn=PN, kc=KC, mbon=MBON, dan=DAN).items()},
        soma={k: [parse_soma(s) for s in meta.loc[v, "somaLocation"]] for k, v in
              dict(pn=PN, kc=KC, mbon=MBON, dan=DAN).items()},
        nt={k: [str(x) for x in meta.loc[v, "top_nt"].fillna("unknown")] for k, v in
            dict(pn=PN, kc=KC, mbon=MBON, dan=DAN).items()},
        signs={k: sign_of(v).tolist() for k, v in
               dict(pn=PN, kc=KC, mbon=MBON, dan=DAN).items()},
        mbon_valence=valence.tolist(),
        dan_is_punishment=([0] * len(PAM) + [1] * len(PPL1)),
    )
    return blocks, out


def save(blocks, info, path="data/mb_circuit.npz", manifest="data/mb_manifest.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, block in blocks.items():
        coo = block.tocoo()
        arrays[f"{name}_row"] = coo.row.astype(np.int32)
        arrays[f"{name}_col"] = coo.col.astype(np.int32)
        arrays[f"{name}_val"] = coo.data.astype(np.float32)
        arrays[f"{name}_shape"] = np.array(block.shape, np.int32)
    arrays["mbon_valence"] = np.array(info["mbon_valence"], np.int8)
    arrays["dan_is_punishment"] = np.array(info["dan_is_punishment"], np.int8)
    np.savez_compressed(path, **arrays)

    counts = {k: len(v) for k, v in info["populations"].items()}
    man = dict(
        dataset="MaleCNS v1.0 (Berg et al. 2025 / Nern et al. 2024), CC BY 4.0",
        mirror="github.com/YijieYin/connectome_data_prep",
        counts=counts,
        edges={k: int(b.nnz) for k, b in blocks.items()},
        synapses={k: int(b.sum()) for k, b in blocks.items()},
        mbon_approach=int(sum(1 for v in info["mbon_valence"] if v > 0)),
        mbon_avoid=int(sum(1 for v in info["mbon_valence"] if v < 0)),
        selection="ALPN by class; KCs with both ALPN input and MBON output, "
                  "evenly sampled across pathway-involvement rank; all MBONs; "
                  "all PAM+PPL1. KC->KC excluded (see extract.py). "
                  "No behavioural measure used at any point.",
    )
    Path(manifest).write_text(json.dumps(man, indent=2))
    Path("data/mb_cells.json").write_text(json.dumps(
        {k: info[k] for k in ("types", "soma", "nt", "mbon_valence", "dan_is_punishment")},
        separators=(",", ":")))
    return man


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kenyon", default="1200", help="number of Kenyon cells, or 'all'")
    a = p.parse_args()
    n = None if a.kenyon == "all" else int(a.kenyon)
    blocks, info = extract(kenyon=n)
    man = save(blocks, info)
    print(json.dumps(man, indent=2))
