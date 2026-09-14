"""Export the mushroom body for the browser: sparse weights + real soma positions."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from .model import MushroomBody, KC_SPARSITY, LEARNING_RATE, DECAY, LATERAL
from .odour import Odours, ACTIVE_GLOMERULI


def sparse(W, keep_all=True):
    r, c = np.nonzero(W)
    return dict(r=r.astype(int).tolist(), c=c.astype(int).tolist(),
                v=[round(float(x), 6) for x in W[r, c]], shape=list(W.shape))


def main(out="web/mb", odours=("geraniol", "octanol")):
    mb = MushroomBody()
    od = Odours()
    cells = json.loads(Path("data/mb_cells.json").read_text())

    def positions(key):
        pts = cells["soma"][key]
        good = [p for p in pts if p]
        if not good:
            return []
        fill = np.mean(good, axis=0).tolist()
        return [[round(v, 1) for v in (p or fill)] for p in pts]

    payload = dict(
        meta=dict(sparsity=KC_SPARSITY, learning_rate=LEARNING_RATE,
                  decay=DECAY, lateral=LATERAL, active_glomeruli=ACTIVE_GLOMERULI,
                  counts={k: len(cells["types"][k]) for k in cells["types"]}),
        w_pn_kc=sparse(mb.W_pn_kc),
        w_kc_mbon=sparse(mb.W_kc_mbon),
        w_dan_mbon=sparse(mb.W_dan_mbon),
        w_mbon_mbon=sparse(mb.W_mbon_mbon),
        mbon_valence=[int(v) for v in mb.valence],
        dan_is_punishment=[int(v) for v in mb.punish_mask],
        soma={k: positions(k) for k in ("pn", "kc", "mbon", "dan")},
        types={k: cells["types"][k] for k in ("mbon", "dan")},
        odours={name: [round(float(x), 4) for x in od.vector(name)] for name in odours},
    )
    Path(out).mkdir(parents=True, exist_ok=True)
    f = Path(out) / "mb.json"
    f.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {f} ({f.stat().st_size/1e6:.2f} MB)")
    for k in ("w_pn_kc", "w_kc_mbon", "w_dan_mbon", "w_mbon_mbon"):
        print(f"  {k:14s} {len(payload[k]['v']):7,} weights  shape {payload[k]['shape']}")
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="web/mb")
    a = p.parse_args()
    main(a.out)
