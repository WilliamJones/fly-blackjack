"""The T-maze assay: give a fly a choice between two odours and count.

This is the shape of the classic Drosophila olfactory conditioning experiment
(Quinn, Harris & Benzer 1974; Tully & Quinn 1985). A population of flies is
trained with one odour paired with shock, then released at a choice point
between the two odours. The preference index is

    PI = (flies choosing the safe odour - flies choosing the trained odour)
         / total flies

0 means indifferent, 1 means every fly avoided the punished odour.

We do NOT claim a numerical match to any published figure. Reported PIs depend
on training cycles, shock voltage and intensity, timing, and the odour pair, and
we have not matched any specific paradigm. What is comparable is the qualitative
result: naive flies are near indifferent, trained flies avoid, and the effect
disappears in every control.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from .experiment import acquisition, build, ODOURS
from .odour import Odours

CHOICE_SHARPNESS = 2.5      # how decisively a valence difference drives choice


def preference_index(mb, A, B, flies=100, rng=None, sharpness=CHOICE_SHARPNESS):
    rng = rng or np.random.default_rng(0)
    va, vb = mb.valence_of(A), mb.valence_of(B)
    p_a = 1.0 / (1.0 + np.exp(-sharpness * (va - vb)))     # chance of the trained arm
    chose_a = rng.random(flies) < p_a
    return float((np.sum(~chose_a) - np.sum(chose_a)) / flies)


def assay(seeds=12, trials=20, flies=100, control="none", unpaired=False):
    naive, trained = [], []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        run = acquisition(s, trials, control, unpaired)
        od = Odours()
        A, B = run["A"], run["B"]
        fresh = build(control, s)
        naive.append(preference_index(fresh, A, B, flies, rng))
        trained.append(preference_index(run["model"], A, B, flies, rng))
    return np.array(naive), np.array(trained)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--flies", type=int, default=100)
    p.add_argument("--out", default="data/mb_tmaze.json")
    a = p.parse_args()

    print(f"\nT-MAZE  ({a.seeds} seeds x {a.flies} flies, {a.trials} training trials)\n")
    print(f"{'condition':30s} {'naive PI':>10s} {'trained PI':>12s} {'shift':>9s}")
    print("-" * 64)
    out = {}
    for label, kw in [("connectome + plasticity", {}),
                      ("plasticity off", dict(control="plasticity-off")),
                      ("dopamine disconnected", dict(control="dopamine-cut")),
                      ("KC->MBON shuffled", dict(control="shuffled")),
                      ("unpaired odour and shock", dict(unpaired=True))]:
        n, t = assay(a.seeds, a.trials, a.flies, **kw)
        out[label] = dict(naive=n.tolist(), trained=t.tolist())
        print(f"{label:30s} {n.mean():>10.3f} {t.mean():>12.3f} {t.mean()-n.mean():>9.3f}")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")
