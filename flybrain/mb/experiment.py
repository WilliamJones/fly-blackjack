"""Conditioning experiments, each run against every control.

The headline number is the LEARNING INDEX: how far the punished odour's valence
moved, minus how far the unpaired odour's moved. Odours have idiosyncratic naive
valence, so only the change is meaningful.

A model that still learns with dopamine disconnected has a bug, not a result.
That control is the one to read first.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from .model import MushroomBody
from .odour import Odours

ODOURS = ["geraniol", "octanol", "benzaldehyde", "ethyl-butyrate",
          "methylcyclohexanol", "isoamyl-acetate", "pentyl-acetate", "linalool"]


def build(control="none", seed=0):
    kw = dict(seed=seed)
    if control == "plasticity-off":
        kw["learning_rate"] = 0.0
    if control == "shuffled":
        kw["shuffle_kc_mbon"] = True
    mb = MushroomBody(**kw)
    if control == "dopamine-cut":
        mb.W_dan_mbon = np.zeros_like(mb.W_dan_mbon)
    return mb


def acquisition(seed=0, trials=20, control="none", unpaired=False):
    od = Odours()
    rng = np.random.default_rng(seed)
    a, b = rng.choice(ODOURS, 2, replace=False)
    A, B = od.vector(str(a)), od.vector(str(b))
    mb = build(control, seed)

    base_a, base_b = mb.valence_of(A), mb.valence_of(B)
    curve = []
    for t in range(trials):
        if unpaired:
            # shock happens, but never while the odour is present
            mb.present(A, learn=True)
            mb.present(np.zeros(mb.n_pn), punish=1.0, learn=True)
        else:
            mb.present(A, punish=1.0, learn=True)
        mb.present(B, learn=True)
        curve.append(dict(trial=t + 1,
                          a=mb.valence_of(A) - base_a,
                          b=mb.valence_of(B) - base_b,
                          depression=mb.depression))
    return dict(odours=[str(a), str(b)], baseline=[base_a, base_b],
                curve=curve, index=curve[-1]["a"] - curve[-1]["b"], model=mb,
                A=A, B=B)


def extinction(run, trials=20):
    """Present the punished odour with no shock. Avoidance should fade."""
    mb, A, B = run["model"], run["A"], run["B"]
    base_a, base_b = run["baseline"]
    out = []
    for t in range(trials):
        mb.present(A, learn=True)
        mb.present(B, learn=True)
        out.append(dict(trial=t + 1, a=mb.valence_of(A) - base_a,
                        b=mb.valence_of(B) - base_b))
    return out


def reversal(run, trials=20):
    """Now punish the other odour instead."""
    mb, A, B = run["model"], run["A"], run["B"]
    base_a, base_b = run["baseline"]
    out = []
    for t in range(trials):
        mb.present(B, punish=1.0, learn=True)
        mb.present(A, learn=True)
        out.append(dict(trial=t + 1, a=mb.valence_of(A) - base_a,
                        b=mb.valence_of(B) - base_b))
    return out


def pooled(control="none", seeds=12, trials=20, unpaired=False):
    idx = [acquisition(s, trials, control, unpaired)["index"] for s in range(seeds)]
    return np.array(idx)


def report(seeds=12, trials=20):
    rows = []
    base = pooled("none", seeds, trials)
    rows.append(("connectome + plasticity", base))
    for name, kw in [("plasticity off", dict(control="plasticity-off")),
                     ("dopamine disconnected", dict(control="dopamine-cut")),
                     ("KC->MBON shuffled", dict(control="shuffled")),
                     ("unpaired odour and shock", dict(unpaired=True))]:
        rows.append((name, pooled(seeds=seeds, trials=trials, **kw)))

    sd = max(base.std(), 1e-9)
    print(f"\nACQUISITION  ({seeds} seeds, {trials} trials, learning index "
          f"= change in punished odour minus change in unpaired odour)\n")
    print(f"{'condition':30s} {'index':>10s} {'sd':>8s} {'vs control':>12s}")
    print("-" * 64)
    for name, vals in rows:
        d = (base.mean() - vals.mean()) / sd if name != rows[0][0] else 0.0
        print(f"{name:30s} {vals.mean():>10.3f} {vals.std():>8.3f}"
              + (f" {d:>11.1f} sd" if name != rows[0][0] else f" {'--':>14s}"))

    run = acquisition(0, trials)
    ext, rev = extinction(run), reversal(acquisition(0, trials))
    print(f"\nodours: {run['odours'][0]} punished, {run['odours'][1]} unpaired")
    print(f"{'phase':16s} {'start':>9s} {'end':>9s}")
    print("-" * 38)
    print(f"{'acquisition':16s} {run['curve'][0]['a']:>9.3f} {run['curve'][-1]['a']:>9.3f}")
    print(f"{'extinction':16s} {ext[0]['a']:>9.3f} {ext[-1]['a']:>9.3f}")
    print(f"{'reversal (B)':16s} {rev[0]['b']:>9.3f} {rev[-1]['b']:>9.3f}")

    return dict(
        acquisition={n: v.tolist() for n, v in rows},
        example=dict(odours=run["odours"],
                     acquisition=[{k: c[k] for k in ("trial", "a", "b", "depression")}
                                  for c in run["curve"]],
                     extinction=ext, reversal=rev))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--out", default="data/mb_results.json")
    a = p.parse_args()
    res = report(a.seeds, a.trials)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {a.out}")
