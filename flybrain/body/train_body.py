"""CEM training for the four body arenas. Same recipe as flybrain/train.py.

Training happens in this fast kinematic simulator, not in MuJoCo. MuJoCo is used
afterwards, to render the trained controller on the real anatomical body
(flybrain/body/mujoco_clip.py). That split is deliberate: MuJoCo's fly runs near
0.7x realtime, which is fine for recording and hopeless for 15,000 episodes.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

from ..circuit import load, load_silenced
from ..readout import n_params, act
from .arenas import ARENAS


def rollout(theta, circuit, env, seed):
    obs = env.reset(seed); circuit.reset()
    done = False; score = 0.0
    while not done:
        activity = circuit.step(obs)
        a = act(theta, activity, circuit.n_outputs, env.N_ACTIONS)
        obs, score, done = env.step(a)
    return score


def mean_score(theta, circuit, env, seeds):
    return float(np.mean([rollout(theta, circuit, env, s) for s in seeds]))


def train(task="targets", seed=20260913, generations=40, population=48,
          elites=6, seconds=18.0, courses=3, out=None, quiet=False):
    rng = np.random.default_rng(seed)
    circuit = load()
    env = ARENAS[task](seconds=seconds)
    d = n_params(circuit.n_outputs, env.N_ACTIONS)

    mean = np.zeros(d); sigma = np.full(d, 0.8)
    champion = rng.normal(0, 0.7, d)
    validation = [1100001, 1100002, 1100003, 1100004]
    best_validation = -np.inf; history = []

    for g in range(generations):
        courses_seeds = [int(rng.integers(1, 900000)) for _ in range(courses)]
        candidates = [champion.copy()] + [mean + sigma * rng.standard_normal(d)
                                          for _ in range(population - 1)]
        fitness = np.array([mean_score(c, circuit, env, courses_seeds) for c in candidates])
        order = np.argsort(-fitness)
        elite = np.array([candidates[i] for i in order[:elites]])
        mean = 0.3 * mean + 0.7 * elite.mean(axis=0)
        sigma = np.maximum(0.3 * sigma + 0.7 * elite.std(axis=0), 0.07)

        best = candidates[order[0]]
        val = mean_score(best, circuit, env, validation)
        if val > best_validation:
            best_validation, champion = val, best.copy()
        history.append(dict(generation=g, train=float(fitness[order[0]]),
                            validation=val, best=float(best_validation)))
        if not quiet:
            print(f"[{task}] gen {g:3d}  train {fitness[order[0]]:10.1f}  "
                  f"val {val:10.1f}  best {best_validation:10.1f}")

    out = Path(out or f"policies/{task}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(task=task, seed=seed, seconds=seconds,
                                   validation=best_validation,
                                   theta=champion.tolist(), history=history), indent=1))
    return champion, best_validation


def benchmark(theta, task, seconds=18.0, n=40, start=2100001, quiet=False):
    seeds = list(range(start, start + n))
    env = ARENAS[task](seconds=seconds)
    rows = []
    rows.append(("connectome + trained readout",
                 [rollout(theta, load(), env, s) for s in seeds]))
    rows.append(("same readout, circuit silenced",
                 [rollout(theta, load_silenced(), env, s) for s in seeds]))
    untrained = np.random.default_rng(7).normal(0, 0.7, theta.shape)
    rows.append(("untrained readout",
                 [rollout(untrained, load(), env, s) for s in seeds]))
    rng = np.random.default_rng(11)
    random_scores = []
    for s in seeds:
        obs = env.reset(s); done = False; sc = 0.0
        while not done:
            obs, sc, done = env.step(int(rng.integers(0, env.N_ACTIONS)))
        random_scores.append(sc)
    rows.append(("uniform random actions", random_scores))
    if not quiet:
        print(f"\n{task}  ({n} held-out seeds)")
        print(f"{'controller':34s} {'mean score':>12s} {'median':>10s}")
        print("-" * 58)
        for name, sc in rows:
            print(f"{name:34s} {np.mean(sc):>12.1f} {np.median(sc):>10.1f}")
    return {name: [float(x) for x in sc] for name, sc in rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all", choices=["all", *ARENAS])
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--population", type=int, default=48)
    p.add_argument("--seconds", type=float, default=18.0)
    p.add_argument("--seed", type=int, default=20260913)
    a = p.parse_args()
    tasks = list(ARENAS) if a.task == "all" else [a.task]
    summary = {}
    for task in tasks:
        t0 = time.time()
        theta, val = train(task, seed=a.seed, generations=a.generations,
                           population=a.population, seconds=a.seconds, quiet=True)
        print(f"[{task}] trained in {time.time()-t0:6.1f}s  validation {val:10.1f}")
        summary[task] = benchmark(theta, task, seconds=a.seconds)
    Path("policies").mkdir(exist_ok=True)
    Path("policies/benchmarks.json").write_text(json.dumps(summary, indent=1))
