"""Cross-entropy method (CEM) training of the readout only.

CEM is simple on purpose: sample a population of weight vectors, keep the best
few, move the sampling distribution toward them, repeat. No gradients, no
backprop, ~120 lines of concept. Easy to read when you are rusty.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

from .circuit import load, load_silenced
from .envs import Dodger
from .readout import n_params, act


def rollout(theta, circuit, env, seed, max_seconds):
    env.max_seconds = max_seconds
    obs = env.reset(seed)
    circuit.reset()
    done = False
    score = 0.0
    while not done:
        activity = circuit.step(obs)
        a = act(theta, activity, circuit.n_outputs, env.N_ACTIONS)
        obs, score, done = env.step(a)
    return score, env.t


def evaluate(theta, circuit, env, seeds, max_seconds):
    out = [rollout(theta, circuit, env, s, max_seconds) for s in seeds]
    scores = np.array([o[0] for o in out])
    times = np.array([o[1] for o in out])
    return scores, times


def train(seed=20260913, generations=40, population=48, elites=6,
          episode_seconds=45.0, courses=3, out="checkpoint.json", quiet=False):
    rng = np.random.default_rng(seed)
    circuit = load()
    env = Dodger()
    d = n_params(circuit.n_outputs, env.N_ACTIONS)

    mean = np.zeros(d)
    sigma = np.full(d, 0.8)
    champion = rng.normal(0, 0.7, d)
    validation_seeds = [1100001, 1100002, 1100003, 1100004]
    best_validation = -np.inf
    history = []

    for g in range(generations):
        course_seeds = [int(rng.integers(1, 900000)) for _ in range(courses)]
        candidates = [champion.copy()] + [
            mean + sigma * rng.standard_normal(d) for _ in range(population - 1)
        ]
        fitness = np.array([
            evaluate(c, circuit, env, course_seeds, episode_seconds)[0].mean()
            for c in candidates
        ])
        order = np.argsort(-fitness)
        elite = np.array([candidates[i] for i in order[:elites]])
        mean = 0.3 * mean + 0.7 * elite.mean(axis=0)
        sigma = np.maximum(0.3 * sigma + 0.7 * elite.std(axis=0), 0.07)

        best = candidates[order[0]]
        val = evaluate(best, circuit, env, validation_seeds, episode_seconds)[0].mean()
        if val > best_validation:
            best_validation, champion = val, best.copy()
        history.append(dict(generation=g, train=float(fitness[order[0]]),
                            validation=float(val), best_validation=float(best_validation)))
        if not quiet:
            print(f"gen {g:3d}  train {fitness[order[0]]:8.1f}  "
                  f"val {val:8.1f}  best {best_validation:8.1f}")

    Path(out).write_text(json.dumps(dict(
        seed=seed, generations=generations, population=population,
        episode_seconds=episode_seconds, validation=best_validation,
        theta=champion.tolist(), history=history), indent=1))
    return champion, best_validation, history


def benchmark(theta, episode_seconds=45.0, n=50, start=2100001):
    """Held-out test PLUS the two controls that make the result mean anything."""
    seeds = list(range(start, start + n))
    env = Dodger()
    rows = []

    circuit = load()
    s, t = evaluate(theta, circuit, env, seeds, episode_seconds)
    rows.append(("connectome + trained readout", s, t))

    s, t = evaluate(theta, load_silenced(), env, seeds, episode_seconds)
    rows.append(("SAME readout, circuit silenced", s, t))

    untrained = np.random.default_rng(7).normal(0, 0.7, theta.shape)
    s, t = evaluate(untrained, circuit, env, seeds, episode_seconds)
    rows.append(("untrained readout", s, t))

    class Random:
        def __init__(self): self.rng = np.random.default_rng(11)
        def reset(self): pass
        n_outputs = 16
        def step(self, o): return np.zeros(16)
    rs, rt = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        obs = env.reset(sd); env.max_seconds = episode_seconds
        done = False; sc = 0.0
        while not done:
            obs, sc, done = env.step(int(rng.integers(0, 3)))
        rs.append(sc); rt.append(env.t)
    rows.append(("uniform random actions", np.array(rs), np.array(rt)))

    print(f"\n{'controller':34s} {'survived/' + str(n):>12s} {'mean seconds':>13s} {'mean score':>11s}")
    print("-" * 74)
    for name, sc, tt in rows:
        print(f"{name:34s} {int((tt >= episode_seconds - 1e-6).sum()):>12d} "
              f"{tt.mean():>13.2f} {sc.mean():>11.1f}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--population", type=int, default=48)
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--out", default="checkpoint.json")
    a = p.parse_args()
    t0 = time.time()
    theta, val, _ = train(seed=a.seed, generations=a.generations,
                          population=a.population, episode_seconds=a.seconds, out=a.out)
    print(f"\ntrained in {time.time() - t0:.1f}s, validation {val:.1f} -> {a.out}")
    benchmark(theta, episode_seconds=a.seconds)
