"""The whole thing, joined up: a fly that learns an odour is dangerous and
then walks away from it.

Two circuits, two timescales, and they are genuinely different kinds of learning:

  * The MUSHROOM BODY learns within the fly's lifetime, by the animal's own
    mechanism -- dopamine depressing Kenyon-cell synapses during the episode.
    Nothing is trained here; it happens while you watch.

  * The STEERING READOUT is shaped across generations by CEM, before the episode
    starts. It never learns what an odour means. It only learns what to do about
    a valence, and the mushroom body tells it the valence.

That split is not a convenience. Evolution builds the machinery that acts on
valence; experience fills in which smell has which. The readout sees the
mushroom body's answer on channels 3 and 4 and has no other access to odour
identity -- so if it avoids the shocked source, it can only be because the
mushroom body told it to.

MEASURED: both circuits. INVENTED: the arena, the odour gradients, the sensor
mapping, and everything already listed in model.py and arenas.py.
"""
from __future__ import annotations
import numpy as np

from ..body.arenas import ArenaBase, ARENA, _bearing_channels
from .model import MushroomBody
from .odour import Odours

ANTENNA = 14.0          # half the spacing between the two sensors, arena units
SOURCE_SIGMA = 95.0     # how far each odour carries
REACH = 30.0            # radius at which a source is reached
MB_EVERY = 5            # run the mushroom body at 6 Hz, steering at 30 Hz
MIN_SEPARATION = 230.0  # keep the two plumes distinguishable (sigma is 95)
SHOCK_PAIRINGS = 4      # presentations of the odour paired with dopamine, per shock

FOOD = 500.0            # reward for reaching the safe source
SHOCK = 250.0           # cost of reaching the dangerous one
HUNGER = 8.0            # per second, so doing nothing is not a winning strategy
ENGAGE = 900.0          # shaping on APPROACH, identical for both sources


class Forage(ArenaBase):
    """Two odour sources. One shocks. Learn which, and stop going there."""
    NAME = "forage"

    def __init__(self, seconds: float = 40.0, mushroom: MushroomBody | None = None,
                 odours: Odours | None = None, punished_odour="geraniol",
                 safe_odour="octanol"):
        super().__init__(seconds)
        self.mb = mushroom if mushroom is not None else MushroomBody()
        # A second copy that never learns. Subtracting it gives the change
        # experience has made to this fly's response -- which is what a
        # behavioural experiment measures, and what the steering circuit sees.
        self.naive = MushroomBody()
        self.naive.plasticityOn = False if hasattr(self.naive, "plasticityOn") else None
        self.od = odours if odours is not None else Odours()
        self.vec_bad = self.od.vector(punished_odour)
        self.vec_safe = self.od.vector(safe_odour)
        # the blend is linear in concentration, so the projection-neuron matrix
        # product is done once here rather than every step
        self.drive_bad = self.mb.pn_drive(self.vec_bad)
        self.drive_safe = self.mb.pn_drive(self.vec_safe)
        self._tick = 0

    # -- arena ---------------------------------------------------------
    def _setup(self):
        self.mb.reset()
        self.naive.reset()
        self.shock_cooldown = 0.0
        self.shocks = 0
        self.food = 0
        self.valence = 0.0
        self.intensity = 0.0
        self._last_smell = 0.0
        self._tick = 0
        self._place(both=True)

    def _bounce(self):
        """Turn away from a wall rather than grinding along it."""
        w = self.walker
        if abs(w.x) >= ARENA - 1 or abs(w.y) >= ARENA - 1:
            w.heading = float(np.arctan2(-w.y, -w.x))    # face the middle
            w.x = float(np.clip(w.x, -ARENA + 6, ARENA - 6))
            w.y = float(np.clip(w.y, -ARENA + 6, ARENA - 6))

    def _spot(self, away_from=None, clearance=MIN_SEPARATION):
        """A new source position, kept clear of the other source.

        Separation matters more than it looks. If the two plumes overlap, the
        fly only ever smells a blend, the Kenyon cell code is a blend too, and
        depressing it teaches "all odour is dangerous" rather than "this odour
        is". An earlier version let sources land within one plume-width a
        quarter of the time and the fly duly learned to avoid everything.
        """
        for _ in range(40):
            angle = self.rng.uniform(-np.pi, np.pi)
            r = self.rng.uniform(120, 190)
            x = float(np.clip(self.walker.x + r * np.cos(angle), -ARENA, ARENA))
            y = float(np.clip(self.walker.y + r * np.sin(angle), -ARENA, ARENA))
            if away_from is None or np.hypot(x - away_from[0], y - away_from[1]) >= clearance:
                return (x, y)
        return (x, y)

    def _place(self, both=False):
        self.bad = self._spot()
        self.safe = self._spot(away_from=self.bad)

    def _concentration(self, x, y, source):
        d = np.hypot(x - source[0], y - source[1])
        return float(np.exp(-0.5 * (d / SOURCE_SIGMA) ** 2))

    def _smell_at(self, x, y):
        return (self._concentration(x, y, self.bad),
                self._concentration(x, y, self.safe))

    # -- the join ------------------------------------------------------
    def _run_mushroom_body(self, punish: float):
        """Whatever is in the air becomes projection-neuron activity.

        The reported valence is the LEARNED response minus the naive one, so it
        is zero for a fly that has never been shocked, whatever the odour's
        idiosyncratic baseline happens to be.
        """
        cb, cs = self._smell_at(self.walker.x, self.walker.y)
        drive = cb * self.drive_bad + cs * self.drive_safe
        self.intensity = cb + cs
        out = self.mb.present(None, punish=punish, learn=True, drive=drive)
        self.valence = out["valence"] - self.naive.present(None, learn=False, drive=drive)["valence"]
        return out

    def _advance(self):
        self._tick += 1
        self._bounce()
        self.shock_cooldown = max(0.0, self.shock_cooldown - self.DT)
        d_bad = np.hypot(self.walker.x - self.bad[0], self.walker.y - self.bad[1])
        d_safe = np.hypot(self.walker.x - self.safe[0], self.walker.y - self.safe[1])

        if d_bad < REACH and self.shock_cooldown <= 0:
            # one shock is one training trial, not one per physics tick
            for _ in range(SHOCK_PAIRINGS):
                self._run_mushroom_body(punish=1.0)
            self.shocks += 1
            self.shock_cooldown = 1.5
            # Discrete events are credited here, not in _reward(). The source
            # moves in the same breath, so by the time _reward() runs the fly is
            # no longer standing on it -- an earlier version lost every food
            # reward exactly that way and the task became unlearnable.
            self.score -= SHOCK
            # move the source. The fly cannot learn "avoid that spot" -- the
            # smell is the only thing that carries over.
            self.bad = self._spot(away_from=self.safe)
            self._last_smell = sum(self._smell_at(self.walker.x, self.walker.y))
        elif self._tick % MB_EVERY == 0:
            self._run_mushroom_body(punish=0.0)

        if d_safe < REACH:
            self.food += 1
            self.score += FOOD
            self.safe = self._spot(away_from=self.bad)
            # do not charge the fly for the odour vanishing when it moves
            self._last_smell = sum(self._smell_at(self.walker.x, self.walker.y))

    def _sense(self):
        o = np.zeros(8)
        h = self.walker.heading
        lx, ly = (self.walker.x - ANTENNA * np.sin(h), self.walker.y + ANTENNA * np.cos(h))
        rx, ry = (self.walker.x + ANTENNA * np.sin(h), self.walker.y - ANTENNA * np.cos(h))
        here = sum(self._smell_at(self.walker.x, self.walker.y))
        left = sum(self._smell_at(lx, ly))
        right = sum(self._smell_at(rx, ry))

        o[0] = min(1.0, here)
        o[1] = float(np.clip(0.5 + 12.0 * (left - right), 0.0, 1.0))   # which way it grows
        o[2] = float(np.clip(0.5 + 60.0 * (left - right), 0.0, 1.0))   # finer gradient
        # channels 3 and 4 are the mushroom body's verdict -- the only route by
        # which odour identity reaches the steering circuit
        o[3] = float(np.clip(0.5 + 0.5 * self.valence, 0.0, 1.0))
        o[4] = float(np.clip(abs(self.valence) * min(1.0, here * 2.0), 0.0, 1.0))
        o[5] = self._speed_channel()
        near = self.bad if np.hypot(self.walker.x - self.bad[0], self.walker.y - self.bad[1]) < \
            np.hypot(self.walker.x - self.safe[0], self.walker.y - self.safe[1]) else self.safe
        o[6], o[7] = _bearing_channels(near[0] - self.walker.x, near[1] - self.walker.y, h)
        return o

    def _reward(self):
        """Deliberately blind to WHICH odour is which.

        An earlier version rewarded approaching the safe plume and penalised the
        dangerous one, which handed the steering circuit the answer directly --
        it could avoid the shock without the mushroom body ever being consulted.
        Now the shaping term is the total odour in the air, identical for both
        sources, so nothing in the reward distinguishes them. The only thing
        that can is the mushroom body on channels 3 and 4.

        Hunger is what makes ignoring odour altogether a losing strategy: sit
        still and starve, engage and risk being wrong until you have learned.
        """
        cb, cs = self._smell_at(self.walker.x, self.walker.y)
        r = -HUNGER
        # shaping on CLOSING THE DISTANCE, not on standing in the plume --
        # otherwise the best strategy is to sit in the odour and never move
        smell = cb + cs
        r += ENGAGE * (smell - self._last_smell)
        self._last_smell = smell
        return r


# ---------------------------------------------------------------------------
def train(seed=20260913, generations=30, population=36, elites=5,
          seconds=40.0, courses=2, out="policies/forage.json", quiet=True):
    """CEM on the steering readout only. The mushroom body is untouched by
    training -- it learns inside each episode and is wiped between them."""
    import json
    from pathlib import Path
    from ..circuit import load as load_circuit
    from ..readout import n_params, act

    rng = np.random.default_rng(seed)
    circuit = load_circuit()
    env = Forage(seconds=seconds)
    d = n_params(circuit.n_outputs, env.N_ACTIONS)

    def rollout(theta, s):
        obs = env.reset(s); circuit.reset(); done = False; score = 0.0
        while not done:
            a = act(theta, circuit.step(obs), circuit.n_outputs, env.N_ACTIONS)
            obs, score, done = env.step(a)
        return score

    mean, sigma = np.zeros(d), np.full(d, 0.8)
    champion = rng.normal(0, 0.7, d)
    validation = [1100001, 1100002, 1100003]
    best = -np.inf
    for g in range(generations):
        cs = [int(rng.integers(1, 900000)) for _ in range(courses)]
        cands = [champion.copy()] + [mean + sigma * rng.standard_normal(d)
                                     for _ in range(population - 1)]
        fit = np.array([np.mean([rollout(c, s) for s in cs]) for c in cands])
        order = np.argsort(-fit)
        elite = np.array([cands[i] for i in order[:elites]])
        mean = 0.3 * mean + 0.7 * elite.mean(axis=0)
        sigma = np.maximum(0.3 * sigma + 0.7 * elite.std(axis=0), 0.07)
        top = cands[order[0]]
        val = np.mean([rollout(top, s) for s in validation])
        if val > best:
            best, champion = val, top.copy()
        if not quiet:
            print(f"gen {g:3d} train {fit[order[0]]:9.1f} val {val:9.1f} best {best:9.1f}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(dict(task="forage", seed=seed, seconds=seconds,
                                         validation=best, theta=champion.tolist()), indent=1))
    return champion, best


def evaluate(theta, seeds=20, seconds=60.0, dopamine=True, plasticity=True):
    """Does the fly get shocked LESS as the episode goes on? That is the whole
    question -- it is the only thing that could come from learning, because the
    steering weights are frozen and the sources move after every shock."""
    from ..circuit import load as load_circuit
    from ..readout import act
    circuit = load_circuit()
    env = Forage(seconds=seconds)
    if not dopamine:
        env.mb.W_dan_mbon = np.zeros_like(env.mb.W_dan_mbon)
    if not plasticity:
        env.mb.lr = 0.0
    thirds, food = [], []
    for s in range(seeds):
        obs = env.reset(2100001 + s); circuit.reset(); done = False
        marks, last = [], 0
        while not done:
            a = act(theta, circuit.step(obs), circuit.n_outputs, env.N_ACTIONS)
            obs, _, done = env.step(a)
            if len(marks) < 3 and env.t >= seconds / 3 * (len(marks) + 1):
                marks.append(env.shocks - last); last = env.shocks
        while len(marks) < 3:
            marks.append(env.shocks - last); last = env.shocks
        thirds.append(marks); food.append(env.food)
    return np.array(thirds, float), np.array(food, float)
