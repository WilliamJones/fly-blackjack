"""Four arenas, one sensor interface.

Every arena reports the same 8 channels, so the same 80-cell circuit drives all
four behaviours without being rebuilt:

  0  primary stimulus proximity      (goal / prey / plume source direction)
  1  primary bearing, left
  2  primary bearing, right
  3  secondary stimulus proximity    (obstacle / threat / wall)
  4  secondary bearing, left
  5  secondary bearing, right
  6  own forward speed               (0.5 = stopped)
  7  scalar field intensity          (odour concentration; 0 where unused)

Channel-to-cell-type assignment is an engineering choice, not biology. The one
place it lines up with the real animal is SWAT: channels 3-5 feed LC4/LPLC2,
which are genuinely looming-sensitive escape cells in a living fly.

Mirrored in web/arenas.js.
"""
from __future__ import annotations
import numpy as np

from .gait import Walker

ARENA = 220.0          # half-width of the square arena
TAU = 2.0 * np.pi


def _bearing_channels(dx: float, dy: float, heading: float):
    """Split a relative bearing into separate left and right channels."""
    angle = np.arctan2(dy, dx) - heading
    s = float(np.sin(angle))
    return max(0.0, s), max(0.0, -s)


class ArenaBase:
    N_OBS = 8
    N_ACTIONS = Walker.N_ACTIONS
    DT = 1.0 / 60.0
    CONTROL_EVERY = 2                 # decide at 30 Hz, integrate at 60 Hz
    NAME = "base"

    def __init__(self, seconds: float = 20.0):
        self.seconds = seconds
        self.walker = Walker()

    # -- subclass hooks ----------------------------------------------------
    def _setup(self): ...
    def _advance(self): ...
    def _sense(self) -> np.ndarray: raise NotImplementedError
    def _reward(self) -> float: return 0.0
    def _terminal(self) -> bool: return False

    # -- common loop -------------------------------------------------------
    def reset(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.score = 0.0
        self.trail = []
        self.walker.reset(0.0, 0.0, float(self.rng.uniform(-np.pi, np.pi)))
        self._setup()
        return self._sense()

    def step(self, action: int):
        for _ in range(self.CONTROL_EVERY):
            self.walker.step(action, self.DT)
            self.walker.x = float(np.clip(self.walker.x, -ARENA, ARENA))
            self.walker.y = float(np.clip(self.walker.y, -ARENA, ARENA))
            self.t += self.DT
            self._advance()
            self.score += self._reward() * self.DT
        self.trail.append((self.walker.x, self.walker.y))
        done = self._terminal() or self.t >= self.seconds
        return self._sense(), self.score, done

    def _speed_channel(self) -> float:
        return float(np.clip(0.5 + 0.5 * self.walker.speed, 0.0, 1.0))


# ---------------------------------------------------------------------------
class Targets(ArenaBase):
    """Walk to a goal, around obstacles. Goal respawns when reached."""
    NAME = "targets"
    GOAL_RADIUS = 16.0
    SIGHT = 320.0

    def _setup(self):
        self.obstacles = [(float(self.rng.uniform(-160, 160)),
                           float(self.rng.uniform(-160, 160)),
                           float(self.rng.uniform(16, 30))) for _ in range(6)]
        self.reached = 0
        self._new_goal()
        self.prev_distance = self._goal_distance()

    def _new_goal(self):
        for _ in range(60):
            gx, gy = self.rng.uniform(-180, 180, 2)
            if all(np.hypot(gx - ox, gy - oy) > r + 28 for ox, oy, r in self.obstacles):
                self.goal = (float(gx), float(gy)); return
        self.goal = (float(gx), float(gy))

    def _goal_distance(self):
        return float(np.hypot(self.goal[0] - self.walker.x, self.goal[1] - self.walker.y))

    def _nearest_obstacle(self):
        best, bd = None, 1e9
        for ox, oy, r in self.obstacles:
            d = np.hypot(ox - self.walker.x, oy - self.walker.y) - r
            if d < bd:
                best, bd = (ox, oy, r), d
        return best, bd

    def _sense(self):
        o = np.zeros(8)
        gx, gy = self.goal
        dx, dy = gx - self.walker.x, gy - self.walker.y
        o[0] = 1.0 - min(1.0, np.hypot(dx, dy) / self.SIGHT)
        o[1], o[2] = _bearing_channels(dx, dy, self.walker.heading)
        (ox, oy, r), d = self._nearest_obstacle()
        o[3] = 1.0 - min(1.0, max(0.0, d) / 120.0)
        o[4], o[5] = _bearing_channels(ox - self.walker.x, oy - self.walker.y, self.walker.heading)
        o[6] = self._speed_channel()
        return o

    def _reward(self):
        d = self._goal_distance()
        progress = (self.prev_distance - d) * 60.0     # per-second closing rate
        self.prev_distance = d
        _, od = self._nearest_obstacle()
        penalty = 40.0 if od < 0 else 0.0
        if d < self.GOAL_RADIUS:
            self.reached += 1
            self._new_goal()
            self.prev_distance = self._goal_distance()
            return 300.0 - penalty
        return progress - penalty


# ---------------------------------------------------------------------------
class Swat(ArenaBase):
    """A looming swatter drops on the fly. Get out from under it.

    This is the arena where the cell types mean what they say: channels 3-5
    drive LC4 and LPLC2, which are looming-responsive escape cells in a real
    Drosophila. The escape they produce here is still learned by the readout,
    not inherited from the connectome.
    """
    NAME = "swat"
    STRIKE_RADIUS = 34.0

    def _setup(self):
        self._arm(first=True)
        self.hits = 0
        self.dodges = 0

    def _arm(self, first=False):
        angle = self.rng.uniform(-np.pi, np.pi)
        reach = self.rng.uniform(90, 150)
        self.sx = float(self.walker.x + reach * np.cos(angle))
        self.sy = float(self.walker.y + reach * np.sin(angle))
        self.sz = 1.0                                    # 1 = high, 0 = struck
        self.descent = float(self.rng.uniform(0.42, 0.72))
        self.wait = 1.2 if first else float(self.rng.uniform(0.4, 1.1))

    def _advance(self):
        if self.wait > 0:
            self.wait -= self.DT
            return
        # swatter tracks the fly's position as it comes down, like a real hand
        self.sx += (self.walker.x - self.sx) * 0.9 * self.DT
        self.sy += (self.walker.y - self.sy) * 0.9 * self.DT
        self.sz -= self.descent * self.DT
        if self.sz <= 0.0:
            d = np.hypot(self.walker.x - self.sx, self.walker.y - self.sy)
            if d < self.STRIKE_RADIUS:
                self.hits += 1
            else:
                self.dodges += 1
            self._arm()

    def _sense(self):
        o = np.zeros(8)
        dx, dy = self.sx - self.walker.x, self.sy - self.walker.y
        distance = float(np.hypot(dx, dy))
        # looming: apparent size grows as the swatter gets lower AND closer
        looming = (1.0 - self.sz) * (1.0 - min(1.0, distance / 180.0))
        o[3] = float(np.clip(looming, 0.0, 1.0))
        o[4], o[5] = _bearing_channels(dx, dy, self.walker.heading)
        o[0] = 1.0 - min(1.0, distance / 260.0)
        o[6] = self._speed_channel()
        return o

    def _reward(self):
        d = np.hypot(self.walker.x - self.sx, self.walker.y - self.sy)
        urgency = (1.0 - self.sz) ** 2
        return float(min(d, 120.0)) * urgency * 0.6

    def _terminal(self):
        return self.hits > 0


# ---------------------------------------------------------------------------
class Plume(ArenaBase):
    """Cast up a wind-borne odour plume to its source.

    Real flies do this: they surge upwind when they hit odour and cast crosswind
    when they lose it. Channel 7 is concentration, 1/2 are the upwind direction
    the antennae report.
    """
    NAME = "plume"
    SOURCE_RADIUS = 22.0

    def _setup(self):
        self.wind = float(self.rng.uniform(-np.pi, np.pi))
        reach = self.rng.uniform(180, 300)
        self.ox = float(self.walker.x - reach * np.cos(self.wind))
        self.oy = float(self.walker.y - reach * np.sin(self.wind))
        self.found = 0
        self.best = float(np.hypot(self.ox - self.walker.x, self.oy - self.walker.y))

    def concentration(self, x: float, y: float) -> float:
        # rotate into plume coordinates: +downwind, cross = lateral offset
        dx, dy = x - self.ox, y - self.oy
        downwind = dx * np.cos(self.wind) + dy * np.sin(self.wind)
        cross = -dx * np.sin(self.wind) + dy * np.cos(self.wind)
        if downwind < 0:
            return 0.0
        width = 12.0 + 0.30 * downwind
        c = np.exp(-0.5 * (cross / width) ** 2) * np.exp(-downwind / 420.0)
        # plumes are intermittent, not smooth -- this is what forces casting
        packet = 0.55 + 0.45 * np.sin(downwind * 0.08 - self.t * 3.1 + cross * 0.02)
        return float(np.clip(c * packet, 0.0, 1.0))

    def _sense(self):
        o = np.zeros(8)
        o[7] = self.concentration(self.walker.x, self.walker.y)
        # antennae report where the wind is coming FROM
        o[1], o[2] = _bearing_channels(np.cos(self.wind + np.pi),
                                       np.sin(self.wind + np.pi), self.walker.heading)
        d = float(np.hypot(self.ox - self.walker.x, self.oy - self.walker.y))
        o[0] = 1.0 - min(1.0, d / 420.0)
        o[6] = self._speed_channel()
        return o

    def _reward(self):
        d = float(np.hypot(self.ox - self.walker.x, self.oy - self.walker.y))
        gain = max(0.0, self.best - d) * 60.0
        self.best = min(self.best, d)
        if d < self.SOURCE_RADIUS and self.found == 0:
            self.found = 1
            return 600.0
        return gain + 18.0 * self.concentration(self.walker.x, self.walker.y)

    def _terminal(self):
        return self.found > 0


# ---------------------------------------------------------------------------
class Chase(ArenaBase):
    """Pursue another fly that keeps trying to leave."""
    NAME = "chase"
    CAPTURE = 26.0
    SIGHT = 340.0

    def _setup(self):
        angle = self.rng.uniform(-np.pi, np.pi)
        self.px = float(self.walker.x + 150 * np.cos(angle))
        self.py = float(self.walker.y + 150 * np.sin(angle))
        self.pheading = float(self.rng.uniform(-np.pi, np.pi))
        self.contact = 0.0

    def _advance(self):
        # prey: wanders, and veers away when the pursuer gets close
        dx, dy = self.px - self.walker.x, self.py - self.walker.y
        d = np.hypot(dx, dy)
        self.pheading += float(self.rng.normal(0, 1.4)) * self.DT
        if d < 110.0:
            away = np.arctan2(dy, dx)
            delta = float(np.mod(away - self.pheading + np.pi, TAU) - np.pi)
            self.pheading += np.clip(delta, -3.0, 3.0) * 2.2 * self.DT
        speed = 16.0 if d < 110.0 else 9.0
        self.px = float(np.clip(self.px + speed * np.cos(self.pheading) * self.DT, -ARENA, ARENA))
        self.py = float(np.clip(self.py + speed * np.sin(self.pheading) * self.DT, -ARENA, ARENA))

    def _sense(self):
        o = np.zeros(8)
        dx, dy = self.px - self.walker.x, self.py - self.walker.y
        d = float(np.hypot(dx, dy))
        o[0] = 1.0 - min(1.0, d / self.SIGHT)
        o[1], o[2] = _bearing_channels(dx, dy, self.walker.heading)
        o[3] = float(np.clip((d - 40.0) / 200.0, 0.0, 1.0))   # "it is getting away"
        o[6] = self._speed_channel()
        return o

    def _reward(self):
        d = float(np.hypot(self.px - self.walker.x, self.py - self.walker.y))
        if d < self.CAPTURE:
            self.contact += self.DT
            return 200.0
        return max(0.0, 1.0 - d / self.SIGHT) * 60.0


ARENAS = {"targets": Targets, "swat": Swat, "plume": Plume, "chase": Chase}
