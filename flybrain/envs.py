"""A small clean-room obstacle-dodging environment.

Deliberately simple and dependency-free so you can read it end to end and then
replace it. The only contract the rest of the kit depends on is:

    env.reset(seed) -> observation (8 floats, roughly 0..1)
    env.step(action: int 0..2) -> observation, reward, done
    env.N_OBS == 8, env.N_ACTIONS == 3

Swap this file for a different game and everything else keeps working.
"""
from __future__ import annotations
import numpy as np

RUN, JUMP, DUCK = 0, 1, 2


class Dodger:
    N_OBS = 8
    N_ACTIONS = 3

    TICK = 1.0 / 60.0
    DECISION_EVERY = 2          # act at 30 Hz, simulate at 60 Hz
    GRAVITY = 2000.0
    JUMP_VELOCITY = 780.0
    FAST_FALL = 2600.0
    PLAYER_X = 50.0
    PLAYER_W = 20.0
    STAND_H = 44.0
    DUCK_H = 24.0
    WIDTH = 600.0
    SIGHT = 300.0          # how far ahead the proximity channel resolves

    def __init__(self, max_seconds: float = 60.0):
        self.max_seconds = max_seconds
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    def reset(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.speed = 200.0
        self.y = 0.0
        self.vy = 0.0
        self.ducking = False
        self.obstacles = []
        self.next_spawn = 320.0
        self.score = 0.0
        self._spawn_until_full()
        return self._observe()

    # ------------------------------------------------------------------
    def _spawn_until_full(self):
        while not self.obstacles or self.obstacles[-1]["x"] < self.WIDTH:
            last = self.obstacles[-1]["x"] if self.obstacles else self.PLAYER_X
            gap = self.next_spawn * self.rng.uniform(0.85, 1.6)
            gap = max(gap, 140.0 + self.speed * 0.25)
            flying = self.rng.random() < 0.28
            if flying:
                ob = dict(x=last + gap, w=self.rng.uniform(24, 40),
                          h=22.0, bottom=float(self.rng.choice([34.0, 56.0])))
            else:
                ob = dict(x=last + gap, w=self.rng.uniform(14, 38),
                          h=self.rng.uniform(30, 48), bottom=0.0)
            self.obstacles.append(ob)

    def _nearest(self):
        for ob in self.obstacles:
            if ob["x"] + ob["w"] > self.PLAYER_X:
                return ob
        return None

    def _observe(self):
        ob = self._nearest()
        o = np.zeros(self.N_OBS)
        if ob is not None:
            o[0] = 1.0 - min(max((ob["x"] - self.PLAYER_X) / self.SIGHT, 0.0), 1.0)
            o[1] = ob["w"] / 75.0
            o[2] = ob["h"] / 60.0
            o[3] = ob["bottom"] / 60.0
        o[4] = self.speed / 620.0
        o[5] = self.y / 100.0
        o[6] = (self.vy / 900.0 + 1.0) / 2.0
        o[7] = 1.0 if self.y <= 0.0 else 0.0
        return o

    # ------------------------------------------------------------------
    def step(self, action: int):
        for _ in range(self.DECISION_EVERY):
            grounded = self.y <= 0.0
            if action == JUMP and grounded:
                self.vy = self.JUMP_VELOCITY
                self.ducking = False
            elif action == DUCK:
                self.ducking = True
                if not grounded:
                    self.vy -= self.FAST_FALL * self.TICK
            else:
                self.ducking = False

            if not grounded or self.vy > 0:
                self.vy -= self.GRAVITY * self.TICK
                self.y += self.vy * self.TICK
                if self.y <= 0.0:
                    self.y, self.vy = 0.0, 0.0

            self.speed = min(620.0, self.speed + 6.0 * self.TICK)
            dx = self.speed * self.TICK
            for o in self.obstacles:
                o["x"] -= dx
            self.obstacles = [o for o in self.obstacles if o["x"] + o["w"] > -40]
            self._spawn_until_full()

            self.t += self.TICK
            self.score += dx * 0.025

            if self._collides():
                return self._observe(), self.score, True
            if self.t >= self.max_seconds:
                return self._observe(), self.score, True
        return self._observe(), self.score, False

    def _collides(self) -> bool:
        ph = self.DUCK_H if (self.ducking and self.y <= 0.0) else self.STAND_H
        ptop, pbot = self.y + ph, self.y
        for o in self.obstacles:
            if o["x"] > self.PLAYER_X + self.PLAYER_W:
                break
            if o["x"] + o["w"] < self.PLAYER_X:
                continue
            if pbot < o["bottom"] + o["h"] and ptop > o["bottom"]:
                return True
        return False

    # ------------------------------------------------------------------
    def render(self) -> str:
        cols, rows = 78, 7
        grid = [[" "] * cols for _ in range(rows)]
        scale = self.WIDTH / cols
        def row_of(height):
            return rows - 1 - min(rows - 1, int(height / 16.0))
        for o in self.obstacles:
            c = int(o["x"] / scale)
            if 0 <= c < cols:
                for hh in range(int(o["bottom"]), int(o["bottom"] + o["h"]), 8):
                    grid[row_of(hh)][c] = "#" if o["bottom"] == 0 else "v"
        pc = int(self.PLAYER_X / scale)
        grid[row_of(self.y)][pc] = "_" if (self.ducking and self.y <= 0) else "T"
        body = "\n".join("".join(r) for r in grid)
        return f"{body}\n{'-' * cols}\nt={self.t:5.1f}s  score={self.score:7.1f}  speed={self.speed:5.0f}"
