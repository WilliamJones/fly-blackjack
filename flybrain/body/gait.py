"""Tripod gait central pattern generator + simple leg kinematics.

IMPORTANT, and say it out loud whenever you show this: the GAIT IS MINE, not
the connectome's. Real insects generate walking rhythm in the ventral nerve
cord, and descending neurons from the brain modulate speed and turning. That is
what is modelled here: a hand-built CPG produces the rhythm, and the fly circuit
steers it. Nothing in this file is learned, and nothing in it is measured.

This file is mirrored exactly in web/gait.js. If you change one, change both --
`python3 -m flybrain.body.check_parity` verifies they still agree.
"""
from __future__ import annotations
import numpy as np

# leg order: T1=front, T2=middle, T3=hind
LEGS = ["T1_left", "T2_left", "T3_left", "T1_right", "T2_right", "T3_right"]
# tripod A = front-left, middle-right, hind-left. tripod B = the other three.
TRIPOD_B = {"T1_right", "T2_left", "T3_right"}

DUTY = 0.62           # fraction of the cycle a leg spends on the ground
SWING_LIFT = 1.0      # normalised lift height during swing
STRIDE_RATE = 7.5     # gait cycles per second at full speed
IDLE_RATE = 0.6       # legs keep shuffling a little when stopped

# Foot trajectory, in the body frame, in MuJoCo length units.
# The fly is 0.262 long nose to tail, so a 0.055 stride is about a fifth of a
# body length -- close to a real Drosophila step.
STRIDE = 0.055
LIFT = 0.030
LATERAL_TURN = 0.028      # how far the feet shift sideways in a hard turn

LEG_JOINTS = ["coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur", "tibia"]


def leg_phase(phase: float, leg: str) -> float:
    """Where this leg sits in the cycle. The two tripods run half a cycle apart."""
    p = phase + (np.pi if leg in TRIPOD_B else 0.0)
    return float(np.mod(p, 2.0 * np.pi))


def foot_offset(phase: float, leg: str, speed: float, turn: float):
    """Where this foot should be, relative to its neutral stance position.

    Stance: planted, sliding backwards under the body. Swing: lifted, swung
    forward. Turning lengthens the outer stride and shortens the inner one,
    which is how insects actually turn -- they do not steer, they limp on
    purpose.
    """
    segment = leg.split("_")[0]
    side = 1.0 if leg.endswith("left") else -1.0
    p = leg_phase(phase, leg) / (2.0 * np.pi)

    stride = float(np.clip(1.0 + 0.55 * turn * side, 0.2, 1.8))
    stride *= float(np.clip(abs(speed), 0.12, 1.0))
    direction = 1.0 if speed >= 0 else -1.0

    if p < DUTY:
        s = p / DUTY
        along = (0.5 - s) * STRIDE
        lift = 0.0
    else:
        s = (p - DUTY) / (1.0 - DUTY)
        along = (s - 0.5) * STRIDE
        lift = float(np.sin(np.pi * s)) * LIFT

    return (along * stride * direction,
            -side * turn * LATERAL_TURN * float(np.clip(abs(speed), 0.0, 1.0)),
            lift)


class Walker:
    """Planar body state driven by discrete actions. Shared by training and JS."""

    # action -> (target forward speed, target turn rate), both -1..1
    ACTIONS = [
        (0.0, 0.0),      # 0 stop
        (1.0, 0.0),      # 1 forward
        (0.75, 1.0),     # 2 forward-left
        (0.75, -1.0),    # 3 forward-right
        (-0.6, 0.0),     # 4 backward
    ]
    N_ACTIONS = len(ACTIONS)

    MAX_SPEED = 22.0        # body lengths-ish per second, arena units
    MAX_TURN = 4.2          # radians per second
    RESPONSE = 0.28         # how fast speed/turn approach their target

    def __init__(self):
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x: float, y: float, heading: float):
        self.x, self.y, self.heading = float(x), float(y), float(heading)
        self.speed = 0.0
        self.turn = 0.0
        self.phase = 0.0

    def step(self, action: int, dt: float):
        target_speed, target_turn = self.ACTIONS[int(action)]
        self.speed += (target_speed - self.speed) * self.RESPONSE
        self.turn += (target_turn - self.turn) * self.RESPONSE
        self.heading += self.turn * self.MAX_TURN * dt
        self.heading = float(np.mod(self.heading + np.pi, 2 * np.pi) - np.pi)
        self.x += self.speed * self.MAX_SPEED * np.cos(self.heading) * dt
        self.y += self.speed * self.MAX_SPEED * np.sin(self.heading) * dt
        rate = IDLE_RATE + STRIDE_RATE * abs(self.speed)
        self.phase = float(np.mod(self.phase + 2 * np.pi * rate * dt, 2 * np.pi))

    def joint_pose(self) -> dict:
        return pose(self.phase, self.speed, self.turn)
