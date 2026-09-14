"""The only part that learns: a tiny MLP reading the circuit's output cells.

16 readout cells -> 12 tanh hidden -> 3 action scores = 243 parameters.
The connectome wiring never changes. This is important and you should say so
out loud whenever you show the result to anyone.
"""
from __future__ import annotations
import numpy as np

HIDDEN = 12
INPUT_GAIN = 4.0


def n_params(n_in: int, n_out: int, hidden: int = HIDDEN) -> int:
    return n_in * hidden + hidden + hidden * n_out + n_out


def unpack(theta: np.ndarray, n_in: int, n_out: int, hidden: int = HIDDEN):
    i = 0
    w1 = theta[i:i + n_in * hidden].reshape(n_in, hidden); i += n_in * hidden
    b1 = theta[i:i + hidden]; i += hidden
    w2 = theta[i:i + hidden * n_out].reshape(hidden, n_out); i += hidden * n_out
    b2 = theta[i:i + n_out]
    return w1, b1, w2, b2


def act(theta: np.ndarray, activity: np.ndarray, n_in: int, n_out: int) -> int:
    w1, b1, w2, b2 = unpack(theta, n_in, n_out)
    h = np.tanh(INPUT_GAIN * activity @ w1 + b1)
    return int(np.argmax(h @ w2 + b2))
