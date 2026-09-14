"""Run a measured fruit-fly connectome circuit as a recurrent rate network.

The wiring (which cell talks to which, and how strongly) is REAL measured data
from the MaleCNS v1.0 connectome (FlyEM / HHMI Janelia, CC BY 4.0).

Everything else in this file -- the neuron dynamics, the sign convention, the
gain, the leak -- is a simplified ENGINEERING model, not physiology. See
README.md "What's real and what isn't".
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


class Circuit:
    def __init__(self, graph: dict, gain: float = 1.4, leak: float = 0.3, steps: int = 3):
        self.nodes = graph["nodes"]
        self.channels = graph["channels"]          # the 8 input channels (cell types)
        self.inputs = graph["inputs"]              # [node_index, channel_index]
        self.outputs = np.array(graph["outputs"])  # readout (descending) cells
        self.n = len(self.nodes)
        self.gain, self.leak, self.steps = gain, leak, steps

        # --- build the signed, normalised weight matrix ----------------------
        # sign comes from each cell's predicted neurotransmitter:
        #   acetylcholine -> +1 (excitatory), GABA/glutamate -> -1, unknown -> 0
        sign = np.array([nd["sign"] for nd in self.nodes], dtype=np.float64)
        signed = np.zeros((self.n, self.n))   # signed[pre, post]
        absolute = np.zeros((self.n, self.n))
        for pre, post, contacts in graph["edges"]:
            signed[pre, post] += contacts * sign[pre]
            absolute[pre, post] += contacts * abs(sign[pre])
        denom = absolute.sum(axis=0)
        denom[denom == 0] = 1.0               # cells with no signed input get zero drive
        self.W = signed / denom               # W[pre, post]

        self.h = np.zeros(self.n)

    # ------------------------------------------------------------------------
    def reset(self):
        self.h[:] = 0.0

    def step(self, observation: np.ndarray) -> np.ndarray:
        """Inject an 8-channel observation, settle the circuit, return readout."""
        u = np.zeros(self.n)
        for node_index, channel in self.inputs:
            # centre each observation on zero: 0..1 -> -1..+1
            u[node_index] = 2.0 * (observation[channel] - 0.5)
        for _ in range(self.steps):
            drive = u + self.gain * (self.h @ self.W)
            self.h = self.leak * self.h + (1.0 - self.leak) * np.tanh(drive)
        return self.h[self.outputs]

    # ------------------------------------------------------------------------
    @property
    def n_outputs(self) -> int:
        return len(self.outputs)

    def summary(self) -> str:
        roles = {}
        for nd in self.nodes:
            roles[nd["role"]] = roles.get(nd["role"], 0) + 1
        return (f"{self.n} cells ({roles}), channels={self.channels}, "
                f"readout cells={self.n_outputs}")


class SilencedCircuit(Circuit):
    """Control condition: identical shape, but the circuit contributes nothing.

    If your trained readout still works with this, the connectome was decorative
    and you have learned nothing. Always run this control.
    """
    def step(self, observation: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_outputs)


def load(path: str | Path = None) -> Circuit:
    path = Path(path or Path(__file__).resolve().parent.parent / "data" / "circuit.json")
    return Circuit(json.loads(path.read_text()))


def load_silenced(path: str | Path = None) -> SilencedCircuit:
    path = Path(path or Path(__file__).resolve().parent.parent / "data" / "circuit.json")
    return SilencedCircuit(json.loads(path.read_text()))
