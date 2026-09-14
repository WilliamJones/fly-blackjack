"""Odours, as patterns of activity across antennal lobe projection neurons.

Real ALPNs are glomerulus-specific: each carries the output of one glomerulus,
and an odour activates a characteristic subset of glomeruli. So an odour here is
a set of active glomeruli, and every projection neuron of an active glomerulus
responds together -- which is at least the right shape, even though which
glomeruli a given real odour drives is measured data we are not using.

INVENTED: the glomerulus-to-odour assignment. Real ALPN odour responses exist
(Badel et al. 2016 and others) and could replace this wholesale; the model would
not otherwise change.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

ACTIVE_GLOMERULI = 8          # how many glomeruli one odour drives


def _seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


class Odours:
    def __init__(self, cells_path="data/mb_cells.json"):
        cells = json.loads(Path(cells_path).read_text())
        self.pn_types = cells["types"]["pn"]
        self.glomerulus = [t.split("_")[0] for t in self.pn_types]
        self.names = sorted(set(self.glomerulus))
        self.index = {g: np.array([i for i, x in enumerate(self.glomerulus) if x == g])
                      for g in self.names}
        self.n_pn = len(self.pn_types)

    def glomeruli_for(self, odour: str, n: int = ACTIVE_GLOMERULI):
        rng = np.random.default_rng(_seed(odour))
        return sorted(rng.choice(len(self.names), size=n, replace=False).tolist())

    def vector(self, odour: str, n: int = ACTIVE_GLOMERULI) -> np.ndarray:
        rng = np.random.default_rng(_seed(odour) ^ 0x9E37)
        v = np.zeros(self.n_pn)
        for g in self.glomeruli_for(odour, n):
            strength = rng.uniform(0.45, 1.0)
            ids = self.index[self.names[g]]
            v[ids] = strength * rng.uniform(0.85, 1.15, size=len(ids))
        return np.clip(v, 0.0, 1.5)

    def overlapping_pair(self, base: str, other: str, shared: int,
                         n: int = ACTIVE_GLOMERULI):
        """Two odours sharing `shared` glomeruli -- for generalisation tests."""
        a = self.glomeruli_for(base, n)
        rng = np.random.default_rng(_seed(other))
        pool = [g for g in range(len(self.names)) if g not in a]
        b = sorted(a[:shared] + rng.choice(pool, n - shared, replace=False).tolist())
        def build(gl, key):
            r = np.random.default_rng(_seed(key) ^ 0x9E37)
            v = np.zeros(self.n_pn)
            for g in gl:
                s = r.uniform(0.45, 1.0)
                ids = self.index[self.names[g]]
                v[ids] = s * r.uniform(0.85, 1.15, size=len(ids))
            return np.clip(v, 0, 1.5)
        return build(a, base), build(b, other)
