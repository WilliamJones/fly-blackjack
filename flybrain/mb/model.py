"""A connectome-constrained model of mushroom body learning.

MEASURED: which projection neurons contact which Kenyon cells, which Kenyon
cells contact which output neurons, and which dopaminergic neurons can modify
which output neurons. That last one matters most -- it means the *anatomy*
decides what a punishment signal is able to change, not a parameter.

INVENTED: rate units instead of spikes; the learning rate, decay and floor;
k-winners-take-all standing in for APL feedback inhibition; the odour mapping in
odour.py. Reproducing the shape of a published behavioural curve with this would
not prove the model is right.

THE LEARNING RULE. Coincidence of Kenyon cell activity and dopamine DEPRESSES
that KC->MBON synapse:

    gain -= LR * outer(kc_activity, dopamine_at_mbon)
    gain += DECAY * (1 - gain)

Note what the decay term is and is not. It makes memory FADE -- forgetting. Real
extinction in Drosophila is an active process in which presenting the odour
without reinforcement drives dopaminergic neurons itself, and that is not
modelled here. Read the extinction curve as forgetting.

Depression, not potentiation -- that is the direction the real synapse moves,
and it is why punishing an odour makes the approach-driving output go quiet
rather than making an avoidance output shout.

VALENCE. In the compartmental model of the mushroom body, punishment (PPL1)
dopamine acts on compartments whose MBONs drive approach, and reward (PAM)
dopamine on those whose MBONs drive avoidance. extract.py therefore labels each
MBON by which dopaminergic population dominates its input. Net valence is the
approach pool minus the avoidance pool.

Note what this does and does not claim. The partition is measured. The
assignment of "approach" to the PPL1-dominant side follows the standard
compartmental account; we did NOT independently verify the mapping between
MaleCNS MBON numbering and individually characterised MBONs in the literature.
The learning result does not depend on which side carries which label -- it
depends on the two pools being differentially modified, which is the anatomy.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

KC_SPARSITY = 0.06      # fraction of Kenyon cells active per odour (real: ~5-10%)
LEARNING_RATE = 0.55
DECAY = 0.012           # passive recovery of depressed synapses, per presentation
GAIN_FLOOR = 0.0
LATERAL = 0.25          # weight of measured MBON->MBON interactions


def _dense(z, name):
    import scipy.sparse as sp
    shape = tuple(z[f"{name}_shape"])
    return sp.coo_matrix((z[f"{name}_val"], (z[f"{name}_row"], z[f"{name}_col"])),
                         shape=shape).toarray().astype(np.float64)


def _normalise(W):
    """Scale each postsynaptic cell's inputs to sum to one, as in circuit.py."""
    s = W.sum(axis=0, keepdims=True)
    s[s == 0] = 1.0
    return W / s


class MushroomBody:
    def __init__(self, path="data/mb_circuit.npz", sparsity=KC_SPARSITY,
                 learning_rate=LEARNING_RATE, decay=DECAY, lateral=LATERAL,
                 shuffle_kc_mbon=False, seed=0):
        z = np.load(path)
        self.W_pn_kc = _normalise(_dense(z, "pn_kc"))
        self.W_kc_mbon_raw = _dense(z, "kc_mbon")
        self.W_dan_mbon = _normalise(_dense(z, "dan_mbon"))
        self.W_mbon_mbon = _normalise(_dense(z, "mbon_mbon"))
        self.valence = z["mbon_valence"].astype(np.float64)
        self.punish_mask = z["dan_is_punishment"].astype(bool)

        if shuffle_kc_mbon:
            # degree-preserving rewire: same number of partners per KC and per
            # MBON, different partners. A lesion control, not a topology claim.
            rng = np.random.default_rng(seed)
            W = self.W_kc_mbon_raw
            flat = W[W > 0]
            rng.shuffle(flat)
            out = np.zeros_like(W)
            out[W > 0] = flat
            for _ in range(3):
                perm = rng.permutation(out.shape[0])
                out = out[perm]
            self.W_kc_mbon_raw = out

        self.W_kc_mbon = _normalise(self.W_kc_mbon_raw)
        self.n_kc, self.n_mbon = self.W_kc_mbon.shape
        self.n_pn = self.W_pn_kc.shape[0]
        self.n_dan = self.W_dan_mbon.shape[0]
        self.k = max(1, int(round(sparsity * self.n_kc)))
        self.lr, self.decay, self.lateral = learning_rate, decay, lateral
        self.reset()

    def reset(self):
        self.gain = np.ones((self.n_kc, self.n_mbon))

    # ------------------------------------------------------------------
    def pn_drive(self, pn_activity: np.ndarray) -> np.ndarray:
        """Projection-neuron input reaching each Kenyon cell.

        Separated out because it is linear: if an odour blend is a weighted sum
        of fixed odours, so is the drive, and callers can precompute one vector
        per odour instead of redoing this matrix product every step.
        """
        return self.W_pn_kc.T @ pn_activity

    def kenyon(self, pn_activity: np.ndarray) -> np.ndarray:
        return self.kenyon_from_drive(self.pn_drive(pn_activity))

    def kenyon_from_drive(self, drive: np.ndarray) -> np.ndarray:
        """Sparse odour code. k-winners-take-all stands in for APL inhibition."""
        if drive.max() <= 0:
            return np.zeros(self.n_kc)
        cut = np.partition(drive, -self.k)[-self.k]
        kc = np.where(drive >= cut, drive, 0.0)
        top = kc.max()
        return kc / top if top > 0 else kc

    def dopamine(self, punish: float, reward: float) -> np.ndarray:
        dan = np.zeros(self.n_dan)
        dan[self.punish_mask] = punish
        dan[~self.punish_mask] = reward
        return self.W_dan_mbon.T @ dan          # only where anatomy allows

    def present(self, pn_activity, punish=0.0, reward=0.0, learn=True, drive=None):
        kc = self.kenyon_from_drive(self.pn_drive(pn_activity) if drive is None else drive)
        mbon = (self.W_kc_mbon * self.gain).T @ kc
        mbon = mbon + self.lateral * (self.W_mbon_mbon.T @ mbon)
        da = self.dopamine(punish, reward)
        if learn and (punish or reward):
            self.gain -= self.lr * np.outer(kc, da)
        if learn:
            self.gain += self.decay * (1.0 - self.gain)
            np.clip(self.gain, GAIN_FLOOR, 1.0, out=self.gain)
        return dict(kc=kc, mbon=mbon, dopamine=da,
                    valence=float(mbon @ self.valence),
                    sparsity=float((kc > 0).mean()))

    def valence_of(self, pn_activity) -> float:
        """Read out without learning -- the test trial."""
        return self.present(pn_activity, learn=False)["valence"]

    @property
    def depression(self) -> float:
        """How far the plastic synapses have moved from naive, 0..1."""
        mask = self.W_kc_mbon_raw > 0
        return float(1.0 - self.gain[mask].mean())
