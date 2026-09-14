"""Heads-up poker, played by a mushroom body.

The mapping is the interesting part, and it is not a stretch. A mushroom body
turns a sensory pattern into a predicted value, and updates that prediction from
dopamine. That is a value function. So:

    poker situation + a candidate action  ->  pattern over 682 projection neurons
                                          ->  ~72 Kenyon cells (sparse "smell")
                                          ->  MBON valence = how good this feels
    win the pot   -> reward neurons fire  -> that pattern feels better next time
    lose the pot  -> punishment neurons   -> it feels worse

The fly plays by imagining each legal action, smelling the result, and taking
the one that smells best. NOTHING IS PRE-TRAINED. It sits down knowing nothing
and every hand is a conditioning trial.

Because similar situations share Kenyon cells, what it learns about one hand
bleeds into hands that look like it -- which is generalisation, and is also why
it can be exploited. It will not become good at poker. It will become visibly
less bad, which is the honest and more interesting claim.
"""
from __future__ import annotations
import numpy as np

from .cards import DECK, best_of, equity
from .model import MushroomBody
from .odour import Odours

FOLD, CALL, RAISE, ALLIN = 0, 1, 2, 3
ACTION_NAMES = ["fold", "call", "raise", "all-in"]
STREETS = ["preflop", "flop", "turn", "river"]

SMALL_BLIND, BIG_BLIND, STACK = 1, 2, 100
MAX_RAISES = 3


# ---------------------------------------------------------------- the table
class Hand:
    """One heads-up hand. Seat 0 posts the small blind and acts first preflop."""

    def __init__(self, rng):
        self.rng = rng
        deck = list(DECK)
        rng.shuffle(deck)
        self.hole = [deck[0:2], deck[2:4]]
        self.deck = deck[4:]
        self.board: list[str] = []
        self.street = 0
        self.pot = SMALL_BLIND + BIG_BLIND
        self.committed = [SMALL_BLIND, BIG_BLIND]
        self.stacks = [STACK - SMALL_BLIND, STACK - BIG_BLIND]
        self.folded = None
        self.all_in = False

    # -- helpers -------------------------------------------------------
    def to_call(self, seat):
        return max(0, self.committed[1 - seat] - self.committed[seat])

    def legal(self, seat, raises):
        acts = [CALL]
        if self.to_call(seat) > 0:
            acts.append(FOLD)
        if raises < MAX_RAISES and self.stacks[seat] > self.to_call(seat):
            acts.append(RAISE)
        if self.stacks[seat] > 0:
            acts.append(ALLIN)
        return acts

    def apply(self, seat, action, raises):
        need = self.to_call(seat)
        if action == FOLD:
            self.folded = seat
            return 0
        if action == CALL:
            pay = min(need, self.stacks[seat])
        elif action == RAISE:
            pay = min(need + max(BIG_BLIND, self.pot // 2), self.stacks[seat])
            raises += 1
        else:
            pay = self.stacks[seat]
            raises += 1
        self.stacks[seat] -= pay
        self.committed[seat] += pay
        self.pot += pay
        if self.stacks[seat] == 0:
            self.all_in = True
        return raises

    def deal(self):
        n = {1: 3, 2: 1, 3: 1}[self.street]
        self.board += self.deck[:n]
        self.deck = self.deck[n:]

    def showdown(self):
        a = best_of(self.hole[0] + self.board)
        b = best_of(self.hole[1] + self.board)
        return 0 if a > b else (1 if b > a else -1)


def play_hand(players, rng, button=0, trace=None):
    """Returns net chips for seat 0. `players` are callables (hand, seat, ...)."""
    h = Hand(rng)
    for street in range(4):
        h.street = street
        if street > 0:
            h.deal()
        if h.all_in:
            continue
        raises = 0
        # preflop the small blind acts first; afterwards the big blind does
        turn = 0 if street == 0 else 1
        acted = 0
        while True:
            legal = h.legal(turn, raises)
            act = players[turn](h, turn, legal, raises)
            if trace is not None:
                trace.append((turn, street, act))
            new_raises = h.apply(turn, act, raises)
            if h.folded is not None:
                won = h.pot - h.committed[0] if h.folded == 1 else -h.committed[0]
                return won
            aggressive = new_raises > raises
            raises = new_raises
            acted += 1
            turn = 1 - turn
            if h.all_in:
                break
            if acted >= 2 and not aggressive and h.to_call(turn) == 0:
                break
    result = h.showdown()
    if result == 0:
        return h.pot - h.committed[0]
    if result == 1:
        return -h.committed[0]
    return h.pot // 2 - h.committed[0]


# ------------------------------------------------------------- the encoder
class Situation:
    """Turn a poker spot plus a candidate action into projection-neuron input.

    Every feature value gets its own set of glomeruli, the same way an odour
    does. Overlapping glomeruli between nearby values is deliberate: 'equity
    0.6' and 'equity 0.7' should smell similar, so what the fly learns about one
    carries to the other.
    """

    FEATURES = {
        "street": 4, "equity": 10, "potodds": 5, "facing": 5,
        "seat": 2, "raises": 4, "action": 4, "stack": 4,
    }

    # How many projection neurons each feature value drives. The action and the
    # hand strength get the most, because they are what the decision turns on --
    # with every feature equal, two different actions produced 93% identical
    # Kenyon codes and the fly could not tell its own choices apart.
    WIDTH = {"action": 34, "equity": 22, "street": 10, "potodds": 10,
             "facing": 10, "seat": 6, "raises": 8, "stack": 8}

    def __init__(self, mb: MushroomBody, odours: Odours | None = None, seed=20260914):
        self.od = odours or Odours()
        self.mb = mb
        rng = np.random.default_rng(seed)
        self.drive = {}
        for feature, levels in self.FEATURES.items():
            # ordered features get overlapping codes, so neighbouring levels
            # smell alike and what the fly learns about one carries to the next
            width = self.WIDTH[feature]
            pool = rng.permutation(self.od.n_pn)
            ordered = feature in ("equity", "potodds", "facing", "stack", "street", "raises")
            for level in range(levels):
                pn = np.zeros(self.od.n_pn)
                if ordered:
                    start = int(level * width * 0.6)
                    cells = pool[start:start + width]
                else:
                    cells = rng.choice(self.od.n_pn, width, replace=False)
                amp = 1.3 if feature == "action" else 1.0
                pn[cells] = amp * rng.uniform(0.7, 1.0, size=len(cells))
                self.drive[(feature, level)] = mb.pn_drive(pn)

    @staticmethod
    def bucket(x, n, lo=0.0, hi=1.0):
        return int(np.clip((x - lo) / (hi - lo) * n, 0, n - 1))

    def encode(self, h: Hand, seat: int, action: int, raises: int, eq: float):
        need = h.to_call(seat)
        odds = need / max(1, h.pot + need)
        levels = {
            "street": h.street,
            "equity": self.bucket(eq, 10),
            "potodds": self.bucket(odds, 5, 0, 0.6),
            "facing": self.bucket(need / max(1, h.pot), 5, 0, 1.5),
            "seat": seat,
            "raises": min(raises, 3),
            "action": action,
            "stack": self.bucket(h.stacks[seat] / STACK, 4),
        }
        total = np.zeros(self.mb.n_kc)
        for feature, level in levels.items():
            total += self.drive[(feature, level)]
        return total


# ---------------------------------------------------------------- the fly
class FlyPlayer:
    def __init__(self, mb: MushroomBody | None = None, seed=0, explore=0.35,
                 learn=True, dopamine=True, lr=0.08, explore_floor=0.06,
                 explore_halflife=1500):
        self.mb = mb or MushroomBody()
        self.mb.lr = lr
        self.explore0 = explore
        self.explore_floor = explore_floor
        self.explore_halflife = explore_halflife
        if not dopamine:
            self.mb.W_dan_mbon = np.zeros_like(self.mb.W_dan_mbon)
        self.sit = Situation(self.mb)
        self.rng = np.random.default_rng(seed)
        self.explore = explore
        self.learn = learn
        self.memory: list[np.ndarray] = []
        self.hands = 0
        self._eq_cache: dict = {}

    def _equity(self, h, seat):
        # the board only changes between streets, so sample once per street
        key = (tuple(h.hole[seat]), tuple(h.board))
        if key not in self._eq_cache:
            if len(self._eq_cache) > 4000:
                self._eq_cache.clear()
            self._eq_cache[key] = equity(h.hole[seat], h.board, self.rng, 80)
        return self._eq_cache[key]

    def __call__(self, h: Hand, seat: int, legal, raises):
        eq = self._equity(h, seat)
        drives = {a: self.sit.encode(h, seat, a, raises, eq) for a in legal}
        values = {a: self.mb.present(None, learn=False, drive=d)["valence"]
                  for a, d in drives.items()}
        if self.rng.random() < self.exploration():
            action = int(self.rng.choice(legal))
        else:
            action = max(values, key=values.get)
        self.memory.append(drives[action])
        return action

    def exploration(self) -> float:
        """Curious early, settled later -- otherwise the first arbitrary
        preference the untrained circuit happens to hold becomes permanent."""
        decay = 0.5 ** (self.hands / self.explore_halflife)
        return self.explore_floor + (self.explore0 - self.explore_floor) * decay

    def outcome(self, net: float):
        """Dopamine. Winning the pot is reward, losing it is punishment, and a
        big pot teaches more than a small one."""
        self.hands += 1
        if self.learn and self.memory:
            strength = float(np.clip(abs(net) / 25.0, 0.05, 1.0))
            reward = strength if net > 0 else 0.0
            punish = strength if net < 0 else 0.0
            # the last decision carries most of the blame
            for i, drive in enumerate(reversed(self.memory)):
                w = 0.65 ** i
                self.mb.present(None, punish=punish * w, reward=reward * w,
                                learn=True, drive=drive)
        self.memory.clear()


# ------------------------------------------------------------- opponents
class RandomBot:
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def __call__(self, h, seat, legal, raises): return int(self.rng.choice(legal))
    def outcome(self, net): pass


class CallingStation:
    def __call__(self, h, seat, legal, raises): return CALL
    def outcome(self, net): pass


class TightAggressive:
    """A simple honest player: bets good hands, folds bad ones to pressure."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed); self._c = {}
    def __call__(self, h, seat, legal, raises):
        key = (tuple(h.hole[seat]), tuple(h.board))
        if key not in self._c:
            if len(self._c) > 4000: self._c.clear()
            self._c[key] = equity(h.hole[seat], h.board, self.rng, 80)
        eq = self._c[key]
        need = h.to_call(seat)
        if eq > 0.72 and RAISE in legal and raises < 2:
            return RAISE
        if need == 0:
            return CALL
        odds = need / max(1, h.pot + need)
        if eq > odds + 0.08:
            return CALL
        return FOLD if FOLD in legal else CALL
    def outcome(self, net): pass


# ----------------------------------------------------------------- session
def session(fly, villain, hands=2000, seed=0, block=250):
    """Alternate seats so position cancels out, and report winrate per block."""
    rng = np.random.default_rng(seed)
    nets, blocks = [], []
    for i in range(hands):
        if i % 2 == 0:
            net = play_hand([fly, villain], rng)
        else:
            net = -play_hand([villain, fly], rng)
        fly.outcome(net)
        if hasattr(villain, "outcome"):
            villain.outcome(-net)
        nets.append(net)
        if len(nets) % block == 0:
            blocks.append(float(np.mean(nets[-block:])))
    return np.array(nets), blocks


# ---------------------------------------------------------------------------
# RESULT: THIS DOES NOT WORK, AND THE CONTROL IS WHY.
#
# Over 2,400 hands against a calling station the fly appears to improve, from
# about -15 bb/hand to about -8. It is not learning. Running the same session
# with the dopamine disconnected improves by +6.8 and with dopamine by +6.0 --
# identical within noise.
#
# The diagnosis is clean. An untrained fly playing greedily is already close to
# break-even (-0.38 bb/hand; it calls about 85% of the time). Every bit of the
# loss came from EXPLORATION -- random all-ins at 35% early on. As exploration
# decayed the fly simply returned to its untrained baseline, and that return is
# the whole "learning curve".
#
# Why poker is the wrong task for this circuit: a mushroom body is a one-shot
# associative learner. Odour, then shock, immediately, every time. Poker is
# high-variance sequential credit assignment -- a single hand's result says
# almost nothing about whether the decision was good, and the same decision
# wins and loses at random. Depressing synapses on a signal that flips sign at
# random just averages back to nothing, which is exactly what the 2% depression
# figure shows.
#
# What would work, in order:
#   1. A task with immediate, low-variance, one-decision-per-outcome feedback.
#      Rock paper scissors is the extreme case and is the right fit.
#   2. Poker with a coaching signal -- reward the decision if it was +EV given
#      the equity actually held, rather than whether the pot was won. That
#      learns, but the honest claim becomes "taught by a coach" rather than
#      "learned from its mistakes".
#   3. Far more hands with a value baseline, i.e. proper RL, at which point the
#      mushroom body is doing none of the work.
#
# Kept in the repo because the engine, the vectorised evaluator and the sparse
# conjunctive encoder are all sound and reusable, and because a negative result
# with a control that explains it is worth more than a tuned curve.
