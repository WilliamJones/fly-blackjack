"""Blackjack, played by a mushroom body -- and graded against the real chart.

Why this task and not poker (see the note at the bottom of poker.py): the
situation space is tiny -- your total, whether it is soft, the dealer's upcard --
so every spot is revisited hundreds of times, the outcome lands within the same
hand, and the correct action per spot is stable. That is the shape of thing a
one-shot associative learner can average its way into. Poker had none of those.

And there is a ground truth. Basic strategy is a published table of the right
action in every spot, so the claim is not "it wins more" (noisy) but "its
choices agree with the chart more" (not noisy). We report both.

Rules: infinite deck, dealer stands on soft 17, double on any first two cards,
no splits, no surrender, blackjack pays 3:2. The chart below is the standard
one for those rules.
"""
from __future__ import annotations
import numpy as np

from .model import MushroomBody

HIT, STAND, DOUBLE = 0, 1, 2
ACTION_NAMES = ["hit", "stand", "double"]
CARDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]      # 11 = ace


def total(cards):
    """(best total, is_soft). Aces count 11 until that busts."""
    t = sum(cards)
    aces = cards.count(11)
    while t > 21 and aces:
        t -= 10
        aces -= 1
    return t, (aces > 0)


# ----------------------------------------------------------- basic strategy
def basic_strategy(player_total, soft, upcard, can_double):
    """Dealer stands on soft 17. Returns HIT, STAND or DOUBLE."""
    up = upcard
    if soft:
        if player_total >= 19:
            return STAND
        if player_total == 18:
            if up in (3, 4, 5, 6):
                return DOUBLE if can_double else STAND
            return STAND if up in (2, 7, 8) else HIT
        if player_total == 17:
            return DOUBLE if (can_double and up in (3, 4, 5, 6)) else HIT
        if player_total in (15, 16):
            return DOUBLE if (can_double and up in (4, 5, 6)) else HIT
        # soft 13, 14
        return DOUBLE if (can_double and up in (5, 6)) else HIT
    if player_total >= 17:
        return STAND
    if 13 <= player_total <= 16:
        return STAND if up <= 6 else HIT
    if player_total == 12:
        return STAND if up in (4, 5, 6) else HIT
    if player_total == 11:
        return DOUBLE if (can_double and up <= 10) else HIT
    if player_total == 10:
        return DOUBLE if (can_double and up <= 9) else HIT
    if player_total == 9:
        return DOUBLE if (can_double and up in (3, 4, 5, 6)) else HIT
    return HIT


def chart_cells():
    """Every (total, soft, upcard) spot a first decision can land on."""
    cells = []
    for up in range(2, 12):
        for t in range(5, 22):
            cells.append((t, False, up))
        for t in range(13, 22):
            cells.append((t, True, up))
    return cells


# --------------------------------------------------------------- the table
def play_hand(policy, rng):
    """policy(total, soft, upcard, can_double, legal) -> action.
    Returns (net units, list of (state, action) decisions)."""
    draw = lambda: CARDS[rng.integers(len(CARDS))]
    player = [draw(), draw()]
    dealer = [draw(), draw()]
    up = dealer[0]
    decisions = []

    pt, ps = total(player)
    dt, _ = total(dealer)
    if pt == 21 and dt == 21:
        return 0.0, decisions
    if pt == 21:
        return 1.5, decisions
    if dt == 21:
        return -1.0, decisions

    bet = 1.0
    first = True
    while True:
        pt, ps = total(player)
        legal = [HIT, STAND] + ([DOUBLE] if first else [])
        a = policy(pt, ps, up, first, legal)
        decisions.append(((pt, ps, up, first), a))
        if a == STAND:
            break
        player.append(draw())
        pt, ps = total(player)
        if a == DOUBLE:
            bet = 2.0
            break
        first = False
        if pt > 21:
            return -bet, decisions
    pt, _ = total(player)
    if pt > 21:
        return -bet, decisions
    while True:
        dt, ds = total(dealer)
        if dt >= 17:
            break
        dealer.append(draw())
    dt, _ = total(dealer)
    if dt > 21 or pt > dt:
        return bet, decisions
    if pt < dt:
        return -bet, decisions
    return 0.0, decisions


# ------------------------------------------------------------- the encoder
class Situation:
    """A spot plus a candidate action, as projection-neuron input.

    Ordered features (total, upcard) get overlapping codes so neighbouring
    totals smell alike -- what the fly learns about hard 15 carries a little
    to hard 14 and 16, which is how the chart is actually shaped. The action
    gets the widest code, because the choice is the thing being learned.
    """
    WIDTH = {"total": 26, "upcard": 22, "action": 36, "soft": 12, "first": 8}

    def __init__(self, mb: MushroomBody, seed=20260914):
        self.mb = mb
        rng = np.random.default_rng(seed)
        n = mb.n_pn
        self.drive = {}
        for feature, levels, ordered in [("total", 18, True), ("upcard", 10, True),
                                         ("action", 3, False), ("soft", 2, False),
                                         ("first", 2, False)]:
            width = self.WIDTH[feature]
            pool = rng.permutation(n)
            for level in range(levels):
                pn = np.zeros(n)
                if ordered:
                    start = int(level * width * 0.55)
                    cells = pool[start:start + width]
                else:
                    cells = rng.choice(n, width, replace=False)
                amp = 1.3 if feature == "action" else 1.0
                pn[cells] = amp * rng.uniform(0.7, 1.0, size=len(cells))
                self.drive[(feature, level)] = mb.pn_drive(pn)

    def encode(self, pt, soft, up, first, action):
        d = (self.drive[("total", int(np.clip(pt - 4, 0, 17)))]
             + self.drive[("upcard", up - 2)]
             + self.drive[("action", action)]
             + self.drive[("soft", int(soft))]
             + self.drive[("first", int(first))])
        return d


# ---------------------------------------------------------------- the fly
class FlyPlayer:
    def __init__(self, seed=0, lr=0.05, decay=0.00005, explore=0.35,
                 explore_floor=0.03, explore_halflife=3000,
                 learn=True, dopamine=True, credit_decay=1.0):
        self.mb = MushroomBody()
        self.mb.lr = lr
        self.mb.decay = decay
        if not dopamine:
            self.mb.W_dan_mbon = np.zeros_like(self.mb.W_dan_mbon)
        self.sit = Situation(self.mb)
        self.rng = np.random.default_rng(seed)
        self.learn = learn
        self.explore0, self.explore_floor, self.halflife = explore, explore_floor, explore_halflife
        self.credit_decay = credit_decay
        self.hands = 0

    def exploration(self):
        return self.explore_floor + (self.explore0 - self.explore_floor) * 0.5 ** (self.hands / self.halflife)

    def value(self, pt, soft, up, first, action):
        return self.mb.present(None, learn=False,
                               drive=self.sit.encode(pt, soft, up, first, action))["valence"]

    def greedy(self, pt, soft, up, first, legal):
        vals = {a: self.value(pt, soft, up, first, a) for a in legal}
        return max(vals, key=vals.get)

    def policy(self, pt, soft, up, first, legal):
        if self.rng.random() < self.exploration():
            return int(self.rng.choice(legal))
        return self.greedy(pt, soft, up, first, legal)

    def outcome(self, net, decisions):
        self.hands += 1
        if not self.learn or not decisions:
            return
        strength = float(np.clip(abs(net) / 2.0, 0.15, 1.0))
        reward = strength if net > 0 else 0.0
        punish = strength if net < 0 else 0.0
        if net == 0:
            return
        # Equal credit by default. Weighting the LAST decision most (an earlier
        # version used 0.7**i) is a trap in blackjack: the last decision of a
        # busted hand is always a hit and of a won hand is nearly always a stand,
        # so "hit = lose, stand = win" became a global correlation that drowned
        # every spot-specific lesson and the fly converged on always standing.
        for i, ((pt, soft, up, first), a) in enumerate(reversed(decisions)):
            w = self.credit_decay ** i
            self.mb.present(None, punish=punish * w, reward=reward * w, learn=True,
                            drive=self.sit.encode(pt, soft, up, first, a))

    # -- grading -------------------------------------------------------
    def agreement(self):
        """Fraction of chart spots where the greedy choice matches basic strategy."""
        hits = 0
        cells = chart_cells()
        for (t, soft, up) in cells:
            legal = [HIT, STAND, DOUBLE]
            want = basic_strategy(t, soft, up, True)
            got = self.greedy(t, soft, up, True, legal)
            hits += (want == got)
        return hits / len(cells)


# ----------------------------------------------------------------- session
def session(fly, hands=20000, seed=0, block=2000):
    rng = np.random.default_rng(seed)
    nets, blocks, agree = [], [], []
    for i in range(hands):
        net, decisions = play_hand(fly.policy, rng)
        fly.outcome(net, decisions)
        nets.append(net)
        if (i + 1) % block == 0:
            blocks.append(float(np.mean(nets[-block:])))
            agree.append(fly.agreement())
    return np.array(nets), blocks, agree


def reference_winrates(rng, hands=20000):
    """What the chart, a random player, and a never-bust player earn."""
    out = {}
    out["basic strategy"] = np.mean([play_hand(lambda t, s, u, f, L: basic_strategy(t, s, u, f), rng)[0]
                                     for _ in range(hands)])
    out["random"] = np.mean([play_hand(lambda t, s, u, f, L: int(rng.choice(L)), rng)[0]
                             for _ in range(hands)])
    out["always stand"] = np.mean([play_hand(lambda t, s, u, f, L: STAND, rng)[0]
                                   for _ in range(hands)])
    return out


# ---------------------------------------------------------------------------
# RESULT (3 seeds x 30,000 hands each; controls ran FIRST)
#
#   condition                        winrate            chart agreement
#   untrained greedy                 -45%               45.0% (flat)
#   no learning + exploration decay  -45%  -> -45%      45.0% (flat)
#   dopamine cut + exploration decay -46%  -> -47%      45.0% (flat)
#   connectome + dopamine            -45%  -> -16%      44.6% -> 48.1%
#                                                       (49.8% weighted by how
#                                                        often each spot occurs)
#
# The winrate gain is real and entirely dopamine-dependent: the control with
# the same exploration schedule does not move. This is the opposite of the
# poker result, and for the reason given in the module docstring -- small
# revisited state space, outcome within the hand.
#
# WHAT IT LEARNED is more honest than "basic strategy". The untrained fly hits
# 239 of 260 spots and busts constantly. The trained fly stands on ~227 and
# hits only on hard 10 and 11 -- the two totals that cannot bust. It learned
# "don't bust", which is the first thing a human learns at the table, and its
# winrate landed at -16%, which is exactly the always-stand policy. It has not
# learned doubling at all (0 of 39 spots): doubles are rare and high-variance,
# so their signal is the one this circuit is worst at extracting.
#
# Two bugs on the way, both mine and both instructive:
#   1. Memory decay was tuned for odour conditioning and erased a spot-specific
#      lesson before that spot came round again, so only the global lesson
#      ("hitting hurts") survived. Fixed by lengthening memory.
#   2. Crediting the LAST decision most is a trap here: the last decision of a
#      busted hand is always a hit and of a won hand nearly always a stand, so
#      "hit = lose" became a global correlation. Fixed with equal credit, and
#      chart agreement started rising only after that.
