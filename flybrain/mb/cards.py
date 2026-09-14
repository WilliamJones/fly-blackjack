"""Cards and hand evaluation. Plain poker, nothing to do with the connectome."""
from __future__ import annotations
import itertools
import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = [r + s for r in RANKS for s in SUITS]
RANK_OF = {r: i for i, r in enumerate(RANKS)}

CATEGORIES = ["high card", "pair", "two pair", "trips", "straight",
              "flush", "full house", "quads", "straight flush"]


def _straight_high(ranks: set[int]) -> int | None:
    """Highest card of a straight, or None. Ace plays low for the wheel."""
    if 12 in ranks:
        ranks = ranks | {-1}
    best = None
    for high in range(12, 2, -1):
        if all(high - k in ranks for k in range(5)):
            best = high
            break
    if best is None and all(k in ranks for k in (-1, 0, 1, 2, 3)):
        best = 3
    return best


def score5(cards) -> tuple:
    """Comparable score for exactly five cards. Bigger is better."""
    rs = sorted((RANK_OF[c[0]] for c in cards), reverse=True)
    suits = [c[1] for c in cards]
    flush = len(set(suits)) == 1
    straight = _straight_high(set(rs))
    counts: dict[int, int] = {}
    for r in rs:
        counts[r] = counts.get(r, 0) + 1
    # order by how many, then by rank -- so trips beat the pair beside them
    grouped = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    shape = [n for _, n in grouped]
    kickers = tuple(r for r, _ in grouped)

    if flush and straight is not None:
        return (8, straight)
    if shape[0] == 4:
        return (7,) + kickers
    if shape[:2] == [3, 2]:
        return (6,) + kickers
    if flush:
        return (5,) + tuple(rs)
    if straight is not None:
        return (4, straight)
    if shape[0] == 3:
        return (3,) + kickers
    if shape[:2] == [2, 2]:
        return (2,) + kickers
    if shape[0] == 2:
        return (1,) + kickers
    return (0,) + tuple(rs)


def best_of(cards) -> tuple:
    """Best five-card score from five, six or seven cards."""
    if len(cards) == 5:
        return score5(cards)
    return max(score5(c) for c in itertools.combinations(cards, 5))


def category(cards) -> str:
    return CATEGORIES[best_of(cards)[0]]


def equity(hole, board, rng, samples=120) -> float:
    """Rough chance of beating one random hand, by sampling the rest.

    Deliberately cheap: the fly does not get a solver, it gets a noisy feel for
    how strong its hand is, which is what the encoder needs. Runs through the
    vectorised evaluator in fasteval.py; score5/best_of above stay the readable
    reference that fast path is validated against.
    """
    from .fasteval import eval7, _R, _S
    known = set(hole) | set(board)
    deck = [c for c in DECK if c not in known]
    deck_r = np.fromiter((_R[c[0]] for c in deck), np.int64, len(deck))
    deck_s = np.fromiter((_S[c[1]] for c in deck), np.int64, len(deck))

    need = 5 - len(board)
    # one random permutation per trial; the first need+2 cards are the draw
    pick = np.argsort(rng.random((samples, len(deck))), axis=1)[:, :need + 2]
    dr, ds = deck_r[pick], deck_s[pick]

    tile = lambda vals: np.tile(np.array(vals, np.int64).reshape(1, -1), (samples, 1))
    board_r = tile([_R[c[0]] for c in board])
    board_s = tile([_S[c[1]] for c in board])
    full_r = np.hstack([board_r, dr[:, :need]])
    full_s = np.hstack([board_s, ds[:, :need]])

    me = eval7(np.hstack([tile([_R[c[0]] for c in hole]), full_r]),
               np.hstack([tile([_S[c[1]] for c in hole]), full_s]))
    them = eval7(np.hstack([dr[:, need:], full_r]),
                 np.hstack([ds[:, need:], full_s]))
    return float(((me > them).sum() + 0.5 * (me == them).sum()) / samples)
