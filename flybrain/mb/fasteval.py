"""Vectorised 7-card evaluator. Scores thousands of hands in one numpy call.

Exists purely for speed: sampling equity with the readable evaluator in
cards.py spent 96% of the poker loop inside itertools.combinations. Validated
against that evaluator on random hands -- cards.py stays the reference.
"""
from __future__ import annotations
import numpy as np

# straight lookup: 13-bit rank-presence mask -> high card rank, or -1
_STRAIGHT = np.full(1 << 13, -1, np.int8)
for _mask in range(1 << 13):
    for _hi in range(12, 3, -1):
        if all(_mask >> (_hi - k) & 1 for k in range(5)):
            _STRAIGHT[_mask] = _hi
            break
    else:
        # the wheel: A5432
        if all(_mask >> r & 1 for r in (12, 0, 1, 2, 3)):
            _STRAIGHT[_mask] = 3

# how many distinct ranks actually matter, per category
_SIGNIFICANT = np.array([5, 4, 3, 3, 1, 5, 2, 2, 1], np.int64)


def pack(cat, keys):
    """cat and up to five ranks into one sortable integer."""
    out = cat.astype(np.int64) << 25
    for i in range(5):
        out |= (keys[:, i].astype(np.int64) & 0x1F) << (20 - 5 * i)
    return out


def eval7(ranks: np.ndarray, suits: np.ndarray) -> np.ndarray:
    """ranks, suits: (N, 7) ints. Returns (N,) comparable scores."""
    n = ranks.shape[0]
    rows = np.arange(n)[:, None]

    counts = np.zeros((n, 13), np.int8)
    np.add.at(counts, (rows, ranks), 1)
    suit_counts = np.zeros((n, 4), np.int8)
    np.add.at(suit_counts, (rows, suits), 1)

    rank_mask = (counts > 0).astype(np.int64) @ (1 << np.arange(13))
    straight_hi = _STRAIGHT[rank_mask]

    flush_suit = suit_counts.argmax(axis=1)
    has_flush = suit_counts.max(axis=1) >= 5
    in_flush = suits == flush_suit[:, None]
    flush_mask = np.where(in_flush, 1 << ranks, 0).sum(axis=1)
    sf_hi = _STRAIGHT[flush_mask]
    has_sf = has_flush & (sf_hi >= 0)

    # distinct ranks ordered by (how many, then rank), both descending
    order = np.argsort(-(counts.astype(np.int64) * 16 + np.arange(13)), axis=1)
    ordered = order[:, :5]
    ordered_counts = np.take_along_axis(counts, ordered, axis=1)

    shape0 = ordered_counts[:, 0]
    shape1 = ordered_counts[:, 1]
    cat = np.zeros(n, np.int64)
    cat = np.where(shape0 == 2, 1, cat)
    cat = np.where((shape0 == 2) & (shape1 == 2), 2, cat)
    cat = np.where(shape0 == 3, 3, cat)
    cat = np.where(straight_hi >= 0, 4, cat)
    cat = np.where(has_flush, 5, cat)
    cat = np.where((shape0 == 3) & (shape1 >= 2), 6, cat)
    cat = np.where(shape0 == 4, 7, cat)
    cat = np.where(has_sf, 8, cat)

    keys = ordered.copy()
    # straights and flushes are scored by their own cards, not by rank counts
    top5_flush = np.zeros((n, 5), np.int64)
    if has_flush.any():
        idx = np.flatnonzero(has_flush)
        fr = np.where(in_flush[idx], ranks[idx], -1)
        fr = -np.sort(-fr, axis=1)[:, :5]
        top5_flush[idx] = fr
    keys = np.where((cat == 5)[:, None], top5_flush, keys)
    line = np.zeros((n, 5), np.int64)
    line[:, 0] = np.where(cat == 8, sf_hi, straight_hi)
    keys = np.where(((cat == 4) | (cat == 8))[:, None], line, keys)

    sig = _SIGNIFICANT[cat]
    keys = np.where(np.arange(5)[None, :] < sig[:, None], keys, 0)
    return pack(cat, keys)


_R = {c: i for i, c in enumerate("23456789TJQKA")}
_S = {c: i for i, c in enumerate("cdhs")}


def to_arrays(hands):
    """List of card-string lists -> (ranks, suits) arrays."""
    r = np.array([[_R[c[0]] for c in h] for h in hands], np.int64)
    s = np.array([[_S[c[1]] for c in h] for h in hands], np.int64)
    return r, s
