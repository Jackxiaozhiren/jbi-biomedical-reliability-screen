"""Tests for the ReliK baseline."""
import numpy as np
import pytest
import torch

from baseline_relik import build_sets, relik_scores, relik_curve
from stats_core import realized_error_metrics


class DummyRelik:
    """Score = 3.0 + noise for marked true triples, else 0.0 + noise."""

    def __init__(self, true_triples):
        self.true = {tuple(map(int, t)) for t in true_triples}

    def score_hrt(self, hrt):
        rng = np.random.default_rng(12345)
        base = np.array([3.0 if tuple(map(int, r)) in self.true else 0.0
                         for r in hrt.tolist()])
        return torch.tensor(base + rng.uniform(0.0, 0.01, size=base.shape))


def _setup(n_ent=50, n_rel=5):
    rng = np.random.default_rng(7)
    true = set()
    for _ in range(80):
        true.add((int(rng.integers(n_ent)), int(rng.integers(n_rel)),
                  int(rng.integers(n_ent))))
    true = list(true)
    return true, n_ent, n_rel


def test_relik_separates_true_and_false():
    true, n_ent, n_rel = _setup()
    model = DummyRelik(true)
    hr = {}; rt = {}
    for h, r, t in true:
        hr.setdefault((h, r), set()).add(t)
        rt.setdefault((r, t), set()).add(h)
    true_arr = np.asarray(true)
    false_arr = np.asarray([(int(np.random.randint(n_ent)), int(np.random.randint(n_rel)),
                             int(np.random.randint(n_ent)))
                            for _ in range(60)])
    s_true = relik_scores(model, true_arr[:, 0], true_arr[:, 1], true_arr[:, 2],
                          hr, rt, n_ent, n_rel, m=25)
    s_false = relik_scores(model, false_arr[:, 0], false_arr[:, 1], false_arr[:, 2],
                           hr, rt, n_ent, n_rel, m=25)
    assert np.mean(s_true) > 0.8
    assert np.mean(s_false) < 0.2
    assert np.mean(s_true) > np.mean(s_false) + 0.5


def test_relik_curve_screens_mostly_true():
    true, n_ent, n_rel = _setup()
    model = DummyRelik(true)
    hr = {}; rt = {}
    for h, r, t in true:
        hr.setdefault((h, r), set()).add(t)
        rt.setdefault((r, t), set()).add(h)
    rng = np.random.default_rng(1)
    # build a mixed pool: 80 true + 120 false, split half cal / half eval
    true_arr = np.asarray(true)
    fh = rng.integers(0, n_ent, 120)
    fr = rng.integers(0, n_rel, 120)
    ft = rng.integers(0, n_ent, 120)
    h = np.concatenate([true_arr[:, 0], fh])
    r = np.concatenate([true_arr[:, 1], fr])
    t = np.concatenate([true_arr[:, 2], ft])
    labels = np.concatenate([np.ones(len(true_arr), bool), np.zeros(120, bool)])
    perm = rng.permutation(labels.size)
    h, r, t, labels = h[perm], r[perm], t[perm], labels[perm]
    n = labels.size
    half = n // 2
    rows = relik_curve(h[:half], r[:half], t[:half], labels[:half],
                       h[half:], r[half:], t[half:], labels[half:],
                       model, hr, rt, n_ent, n_rel, m=25, seed=3)
    for row in rows:
        if row["n_screened_in"] > 0:
            assert row["precision"] >= 0.5
