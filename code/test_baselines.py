"""Tests for the two P4 baselines (calibrator, split-conformal)."""
import numpy as np
import pytest

from baseline_calibrator import calibrator_curve, fit_calibrator
from baseline_conformal import conformal_curve, conformal_quantile
from stats_core import realized_error_metrics, screening_decisions


def _synthetic_pool(n_true=800, n_false=3200, seed=0):
    rng = np.random.default_rng(seed)
    true_s = rng.normal(3.0, 1.0, size=n_true)
    false_s = rng.normal(0.0, 1.0, size=n_false)
    scores = np.concatenate([true_s, false_s])
    labels = np.concatenate([np.ones(n_true, bool), np.zeros(n_false, bool)])
    order = rng.permutation(scores.size)
    return scores[order], labels[order]


def test_calibrator_probabilities_are_monotone():
    s, y = _synthetic_pool()
    cal = fit_calibrator(s[:2000], y[:2000], method="platt")
    probs = cal(np.linspace(-2, 6, 9))
    assert np.all(np.diff(probs) >= -1e-9)  # increasing in score


def test_calibrator_screens_mostly_true():
    s, y = _synthetic_pool()
    rows = calibrator_curve(s[:2000], y[:2000], s[2000:], y[2000:],
                            method="isotonic", targets=(0.5, 0.8))
    for r in rows:
        assert r["precision"] >= 0.85
        assert r["realized_fdr"] <= 0.15


def test_conformal_quantile_coverage():
    rng = np.random.default_rng(1)
    cal_true = rng.normal(3.0, 1.0, size=1000)
    # eval true tails drawn from same distribution -> coverage should be ~0.90
    ev_true = rng.normal(3.0, 1.0, size=5000)
    q = conformal_quantile(cal_true, alpha=0.10)
    cov = np.mean(ev_true >= -q)
    assert 0.86 <= cov <= 0.94


def test_conformal_beats_topk_at_matched_coverage():
    s, y = _synthetic_pool()
    half = s.size // 2
    rows = conformal_curve(s[:half], y[:half], s[half:], y[half:], alphas=(0.10,))
    r = rows[0]
    n_kept = r["n_screened_in"]
    order = np.argsort(-s[half:], kind="mergesort")
    topk = np.zeros(s[half:].size, bool)
    topk[order[:n_kept]] = True
    tk = realized_error_metrics(topk, y[half:])
    assert r["realized_fdr"] <= tk["realized_fdr"] + 1e-9
    assert r["true_coverage"] >= 0.86


class TestRelik:
    def test_scores_shape_and_range(self):
        import numpy as np
        import torch
        from pykeen.triples import TriplesFactory
        from pykeen.models import TransE
        from baseline_relik import build_sets, relik_scores
        triples = np.array([['a', 'r1', 'b'], ['b', 'r2', 'c'], ['a', 'r2', 'c'],
                            ['c', 'r1', 'a'], ['d', 'r1', 'b']], dtype=object)
        tf = TriplesFactory.from_labeled_triples(triples)
        m = TransE(triples_factory=tf, embedding_dim=8, random_seed=0)
        hr, rt = build_sets(tf, tf, tf)
        h = np.array([0, 1]); r = np.array([0, 1]); t = np.array([1, 2])
        s = relik_scores(m, h, r, t, hr, rt, tf.num_entities, tf.num_relations,
                         m=5, rng=np.random.default_rng(0))
        assert s.shape == (2,)
        assert np.all(np.isfinite(s))
        assert np.all(s > 0.0) and np.all(s <= 1.0)
