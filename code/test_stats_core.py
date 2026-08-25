"""Unit tests for the screening-layer statistical core."""
import numpy as np
import pytest

from stats_core import (
    benjamini_hochberg, calibrate_threshold, calibrate_threshold_by_cost,
    claimed_vs_realized, empirical_p_values, expected_decision_cost,
    realized_error, relation_stratified, screen_by_pvalue, storey_pi0,
)


class TestEmpiricalPValues:
    def test_all_candidates_higher(self):
        p = empirical_p_values(np.array([0.0]), np.array([[1.0, 2.0, 3.0]]))
        assert p[0] == pytest.approx(1.0)  # observed beaten by all 3 -> (3+1)/4

    def test_observed_beats_all(self):
        p = empirical_p_values(np.array([5.0]), np.array([[1.0, 2.0, 3.0]]))
        assert p[0] == pytest.approx(1 / 4)  # floor p-value

    def test_ties_count_as_ge(self):
        p = empirical_p_values(np.array([2.0]), np.array([[1.0, 2.0, 3.0]]))
        assert p[0] == pytest.approx(3 / 4)  # candidates >= 2: two of three

    def test_uniform_scores_give_uniform_grid(self):
        rng = np.random.default_rng(0)
        obs = rng.normal(size=2000)
        cand = rng.normal(size=(2000, 100))
        p = empirical_p_values(obs, cand)
        assert p.min() == pytest.approx(1 / 101)
        assert p.max() <= 1.0
        # under iid null scores, p is ~uniform on the (K+1)-grid
        assert np.mean(p) == pytest.approx(0.5, abs=0.02)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            empirical_p_values(np.zeros(3), np.zeros((2, 5)))


class TestBenjaminiHochberg:
    def test_hand_computed(self):
        p = np.array([0.01, 0.02, 0.03, 0.5])
        bh = benjamini_hochberg(p, alpha=0.05)
        # k*alpha/m: 0.0125, 0.025, 0.0375, 0.05 -> k*=3, threshold=0.03
        assert bh.n_rejected == 3
        assert bh.threshold == pytest.approx(0.03)
        assert list(bh.rejected) == [True, True, True, False]

    def test_no_rejections(self):
        bh = benjamini_hochberg(np.array([0.4, 0.5, 0.6, 0.9]), alpha=0.05)
        assert bh.n_rejected == 0
        assert bh.threshold == 0.0

    def test_adaptive_m0_rejects_more(self):
        p = np.array([0.01, 0.02, 0.6, 0.9])
        bh_std = benjamini_hochberg(p, alpha=0.05)
        bh_adapt = benjamini_hochberg(p, alpha=0.05, m0=2)
        assert bh_adapt.n_rejected >= bh_std.n_rejected

    def test_all_signal(self):
        bh = benjamini_hochberg(np.array([1e-6, 1e-7, 1e-8]), alpha=0.05)
        assert bh.n_rejected == 3


class TestStoreyPi0:
    def test_uniform_is_all_null(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(size=10000)
        assert storey_pi0(p, 0.5) == pytest.approx(1.0, abs=0.03)

    def test_strong_signal_is_low_null(self):
        p = np.full(10000, 1e-4)
        assert storey_pi0(p, 0.5) == pytest.approx(0.0, abs=0.001)


class TestRealizedError:
    def test_hand_computed(self):
        kept = np.array([True, False, True, True])
        labels = np.array([1, 0, 1, 1])
        r = realized_error(kept, labels)
        assert r.n_kept == 3
        assert r.n_kept_positive == 3
        assert r.n_kept_negative == 0
        assert r.realized_fdr == 0.0
        assert r.precision == 1.0

    def test_fdr_and_precision(self):
        kept = np.array([True, True, True, False])
        labels = np.array([1, 0, 1, 1])
        r = realized_error(kept, labels)
        assert r.n_kept == 3
        assert r.n_kept_positive == 2
        assert r.n_kept_negative == 1
        assert r.realized_fdr == pytest.approx(1 / 3)
        assert r.precision == pytest.approx(2 / 3)
        assert r.recall == pytest.approx(2 / 3)

    def test_nothing_kept(self):
        r = realized_error(np.zeros(5, dtype=bool), np.ones(5, dtype=bool))
        assert r.n_kept == 0
        assert r.realized_fdr == 0.0
        assert r.precision == 0.0
        assert r.recall == 0.0


class TestCalibrateThreshold:
    def test_hits_target_fdr(self):
        rng = np.random.default_rng(2)
        m = 5000
        # 30% signal (p near 0), 70% null (p uniform)
        sig = rng.uniform(0.0, 0.01, size=int(0.3 * m))
        nul = rng.uniform(0.0, 1.0, size=int(0.7 * m))
        p = np.concatenate([sig, nul])
        labels = np.concatenate([np.ones(sig.size), np.zeros(nul.size)])
        th = calibrate_threshold(p, labels, target=0.10)
        kept = screen_by_pvalue(p, th)
        r = realized_error(kept, labels)
        assert r.realized_fdr <= 0.10 + 0.02
        # regression: the threshold must keep a non-empty set with real coverage,
        # not the degenerate "keep nothing => vacuous FDR 0" that the old scan
        # returned for every target
        assert r.n_kept > 0
        assert r.n_kept / r.n_pool > 0.01

    def test_returns_threshold_keeping_nothing_when_signal_absent(self):
        # pure null p-values: no threshold with positive coverage meets even 0.5
        rng = np.random.default_rng(4)
        p = rng.uniform(size=4000)
        labels = np.zeros(4000, dtype=bool)
        th = calibrate_threshold(p, labels, target=0.10)
        # keeping nothing is the honest answer (threshold 0 keeps nothing)
        assert th == 0.0
        assert np.sum(screen_by_pvalue(p, th)) == 0


class TestDecisionCost:
    def test_hand_computed(self):
        kept = np.array([True, False, True])
        labels = np.array([0, 1, 1])
        # FP: idx0 (kept,neg); FN: idx1 (withheld,pos) -> cost = (c_fp + c_fn)/3
        cost = expected_decision_cost(kept, labels, c_fp=2.0, c_fn=3.0)
        assert cost == pytest.approx(5.0 / 3.0)

    def test_cost_aware_threshold_trades_off(self):
        rng = np.random.default_rng(1)
        sig = rng.uniform(0.1, 0.2, 400)   # true signal at moderate p
        nul = rng.uniform(0.0, 1.0, 1600)  # null uniform
        p = np.concatenate([sig, nul])
        lab = np.concatenate([np.ones(400, bool), np.zeros(1600, bool)])
        th_fp = calibrate_threshold_by_cost(p, lab, c_fp=5, c_fn=1)
        th_fn = calibrate_threshold_by_cost(p, lab, c_fp=1, c_fn=5)
        assert th_fp <= th_fn  # stricter when false-positive cost dominates
        # each threshold minimizes its own expected cost
        c = expected_decision_cost(screen_by_pvalue(p, th_fp), lab, 5, 1)
        for th in np.unique(np.sort(p)[::50]):
            c_ = expected_decision_cost(screen_by_pvalue(p, th), lab, 5, 1)
            assert c_ >= c - 1e-9


class TestClaimedVsRealized:
    def test_well_calibrated_pool(self):
        rng = np.random.default_rng(3)
        m = 3000
        # signal p~0 ; null p~U(0,1)  -- a correctly "controlled" pool
        sig = rng.uniform(0.0, 0.001, size=1000)
        nul = rng.uniform(0.0, 1.0, size=2000)
        p = np.concatenate([sig, nul])
        labels = np.concatenate([np.ones(1000), np.zeros(2000)])
        out = claimed_vs_realized(p, labels, alpha=0.05)
        assert out["n_rejected"] > 0
        assert out["realized_fdr"] <= 0.10

    def test_overclaimed_when_labels_reversed(self):
        # if the "signal" p-values are actually false positives, realized FDR is high
        p = np.array([1e-4, 1e-4, 1e-4, 0.5, 0.6, 0.7])
        labels = np.array([0, 0, 0, 1, 1, 1])
        out = claimed_vs_realized(p, labels, alpha=0.05)
        assert out["realized_fdr"] == pytest.approx(1.0)


class TestRelationStratified:
    def test_splits_by_relation(self):
        p = np.array([0.01, 0.5, 0.02, 0.6])
        labels = np.array([1, 0, 1, 0])
        rel = np.array([0, 0, 1, 1])
        out = relation_stratified(p, labels, rel)
        assert set(out.keys()) == {0, 1}
        assert out[0]["n_rejected"] >= 1
