"""Orakel-Tests für CluBS (unabhängige Rechenwege):

1. find_function / r_squared gegen numpy.linalg.lstsq auf der Vandermonde-Matrix und sklearn r2_score.
2. ARI, NMI, Accuracy/F1_macro gegen sklearn (adjusted_rand_score, normalized_mutual_info_score, f1_score
   unter allen optimalen Label-Zuordnungen), auch bei ungleicher Clusterzahl und Randfällen.
3. Die ganze Pipeline (Entdeckung, Reassign/Retrain, Merge) gegen eine zweite Implementierung auf
   Mengen/lstsq-Basis mit demselben Zufallsstrom: identische Partition."""

import itertools
import warnings

import numpy as np
import pytest

from cb_algorithm import find_function, r_squared, run, labels_from_clubs
from cb_evaluation import accuracy_and_f1_macro, adjusted_rand_index, normalized_mutual_information
from cb_scenario import generate_polynomial_mixture

metrics = pytest.importorskip("sklearn.metrics")


def _ref_fit(x, y, max_degree=3, threshold=0.9):
    best = None
    for degree in range(1, min(max_degree, len(x) - 1) + 1):
        vander = np.vander(x, degree + 1)
        coeffs, *_ = np.linalg.lstsq(vander, y, rcond=None)
        res = ((y - vander @ coeffs) ** 2).sum()
        tot = ((y - y.mean()) ** 2).sum()
        r2 = (1.0 if res < 1e-12 else 0.0) if tot < 1e-12 else 1 - res / tot
        if best is None or r2 > best[1]:
            best = (degree, r2, coeffs)
        if r2 >= threshold:
            return degree, r2, coeffs
    return best


def test_find_function_matches_lstsq_and_sklearn_r2():
    rng = np.random.default_rng(21)
    for _ in range(120):
        n = int(rng.integers(2, 50))
        x = rng.uniform(-5, 5, n)
        y = np.polyval(rng.uniform(-2, 2, int(rng.integers(2, 5))), x) + rng.normal(0, rng.uniform(0.01, 4), n)
        function, r2 = find_function(x, y)
        degree, ref_r2, coeffs = _ref_fit(x, y)
        assert function.degree == degree
        assert r2 == pytest.approx(ref_r2, abs=1e-8)
        np.testing.assert_allclose(function.coeffs, coeffs, rtol=1e-5, atol=1e-6)
        if n >= 3:
            assert r_squared(y, function(x)) == pytest.approx(metrics.r2_score(y, function(x)), abs=1e-9)


def test_cluster_metrics_match_sklearn():
    rng = np.random.default_rng(4)
    for _ in range(150):
        n = int(rng.integers(1, 30))
        true = rng.integers(0, int(rng.integers(1, 5)), n)
        pred = rng.integers(0, int(rng.integers(1, 6)), n)
        assert adjusted_rand_index(true, pred) == pytest.approx(metrics.adjusted_rand_score(true, pred), abs=1e-9)
        assert normalized_mutual_information(true, pred) == pytest.approx(
            metrics.normalized_mutual_info_score(true, pred), abs=1e-9
        )
        true_ids, pred_ids = np.unique(true), np.unique(pred)
        if len(pred_ids) <= len(true_ids):
            mappings = [dict(zip(pred_ids, p)) for p in itertools.permutations(true_ids, len(pred_ids))]
        else:
            mappings = [dict(zip(p, true_ids)) for p in itertools.permutations(pred_ids, len(true_ids))]
        scored = []
        for mapping in mappings:
            mapped = np.array([mapping.get(q, -1) for q in pred])
            f1 = metrics.f1_score(true, mapped, labels=true_ids, average="macro", zero_division=0)
            scored.append((float(np.mean(mapped == true)), round(f1, 9)))
        best_acc = max(s[0] for s in scored)
        acc, f1 = accuracy_and_f1_macro(true, pred)
        assert acc == pytest.approx(best_acc, abs=1e-12)
        assert round(f1, 9) in {s[1] for s in scored if abs(s[0] - best_acc) < 1e-12}


def _ref_pipeline(x, y, n_min, tau_j, u_max, k_nb, tau_eps, r_merge, seed):
    rng = np.random.default_rng(seed)

    def fit(idx):
        idx = np.asarray(sorted(idx))
        if len(idx) < 2:
            const = y[idx[0]] if len(idx) else 0.0
            return (lambda q: np.full(len(np.atleast_1d(q)), const)), (1.0 if len(idx) == 1 else 0.0)
        _, r2, coeffs = _ref_fit(x[idx], y[idx])
        return (lambda q, c=coeffs: np.polyval(c, np.asarray(q, float))), r2

    remaining = list(range(len(x)))
    raw = []
    while len(remaining) >= n_min:
        rx = x[remaining]
        corner = rng.choice([rx.min(), rx.max()])
        order = np.argsort(np.abs(rx - rx[int(np.argmin(np.abs(rx - corner)))]), kind="stable")
        current = set(order[: min(k_nb, len(rx))].tolist())
        u = 0
        while True:
            f, _ = fit([remaining[i] for i in current])
            nxt = set(np.nonzero(np.abs(f(rx) - y[remaining]) <= tau_eps)[0].tolist()) or set(current)
            union = current | nxt
            jac = len(current & nxt) / len(union)
            u += 1
            current = nxt
            if jac >= 1 - tau_j or u > u_max:
                break
        members = [remaining[i] for i in sorted(current)]
        raw.append((members, fit(members)[0]))
        remaining = [r for r in remaining if r not in set(members)]
    asg = np.stack([np.abs(f(x) - y) for _, f in raw], axis=1).argmin(axis=1)
    clubs = []
    for i in range(len(raw)):
        idx = list(np.nonzero(asg == i)[0])
        if idx:
            clubs.append((idx, fit(idx)[1]))
    while len(clubs) > 1:
        best = None
        for i in range(len(clubs)):
            for j in range(i + 1, len(clubs)):
                idx = clubs[i][0] + clubs[j][0]
                r2 = fit(idx)[1]
                ni, nj = len(clubs[i][0]), len(clubs[j][0])
                if r2 >= (ni * clubs[i][1] + nj * clubs[j][1]) / (ni + nj) or r2 >= r_merge:
                    if best is None or r2 > best[0]:
                        best = (r2, i, j, idx)
        if best is None:
            break
        r2, i, j, idx = best
        clubs = [c for k, c in enumerate(clubs) if k not in (i, j)] + [(idx, r2)]
    labels = np.full(len(x), -1)
    for label, (idx, _) in enumerate(clubs):
        labels[idx] = label
    return labels


def test_pipeline_matches_second_implementation():
    rng = np.random.default_rng(21)
    for _ in range(20):
        n = int(rng.integers(60, 160))
        scenario = generate_polynomial_mixture(
            n, int(rng.integers(1, 4)), int(rng.integers(1, 4)), float(rng.uniform(0, 30)),
            float(rng.uniform(0.05, 3)), int(rng.integers(0, 10**6)),
        )
        params = dict(
            n_min=max(12, int(0.1 * n)), tau_jaccard=0.01, u_max=int(rng.integers(2, 21)),
            k_neighbors=int(rng.integers(6, 40)), tau_eps=float(rng.uniform(0.3, 6)), r_merge=float(rng.uniform(0.5, 0.99)),
        )
        seed = int(rng.integers(0, 10**6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run(scenario.x, scenario.y, seed=seed, **params)
            reference = _ref_pipeline(
                scenario.x, scenario.y, params["n_min"], 0.01, params["u_max"], params["k_neighbors"],
                params["tau_eps"], params["r_merge"], seed,
            )
        mine = labels_from_clubs(n, result.final_clubs)
        assert (mine >= 0).all()
        assert metrics.adjusted_rand_score(mine, reference) == 1.0
