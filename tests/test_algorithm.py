import numpy as np
import pytest

from cb_algorithm import (
    Function, find_function, r_squared, jaccard_index, select_initial_cluster,
    update_cluster, find_club, run, reassign, merge_clubs, labels_from_clubs,
)
from cb_scenario import generate_polynomial_mixture


# ---------------------------------------------------------------------------
# Handgerechnete Beispiele
# ---------------------------------------------------------------------------

def test_find_function_exact_line_gives_degree_one_and_correct_coefficients():
    # y = 2x + 3, exakt (kein Rauschen) - FindFunction muss Grad 1 mit den exakten
    # Koeffizienten liefern, R^2 = 1.0.
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 3.0
    function, r2 = find_function(x, y)
    assert function.degree == 1
    assert function.coeffs[0] == pytest.approx(2.0, abs=1e-9)
    assert function.coeffs[1] == pytest.approx(3.0, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_find_function_prefers_lower_degree_when_it_already_fits_well():
    # Fast-lineare Daten (winziges Rauschen) - Grad 1 erreicht bereits die
    # Guetegrenze, Grad 2/3 duerfen NICHT bevorzugt werden (Sparsamkeitsprinzip).
    rng = np.random.default_rng(0)
    x = np.linspace(-5, 5, 40)
    y = 3.0 * x - 1.0 + rng.normal(0, 0.01, size=x.shape)
    function, r2 = find_function(x, y)
    assert function.degree == 1
    assert r2 > 0.999


def test_r_squared_hand_computed():
    # y_true = [1,2,3,4], y_pred = [1,2,3,6] -> SS_res=4, SS_tot = sum((y-2.5)^2) = 5.0
    # R^2 = 1 - 4/5 = 0.2
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 6.0])
    assert r_squared(y_true, y_pred) == pytest.approx(0.2, abs=1e-9)


def test_jaccard_index_hand_computed():
    a = np.array([0, 1, 2, 3])
    b = np.array([2, 3, 4])
    # Schnitt = {2,3} (2), Vereinigung = {0,1,2,3,4} (5) -> J = 2/5 = 0.4
    assert jaccard_index(a, b) == pytest.approx(0.4, abs=1e-12)


def test_jaccard_index_identical_sets_is_one():
    a = np.array([5, 6, 7])
    assert jaccard_index(a, a) == pytest.approx(1.0)


def test_jaccard_index_disjoint_sets_is_zero():
    a = np.array([0, 1])
    b = np.array([2, 3])
    assert jaccard_index(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Struktur-Invarianten
# ---------------------------------------------------------------------------

def test_select_initial_cluster_picks_points_near_a_bounding_box_corner():
    rng = np.random.default_rng(1)
    x = np.linspace(-10, 10, 50)
    idx = select_initial_cluster(x, k_neighbors=5, rng=np.random.default_rng(2))
    assert len(idx) == 5
    selected = x[idx]
    # Alle gewaehlten Punkte muessen naeher an EINEM Rand liegen als am Zentrum.
    assert (selected.min() <= -8.0) or (selected.max() >= 8.0)


def test_update_cluster_includes_all_points_within_threshold_across_full_dataset():
    x = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
    y = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
    function = Function(degree=1, coeffs=(1.0, 0.0))  # y = x, perfekt fuer alle Punkte
    idx = update_cluster(x, y, function, tau_eps=0.5)
    np.testing.assert_array_equal(idx, np.array([0, 1, 2, 3, 4]))


def test_update_cluster_excludes_points_above_threshold():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 100.0])  # letzter Punkt ist ein Ausreisser
    function = Function(degree=1, coeffs=(1.0, 0.0))
    idx = update_cluster(x, y, function, tau_eps=0.5)
    np.testing.assert_array_equal(idx, np.array([0, 1]))


def test_run_partitions_every_point_exactly_once_after_reassign():
    s = generate_polynomial_mixture(150, 1, 2, offset=20.0, noise_sigma=0.3, seed=1)
    result = run(
        s.x, s.y, n_min=15, tau_jaccard=0.01, u_max=20,
        k_neighbors=15, tau_eps=1.5, r_merge=0.85, seed=1,
    )
    labels = labels_from_clubs(len(s.x), result.final_clubs)
    assert np.all(labels >= 0)  # jeder Punkt zugeordnet, keiner verloren
    all_indices = np.concatenate([c.point_indices for c in result.final_clubs])
    assert len(all_indices) == len(s.x)
    assert len(set(all_indices.tolist())) == len(s.x)  # keine Doppelzuordnung


def test_merge_can_only_reduce_or_keep_club_count():
    s = generate_polynomial_mixture(150, 1, 1, offset=0.5, noise_sigma=0.2, seed=2)
    result = run(
        s.x, s.y, n_min=15, tau_jaccard=0.01, u_max=20,
        k_neighbors=15, tau_eps=1.0, r_merge=0.85, seed=2,
    )
    assert len(result.final_clubs) <= len(result.reassigned_clubs)


def test_reassign_never_increases_club_count():
    s = generate_polynomial_mixture(150, 1, 2, offset=20.0, noise_sigma=0.3, seed=1)
    raw = run(
        s.x, s.y, n_min=15, tau_jaccard=0.01, u_max=20,
        k_neighbors=15, tau_eps=1.5, r_merge=0.85, seed=1,
    ).raw_clubs
    reassigned = reassign(s.x, s.y, raw)
    assert len(reassigned) <= len(raw)


# ---------------------------------------------------------------------------
# Kernbehauptungen: Ground-Truth Function Recovery (Paper Abschnitt 5.1) und
# ehrlich fehlende Konvergenzgarantie
# ---------------------------------------------------------------------------

def _ari(true_labels, pred_labels):
    # lokale Mini-Implementierung nur fuer diesen Test - die vollstaendige,
    # Label-permutations-robuste Version lebt in cb_evaluation.py und wird dort
    # eigenstaendig getestet.
    from cb_evaluation import adjusted_rand_index
    return adjusted_rand_index(true_labels, pred_labels)


def test_ground_truth_recovery_matches_paper_methodology_well_separated_case():
    # Zwei klar getrennte Polynome (grosser Offset, wenig Rauschen) - CluBS soll die
    # wahre Partition mit hohem ARI rekonstruieren, ueber mehrere Seeds hinweg.
    aris = []
    for seed in range(5):
        s = generate_polynomial_mixture(200, 1, 2, offset=25.0, noise_sigma=0.4, seed=seed)
        result = run(
            s.x, s.y, n_min=20, tau_jaccard=0.01, u_max=20,
            k_neighbors=20, tau_eps=2.0, r_merge=0.85, seed=seed,
        )
        labels = labels_from_clubs(len(s.x), result.final_clubs)
        aris.append(_ari(s.true_labels, labels))
    assert np.median(aris) > 0.9


def test_no_convergence_guarantee_shown_honestly_on_hard_case():
    # Haerte-Fall analog zu Figure 6 im Appendix: zwei Funktionen, die sich ueber den
    # gesamten Bereich stark aehneln (kein Offset, gleicher Grad, aehnliche
    # Koeffizienten) - CluBS' eigene Konvergenzpruefung bietet KEINE Garantie, dass
    # dabei die wahre Partition gefunden wird. Wir zeigen das ehrlich: der ARI muss
    # hier deutlich schlechter sein als im gut trennbaren Fall oben.
    aris = []
    for seed in range(5):
        s = generate_polynomial_mixture(200, 1, 1, offset=0.3, noise_sigma=1.5, seed=seed)
        result = run(
            s.x, s.y, n_min=20, tau_jaccard=0.01, u_max=20,
            k_neighbors=20, tau_eps=2.0, r_merge=0.85, seed=seed,
        )
        labels = labels_from_clubs(len(s.x), result.final_clubs)
        aris.append(_ari(s.true_labels, labels))
    assert np.median(aris) < 0.5


def test_kmeans_baseline_fails_where_clubs_succeeds_on_overlapping_scatter():
    # Figure-1-Kernszenario: kein Offset, sich kreuzende Funktionen - die
    # (x,y)-Punktwolken beider Funktionen ueberlappen im Rohraum. Eine
    # k-Means-dann-Fit-Baseline (siehe cb_evaluation.py) soll hier klar
    # schlechter abschneiden als CluBS selbst.
    from cb_evaluation import kmeans_then_fit_labels

    s = generate_polynomial_mixture(300, 1, 2, offset=0.0, noise_sigma=0.5, seed=3)
    result = run(
        s.x, s.y, n_min=20, tau_jaccard=0.01, u_max=20,
        k_neighbors=20, tau_eps=1.5, r_merge=0.85, seed=3,
    )
    clubs_labels = labels_from_clubs(len(s.x), result.final_clubs)
    kmeans_labels = kmeans_then_fit_labels(s.x, s.y, k=2, seed=3)

    clubs_ari = _ari(s.true_labels, clubs_labels)
    kmeans_ari = _ari(s.true_labels, kmeans_labels)
    assert clubs_ari > kmeans_ari + 0.2
