import numpy as np
import pytest

from cb_evaluation import (
    adjusted_rand_index, normalized_mutual_information, accuracy_and_f1_macro,
    kmeans_1d_then_fit_labels,
)


# ---------------------------------------------------------------------------
# Handgerechnete Beispiele
# ---------------------------------------------------------------------------

def test_ari_perfect_agreement_is_one():
    true = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 1])
    assert adjusted_rand_index(true, pred) == pytest.approx(1.0, abs=1e-9)


def test_ari_perfect_agreement_is_one_under_label_permutation():
    # ARI ist per Konstruktion (Kontingenztabelle) unabhaengig von Label-Namen.
    true = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([1, 1, 1, 0, 0, 0])
    assert adjusted_rand_index(true, pred) == pytest.approx(1.0, abs=1e-9)


def test_ari_hand_computed_partial_agreement():
    # Tabelle: true0=[2,1], true1=[1,2] (Zeilensumme je 3), n=6
    # sum_comb=2, sum_comb_c=6, sum_comb_k=6, comb2(n)=15, erwartet=2.4
    # ARI = (2 - 2.4) / (6 - 2.4) = -0.4/3.6 = -1/9
    true = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 1, 0, 1, 1])
    assert adjusted_rand_index(true, pred) == pytest.approx(-1.0 / 9.0, abs=1e-9)


def test_nmi_perfect_agreement_is_one():
    true = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    assert normalized_mutual_information(true, pred) == pytest.approx(1.0, abs=1e-9)


def test_nmi_hand_computed_independent_case_is_zero():
    # true=[0,0,1,1], pred=[0,1,1,0]: Kontingenztabelle ist [[1,1],[1,1]] -> jede
    # Zelle = p_true*p_pred*n, d.h. statistisch unabhaengig -> MI = 0 -> NMI = 0.
    true = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 0])
    assert normalized_mutual_information(true, pred) == pytest.approx(0.0, abs=1e-9)


def test_accuracy_and_f1_handle_arbitrary_label_permutation():
    true = np.array([0, 0, 0, 1, 1])
    pred = np.array([1, 1, 1, 0, 0])  # Label komplett vertauscht, sonst perfekt
    acc, f1 = accuracy_and_f1_macro(true, pred)
    assert acc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_accuracy_and_f1_penalize_real_mismatches():
    true = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 1, 0])  # je ein Punkt pro wahrer Gruppe falsch
    acc, f1 = accuracy_and_f1_macro(true, pred)
    assert acc == pytest.approx(4.0 / 6.0)
    assert 0.0 < f1 < 1.0


def test_accuracy_handles_more_predicted_clusters_than_true_clusters():
    # CluBS kann mehr CluBs finden als es wahre Funktionen gibt (z.B. vor
    # vollstaendiger Verschmelzung) - die Metrik darf dabei nicht abstuerzen.
    true = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 2, 1, 1, 1])
    acc, f1 = accuracy_and_f1_macro(true, pred)
    assert acc == pytest.approx(5.0 / 6.0)


# ---------------------------------------------------------------------------
# k-Means-dann-Fit-Baseline
# ---------------------------------------------------------------------------

def test_kmeans_baseline_reproducible_and_covers_all_points():
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, 80)
    y = 2 * x + rng.normal(0, 0.3, 80)
    y[40:] += 20  # zweite, klar getrennte Gruppe
    labels_a = kmeans_1d_then_fit_labels(x, y, k=2, seed=1)
    labels_b = kmeans_1d_then_fit_labels(x, y, k=2, seed=1)
    np.testing.assert_array_equal(labels_a, labels_b)
    assert len(labels_a) == 80
    assert set(np.unique(labels_a).tolist()) <= {0, 1}


def test_kmeans_baseline_recovers_spatially_separated_groups():
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 5, 200)
    y = np.where(np.arange(200) < 100, 0.0, 50.0) + rng.normal(0, 0.5, 200)
    true_labels = np.where(np.arange(200) < 100, 0, 1)
    labels = kmeans_1d_then_fit_labels(x, y, k=2, seed=0)
    ari = adjusted_rand_index(true_labels, labels)
    assert ari > 0.9
