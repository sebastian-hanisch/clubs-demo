import numpy as np

from cb_scenario import generate_polynomial_mixture


def test_reproducible_for_fixed_seed():
    a = generate_polynomial_mixture(100, 1, 2, offset=10.0, noise_sigma=0.5, seed=7)
    b = generate_polynomial_mixture(100, 1, 2, offset=10.0, noise_sigma=0.5, seed=7)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.true_labels, b.true_labels)


def test_different_seed_gives_different_data():
    a = generate_polynomial_mixture(100, 1, 2, offset=10.0, noise_sigma=0.5, seed=1)
    b = generate_polynomial_mixture(100, 1, 2, offset=10.0, noise_sigma=0.5, seed=2)
    assert not np.array_equal(a.x, b.x) or not np.array_equal(a.y, b.y)


def test_point_count_and_label_balance():
    s = generate_polynomial_mixture(101, 1, 1, offset=5.0, noise_sigma=0.1, seed=3)
    assert len(s.x) == 101
    assert len(s.y) == 101
    assert len(s.true_labels) == 101
    n_a = int(np.sum(s.true_labels == 0))
    n_b = int(np.sum(s.true_labels == 1))
    assert n_a + n_b == 101
    assert abs(n_a - n_b) <= 1


def test_offset_shifts_function_b_upward_on_average():
    low_offset = generate_polynomial_mixture(400, 1, 1, offset=0.0, noise_sigma=0.05, seed=5)
    high_offset = generate_polynomial_mixture(400, 1, 1, offset=50.0, noise_sigma=0.05, seed=5)
    mean_b_low = low_offset.y[low_offset.true_labels == 1].mean()
    mean_b_high = high_offset.y[high_offset.true_labels == 1].mean()
    assert mean_b_high - mean_b_low > 40.0


def test_noise_increases_residual_scatter_around_true_function():
    low_noise = generate_polynomial_mixture(300, 1, 1, offset=10.0, noise_sigma=0.05, seed=9)
    high_noise = generate_polynomial_mixture(300, 1, 1, offset=10.0, noise_sigma=3.0, seed=9)
    resid_low = low_noise.y[low_noise.true_labels == 0] - np.polyval(
        low_noise.coeffs_a, low_noise.x[low_noise.true_labels == 0]
    )
    resid_high = high_noise.y[high_noise.true_labels == 0] - np.polyval(
        high_noise.coeffs_a, high_noise.x[high_noise.true_labels == 0]
    )
    assert resid_high.std() > resid_low.std() * 5


def test_degree_controls_number_of_fitted_coefficients():
    s = generate_polynomial_mixture(50, 2, 3, offset=5.0, noise_sigma=0.1, seed=1)
    assert len(s.coeffs_a) == 3  # Grad 2 -> 3 Koeffizienten
    assert len(s.coeffs_b) == 4  # Grad 3 -> 4 Koeffizienten
    assert s.degree_a == 2
    assert s.degree_b == 3


def test_x_within_configured_range():
    s = generate_polynomial_mixture(200, 1, 2, offset=5.0, noise_sigma=0.1, seed=1, x_range=(-3.0, 3.0))
    assert s.x.min() >= -3.0
    assert s.x.max() <= 3.0
