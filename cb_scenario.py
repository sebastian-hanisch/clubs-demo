"""Szenario-Generator: Mischungen aus zwei zufaelligen Polynom-Grundwahrheiten, direkt
im Geist des CluBS-Papers (Zdankin, Kummerow & Weis, KDD'26)
nachempfunden - Grad 1/2/3, mit einstellbarem Offset (Trennbarkeit) und Rauschen."""

from dataclasses import dataclass

import numpy as np

X_RANGE = (-5.0, 5.0)
COEFF_SCALE = 1.8


@dataclass(frozen=True)
class Scenario:
    x: np.ndarray
    y: np.ndarray
    true_labels: np.ndarray  # 0 = Funktion A, 1 = Funktion B
    coeffs_a: np.ndarray
    coeffs_b: np.ndarray
    degree_a: int
    degree_b: int


def generate_polynomial_mixture(
    n_points, degree_a, degree_b, offset, noise_sigma, seed,
    x_range=X_RANGE, coeff_scale=COEFF_SCALE,
):
    """Erzeugt eine Punktwolke aus zwei zufaelligen Polynomen vom Grad `degree_a`/
    `degree_b` (je 1-3). `offset` wird additiv auf Funktion B addiert,
    um die Trennbarkeit zu erhoehen. Rueckgabe ist deterministisch fuer festen `seed`."""
    rng = np.random.default_rng(seed)
    coeffs_a = rng.uniform(-coeff_scale, coeff_scale, degree_a + 1)
    coeffs_b = rng.uniform(-coeff_scale, coeff_scale, degree_b + 1)

    n_a = n_points // 2
    n_b = n_points - n_a
    x_a = rng.uniform(x_range[0], x_range[1], n_a)
    x_b = rng.uniform(x_range[0], x_range[1], n_b)
    y_a = np.polyval(coeffs_a, x_a) + rng.normal(0.0, noise_sigma, n_a)
    y_b = np.polyval(coeffs_b, x_b) + offset + rng.normal(0.0, noise_sigma, n_b)

    x = np.concatenate([x_a, x_b])
    y = np.concatenate([y_a, y_b])
    true_labels = np.concatenate([np.zeros(n_a, dtype=int), np.ones(n_b, dtype=int)])

    # Reihenfolge mischen, damit kein Algorithmus zufaellig von der Eingabereihenfolge
    # profitiert (z.B. SelectInitialCluster's Eckenwahl darf keine sortierte Eingabe
    # als versteckten Vorteil bekommen).
    order = rng.permutation(n_points)
    return Scenario(
        x=x[order], y=y[order], true_labels=true_labels[order],
        coeffs_a=coeffs_a, coeffs_b=coeffs_b, degree_a=degree_a, degree_b=degree_b,
    )
