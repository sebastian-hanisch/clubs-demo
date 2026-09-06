"""Defaults, Regler-Grenzen, Sicherheitsgrenzen und Presets fuer die CluBS-
(Clustering-Behavioural-Similarity-)Demo."""

DEFAULT_N_POINTS = 200
N_POINTS_MIN, N_POINTS_MAX = 60, 400
N_POINTS_HARD_MAX = 400  # Sicherheitsgrenze fuer Merge (O(n*c^2)) und den Live-Regler

DEGREE_MIN, DEGREE_MAX = 1, 3
DEFAULT_DEGREE_A = 1
DEFAULT_DEGREE_B = 2
DEGREE_LABELS = {1: "linear", 2: "quadratisch", 3: "kubisch"}

DEFAULT_OFFSET = 20.0
OFFSET_MIN, OFFSET_MAX = 0.0, 40.0

DEFAULT_NOISE = 0.6
NOISE_MIN, NOISE_MAX = 0.05, 4.0

DEFAULT_SEED = 1

DEFAULT_N_NEIGHBORS = 18
N_NEIGHBORS_MIN, N_NEIGHBORS_MAX = 6, 60

DEFAULT_TAU_EPS = 1.8
TAU_EPS_MIN, TAU_EPS_MAX = 0.2, 15.0

DEFAULT_R_MERGE = 0.85
R_MERGE_MIN, R_MERGE_MAX = 0.5, 0.99

# Feste Algorithmus-Hyperparameter aus Algorithm 1 im Paper - dort ebenfalls fest
# gewaehlt (n_min proportional zur Datensatzgroesse skaliert, da unsere Demo mit
# deutlich weniger Punkten arbeitet als die 1000-2000 im Paper).
TAU_JACCARD = 0.01
U_MAX = 20
N_MIN_FRACTION = 0.1
N_MIN_FLOOR = 12

X_RANGE = (-5.0, 5.0)
COEFF_SCALE = 1.8

# Fester Vergleichs-Seed fuer die k-Means-dann-Fit-Baseline, unabhaengig vom
# Szenario-Seed (etabliertes Muster aus gmm-/dpmm-/spectral-/leiden-/divisive-demo).
COMPARISON_SEED = 1


def n_min_for(n_points):
    return max(N_MIN_FLOOR, int(N_MIN_FRACTION * n_points))


PRESETS = {
    "Einfaches Beispiel": {
        "n_points": 160, "degree_a": 1, "degree_b": 2, "offset": 25.0,
        "noise": 0.5, "n_neighbors": 16, "tau_eps": 1.5, "r_merge": 0.85, "seed": 17,
    },
    "k-Means scheitert, CluBS nicht": {
        "n_points": 300, "degree_a": 1, "degree_b": 2, "offset": 0.0,
        "noise": 0.5, "n_neighbors": 20, "tau_eps": 1.5, "r_merge": 0.85, "seed": 19,
    },
    "Ähnliche Funktionen verschmelzen": {
        "n_points": 200, "degree_a": 1, "degree_b": 1, "offset": 2.0,
        "noise": 0.5, "n_neighbors": 18, "tau_eps": 1.8, "r_merge": 0.85, "seed": 10,
    },
    "Keine Konvergenzgarantie": {
        "n_points": 200, "degree_a": 1, "degree_b": 1, "offset": 0.5,
        "noise": 1.2, "n_neighbors": 20, "tau_eps": 2.0, "r_merge": 0.85, "seed": 3,
    },
}
