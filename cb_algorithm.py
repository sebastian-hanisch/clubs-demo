"""Von Grund auf neu implementierter CluBS-Algorithmus (Algorithm 1 aus Zdankin,
Kummerow & Weis, "CluBS: Clustering Behavioural Similarity", KDD'26).

Eine bewusste, im README begruendete Vereinfachung: `find_function` (der Symbolic-
Regression-Baustein des Papers) durchsucht hier eine kleine Bibliothek von
Polynomgraden (1-3) statt einer offenen genetischen Ausdruckssuche - siehe README.

Das Paper selbst benennt explizit: "We do not provide any guarantees for
convergence" - die Konvergenzpruefung (Jaccard-Index >= 1-tau oder u > u_max) ist
daher ein Abbruchkriterium, kein Optimalitaetsbeweis.
"""

from dataclasses import dataclass, field

import numpy as np

MAX_DEGREE = 3
R2_ACCEPT_THRESHOLD = 0.9


@dataclass(frozen=True)
class Function:
    """Ein gefittetes Polynom (`degree`-ter Grad, `coeffs` wie `numpy.polyfit`:
    hoechster Grad zuerst)."""
    degree: int
    coeffs: tuple

    def __call__(self, x):
        return np.polyval(self.coeffs, np.asarray(x, dtype=float))


def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def find_function(x, y, max_degree=MAX_DEGREE, r2_threshold=R2_ACCEPT_THRESHOLD):
    """FindFunction (Gleichung 6 im Paper): sucht die Funktion, die die kleinste
    quadratische Fehlersumme auf (x, y) erreicht. Vereinfachung dieser Demo: statt
    offener symbolischer Regression wird ueber Polynomgrade 1..max_degree gesucht -
    der NIEDRIGSTE Grad, der eine R^2-Guetegrenze erreicht, wird bevorzugt (entspricht
    der im Paper selbst genannten Design-Absicht "kleinere Modelle vermeiden
    Overfitting und verbessern Extrapolation"). Erreicht kein Grad die Guetegrenze,
    wird der Grad mit dem besten R^2 zurueckgegeben. Bei zu wenigen Punkten fuer
    Grad 1 (< 2 Punkte) wird eine konstante Funktion (Grad 0, Mittelwert) geliefert.

    Gibt (Function, r2) zurueck."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if n < 2:
        mean_y = float(y[0]) if n == 1 else 0.0
        return Function(degree=0, coeffs=(mean_y,)), (1.0 if n == 1 else 0.0)

    usable_max_degree = min(max_degree, n - 1)
    best = None
    for degree in range(1, usable_max_degree + 1):
        coeffs = np.polyfit(x, y, degree)
        pred = np.polyval(coeffs, x)
        r2 = r_squared(y, pred)
        candidate = (Function(degree=degree, coeffs=tuple(coeffs)), r2)
        if best is None or r2 > best[1]:
            best = candidate
        if r2 >= r2_threshold:
            return candidate
    return best


def jaccard_index(a_indices, b_indices):
    """Jaccard-Index (Gleichung 7 im Paper) zweier Punktmengen (als Indexmengen)."""
    a, b = set(a_indices.tolist()), set(b_indices.tolist())
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def select_initial_cluster(x, k_neighbors, rng):
    """SelectInitialCluster, kNN-Variante (Abschnitt 4.1.1): eine zufaellig gewaehlte
    Ecke der (eindimensionalen) Bounding-Box von x, plus deren k naechste Nachbarn im
    Eingaberaum (hier: nach |x_i - x_seed| sortiert)."""
    corner = rng.choice([x.min(), x.max()])
    seed_idx = int(np.argmin(np.abs(x - corner)))
    dists = np.abs(x - x[seed_idx])
    order = np.argsort(dists, kind="stable")
    k = min(k_neighbors, len(x))
    return order[:k]


def update_cluster(x, y, function, tau_eps):
    """UpdateCluster (Abschnitt 4.1.3, Gleichung 8): alle Punkte aus D_t mit
    Vorhersagefehler unter der Schwelle tau_eps."""
    errs = np.abs(function(x) - y)
    return np.where(errs <= tau_eps)[0]


@dataclass(frozen=True)
class InnerStep:
    """Ein Schritt der inneren Schleife (FindFunction + UpdateCluster)."""
    u: int
    cluster_indices: np.ndarray  # Indizes in D_t, auf denen `function` gefittet wurde
    function: Function
    r2: float
    next_cluster_indices: np.ndarray  # Ergebnis von UpdateCluster
    jaccard: float
    converged: bool


@dataclass(frozen=True)
class ClubDiscovery:
    """Ergebnis der inneren Schleife fuer einen CluB (vor Reassign/Merge)."""
    cluster_indices: np.ndarray  # Indizes in D_t
    function: Function
    r2: float
    steps: list
    converged: bool


def find_club(x, y, k_neighbors, tau_eps, tau_jaccard, u_max, rng):
    """Die innere Schleife aus Algorithm 1 (Zeilen 6-11): abwechselnd FindFunction und
    UpdateCluster, bis der Jaccard-Index zwischen zwei Iterationen >= 1-tau ist oder
    u_max ueberschritten wird."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_prev = select_initial_cluster(x, k_neighbors, rng)
    steps = []
    u = 0
    converged = False
    while True:
        function, r2 = find_function(x[x_prev], y[x_prev])
        x_next = update_cluster(x, y, function, tau_eps)
        if x_next.size == 0:
            # Sicherheitsgrenze dieser Demo (im Paper nicht spezifiziert): eine leere
            # Aktualisierung wuerde FindFunction im naechsten Schritt zum Absturz
            # bringen - wir brechen stattdessen mit der letzten nicht-leeren Menge ab.
            x_next = x_prev
        j = jaccard_index(x_next, x_prev)
        u += 1
        converged = j >= 1 - tau_jaccard
        steps.append(InnerStep(
            u=u, cluster_indices=x_prev, function=function, r2=r2,
            next_cluster_indices=x_next, jaccard=j, converged=converged,
        ))
        x_prev = x_next
        if converged or u > u_max:
            break

    function, r2 = find_function(x[x_prev], y[x_prev])
    return ClubDiscovery(cluster_indices=x_prev, function=function, r2=r2, steps=steps, converged=converged)


@dataclass(frozen=True)
class Club:
    """Ein fertiger CluB: Punktindizes (in den ORIGINALEN x/y-Arrays) plus Funktion."""
    point_indices: np.ndarray
    function: Function
    r2: float


@dataclass(frozen=True)
class OuterStep:
    """Ein Durchlauf der aeusseren Schleife: Entdeckung genau eines CluBs."""
    club_index: int
    inner_steps: list
    final_indices: np.ndarray     # Indizes in den ORIGINALEN x/y-Arrays
    remaining_indices: np.ndarray  # D_t zu Beginn dieses Durchlaufs, als Indizes in x/y
    # (inner_steps' eigene cluster_indices/next_cluster_indices sind lokale Indizes IN
    # `remaining_indices`, nicht in den originalen x/y-Arrays - fuer die Visualisierung)


@dataclass(frozen=True)
class RunResult:
    outer_steps: list
    raw_clubs: list          # CluBs direkt aus der aeusseren Schleife, vor Post-Processing
    reassigned_clubs: list   # nach Reassign + Retrain
    final_clubs: list        # nach Merge - das Endergebnis
    n_min: int
    tau_jaccard: float
    u_max: int


def reassign(x, y, clubs):
    """Reassign + Retrain (Abschnitt 4.3): jeder Punkt aus dem GESAMTEN Datensatz wird
    dem CluB mit kleinstem Vorhersagefehler zugeordnet, jede Funktion wird danach auf
    ihrer neuen Mitgliedschaft neu gefittet ("Retrain")."""
    if not clubs:
        return []
    errs = np.stack([np.abs(c.function(x) - y) for c in clubs], axis=1)
    assignment = np.argmin(errs, axis=1)
    new_clubs = []
    for i in range(len(clubs)):
        idx = np.where(assignment == i)[0]
        if idx.size == 0:
            continue
        function, r2 = find_function(x[idx], y[idx])
        new_clubs.append(Club(point_indices=idx, function=function, r2=r2))
    return new_clubs


def merge_clubs(x, y, clubs, r_merge):
    """Merge (Abschnitt 4.3): wiederholt das beste Verschmelzungs-Kandidatenpaar
    zusammenfuehren, solange eines die Kriterien erfuellt - verschmolzenes R^2
    verbessert den gewichteten Mittelwert der Einzel-R^2, ODER uebersteigt r_merge.
    Terminiert, wenn keine Verschmelzung mehr das Kriterium erfuellt (oder nur noch
    ein CluB uebrig ist)."""
    clubs = list(clubs)
    while len(clubs) > 1:
        best = None
        for i in range(len(clubs)):
            for j in range(i + 1, len(clubs)):
                idx = np.concatenate([clubs[i].point_indices, clubs[j].point_indices])
                function, r2 = find_function(x[idx], y[idx])
                n_i, n_j = len(clubs[i].point_indices), len(clubs[j].point_indices)
                weighted_avg = (n_i * clubs[i].r2 + n_j * clubs[j].r2) / (n_i + n_j)
                if r2 >= weighted_avg or r2 >= r_merge:
                    if best is None or r2 > best[0]:
                        best = (r2, i, j, idx, function)
        if best is None:
            break
        r2, i, j, idx, function = best
        merged = Club(point_indices=idx, function=function, r2=r2)
        clubs = [c for k_, c in enumerate(clubs) if k_ not in (i, j)] + [merged]
    return clubs


def run(x, y, n_min, tau_jaccard, u_max, k_neighbors, tau_eps, r_merge, seed):
    """Algorithm 1 vollstaendig: aeussere Schleife (CluB-Entdeckung) gefolgt von
    Reassign, Retrain und Merge."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)

    remaining = np.arange(len(x))
    raw_clubs = []
    outer_steps = []

    while remaining.size >= n_min:
        discovery = find_club(x[remaining], y[remaining], k_neighbors, tau_eps, tau_jaccard, u_max, rng)
        global_indices = remaining[discovery.cluster_indices]
        raw_clubs.append(Club(point_indices=global_indices, function=discovery.function, r2=discovery.r2))
        outer_steps.append(OuterStep(
            club_index=len(raw_clubs) - 1, inner_steps=discovery.steps, final_indices=global_indices,
            remaining_indices=remaining,
        ))
        remaining = np.setdiff1d(remaining, global_indices, assume_unique=True)
        if global_indices.size == 0:
            # Sicherheitsgrenze dieser Demo: ein CluB ohne Punkte wuerde D_t nie
            # verkleinern und die aeussere Schleife nie terminieren lassen.
            break

    reassigned_clubs = reassign(x, y, raw_clubs) if raw_clubs else []
    final_clubs = merge_clubs(x, y, reassigned_clubs, r_merge) if reassigned_clubs else []

    return RunResult(
        outer_steps=outer_steps, raw_clubs=raw_clubs, reassigned_clubs=reassigned_clubs,
        final_clubs=final_clubs, n_min=n_min, tau_jaccard=tau_jaccard, u_max=u_max,
    )


def labels_from_clubs(n_points, clubs):
    """Wandelt eine CluB-Liste in ein Label-Array (n_points,) um - -1 fuer nicht
    zugeordnete Punkte (sollte nach `reassign` nicht mehr vorkommen)."""
    labels = np.full(n_points, -1, dtype=int)
    for label, club in enumerate(clubs):
        labels[club.point_indices] = label
    return labels
