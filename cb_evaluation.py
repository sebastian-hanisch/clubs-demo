"""Bewertungsmetriken (from scratch, kein sklearn) fuer den Ground-Truth-Recovery-
Nachweis - ARI, NMI, F1_macro (mit Label-
Permutation) - plus eine frische k-Means-dann-Fit-Baseline (kein Cross-Import) fuer
den Kernvergleich dieser Demo."""

from itertools import permutations

import numpy as np

from cb_algorithm import find_function


def _contingency_matrix(true_labels, pred_labels):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    true_ids = np.unique(true_labels)
    pred_ids = np.unique(pred_labels)
    table = np.zeros((len(true_ids), len(pred_ids)), dtype=float)
    for i, t in enumerate(true_ids):
        for j, p in enumerate(pred_ids):
            table[i, j] = np.sum((true_labels == t) & (pred_labels == p))
    return table


def adjusted_rand_index(true_labels, pred_labels):
    """Adjusted Rand Index, aus der Kontingenztabelle (Standardformel, from scratch)."""
    table = _contingency_matrix(true_labels, pred_labels)
    n = table.sum()
    if n == 0:
        return 1.0

    def comb2(v):
        return v * (v - 1) / 2.0

    sum_comb_c = np.sum([comb2(v) for v in table.sum(axis=1)])
    sum_comb_k = np.sum([comb2(v) for v in table.sum(axis=0)])
    sum_comb = np.sum(comb2(table))
    expected = sum_comb_c * sum_comb_k / comb2(n) if n >= 2 else 0.0
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return (sum_comb - expected) / denom


def normalized_mutual_information(true_labels, pred_labels):
    """NMI (arithmetisches Mittel der Normierung), from scratch."""
    table = _contingency_matrix(true_labels, pred_labels)
    n = table.sum()
    if n == 0:
        return 1.0
    p_true = table.sum(axis=1) / n
    p_pred = table.sum(axis=0) / n
    p_joint = table / n

    mi = 0.0
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if p_joint[i, j] > 0:
                mi += p_joint[i, j] * np.log(p_joint[i, j] / (p_true[i] * p_pred[j]))

    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    h_true = entropy(p_true)
    h_pred = entropy(p_pred)
    if h_true == 0 and h_pred == 0:
        return 1.0
    if h_true + h_pred == 0:
        return 0.0
    return 2.0 * mi / (h_true + h_pred)


def _best_label_permutation_accuracy_and_f1(true_labels, pred_labels):
    """Findet - per Bruteforce ueber alle Permutationen der (wenigen) Cluster-Labels -
    die Zuordnung Vorhersage->Wahrheit, die die Genauigkeit maximiert, und liefert
    Accuracy plus F1_macro unter dieser Zuordnung. Nur fuer kleine Clusterzahlen
    praktikabel (hier: 2-6 CluBs), ueblich fuer Clustering-Metriken mit beliebigen Cluster-Indizes."""
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    true_ids = np.unique(true_labels)
    pred_ids = np.unique(pred_labels)

    best_acc = -1.0
    best_mapping = None
    # Permutiere ueber die laengere Label-Menge, damit auch bei ungleicher
    # Clusterzahl (z.B. CluBS findet mehr/weniger CluBs als es wahre Funktionen gibt)
    # jede Vorhersage-ID einer wahren ID zugeordnet werden kann.
    if len(pred_ids) <= len(true_ids):
        for perm in permutations(true_ids, len(pred_ids)):
            mapping = dict(zip(pred_ids, perm))
            mapped = np.array([mapping[p] for p in pred_labels])
            acc = np.mean(mapped == true_labels)
            if acc > best_acc:
                best_acc, best_mapping = acc, mapped
    else:
        for perm in permutations(pred_ids, len(true_ids)):
            mapping = dict(zip(perm, true_ids))
            mapped = np.array([mapping.get(p, -1) for p in pred_labels])
            acc = np.mean(mapped == true_labels)
            if acc > best_acc:
                best_acc, best_mapping = acc, mapped

    mapped_pred = best_mapping
    f1s = []
    for t in true_ids:
        tp = np.sum((mapped_pred == t) & (true_labels == t))
        fp = np.sum((mapped_pred == t) & (true_labels != t))
        fn = np.sum((mapped_pred != t) & (true_labels == t))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return best_acc, float(np.mean(f1s))


def accuracy_and_f1_macro(true_labels, pred_labels):
    return _best_label_permutation_accuracy_and_f1(true_labels, pred_labels)


def kmeans_1d_then_fit_labels(x, y, k, seed, n_restarts=5, max_iter=100):
    """Frische, eigenstaendige k-Means-Implementierung (kein Cross-Import aus
    kmeans-demo) auf der (x,y)-Punktwolke - genau der Baseline-Ansatz des Kernvergleichs:
    erst raeumlich clustern, dann PRO Cluster eine Funktion fitten. Liefert
    nur die Cluster-Zuordnung (Rueckgabe passend zu `find_function`-basierten Labels)."""
    points = np.column_stack([x, y])
    rng = np.random.default_rng(seed)

    best_labels, best_inertia = None, np.inf
    for _ in range(n_restarts):
        centers = points[rng.choice(len(points), size=k, replace=False)]
        labels = np.zeros(len(points), dtype=int)
        for _ in range(max_iter):
            dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels) and _ > 0:
                labels = new_labels
                break
            labels = new_labels
            for c in range(k):
                if np.any(labels == c):
                    centers[c] = points[labels == c].mean(axis=0)
        dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        inertia = np.sum(np.min(dists, axis=1) ** 2)
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels


def kmeans_then_fit_labels(x, y, k, seed, n_restarts=5):
    """Alias mit dem kuerzeren Namen (`kmeans_then_fit`): raeumliches
    k-Means auf (x,y), gefolgt von je einer `find_function`-Anpassung pro Cluster
    (zur Vollstaendigkeit hier zurueckgegeben, fuer den reinen Partitions-Vergleich
    reicht die Cluster-Zuordnung selbst)."""
    return kmeans_1d_then_fit_labels(x, y, k, seed, n_restarts=n_restarts)


def kmeans_then_fit_functions(x, y, k, seed, n_restarts=5):
    """Wie `kmeans_then_fit_labels`, liefert zusaetzlich die pro Cluster gefittete
    Funktion (fuer die Visualisierung des Methodenvergleichs)."""
    labels = kmeans_1d_then_fit_labels(x, y, k, seed, n_restarts=n_restarts)
    functions = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        function, r2 = find_function(x[idx], y[idx])
        functions[int(c)] = (function, r2)
    return labels, functions
