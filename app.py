"""CluBS (Clustering Behavioural Similarity) - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Elftes Stück der "Konzepte"-Reihe und das ERSTE, das keinem der beiden bisherigen
Wurzelknoten (kmeans-demo, agglomerative-demo) entstammt: statt nach raeumlicher
Naehe/Dichte/Hierarchie zu clustern, gruppiert CluBS Datenpunkte danach, welcher
gemeinsamen mathematischen Funktion sie folgen - "Verhaltens-Aehnlichkeit" statt
geometrischer Naehe.

⚠️ Frisch aus der Forschung: Zdankin, Kummerow & Weis, "CluBS: Clustering
Behavioural Similarity", KDD'26 (DOI 10.1145/3770855.3817875, August 2026,
Referenzimplementierung: https://github.com/TheRustyStorm/CluBS). Das ist
unveroeffentlichte, noch nicht unabhaengig reproduzierte Forschung - KEIN
etablierter De-facto-Standard wie Leiden oder HDBSCAN. Diese Demo baut Algorithm 1
des Papers von Grund auf nach, mit EINER bewusst benannten Vereinfachung (siehe
README/Formulierungs-Expander): `FindFunction` durchsucht eine kleine Bibliothek von
Polynomgraden (1-3) statt offener symbolischer Regression.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import cb_constants as C
from cb_algorithm import labels_from_clubs
from cb_algorithm import run as run_clubs
from cb_evaluation import adjusted_rand_index, kmeans_then_fit_functions
from cb_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from cb_scenario import generate_polynomial_mixture
from cb_visualization import (
    build_inner_step_figure,
    build_metric_comparison_chart,
    build_scatter_with_curves,
)

st.set_page_config(page_title="CluBS – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _compute_scenario(n_points, degree_a, degree_b, offset, noise, seed):
    return generate_polynomial_mixture(n_points, degree_a, degree_b, offset=offset, noise_sigma=noise, seed=seed)


@st.cache_data(show_spinner=False)
def _compute_run(scenario, n_neighbors, tau_eps, r_merge, seed):
    n_min = C.n_min_for(len(scenario.x))
    return run_clubs(
        scenario.x, scenario.y, n_min=n_min, tau_jaccard=C.TAU_JACCARD, u_max=C.U_MAX,
        k_neighbors=n_neighbors, tau_eps=tau_eps, r_merge=r_merge, seed=seed,
    )


@st.cache_data(show_spinner=False)
def _compute_kmeans_baseline(scenario, seed):
    return kmeans_then_fit_functions(scenario.x, scenario.y, k=2, seed=seed)


st.title("📐 CluBS: Clustering nach Verhalten statt nach Position")

st.warning(
    "⚠️ **Frisch aus der Forschung, kein etablierter Standard.** Diese Demo baut "
    "*Zdankin, Kummerow & Weis, \"CluBS: Clustering Behavioural Similarity\", "
    "KDD'26* nach ([DOI 10.1145/3770855.3817875](https://doi.org/10.1145/3770855.3817875), "
    "August 2026, [Referenzimplementierung](https://github.com/TheRustyStorm/CluBS)) - "
    "unveröffentlichte, noch nicht unabhängig reproduzierte Forschung, kein De-facto-"
    "Standard wie Leiden oder HDBSCAN in dieser Reihe. Das Paper selbst benennt offen: "
    "*\"We do not provide any guarantees for convergence.\"*"
)

st.markdown(
    """
Jedes bisherige Stück dieser Reihe gruppiert nach **räumlicher Nähe, Dichte oder
Hierarchie**. CluBS stellt eine völlig andere Frage: gegeben Punkte $(x, y)$ - welche
Punkte wurden von **derselben mathematischen Funktion** erzeugt? Das ist eine echte
dritte Wurzel dieser Konzepte-Reihe, unabhängig von k-Means UND von agglomerativem
Clustering: eine Punktwolke, die räumlich völlig verschränkt aussieht, kann trotzdem
sauber nach Funktionszugehörigkeit trennbar sein - und umgekehrt.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere "
    "Verfahren vergleichen, zeigt diese Demo - Teil der wachsenden \"Konzepte\"-Reihe - "
    "**ein** Verfahren an einem wachsenden Beispiel."
)

with st.expander("So funktioniert CluBS", expanded=True):
    st.markdown(
        """
CluBS sucht wiederholt einen neuen **CluB** (Clustered Behaviour), solange genug
unzugeordnete Punkte übrig sind:

1. **Startcluster**: eine zufällig gewählte Ecke der Punktwolke plus ihre $k$
   nächsten Nachbarn.
2. **Innere Schleife**: abwechselnd eine Funktion an den aktuellen Cluster anpassen
   (`FindFunction`) und ALLE Restpunkte mit kleinem Vorhersagefehler neu einschließen
   (`UpdateCluster`) - bis sich der Cluster kaum noch ändert (Jaccard-Index) oder ein
   Iterationslimit erreicht ist.
3. Der gefundene CluB wird gespeichert, seine Punkte werden entfernt, weiter mit
   Schritt 1.

Am Ende folgt eine Nachbearbeitung: **jeder** Punkt wird dem CluB mit kleinstem
Fehler neu zugeordnet (Reassign), jede Funktion neu gefittet (Retrain), und ähnliche
CluB-Paare werden verschmolzen (Merge), solange das die Erklärgüte nicht verschlechtert.

**Vereinfachung dieser Demo**: `FindFunction` (im Paper offene Symbolische Regression)
sucht hier über eine kleine Bibliothek von Polynomgraden (1-3) - der niedrigste Grad,
der eine Gütegrenze erreicht, gewinnt. Das deckt sich mit dem sauberesten Benchmark
des Papers selbst (Abschnitt 5.1), der ebenfalls Polynom-Mischungen verwendet.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = {
    "Einfaches Beispiel": "Zwei klar getrennte Verhaltensweisen (großer Offset) - CluBS rekonstruiert beide Funktionen und die Partition zuverlässig.",
    "k-Means scheitert, CluBS nicht": "Die Kernaussage des Papers (Figure 1): die Punktwolken überlappen räumlich - eine k-Means-dann-Fit-Baseline scheitert, CluBS trennt weiterhin nach Funktionsverhalten.",
    "Ähnliche Funktionen verschmelzen": "Zwei sich stark ähnelnde Funktionen - der Verschmelzungs-Schwellenwert r_merge entscheidet live, ob sie getrennt bleiben.",
    "Keine Konvergenzgarantie": "Ehrlicher Härtefall: sehr ähnliche, verrauschte Funktionen - CluBS liefert hier eine sichere, aber falsche Partition. Keine beschönigte Erfolgsgarantie.",
}
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach "
    "kopieren, um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_points = st.slider("Anzahl Punkte", *bounds("n_points_slider"), key="n_points_slider")
    degree_a = st.selectbox(
        "Grad Funktion A", options=(1, 2, 3), key="degree_a_select",
        format_func=lambda d: f"{d} ({C.DEGREE_LABELS[d]})",
    )
    degree_b = st.selectbox(
        "Grad Funktion B", options=(1, 2, 3), key="degree_b_select",
        format_func=lambda d: f"{d} ({C.DEGREE_LABELS[d]})",
    )
    offset = st.slider(
        "Offset (Trennbarkeit)", *bounds("offset_slider"), key="offset_slider", step=0.5,
        help="Additiver Versatz auf Funktion B - im Paper genutzt, um die Trennbarkeit "
        "zu erhöhen. Bei 0 können sich die Punktwolken im Rohraum stark überlappen.",
    )
    noise = st.slider("Rauschen σ", *bounds("noise_slider"), key="noise_slider", step=0.05)
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**CluBS-Parameter**")
    n_neighbors = st.slider(
        "Startcluster-Größe k (kNN)", *bounds("n_neighbors_slider"), key="n_neighbors_slider",
    )
    tau_eps = st.slider(
        "Fehlerschwelle τ_ε", *bounds("tau_eps_slider"), key="tau_eps_slider", step=0.1,
        help="Punkte mit Vorhersagefehler unter dieser Schwelle gelten als demselben "
        "Verhalten zugehörig. Muss zur Rauschstärke passen.",
    )
    r_merge = st.slider(
        "Verschmelzungs-Schwelle r_merge", *bounds("r_merge_slider"), key="r_merge_slider", step=0.01,
        help="Zwei CluBs werden verschmolzen, wenn eine gemeinsame Funktion mindestens "
        "dieses R² erreicht. Höher = strenger, weniger Verschmelzungen.",
    )

    st.button(
        "🎲 Neue Punktwolke generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für die Punktwolke.",
    )

sync_query_params(n_points, degree_a, degree_b, offset, noise, seed, n_neighbors, tau_eps, r_merge)

with st.spinner("Führe CluBS aus..."):
    scenario = _compute_scenario(int(n_points), int(degree_a), int(degree_b), offset, noise, int(seed))
    result = _compute_run(scenario, int(n_neighbors), tau_eps, r_merge, int(seed))

st.markdown("## 🎯 CluBS in Aktion")

n_outer = len(result.outer_steps)
if n_outer == 0:
    st.error(
        "Bei dieser Konfiguration wurde kein einziger CluB gefunden (zu wenige Punkte "
        "relativ zur Mindestclustergröße). Bitte mehr Punkte oder andere Einstellungen wählen."
    )
    st.stop()

run_key = (n_points, degree_a, degree_b, offset, noise, seed, n_neighbors, tau_eps, r_merge)
if "cb_outer_step" not in st.session_state or st.session_state.get("cb_run_owner") != run_key:
    st.session_state["cb_outer_step"] = n_outer - 1
    st.session_state["cb_run_owner"] = run_key
    st.session_state["cb_inner_step"] = len(result.outer_steps[n_outer - 1].inner_steps) - 1

if n_outer == 1:
    outer_idx = 0
    st.caption("Genau ein CluB gefunden - keine Navigation zwischen mehreren Funden nötig.")
else:
    outer_idx = st.slider(
        "CluB-Fund (äußere Schleife)", 0, n_outer - 1, key="cb_outer_step",
        help="Welcher CluB-Fund gerade betrachtet wird - jeder Durchlauf der äußeren "
        "Schleife entdeckt genau einen neuen CluB.",
    )
outer_step = result.outer_steps[outer_idx]
n_inner = len(outer_step.inner_steps)

if st.session_state.get("cb_inner_step_owner") != outer_idx:
    st.session_state["cb_inner_step"] = n_inner - 1
    st.session_state["cb_inner_step_owner"] = outer_idx

if n_inner == 1:
    inner_idx = 0
    st.caption("Nach genau einem Verfeinerungsschritt konvergiert oder abgebrochen - kein Regler nötig.")
else:
    inner_idx = st.slider(
        "Verfeinerungsschritt (innere Schleife)", 0, n_inner - 1, key="cb_inner_step",
        help="FindFunction + UpdateCluster abwechselnd, bis der Jaccard-Index konvergiert.",
    )
inner_step = outer_step.inner_steps[inner_idx]

dt_x = scenario.x[outer_step.remaining_indices]
dt_y = scenario.y[outer_step.remaining_indices]

st.plotly_chart(
    build_inner_step_figure(
        dt_x, dt_y, inner_step.cluster_indices, inner_step.function,
        inner_step.next_cluster_indices, x_range=(scenario.x.min(), scenario.x.max()),
    ),
    width="stretch", key=f"inner_{outer_idx}_{inner_idx}",
)

im1, im2, im3 = st.columns(3)
im1.metric("Fit-Grad", inner_step.function.degree)
im2.metric("R² (Fit-Menge)", f"{inner_step.r2:.4f}")
im3.metric("Jaccard-Index", f"{inner_step.jaccard:.4f}", help="Konvergenz bei ≥ {:.2f}".format(1 - C.TAU_JACCARD))
if inner_step.converged:
    st.success(f"✅ Konvergiert nach {inner_idx + 1} Verfeinerungsschritt(en).")
elif inner_idx == n_inner - 1:
    st.info(f"ℹ️ Abgebrochen nach u_max={C.U_MAX} Schritten (keine Konvergenzgarantie, siehe Formulierungs-Expander).")

st.markdown("---")
st.markdown("### Endergebnis nach Reassign, Retrain und Merge")

final_labels_col, raw_col = st.columns(2)
final_functions = {i: c.function for i, c in enumerate(result.final_clubs)}
final_labels = labels_from_clubs(len(scenario.x), result.final_clubs)
final_labels_col.plotly_chart(
    build_scatter_with_curves(scenario.x, scenario.y, final_labels, final_functions),
    width="stretch", key="final_result",
)
with final_labels_col:
    st.caption(
        f"**{len(result.final_clubs)} CluB(s)** nach Post-Processing "
        f"(vorher: {len(result.raw_clubs)} aus der äußeren Schleife, "
        f"{len(result.reassigned_clubs)} nach Reassign)."
    )

raw_functions = {i: c.function for i, c in enumerate(result.raw_clubs)}
raw_labels = labels_from_clubs(len(scenario.x), result.raw_clubs)
raw_col.plotly_chart(
    build_scatter_with_curves(scenario.x, scenario.y, raw_labels, raw_functions),
    width="stretch", key="raw_result",
)
with raw_col:
    st.caption(
        "Direkt aus der äußeren Schleife, VOR Reassign/Merge - Punkte, die während der "
        "iterativen Suche nie in einen CluB aufgenommen wurden, erscheinen hier als "
        "'nicht zugeordnet' in der Legende; Reassign behebt das."
    )

st.markdown("---")

st.subheader("📐 Warum reicht räumliche Nähe nicht?")
st.markdown(
    """
Der Kernvergleich aus **Figure 1** des Papers: eine klassische **k-Means-dann-Fit**-
Baseline clustert zuerst nach räumlicher Nähe im $(x,y)$-Raum und passt danach pro
Cluster eine Funktion an. CluBS clustert direkt nach Funktionsverhalten. Live für Ihr
aktuelles Szenario:
"""
)

kmeans_labels, kmeans_functions = _compute_kmeans_baseline(scenario, C.COMPARISON_SEED)
kmeans_functions_only = {k: f for k, (f, r2) in kmeans_functions.items()}
clubs_ari = adjusted_rand_index(scenario.true_labels, final_labels)
kmeans_ari = adjusted_rand_index(scenario.true_labels, kmeans_labels)

cmp_col1, cmp_col2 = st.columns(2)
cmp_col1.plotly_chart(
    build_scatter_with_curves(scenario.x, scenario.y, kmeans_labels, kmeans_functions_only),
    width="stretch", key="kmeans_scatter",
)
cmp_col1.caption("k-Means (räumlich) dann Funktion pro Cluster gefittet.")
cmp_col2.plotly_chart(
    build_scatter_with_curves(scenario.x, scenario.y, final_labels, final_functions),
    width="stretch", key="clubs_scatter",
)
cmp_col2.caption("CluBS: direkt nach Funktionsverhalten geclustert.")

st.plotly_chart(build_metric_comparison_chart(clubs_ari, kmeans_ari), width="stretch", key="ari_comparison")

gap = clubs_ari - kmeans_ari
if gap > 0.2:
    st.success(
        f"✅ CluBS erreicht einen Adjusted Rand Index von {clubs_ari:.2f} gegenüber "
        f"{kmeans_ari:.2f} für k-Means-dann-Fit - bei überlappenden Punktwolken zahlt "
        f"sich das direkte Clustern nach Verhalten klar aus."
    )
elif clubs_ari < 0.5:
    st.warning(
        f"⚠️ Ehrlicher Befund: CluBS erreicht hier nur einen Adjusted Rand Index von "
        f"{clubs_ari:.2f} - das Paper benennt selbst offen, dass keine "
        f"Konvergenzgarantie besteht. Bei sehr ähnlichen Funktionen kann die "
        f"Fehlerschwelle τ_ε oder die Startcluster-Größe helfen, ist aber kein Free Lunch."
    )
else:
    st.info(
        f"ℹ️ CluBS ({clubs_ari:.2f}) und k-Means-dann-Fit ({kmeans_ari:.2f}) liegen hier "
        f"nah beieinander - bei dieser Trennbarkeit macht der Unterschied zwischen "
        f"räumlicher und funktionaler Clusterung wenig aus. Probieren Sie das Preset "
        f"\"k-Means scheitert, CluBS nicht\" für den Kontrastfall."
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem** (Zdankin, Kummerow & Weis, KDD'26): Datensatz
$D=\{(x_i,y_i)\}\subset\mathbb{R}\times\mathbb{R}$. Gesucht: eine Partition
$X_1,\dots,X_c$ und Funktionen $f_1,\dots,f_c$, die

$$
\sum_{i} \left(f_{c(i)}(x_i) - y_i\right)^2
$$

minimieren - wobei WEDER $c$ noch die Funktionen vorab bekannt sind.

**Algorithm 1** (vollständig nachgebaut in `cb_algorithm.py`):

1. **SelectInitialCluster** (kNN-Variante, laut Paper die bessere der beiden
   getesteten Varianten): eine zufällige Ecke der Bounding-Box der Restdaten $D_t$,
   plus deren $k$ nächste Nachbarn.
2. **Innere Schleife**: abwechselnd
   $f_{t,u} = \arg\min_{f\in\Omega}\sum_{x_i\in X_{t,u}}(f(x_i)-y_i)^2$
   (`FindFunction`) und $X_{t,u+1}=\{(x_i,y_i)\in D_t : |f_{t,u}(x_i)-y_i|\le\tau_\epsilon\}$
   (`UpdateCluster`), bis der Jaccard-Index
   $J(X_{t,u},X_{t,u-1})=\frac{|X_{t,u}\cap X_{t,u-1}|}{|X_{t,u}\cup X_{t,u-1}|}\ge 1-\tau$
   oder $u>u_{max}$.
3. **Reassign, Retrain, Merge**: jeder Punkt aus dem GESAMTEN $D$ wird dem CluB mit
   kleinstem Fehler neu zugeordnet, jede Funktion neu gefittet, und CluB-Paare werden
   verschmolzen, wenn das verschmolzene $R^2$ den gewichteten Mittelwert beider
   Einzel-$R^2$ übertrifft ODER eine feste Schwelle $r_{merge}$ übersteigt.

**Vereinfachung dieser Demo**: $\Omega$ ist im Paper eine offene, per genetischer
Programmierung durchsuchte Menge symbolischer Ausdrücke. Diese Demo ersetzt das durch
eine Bibliothek von Polynomgraden 1-3 (per kleinstem Quadrat gefittet) - der
niedrigste Grad, der eine $R^2$-Gütegrenze erreicht, gewinnt. Das deckt sich mit dem
saubersten, am leichtesten reproduzierbaren Benchmark des Papers selbst (Abschnitt
5.1, Ground-Truth Function Recovery), der ebenfalls Mischungen aus Polynomen
verwendet.

**Ehrlich benannt, nicht versteckt**: das Paper selbst gibt an, *"We do not provide
any guarantees for convergence"* - CluBS ist damit das erste Stück dieser Reihe ganz
ohne Konvergenz- oder Monotonie-Beweis (anders als k-Means' Trägheit, EM/VI's ELBO
oder Leidens Modularität). Das Preset "Keine Konvergenzgarantie" zeigt einen echten
Fehlschlag, analog zu Figure 6 im Appendix des Papers.

Implementiert in `cb_algorithm.py` (Algorithm 1), `cb_evaluation.py` (ARI/NMI/F1_macro,
k-Means-dann-Fit-Baseline) und `cb_scenario.py` (Polynom-Mischungen, Abschnitt 5.1).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
