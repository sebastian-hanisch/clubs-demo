"""Plotly-Visualisierungen: Punktwolke mit ueberlagerten gefitteten Funktionskurven (neuer
Diagrammtyp fuer diese Reihe), verschachtelte Schritt-fuer-Schritt-Ansicht des inneren
CluBS-Loops (im Illustrationsstil von Figure 2 des Papers) und der direkte
Figure-1-Methodenvergleich gegen eine k-Means-dann-Fit-Baseline."""

import numpy as np

CLUSTER_PALETTE = [
    "#1f77b4", "#d68a2e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def _cluster_color(index, n_clusters):
    if n_clusters <= len(CLUSTER_PALETTE):
        return CLUSTER_PALETTE[index % len(CLUSTER_PALETTE)]
    hue = (index * 360.0 / n_clusters) % 360
    return f"hsl({hue:.1f}, 65%, 50%)"


def _x_grid(x, padding=0.05, n=200):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    pad = (xmax - xmin) * padding or 1.0
    return np.linspace(xmin - pad, xmax + pad, n)


def build_scatter_with_curves(x, y, labels, functions, height=460, legend=True, curve_x=None):
    """Punktwolke, eingefaerbt nach `labels` (ein CluB je Farbe), mit der je Label in
    `functions` (dict label -> Function) gefitteten Kurve ueberlagert.

    Die y-Achse wird bewusst auf den Datenbereich begrenzt (mit Polsterung), NICHT auf
    den Wertebereich der Kurven: ein CluB mit sehr wenigen Punkten kann (v.a. VOR
    Reassign/Merge) ein Polynom liefern, das ausserhalb der beobachteten Punkte massiv
    extrapoliert - ohne diese Grenze wuerde eine einzige entartete Kurve die gesamte
    Achse stauchen und alle anderen CluBs unlesbar machen."""
    import plotly.graph_objects as go

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    labels = np.asarray(labels)
    grid = curve_x if curve_x is not None else _x_grid(x)

    y_min, y_max = float(y.min()), float(y.max())
    y_pad = (y_max - y_min) * 0.1 or 1.0
    y_range = [y_min - y_pad, y_max + y_pad]

    cluster_ids = sorted(set(labels.tolist()))
    n_clusters = len(cluster_ids)
    show_legend = legend and n_clusters <= 10

    fig = go.Figure()
    for index, cid in enumerate(cluster_ids):
        mask = labels == cid
        color = _cluster_color(index, n_clusters)
        name = f"CluB {cid + 1}" if cid >= 0 else "nicht zugeordnet"
        fig.add_trace(go.Scatter(
            x=x[mask], y=y[mask], mode="markers", name=name, showlegend=show_legend,
            marker=dict(color=color, size=7, line=dict(width=0.5, color="white")),
            hoverinfo="skip", legendgroup=str(cid),
        ))
        if functions is not None and cid in functions:
            fig.add_trace(go.Scatter(
                x=grid, y=functions[cid](grid), mode="lines",
                line=dict(color=color, width=2.5), showlegend=False,
                legendgroup=str(cid), hoverinfo="skip",
            ))

    fig.update_layout(
        template="plotly_white", height=height,
        xaxis=dict(title="x", fixedrange=True),
        yaxis=dict(title="y", fixedrange=True, range=y_range),
        showlegend=legend,
        margin=dict(t=40 if legend else 5, l=10, r=10, b=10),
    )
    if legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def build_inner_step_figure(dt_x, dt_y, cluster_indices, function, next_indices, x_range):
    """Ein Schritt der inneren Schleife: alle Restpunkte D_t (blass), die aktuelle
    Fit-Menge X_{t,u} (kraeftig, Kreise) und die aus f_{t,u} resultierende naechste
    Menge X_{t,u+1} (kraeftig umrandet) - macht das Fit-dann-Neuzuordnen-Wechselspiel
    direkt sichtbar."""
    import plotly.graph_objects as go

    dt_x = np.asarray(dt_x, dtype=float)
    dt_y = np.asarray(dt_y, dtype=float)
    in_current = np.zeros(len(dt_x), dtype=bool)
    in_current[cluster_indices] = True
    in_next = np.zeros(len(dt_x), dtype=bool)
    in_next[next_indices] = True

    grid = np.linspace(x_range[0], x_range[1], 200)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dt_x[~in_next], y=dt_y[~in_next], mode="markers", name="übrige Restpunkte D_t",
        marker=dict(color="#c7ccd6", size=6), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dt_x[in_current], y=dt_y[in_current], mode="markers", name="Fit-Menge X_(t,u)",
        marker=dict(color="#1f77b4", size=9, symbol="circle-open", line=dict(width=2)),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dt_x[in_next], y=dt_y[in_next], mode="markers", name="neue Menge X_(t,u+1)",
        marker=dict(color="#d62728", size=6), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=grid, y=function(grid), mode="lines", name="f_(t,u)",
        line=dict(color="#1f77b4", width=2.5), hoverinfo="skip",
    ))

    fig.update_layout(
        template="plotly_white", height=420,
        xaxis=dict(title="x", fixedrange=True, range=[x_range[0], x_range[1]]),
        yaxis=dict(title="y", fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=40, l=10, r=10, b=10),
    )
    return fig


def build_metric_comparison_chart(clubs_ari, kmeans_ari):
    """Balkendiagramm: Adjusted Rand Index von CluBS gegen die k-Means-dann-Fit-
    Baseline auf demselben Szenario - der direkte Figure-1-Kernvergleich des Papers."""
    import plotly.graph_objects as go

    labels = ["CluBS", "k-Means dann Fit"]
    values = [clubs_ari, kmeans_ari]
    colors = ["#2ca02c", "#d68a2e"]

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
    fig.update_layout(
        template="plotly_white", height=300,
        yaxis=dict(title="Adjusted Rand Index", range=[min(0, min(values)) - 0.05, 1.05], fixedrange=True),
        xaxis=dict(fixedrange=True), margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig
