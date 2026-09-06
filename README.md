# CluBS – Clustering nach Verhalten statt nach Position – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-clubs-demo.streamlit.app/)**

Elftes Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning". Das **erste Stück, das keinem der beiden bisherigen
Wurzelknoten entstammt** (kmeans-demo, agglomerative-demo) - eine echte dritte,
unabhängige Clustering-Philosophie:

```
kmeans-demo → dbscan-demo ──┐
                             ├──> hdbscan-demo
              agglomerative-demo ──────────┘
kmeans-demo → gmm-demo → dpmm-demo
kmeans-demo → spectral-demo → leiden-demo
kmeans-demo ─┐
             ├──> divisive-demo
agglomerative-demo ─┘

clubs-demo   (eigene Wurzel: Verhaltens- statt Positions-Clustering)
```

Jedes bisherige Stück gruppiert nach **räumlicher Nähe, Dichte oder Hierarchie**.
CluBS stellt eine andere Frage: gegeben Punkte $(x,y)$ - welche wurden von
**derselben mathematischen Funktion** erzeugt? Eine Punktwolke, die räumlich völlig
verschränkt aussieht, kann sauber nach Funktionszugehörigkeit trennbar sein - und
umgekehrt.

## ⚠️ Über diese Forschung

Diese Demo baut **Algorithm 1** aus

> Peter Zdankin, Arne Kummerow, Torben Weis. **"CluBS: Clustering Behavioural
> Similarity."** KDD '26, August 09-13, 2026, Jeju Island, Republic of Korea.
> DOI: [10.1145/3770855.3817875](https://doi.org/10.1145/3770855.3817875)
> Referenzimplementierung: [github.com/TheRustyStorm/CluBS](https://github.com/TheRustyStorm/CluBS)

von Grund auf nach. **Das ist unveröffentlichte, noch nicht unabhängig
reproduzierte Forschung** (Erscheinungsdatum August 2026) - anders als Leiden
(De-facto-Standard der Netzwerk-Community-Detection) oder HDBSCAN in dieser Reihe
ist CluBS **kein etablierter Standard**. Das Paper selbst benennt offen: *"We do not
provide any guarantees for convergence."* Diese Demo verschweigt das nicht, sondern
zeigt es explizit (Preset "Keine Konvergenzgarantie").

**Die eine bewusste Vereinfachung**: `FindFunction` (der Symbolic-Regression-
Baustein des Papers) ist im Original eine offene, per genetischer Programmierung
durchsuchte Menge von Ausdrücken. Diese Demo ersetzt das durch eine Bibliothek von
**Polynomgraden 1-3** (per kleinstem Quadrat gefittet, niedrigster Grad bevorzugt,
der eine R²-Gütegrenze erreicht). Das ist keine willkürliche Abkürzung: der
sauberste, am leichtesten reproduzierbare Benchmark des Papers selbst (Abschnitt
5.1, "Ground-Truth Function Recovery") verwendet exakt Mischungen aus zwei
zufälligen Polynomen - unsere Vereinfachung deckt sich direkt mit diesem Experiment,
das diese Demo nachbaut.

## Der Algorithmus

CluBS sucht wiederholt einen neuen **CluB** (Clustered Behaviour), solange genug
unzugeordnete Punkte übrig sind:

1. **SelectInitialCluster** (kNN-Variante, laut Paper die bessere der beiden
   getesteten Varianten): eine zufällig gewählte Ecke der Bounding-Box der
   Restdaten, plus deren $k$ nächste Nachbarn.
2. **Innere Schleife**: abwechselnd eine Funktion an den aktuellen Cluster anpassen
   (`FindFunction`) und ALLE Restpunkte mit Vorhersagefehler unter einer Schwelle
   $\tau_\epsilon$ neu einschließen (`UpdateCluster`) - bis der Jaccard-Index
   zwischen zwei Iterationen $\ge 1-\tau$ ist oder ein Iterationslimit $u_{max}$
   erreicht wird.
3. **Reassign, Retrain, Merge**: jeder Punkt aus dem GESAMTEN Datensatz wird dem
   CluB mit kleinstem Fehler neu zugeordnet, jede Funktion neu gefittet, und
   CluB-Paare werden verschmolzen, wenn das verschmolzene $R^2$ den gewichteten
   Mittelwert beider Einzel-$R^2$ übertrifft ODER eine feste Schwelle $r_{merge}$
   übersteigt.

## Presets

- **Einfaches Beispiel**: zwei klar getrennte Verhaltensweisen (großer Offset) -
  CluBS rekonstruiert beide Funktionen und die Partition zuverlässig (ARI = 1.0).
- **k-Means scheitert, CluBS nicht** (die Kernaussage aus **Figure 1** des Papers):
  kein Offset, sich kreuzende Funktionen - die Punktwolken überlappen im Rohraum
  stark. Eine k-Means-dann-Fit-Baseline erreicht hier nur ARI ≈ 0.10, CluBS ≈ 0.97.
- **Ähnliche Funktionen verschmelzen**: zwei sich stark ähnelnde Geraden werden bei
  der Standard-Verschmelzungs-Schwelle $r_{merge}=0.85$ fälschlich zu einem CluB
  verschmolzen - der $r_{merge}$-Regler zeigt live, wie ein höherer Wert (≥ 0.92 in
  diesem Szenario) die Trennung wiederherstellt (analog zu Figure 3 im Paper).
- **Keine Konvergenzgarantie**: ein ehrlicher Härtefall (sehr ähnliche, verrauschte
  Funktionen) - CluBS liefert eine sichere, aber falsche Partition (ARI ≈ 0.02).
  Keine beschönigte Erfolgsgarantie, analog zu den im Appendix des Papers
  dokumentierten Fehlschlägen.

## Visualisierung

Ein **verschachtelter Schrittregler**: der äußere Regler wählt den gerade
betrachteten CluB-Fund, der innere blättert durch dessen Verfeinerungs-Iterationen
(Fit-Menge, neu berechnete Menge, angepasste Kurve je Schritt) - im
Illustrationsstil von Figure 2 im Paper. Das Endergebnis wird VOR und NACH
Reassign/Retrain/Merge gegenübergestellt. Der front-and-center Methodenvergleich
reproduziert **Figure 1** des Papers direkt: zwei Streudiagramme (k-Means-dann-Fit
vs. CluBS) mit überlagerten gefitteten Kurven, plus ein Adjusted-Rand-Index-Balken.

## Sicherheitsgrenzen

`N_POINTS_HARD_MAX` (400) hält Merge (worst-case $O(nc^2)$) und den Live-Regler
schnell genug für eine flüssige Bedienung. Ein leeres `UpdateCluster`-Ergebnis oder
ein CluB ohne Punkte brechen die jeweilige Schleife sicher ab (im Paper nicht
spezifiziert, siehe Kommentare in `cb_algorithm.py`).

## Verifikation

- **Handgerechnete Beispiele**: `FindFunction` auf einer exakt linearen Punktmenge
  liefert Grad 1 mit exakten Koeffizienten; $R^2$ und Jaccard-Index von Hand
  nachgerechnet.
- **Ground-Truth Function Recovery, Paper-Methodik direkt nachgebaut** (Abschnitt
  5.1): Mischungen aus zwei zufälligen Polynomen, Adjusted Rand Index über mehrere
  Seeds gegen die wahre Funktionszugehörigkeit.
- **Struktur-Invarianten**: jeder Punkt landet nach Reassign in genau einem CluB
  (Partitions-Eigenschaft), Merge kann die Cluster-Anzahl nur verringern.
- **Ehrlich fehlende Konvergenzgarantie direkt getestet, nicht verschwiegen**: ein
  nachgebauter Härtefall zeigt nachweislich reduzierte, aber ehrlich gezeigte
  Performance.
- **Figure-1-Kernvergleich direkt getestet**: eine frische, eigenständige
  k-Means-dann-Fit-Baseline (kein Cross-Import) scheitert nachweislich bei
  überlappenden Punktwolken, wo CluBS weiterhin korrekt trennt.
- **ARI/NMI/F1_macro from scratch**, inklusive Label-Permutation für beliebige
  Cluster-Indizes (auch bei ungleicher Cluster-Anzahl zwischen Vorhersage und
  Wahrheit) - Handbeispiele in `tests/test_evaluation.py`.
- **Alle vier Presets direkt gegen das tatsächliche App-Verhalten getestet**
  (etabliertes Muster dieser Reihe) plus ein AppTest-Smoke-Test über die
  verschachtelten Schrittregler.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, verschachtelter Schrittregler, Figure-1-Methodenvergleich, Formulierungs-Expander mit Forschungs-Kennzeichnung |
| `cb_constants.py` | Defaults, Regler-Grenzen, Sicherheitsgrenzen, `PRESETS` |
| `cb_presets.py` | `SettingSpec`/`SETTING_SPECS`, Permalink-Logik, Presets, Zufalls-Seed-Button |
| `cb_scenario.py` | Mischungen aus zwei zufälligen Polynomen (Grad 1-3), Abschnitt 5.1 des Papers nachgebaut |
| `cb_algorithm.py` | Algorithm 1 vollständig: `select_initial_cluster`, `find_function`, `update_cluster`, Jaccard-Index, innere+äußere Schleife mit vollem Protokoll, `reassign`, `merge_clubs` |
| `cb_evaluation.py` | ARI, NMI, F1_macro (from scratch, mit Label-Permutation), k-Means-dann-Fit-Baseline (frisch, kein Cross-Import) |
| `cb_visualization.py` | Punktwolke mit überlagerten Funktionskurven, verschachtelte Schritt-Ansicht, Methodenvergleichsdiagramm (Plotly) |
| `tests/` | Handbeispiele, Ground-Truth-Recovery, Struktur-Invarianten, Härtefall-Test, Figure-1-Kernvergleich, Preset-gegen-App-Verhalten-Tests, AppTest-Smoke-Test |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
