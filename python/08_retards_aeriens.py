"""
Projet : Ponctualité de 336 776 vols au départ de New York (2013) — SQL analytique avec DuckDB + modèle prédictif
Données : nycflights13 (vols, compagnies, avions) — Bureau of Transportation Statistics
Méthode : requêtes SQL sur un fichier Parquet (DuckDB) -> KPI de ponctualité (OTP) -> effet cascade dans la journée
          -> âge des avions -> prédire un retard > 15 min avec les seules infos connues avant le vol
"""
import json, pathlib
import duckdb, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = pathlib.Path(__file__).resolve().parents[1]
con = duckdb.connect()
con.execute(f"CREATE VIEW vols AS SELECT * FROM '{ROOT / 'data/nyc_flights.parquet'}'")
con.execute(f"CREATE VIEW compagnies AS SELECT * FROM read_csv_auto('{ROOT / 'data/nyc_airlines.csv'}')")
con.execute(f"CREATE VIEW avions AS SELECT * FROM read_csv_auto('{ROOT / 'data/nyc_planes.csv'}', nullstr='NA')")
con.execute(f"CREATE VIEW aeroports AS SELECT * FROM read_csv_auto('{ROOT / 'data/nyc_airports.csv'}', nullstr='NA')")

Q = {}
def q(name, sql):
    df = con.execute(sql).df()
    Q[name] = {"sql": sql.strip(), "cols": list(df.columns), "rows": df.round(2).astype(object).where(df.notna(), None).values.tolist()}
    return df

kpi = q("KPI globaux", """
SELECT COUNT(*)                                                    AS vols,
       COUNT(DISTINCT carrier)                                     AS compagnies,
       COUNT(DISTINCT dest)                                        AS destinations,
       ROUND(100.0 * AVG(CASE WHEN dep_time IS NULL THEN 1 ELSE 0 END), 2) AS pct_annules,
       ROUND(100.0 * AVG(CASE WHEN arr_delay <= 15 THEN 1 ELSE 0 END) FILTER (WHERE arr_delay IS NOT NULL), 1) AS otp,
       ROUND(AVG(arr_delay), 1)                                    AS retard_moy_arrivee,
       ROUND(MEDIAN(arr_delay), 1)                                 AS retard_median_arrivee
FROM vols;""").iloc[0].to_dict()

cie = q("Ponctualité par compagnie", """
SELECT c.name AS compagnie, COUNT(*) AS vols,
       ROUND(100.0 * AVG(CASE WHEN v.arr_delay <= 15 THEN 1 ELSE 0 END) FILTER (WHERE v.arr_delay IS NOT NULL), 1) AS otp,
       ROUND(AVG(v.arr_delay), 1) AS retard_moy,
       ROUND(100.0 * AVG(CASE WHEN v.dep_time IS NULL THEN 1 ELSE 0 END), 2) AS pct_annules,
       RANK() OVER (ORDER BY AVG(CASE WHEN v.arr_delay <= 15 THEN 1 ELSE 0 END) FILTER (WHERE v.arr_delay IS NOT NULL) DESC) AS rang
FROM vols v JOIN compagnies c USING (carrier)
GROUP BY c.name HAVING COUNT(*) > 1000 ORDER BY otp DESC;""")

heure = q("Effet cascade : retard selon l'heure de départ prévue", """
SELECT hour AS heure, COUNT(*) AS vols,
       ROUND(AVG(dep_delay), 1) AS retard_depart_moy,
       ROUND(100.0 * AVG(CASE WHEN dep_delay > 15 THEN 1 ELSE 0 END), 1) AS pct_retard_15
FROM vols WHERE dep_delay IS NOT NULL AND hour BETWEEN 5 AND 23
GROUP BY hour ORDER BY hour;""")

mois = q("Saisonnalité mensuelle", """
SELECT month AS mois, COUNT(*) AS vols,
       ROUND(AVG(arr_delay), 1) AS retard_moy,
       ROUND(100.0 * AVG(CASE WHEN dep_time IS NULL THEN 1 ELSE 0 END), 2) AS pct_annules
FROM vols GROUP BY month ORDER BY month;""")

orig = q("Aéroports de départ", """
SELECT a.name AS aeroport, v.origin, COUNT(*) AS vols, ROUND(AVG(v.dep_delay), 1) AS retard_depart_moy,
       ROUND(AVG(v.taxi_like), 1) AS duree_vol_moy
FROM (SELECT *, air_time AS taxi_like FROM vols) v JOIN aeroports a ON a.faa = v.origin
GROUP BY a.name, v.origin ORDER BY vols DESC;""")

age = q("Âge de l'avion et retard (jointure avions)", """
WITH v AS (
  SELECT 2013 - p.year AS age, f.arr_delay
  FROM vols f JOIN avions p USING (tailnum)
  WHERE p.year IS NOT NULL AND f.arr_delay IS NOT NULL)
SELECT CASE WHEN age < 5 THEN '0-4 ans' WHEN age < 10 THEN '5-9 ans' WHEN age < 15 THEN '10-14 ans'
            WHEN age < 20 THEN '15-19 ans' ELSE '20 ans et +' END AS tranche_age,
       COUNT(*) AS vols, ROUND(AVG(arr_delay), 1) AS retard_moy,
       ROUND(100.0 * AVG(CASE WHEN arr_delay > 15 THEN 1 ELSE 0 END), 1) AS pct_retard_15
FROM v GROUP BY 1 ORDER BY MIN(age);""")

routes = q("Routes les plus chargées et leur fiabilité", """
SELECT origin || ' → ' || dest AS route, COUNT(*) AS vols,
       ROUND(100.0 * AVG(CASE WHEN arr_delay <= 15 THEN 1 ELSE 0 END) FILTER (WHERE arr_delay IS NOT NULL), 1) AS otp,
       ROUND(QUANTILE_CONT(arr_delay, 0.9), 0) AS p90_retard
FROM vols GROUP BY 1 ORDER BY vols DESC LIMIT 12;""")

# ---------- Modèle : retard > 15 min, avec les infos connues avant le vol ----------
df = con.execute("""
SELECT month, day, dayofweek(make_date(year, month, day)) AS dow, hour, carrier, origin, dest, distance,
       CASE WHEN arr_delay > 15 THEN 1 ELSE 0 END AS y
FROM vols WHERE arr_delay IS NOT NULL""").df()
for c in ["carrier", "origin", "dest"]:
    df[c] = df[c].astype("category")
df["distance"] = df.distance / 1000                          # en milliers de miles (aide la régression à converger)
# Le mois n'est pas utilisé : les mois de test (oct.-déc.) n'existent pas dans l'entraînement
X = df[["dow", "hour", "carrier", "origin", "dest", "distance"]]
tr, te = df.month <= 9, df.month >= 10                        # entraînement janv.-sept., test oct.-déc.
gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, categorical_features="from_dtype", random_state=0)
gb.fit(X[tr], df.y[tr]); p = gb.predict_proba(X[te])[:, 1]
Xd = pd.get_dummies(X, columns=["carrier", "origin", "dest", "hour", "dow"], drop_first=True).astype(float)
lr = LogisticRegression(max_iter=3000).fit(Xd[tr], df.y[tr]); pl = lr.predict_proba(Xd[te])[:, 1]
auc_gb, auc_lr = roc_auc_score(df.y[te], p), roc_auc_score(df.y[te], pl)
fpr, tpr, _ = roc_curve(df.y[te], p); ix = np.linspace(0, len(fpr) - 1, 40).astype(int)
d = pd.DataFrame({"p": p, "y": df.y[te].values})
d["dec"] = pd.qcut(d.p.rank(method="first"), 10, labels=range(1, 11))
dec = d.groupby("dec", observed=True).agg(pred=("p", "mean"), reel=("y", "mean"))

out = {"kpi": {k: (float(v) if v is not None else None) for k, v in kpi.items()}, "queries": Q,
       "model": {"auc_gb": round(auc_gb, 3), "auc_lr": round(auc_lr, 3), "base": round(df.y[te].mean() * 100, 1),
                 "n_train": int(tr.sum()), "n_test": int(te.sum()),
                 "roc": {"fpr": fpr[ix].round(3).tolist(), "tpr": tpr[ix].round(3).tolist()},
                 "dec": {"pred": (dec.pred * 100).round(1).tolist(), "reel": (dec.reel * 100).round(1).tolist()}}}
json.dump(out, open(ROOT / "results/p_flights.json", "w"), ensure_ascii=False, indent=1, default=str)
print(kpi); print(cie); print(heure.head(3), heure.tail(3)); print(age); print(out["model"]["auc_gb"], out["model"]["auc_lr"], out["model"]["base"], dec)
