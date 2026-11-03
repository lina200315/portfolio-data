"""
Projet : Prévoir la demande horaire de vélos en libre-service (Capital Bikeshare, Washington D.C.)
Données : UCI Bike Sharing (17 379 heures, 2011-2012, météo et calendrier)
Méthode : profils d'usage (abonnés vs occasionnels) -> effet météo -> 3 modèles testés sur les 4 derniers mois
          (naïf « même heure la semaine dernière », régression linéaire, gradient boosting)
          -> erreur par heure -> besoin de rééquilibrage des stations
"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/bike_hour.csv", parse_dates=["dteday"])
df["ts"] = df.dteday + pd.to_timedelta(df.hr, unit="h")
df["temp_c"] = df.temp * 47 - 8              # dénormalisation documentée par l'UCI : t = (t - t_min) / (t_max - t_min)
df["hum_pct"] = df.hum * 100
df["vent_kmh"] = df.windspeed * 67

# ---------- Profils ----------
prof = df.groupby(["workingday", "hr"])[["casual", "registered"]].mean().round(1)
profils = {f"{t}_{w}": prof.loc[w][t].tolist() for w in (0, 1) for t in ("casual", "registered")}
meteo = df.groupby("weathersit").cnt.agg(["mean", "size"]).round(1)
meteo.index = meteo.index.map({1: "Dégagé", 2: "Brume / nuages", 3: "Pluie ou neige légère", 4: "Forte pluie"})
df["tbin"] = pd.cut(df.temp_c, range(-10, 45, 4))
temp = df.groupby("tbin", observed=True).cnt.mean().round(1)
mois = df.groupby([df.yr, df.mnth]).cnt.sum()
kpi = {"heures": len(df), "locations": int(df.cnt.sum()), "part_abonnes": round(df.registered.sum() / df.cnt.sum() * 100, 1),
       "croissance": round((df[df.yr == 1].cnt.sum() / df[df.yr == 0].cnt.sum() - 1) * 100, 1),
       "pic": int(df.cnt.max()), "moy_h": round(df.cnt.mean(), 1)}

# ---------- Modèles ----------
df = df.sort_values("ts").set_index("ts")
full = df.asfreq("h")                                         # quelques heures manquent dans les données
df["lag_168"] = full.cnt.shift(168).reindex(df.index)         # même heure, une semaine avant
df["lag_24"] = full.cnt.shift(24).reindex(df.index)
feats = ["hr", "weekday", "workingday", "holiday", "season", "mnth", "yr", "weathersit", "temp_c", "atemp", "hum_pct", "vent_kmh", "lag_24", "lag_168"]
d = df.dropna(subset=["lag_168", "lag_24"])
cut = pd.Timestamp("2012-09-01")
tr, te = d[d.index < cut], d[d.index >= cut]
res = {}
res["Naïf (semaine précédente)"] = te.lag_168.values
X_lin = lambda x: pd.get_dummies(x[feats].astype({"hr": "category", "weekday": "category", "mnth": "category", "weathersit": "category"}), drop_first=True).astype(float)
Xa = X_lin(pd.concat([tr, te])); rg = Ridge(alpha=1).fit(Xa.loc[tr.index], tr.cnt)
res["Régression linéaire"] = rg.predict(Xa.loc[te.index]).clip(min=0)
gb = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=48, loss="poisson", random_state=0)
gb.fit(tr[feats], tr.cnt); res["Gradient boosting"] = gb.predict(te[feats])
perf = {k: {"mae": round(mean_absolute_error(te.cnt, v), 1), "r2": round(r2_score(te.cnt, v), 3),
            "wape": round(np.abs(te.cnt - v).sum() / te.cnt.sum() * 100, 1)} for k, v in res.items()}
best = min(perf, key=lambda k: perf[k]["mae"])
err_h = pd.Series(np.abs(te.cnt.values - res[best]), index=te.hr).groupby(level=0).mean().round(1)
imp = permutation_importance(gb, te[feats], te.cnt, n_repeats=4, random_state=0, scoring="neg_mean_absolute_error")
NOMS = {"hr": "Heure", "weekday": "Jour", "workingday": "Jour ouvré", "holiday": "Férié", "season": "Saison", "mnth": "Mois", "yr": "Année",
        "weathersit": "Météo", "temp_c": "Température", "atemp": "Température ressentie", "hum_pct": "Humidité", "vent_kmh": "Vent",
        "lag_24": "Demande la veille", "lag_168": "Demande sem. précédente"}
imp = pd.Series(imp.importances_mean, index=[NOMS[f] for f in feats]).sort_values(ascending=False).round(1)
wk = te.loc["2012-10-15":"2012-10-21"]
semaine = {"ts": [t.strftime("%a %d/%m %Hh") for t in wk.index], "reel": wk.cnt.tolist(),
           "pred": np.round(gb.predict(wk[feats])).tolist()}

out = {"kpi": kpi, "profils": profils, "meteo": {"labels": meteo.index.tolist(), "moy": meteo["mean"].tolist(), "n": meteo["size"].tolist()},
       "temp": {"labels": [f"{int(i.left)} à {int(i.right)} °C" for i in temp.index], "moy": temp.tolist()},
       "mois": {"labels": [f"{m:02d}/{2011 + y}" for y, m in mois.index], "v": mois.tolist()},
       "perf": perf, "best": best, "err_h": err_h.tolist(), "imp": imp.head(10).to_dict(), "semaine": semaine,
       "test": {"debut": "01/09/2012", "fin": "31/12/2012", "n": len(te)}}
json.dump(out, open(ROOT / "results/p_bike.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(kpi); print(perf); print(meteo); print(imp.head(6)); print(err_h.idxmax(), err_h.max())
