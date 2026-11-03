"""
Projet : Consommation électrique horaire — profils, prévision à J+1 et détection d'anomalies
Données : PJM East (réseau électrique de l'est des États-Unis), 145 366 relevés horaires en MW, 2002-2018
Méthode : contrôle des horodatages (doublons, trous, changement d'heure) -> profils heure × jour, saisons
          -> prévision du lendemain (naïf, gradient boosting avec calendrier et décalages ≥ 24 h)
          -> anomalies : jours où la consommation s'écarte fortement de la prévision
"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "data/pjm_energy.csv", parse_dates=["Datetime"])

# ---------- 1. Qualité des horodatages ----------
dup = int(raw.Datetime.duplicated().sum())                   # heures en double (passage à l'heure d'hiver)
s = raw.groupby("Datetime").PJME_MW.mean().sort_index()
full = s.asfreq("h")
holes = int(full.isna().sum())                               # heures manquantes (passage à l'heure d'été)
s = full.interpolate(limit=3)
quality = {"lignes": len(raw), "doublons": dup, "trous": holes, "debut": s.index.min().strftime("%d/%m/%Y"), "fin": s.index.max().strftime("%d/%m/%Y")}

# ---------- 2. Profils ----------
d = s.to_frame("mw")
d["h"], d["dow"], d["m"], d["y"] = d.index.hour, d.index.dayofweek, d.index.month, d.index.year
heat = d.pivot_table(index="dow", columns="h", values="mw", aggfunc="mean").round(0)
mois = d.groupby("m").mw.mean().round(0)
annuel = d[(d.y > 2002) & (d.y < 2018)].groupby("y").mw.agg(["mean", "max"]).round(0)
saison = {nom: d[d.m.isin(ms)].groupby("h").mw.mean().round(0).tolist()
          for nom, ms in {"Hiver": [12, 1, 2], "Printemps": [3, 4, 5], "Été": [6, 7, 8], "Automne": [9, 10, 11]}.items()}

# ---------- 3. Prévision J+1 (on ne connaît que les données jusqu'à la veille) ----------
d["lag24"] = d.mw.shift(24); d["lag48"] = d.mw.shift(48); d["lag168"] = d.mw.shift(168)
d["moy7j"] = d.mw.shift(24).rolling(168).mean()
d["doy"] = d.index.dayofyear
hol = pd.to_datetime([f"{y}-{md}" for y in range(2002, 2019) for md in ["01-01", "07-04", "12-25", "12-24", "12-31"]])
d["ferie"] = d.index.normalize().isin(hol).astype(int)
feats = ["h", "dow", "m", "doy", "ferie", "lag24", "lag48", "lag168", "moy7j"]
d = d.dropna()
cut = pd.Timestamp("2016-01-01")
tr, te = d[d.index < cut], d[d.index >= cut]
gb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, max_leaf_nodes=48, random_state=0).fit(tr[feats], tr.mw)
p = gb.predict(te[feats])
mape = lambda a, f: float(np.mean(np.abs(f / a - 1)) * 100)
perf = {"Naïf (même heure la veille)": round(mape(te.mw, te.lag24), 2),
        "Naïf (même heure la semaine précédente)": round(mape(te.mw, te.lag168), 2),
        "Gradient boosting": round(mape(te.mw, p), 2)}
te = te.assign(pred=p, err=te.mw - p)
wk = te.loc["2017-07-17":"2017-07-23"]
semaine = {"ts": [t.strftime("%a %Hh") for t in wk.index], "reel": wk.mw.round(0).tolist(), "pred": wk.pred.round(0).tolist(),
           "naif": wk.lag24.round(0).tolist()}

# ---------- 4. Anomalies (sur tout l'historique, en validation "hors échantillon" par année) ----------
allp = pd.Series(index=d.index, dtype=float)
for y in range(2004, 2019):
    m_tr = d[d.y != y]; m_te = d[d.y == y]
    if len(m_te) == 0: continue
    mdl = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=32, random_state=0).fit(m_tr[feats], m_tr.mw)
    allp[m_te.index] = mdl.predict(m_te[feats])
daily = pd.DataFrame({"reel": d.mw, "pred": allp}).dropna().resample("D").sum()
daily["ecart"] = (daily.reel / daily.pred - 1) * 100
z = (daily.ecart - daily.ecart.mean()) / daily.ecart.std()
an = daily[z.abs() > 4].assign(z=z).sort_values("ecart").round(1)
anomalies = [{"date": i.strftime("%d/%m/%Y"), "ecart": r.ecart, "z": round(r.z, 1)} for i, r in an.head(12).iterrows()]
dist = np.histogram(daily.ecart.clip(-40, 20), bins=np.arange(-40, 21, 2))

out = {"kpi": {"heures": len(raw), "annees": 16, "moy": round(s.mean()), "record": round(s.max()), "record_date": s.idxmax().strftime("%d/%m/%Y %Hh"),
               "mape": perf["Gradient boosting"], "anomalies": int(len(an))},
       "quality": quality, "heat": heat.values.tolist(), "mois": mois.tolist(),
       "annuel": {"annees": annuel.index.tolist(), "moy": annuel["mean"].tolist(), "max": annuel["max"].tolist()},
       "saison": saison, "perf": perf, "semaine": semaine, "anomalies": anomalies,
       "dist": {"x": dist[1][:-1].tolist(), "y": dist[0].tolist()}}
json.dump(out, open(ROOT / "results/p_energy.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(out["kpi"], quality); print(perf); print(anomalies)
