"""
Projet : Marketing bancaire — qui appeler pour vendre un dépôt à terme ?
Données : UCI Bank Marketing (bank-additional-full) : 41 188 appels d'une banque portugaise (2008-2010),
          profil client, historique de campagne et contexte économique (Moro, Cortez & Rita, 2014)
Méthode : KPI de campagne -> taux de conversion par segment et par contexte économique
          -> piège de la durée d'appel (connue seulement après l'appel) -> modèle de ciblage
          -> courbe de gain et nombre d'appels économisés à objectif de ventes égal
"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/bank_marketing.csv", sep=";")
df["y"] = (df.y == "yes").astype(int)
df["deja_contacte"] = (df.pdays != 999).astype(int)          # 999 = jamais contacté (valeur sentinelle)

kpi = {"appels": len(df), "conversion": round(df.y.mean() * 100, 2), "ventes": int(df.y.sum()),
       "appels_par_vente": round(len(df) / df.y.sum(), 1), "duree_moy_min": round(df.duration.mean() / 60, 1),
       "inconnus": int((df == "unknown").sum().sum())}

def rate(col, order=None, bins=None, labels=None):
    s = pd.cut(df[col], bins, labels=labels) if bins is not None else df[col]
    g = df.groupby(s, observed=True).y.agg(["mean", "size"])
    if order: g = g.reindex([o for o in order if o in g.index])
    return {"labels": [str(i) for i in g.index], "taux": (g["mean"] * 100).round(1).tolist(), "n": g["size"].tolist()}

facteurs = {
    "Âge": rate("age", bins=[16, 25, 35, 45, 55, 65, 100], labels=["17-25", "26-35", "36-45", "46-55", "56-65", "66+"]),
    "Métier": rate("job"),
    "Canal de contact": rate("contact"),
    "Résultat de la campagne précédente": rate("poutcome", ["nonexistent", "failure", "success"]),
    "Mois de l'appel": rate("month", ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]),
    "Nombre d'appels dans la campagne": rate("campaign", bins=[0, 1, 2, 3, 5, 10, 60], labels=["1", "2", "3", "4-5", "6-10", "11+"]),
    "Taux Euribor 3 mois": rate("euribor3m", bins=[0, 1, 2, 4.9, 6], labels=["< 1 %", "1-2 %", "2-4,9 %", "≥ 4,9 %"]),
}
# Le contexte : conversion et Euribor dans l'ordre chronologique des données (le fichier est trié par date)
df["periode"] = np.arange(len(df)) // 1000
ctx = df.groupby("periode").agg(conv=("y", "mean"), euribor=("euribor3m", "mean"), emploi=("nr.employed", "mean"))
dur = rate("duration", bins=[-1, 60, 120, 180, 300, 600, 5000], labels=["< 1 min", "1-2 min", "2-3 min", "3-5 min", "5-10 min", "> 10 min"])

# ---------- Modèle de ciblage (sans la durée d'appel) ----------
feats = [c for c in df.columns if c not in ["y", "duration", "periode"]]
X = df[feats].copy()
for c in X.select_dtypes(["object", "string"]).columns: X[c] = X[c].astype("category")
X_tr, X_te, y_tr, y_te = train_test_split(X, df.y, test_size=0.3, stratify=df.y, random_state=0)
gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, categorical_features="from_dtype", random_state=0).fit(X_tr, y_tr)
p = gb.predict_proba(X_te)[:, 1]; auc = roc_auc_score(y_te, p)
# Avec la durée : score flatteur mais inutilisable (on ne connaît la durée qu'une fois l'appel terminé)
Xd = X.assign(duration=df.duration)
gbd = HistGradientBoostingClassifier(max_iter=300, categorical_features="from_dtype", random_state=0).fit(Xd.loc[X_tr.index], y_tr)
auc_dur = roc_auc_score(y_te, gbd.predict_proba(Xd.loc[X_te.index])[:, 1])

o = np.argsort(-p); yy = y_te.values[o]
cum = np.cumsum(yy) / yy.sum() * 100
pts = np.linspace(0, len(yy) - 1, 21).astype(int)
gain = {"x": [round(i / len(yy) * 100) for i in pts], "y": [round(cum[i], 1) for i in pts]}
need = int(np.searchsorted(cum, 80) + 1)                     # appels nécessaires pour 80 % des ventes
lift10 = cum[int(len(yy) * .1) - 1] / 10
cout_appel, marge_vente = 5, 120                             # hypothèses : coût d'un appel, marge d'un dépôt souscrit
prof = [{"pct": round(k * 100), "profit": round(yy[:max(1, int(len(yy) * k))].sum() * marge_vente - int(len(yy) * k) * cout_appel)} for k in np.linspace(.05, 1, 20)]

out = {"kpi": kpi, "facteurs": facteurs, "duree": dur,
       "ctx": {"conv": (ctx.conv * 100).round(1).tolist(), "euribor": ctx.euribor.round(2).tolist(), "emploi": ctx.emploi.round(0).tolist()},
       "model": {"auc": round(auc, 3), "auc_duree": round(auc_dur, 3), "gain": gain, "lift10": round(lift10, 2),
                 "appels_80": round(need / len(yy) * 100, 1), "profit": prof, "hyp": {"cout": cout_appel, "marge": marge_vente}}}
json.dump(out, open(ROOT / "results/p_bank.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(kpi); print(facteurs["Résultat de la campagne précédente"], facteurs["Taux Euribor 3 mois"], facteurs["Canal de contact"]); print(dur)
print(out["model"]["auc"], auc_dur, lift10, out["model"]["appels_80"]); print(max(prof, key=lambda x: x["profit"]))
