"""
Projet : Valeur client (CLV), cohortes et prédiction de la valeur future
Données : Superstore Sales — clients identifiés par nom + province (voir le projet qualité des données)
Méthode : Pareto du CA -> cohortes de rétention trimestrielles -> segmentation RFM
          -> prédire le CA 2011-2012 d'un client à partir de son comportement 2009-2010
"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/superstore.csv", encoding="latin1")
df["date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
df["client"] = df["Customer Name"] + " | " + df["Province"]          # clé fiable (projet qualité)
orders = df.groupby(["client", "Order ID"]).agg(date=("date", "min"), ca=("Sales", "sum"),
                                                 profit=("Profit", "sum"), seg=("Customer Segment", "first")).reset_index()

# ---------- 1. Pareto ----------
cl = orders.groupby("client").agg(ca=("ca", "sum"), profit=("profit", "sum"), commandes=("Order ID", "count"),
                                  premiere=("date", "min"), derniere=("date", "max"), seg=("seg", "first"))
s = cl.ca.sort_values(ascending=False)
cum = s.cumsum() / s.sum() * 100
pareto = {"part_clients": [round(x, 1) for x in np.linspace(0, 100, 21)],
          "part_ca": [0.0] + [round(cum.iloc[max(int(len(s) * p / 100) - 1, 0)], 1) for p in np.linspace(5, 100, 20)]}
top20 = round(cum.iloc[int(len(s) * .2) - 1], 1)
neg = int((cl.profit < 0).sum())

# ---------- 2. Cohortes trimestrielles ----------
orders["cohorte"] = orders.groupby("client").date.transform("min").dt.to_period("Q")
orders["periode"] = orders.date.dt.to_period("Q")
orders["age"] = (orders.periode - orders.cohorte).apply(lambda x: x.n)
co = orders.groupby(["cohorte", "age"]).client.nunique().unstack(fill_value=0)
co = co.loc[co.index <= pd.Period("2010Q4")]                  # cohortes avec assez de recul
ret = (co.div(co[0], axis=0) * 100).round(1)
cohort = {"cohortes": [str(c) for c in ret.index], "taille": co[0].tolist(),
          "valeurs": [[None if (ret.index[i] + j) > pd.Period("2012Q4") else v for j, v in enumerate(r)][:9]
                      for i, r in enumerate(ret.values.tolist())]}
avg_ret = [round(ret[j][ret.index + j <= pd.Period("2012Q4")].mean(), 1) for j in range(1, 9)]

# ---------- 3. RFM ----------
ref = df.date.max() + pd.Timedelta(days=1)
cl["recence"] = (ref - cl.derniere).dt.days
cl["R"] = pd.qcut(cl.recence, 5, labels=[5, 4, 3, 2, 1]).astype(int)
cl["F"] = pd.qcut(cl.commandes.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
cl["M"] = pd.qcut(cl.ca, 5, labels=[1, 2, 3, 4, 5]).astype(int)
def seg(r):
    if r.R >= 4 and r.F >= 4: return "Champions"
    if r.R >= 3 and r.F >= 3: return "Fidèles"
    if r.R >= 4 and r.F <= 2: return "Nouveaux / prometteurs"
    if r.R <= 2 and r.F >= 3: return "À risque"
    if r.R <= 2 and r.F <= 2: return "Perdus"
    return "À surveiller"
cl["rfm"] = cl.apply(seg, axis=1)
rfm = cl.groupby("rfm").agg(clients=("ca", "size"), ca=("ca", "sum"), panier=("ca", "mean"), recence=("recence", "mean")).reset_index()
rfm["part_ca"] = rfm.ca / rfm.ca.sum() * 100
rfm = rfm.sort_values("ca", ascending=False).round(1).to_dict("records")

# ---------- 4. CLV prédictive ----------
cut = pd.Timestamp("2011-01-01")
past = orders[orders.date < cut]; fut = orders[orders.date >= cut]
X = past.groupby("client").agg(ca=("ca", "sum"), n=("Order ID", "count"), panier=("ca", "mean"),
                               profit=("profit", "sum"), derniere=("date", "max"), premiere=("date", "min"))
X["recence"] = (cut - X.derniere).dt.days; X["anciennete"] = (cut - X.premiere).dt.days
X = X.drop(columns=["derniere", "premiere"])
X = X.join(pd.get_dummies(cl.seg).astype(int), how="left")
y = fut.groupby("client").ca.sum().reindex(X.index).fillna(0)
ylog = np.log1p(y); Xl = X.copy(); Xl[["ca", "panier"]] = np.log1p(Xl[["ca", "panier"]].clip(lower=0))
kf = KFold(5, shuffle=True, random_state=3)
models = {"Moyenne historique (référence)": None, "Régression linéaire": LinearRegression(),
          "Gradient boosting": GradientBoostingRegressor(loss="huber", n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=3)}
perf, preds = {}, {}
for name, m in models.items():
    if m is None:
        p = X.ca.values * len(fut.date.dt.year.unique()) / len(past.date.dt.year.unique())                                        # "il dépensera autant que les 2 années passées"
    else:
        # Cible en euros (pas en log) : on veut une estimation en valeur, utilisable pour un budget
        p = cross_val_predict(m, Xl, y, cv=kf).clip(min=0)
    preds[name] = p
    perf[name] = {"r2": round(r2_score(y, p), 3), "mae": round(mean_absolute_error(y, p)),
                  "spearman": round(pd.Series(p).corr(pd.Series(y.values), method="spearman"), 3)}
best = max(perf, key=lambda k: perf[k]["spearman"])   # objectif métier : bien classer les clients
p = preds[best]
d = pd.DataFrame({"p": p, "y": y.values})
d["dec"] = pd.qcut(d.p.rank(method="first"), 10, labels=range(1, 11))
calib = d.groupby("dec", observed=True).agg(pred=("p", "mean"), reel=("y", "mean")).round(0)
top10 = d.nlargest(int(len(d) * .1), "p").y.sum() / d.y.sum() * 100

kpi = {"clients": int(len(cl)), "ca_moy_client": round(cl.ca.mean()), "commandes_moy": round(cl.commandes.mean(), 1),
       "top20": top20, "clients_deficitaires": neg, "retention_t1": avg_ret[0], "retention_t4": avg_ret[3],
       "clients_modele": int(len(X)), "top10_capte": round(top10, 1)}
out = {"kpi": kpi, "pareto": pareto, "cohort": cohort, "avg_ret": avg_ret, "rfm": rfm, "perf": perf, "best": best,
       "calib": {"dec": [int(i) for i in calib.index], "pred": calib.pred.tolist(), "reel": calib.reel.tolist()}}
json.dump(out, open(ROOT / "results/p_clv.json", "w"), ensure_ascii=False, indent=1, default=float)
print(kpi, perf, avg_ret, sep="\n"); print(pd.DataFrame(rfm))
