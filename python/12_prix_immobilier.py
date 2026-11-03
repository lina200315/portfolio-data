"""
Projet : Estimer le prix d'une maison — du nettoyage au modèle, et comprendre ses erreurs
Données : Ames Housing (2 930 ventes, Ames, Iowa, 2006-2010, 82 variables) — Dean De Cock, Journal of Statistics Education
Méthode : valeurs manquantes "qui veulent dire quelque chose" -> création de variables -> log du prix
          -> régression Ridge vs gradient boosting en validation croisée -> analyse des erreurs par quartier
"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/ames_housing.csv", index_col=0)
n0 = len(df)

# ---------- 1. Nettoyage ----------
# L'auteur recommande de retirer les maisons de plus de 4 000 sq ft (ventes atypiques)
df = df[df["Gr Liv Area"] <= 4000].copy()
miss = df.isna().mean().sort_values(ascending=False)
miss = (miss[miss > 0].head(12) * 100).round(1)
# NA = "absence de l'équipement" pour ces colonnes (piscine, garage, sous-sol...) : ce n'est pas une donnée manquante
none_cols = ["Pool QC", "Misc Feature", "Alley", "Fence", "Fireplace Qu", "Garage Type", "Garage Finish", "Garage Qual",
             "Garage Cond", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2", "Mas Vnr Type"]
df[none_cols] = df[none_cols].fillna("Aucun")
df["Lot Frontage"] = df.groupby("Neighborhood")["Lot Frontage"].transform(lambda s: s.fillna(s.median()))
num = df.select_dtypes("number").columns.drop(["SalePrice", "PID", "Order"])
df[num] = df[num].fillna(0)

# ---------- 2. Variables créées ----------
qual = {"Aucun": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
for c in ["Exter Qual", "Kitchen Qual", "Bsmt Qual", "Fireplace Qu", "Garage Qual", "Heating QC"]:
    df[c + " (score)"] = df[c].map(qual).fillna(0)
df["Surface totale"] = df["Gr Liv Area"] + df["Total Bsmt SF"]
df["Âge à la vente"] = df["Yr Sold"] - df["Year Built"]
df["Années depuis rénovation"] = df["Yr Sold"] - df["Year Remod/Add"]
df["Salles de bain"] = df["Full Bath"] + 0.5 * df["Half Bath"] + df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"]
df["Prix au m²"] = df.SalePrice / (df["Gr Liv Area"] * 0.0929)

# ---------- 3. Exploration ----------
corr = df.select_dtypes("number").drop(columns=["PID", "Order", "Prix au m²"]).corr()["SalePrice"].drop("SalePrice")
corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(12).round(3)
quart = df.groupby("Neighborhood").agg(prix_median=("SalePrice", "median"), m2=("Prix au m²", "median"), ventes=("SalePrice", "size"))
quart = quart[quart.ventes >= 20].sort_values("prix_median", ascending=False).round(0)
qual_price = df.groupby("Overall Qual").SalePrice.median().round(0)
hist = np.histogram(df.SalePrice / 1000, bins=range(0, 640, 20))

# ---------- 4. Modèles ----------
y = np.log(df.SalePrice)
cat = df.select_dtypes(["object", "string"]).columns
X = pd.get_dummies(df.drop(columns=["SalePrice", "PID", "Order", "Prix au m²"]), columns=cat, drop_first=True).astype(float)
kf = KFold(5, shuffle=True, random_state=42)
models = {"Ridge (linéaire régularisé)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 30))),
          "Gradient boosting": HistGradientBoostingRegressor(max_iter=700, learning_rate=0.04, max_leaf_nodes=20, random_state=0)}
pred, perf = {}, {}
for k, m in models.items():
    p = np.exp(cross_val_predict(m, X, y, cv=kf)); pred[k] = p
    perf[k] = {"mae": round(mean_absolute_error(df.SalePrice, p)), "r2": round(r2_score(df.SalePrice, p), 3),
               "mape": round(np.mean(np.abs(p / df.SalePrice - 1)) * 100, 1)}
avg = (pred[list(models)[0]] + pred[list(models)[1]]) / 2
perf["Moyenne des deux"] = {"mae": round(mean_absolute_error(df.SalePrice, avg)), "r2": round(r2_score(df.SalePrice, avg), 3),
                            "mape": round(np.mean(np.abs(avg / df.SalePrice - 1)) * 100, 1)}
best = min(perf, key=lambda k: perf[k]["mae"])
bp = avg if best == "Moyenne des deux" else pred[best]
df["pred"] = bp; df["err"] = (bp / df.SalePrice - 1) * 100
err_q = df.groupby("Neighborhood").err.agg(["mean", lambda s: s.abs().mean(), "size"])
err_q.columns = ["biais", "erreur_abs", "n"]
err_q = err_q[err_q.n >= 20].sort_values("erreur_abs", ascending=False).round(1)
s = df.sample(600, random_state=1)
scatter = [[round(a / 1000, 1), round(b / 1000, 1)] for a, b in zip(s.SalePrice, s.pred)]
within10 = round((df.err.abs() <= 10).mean() * 100, 1)

rid = models["Ridge (linéaire régularisé)"].fit(X, y)
coef = pd.Series(rid[-1].coef_, index=X.columns)               # effet d'un écart-type de la variable sur log(prix)
coef = coef.reindex(coef.abs().sort_values(ascending=False).index).head(12)
top_coef = {k: round((np.exp(v) - 1) * 100, 1) for k, v in coef.items()}

out = {"kpi": {"ventes": len(df), "retirees": n0 - len(df), "variables": int(df.shape[1]), "variables_modele": int(X.shape[1]),
               "prix_median": int(df.SalePrice.median()), "mae": perf[best]["mae"], "mape": perf[best]["mape"], "within10": within10},
       "miss": miss.to_dict(), "corr": corr.to_dict(), "quartiers": quart.reset_index().to_dict("records"),
       "qual_price": {int(k): v for k, v in qual_price.items()}, "hist": {"x": hist[1][:-1].tolist(), "y": hist[0].tolist()},
       "perf": perf, "best": best, "err_q": err_q.reset_index().head(10).to_dict("records"), "scatter": scatter, "top_coef": top_coef}
json.dump(out, open(ROOT / "results/p_house.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(out["kpi"]); print(perf); print(corr.head(8)); print(err_q.head(5)); print(top_coef)
