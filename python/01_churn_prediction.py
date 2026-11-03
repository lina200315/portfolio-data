"""
Projet : Prédire le churn client d'un opérateur télécom et cibler une campagne de rétention
Données : IBM Telco Customer Churn (7 043 clients, 21 variables)
Méthode : EDA -> KPI de churn -> régression logistique vs gradient boosting
          -> choix du seuil selon la rentabilité de la campagne de rétention
"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.inspection import permutation_importance

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/telco.csv")

# ---------- 1. Nettoyage ----------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_missing = int(df["TotalCharges"].isna().sum())          # nouveaux clients (tenure = 0)
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df["churn"] = (df["Churn"] == "Yes").astype(int)

# ---------- 2. KPI ----------
kpi = {
    "clients": len(df),
    "taux_churn": round(df.churn.mean() * 100, 1),
    "mrr_total": round(df.MonthlyCharges.sum(), 0),
    "mrr_perdu": round(df.loc[df.churn == 1, "MonthlyCharges"].sum(), 0),
    "anciennete_moy": round(df.tenure.mean(), 1),
    "panier_moyen": round(df.MonthlyCharges.mean(), 2),
    "valeurs_manquantes_corrigees": n_missing,
}
kpi["part_mrr_perdu"] = round(kpi["mrr_perdu"] / kpi["mrr_total"] * 100, 1)

def churn_by(col):
    g = df.groupby(col)["churn"].agg(["mean", "size"]).reset_index()
    return [{"label": str(r[col]), "taux": round(r["mean"] * 100, 1), "n": int(r["size"])} for _, r in g.iterrows()]

seg = {c: churn_by(c) for c in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "SeniorCitizen"]}
df["tenure_band"] = pd.cut(df.tenure, [-1, 6, 12, 24, 48, 72],
                           labels=["0-6 mois", "7-12", "13-24", "25-48", "49-72"])
seg["tenure_band"] = churn_by("tenure_band")

# ---------- 3. Modélisation ----------
X = pd.get_dummies(df.drop(columns=["customerID", "Churn", "churn", "tenure_band"]), drop_first=True).astype(float)
y = df.churn
X_tr, X_te, y_tr, y_te, mrr_tr, mrr_te = train_test_split(
    X, y, df.MonthlyCharges, test_size=0.3, stratify=y, random_state=42)

models = {
    "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Gradient boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
}
perf, rocs, probas = {}, {}, {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    probas[name] = p
    fpr, tpr, _ = roc_curve(y_te, p)
    idx = np.linspace(0, len(fpr) - 1, 40).astype(int)
    rocs[name] = {"fpr": fpr[idx].round(3).tolist(), "tpr": tpr[idx].round(3).tolist()}
    perf[name] = {"auc": round(roc_auc_score(y_te, p), 3),
                  "precision@0.5": round(precision_score(y_te, p > .5), 3),
                  "rappel@0.5": round(recall_score(y_te, p > .5), 3)}
best = max(perf, key=lambda k: perf[k]["auc"])
p = probas[best]

# Importance des variables (permutation, sur le jeu de test)
imp = permutation_importance(models[best], X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=0)
importance = (pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False).head(10))
importance = [{"var": k, "imp": round(v, 4)} for k, v in importance.items()]

# Lift par décile : le modèle concentre-t-il les churners en haut de liste ?
d = pd.DataFrame({"p": p, "y": y_te.values})
d["decile"] = pd.qcut(d.p.rank(method="first", ascending=False), 10, labels=range(1, 11))
lift = d.groupby("decile", observed=True).y.mean() / d.y.mean()
lift = [round(v, 2) for v in lift.values]

# ---------- 4. Seuil "business" ----------
# Hypothèses : offre de rétention = 1 mois offert ; acceptée par 35 % des churners ciblés ;
# un client sauvé rapporte 12 mois de facture ; coût de contact 5 €.
COST_CONTACT, SUCCESS, MONTHS = 5, 0.35, 12
rows = []
for s in np.arange(0.05, 0.95, 0.025):
    target = p >= s
    tp = target & (y_te.values == 1)
    gain = (mrr_te.values[tp] * MONTHS * SUCCESS).sum()
    cost = target.sum() * COST_CONTACT + (mrr_te.values[target]).sum() * 1.0  # 1 mois offert
    rows.append({"seuil": round(s, 3), "cibles": int(target.sum()), "profit": round(gain - cost, 0)})
best_row = max(rows, key=lambda r: r["profit"])

out = {"kpi": kpi, "segments": seg, "perf": perf, "best": best, "roc": rocs,
       "importance": importance, "lift": lift, "seuils": rows, "seuil_opt": best_row,
       "hypotheses": {"cout_contact": COST_CONTACT, "taux_succes": SUCCESS, "mois": MONTHS}}
(ROOT / "results").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "results/p_churn.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ["kpi", "perf", "seuil_opt"]}, ensure_ascii=False, indent=1))
print(importance[:5], lift)
