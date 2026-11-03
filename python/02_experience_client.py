"""
Projet : Expérience client d'une compagnie aérienne — quels leviers font la satisfaction ?
Données : Airline Passenger Satisfaction (129 880 passagers, 14 critères notés de 0 à 5)
Méthode : KPI de satisfaction par segment -> impact des retards -> key driver analysis
          (importance x performance) -> modèle prédictif de satisfaction
"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/airline_satisfaction.csv")
df["sat"] = (df["Satisfaction"] == "Satisfied").astype(int)
df["Arrival Delay"] = df["Arrival Delay"].fillna(df["Departure Delay"])  # 393 valeurs manquantes

CRIT = ["Departure and Arrival Time Convenience", "Ease of Online Booking", "Check-in Service",
        "Online Boarding", "Gate Location", "On-board Service", "Seat Comfort", "Leg Room Service",
        "Cleanliness", "Food and Drink", "In-flight Service", "In-flight Wifi Service",
        "In-flight Entertainment", "Baggage Handling"]
FR = {"Departure and Arrival Time Convenience": "Horaires", "Ease of Online Booking": "Réservation en ligne",
      "Check-in Service": "Enregistrement", "Online Boarding": "Embarquement en ligne",
      "Gate Location": "Accès à la porte", "On-board Service": "Service à bord",
      "Seat Comfort": "Confort du siège", "Leg Room Service": "Espace jambes", "Cleanliness": "Propreté",
      "Food and Drink": "Restauration", "In-flight Service": "Service en vol",
      "In-flight Wifi Service": "Wi-Fi", "In-flight Entertainment": "Divertissement",
      "Baggage Handling": "Bagages"}

# ---------- KPI ----------
# Les notes 0 = "non concerné" : on les exclut des moyennes (sinon elles tirent les scores vers le bas)
scores = {FR[c]: round(df.loc[df[c] > 0, c].mean(), 2) for c in CRIT}
csat = df.sat.mean() * 100
kpi = {"passagers": len(df), "csat": round(csat, 1),
       "retard_moy_arrivee": round(df["Arrival Delay"].mean(), 1),
       "part_retard_15": round((df["Arrival Delay"] > 15).mean() * 100, 1),
       "note_moyenne": round(np.mean(list(scores.values())), 2)}

def sat_by(col):
    g = df.groupby(col).sat.agg(["mean", "size"]).reset_index()
    return [{"label": str(r[col]), "taux": round(r["mean"] * 100, 1), "n": int(r["size"])} for _, r in g.iterrows()]

segments = {c: sat_by(c) for c in ["Class", "Type of Travel", "Customer Type"]}
cross = (df.pivot_table(index="Class", columns="Type of Travel", values="sat", aggfunc="mean") * 100).round(1)
cross = {"lignes": cross.index.tolist(), "colonnes": cross.columns.tolist(), "valeurs": cross.values.tolist()}

# Impact du retard à l'arrivée
bins = [-1, 0, 15, 30, 60, 120, 2000]
labels = ["À l'heure", "1-15 min", "16-30", "31-60", "61-120", "> 2 h"]
df["delay_band"] = pd.cut(df["Arrival Delay"], bins, labels=labels)
delay = sat_by("delay_band")

# Âge
df["age_band"] = pd.cut(df.Age, [0, 17, 25, 35, 45, 55, 65, 100],
                        labels=["< 18", "18-25", "26-35", "36-45", "46-55", "56-65", "> 65"])
age = sat_by("age_band")

# ---------- Key driver analysis ----------
# Importance = coefficient standardisé d'une régression logistique (effet d'un écart-type de note)
Xd = df[CRIT].replace(0, np.nan)
Xd = Xd.fillna(Xd.median())
Xs = StandardScaler().fit_transform(Xd)
lr = LogisticRegression(max_iter=2000).fit(Xs, df.sat)
coef = dict(zip(CRIT, lr.coef_[0]))
drivers = [{"critere": FR[c], "importance": round(coef[c], 3), "note": scores[FR[c]],
            "corr": round(np.corrcoef(Xd[c], df.sat)[0, 1], 3)} for c in CRIT]
mi, mn = np.mean([d["importance"] for d in drivers]), np.mean([d["note"] for d in drivers])
for d in drivers:
    hi, good = d["importance"] >= mi, d["note"] >= mn
    d["quadrant"] = ("Priorité d'amélioration" if hi and not good else
                     "Atout à protéger" if hi and good else
                     "Surqualité possible" if good else "Secondaire")

# ---------- Modèle prédictif ----------
X = pd.get_dummies(df[CRIT + ["Age", "Flight Distance", "Departure Delay", "Arrival Delay",
                              "Class", "Type of Travel", "Customer Type", "Gender"]], drop_first=True).astype(float)
X_tr, X_te, y_tr, y_te = train_test_split(X, df.sat, test_size=0.25, stratify=df.sat, random_state=1)
rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=1).fit(X_tr, y_tr)
p = rf.predict_proba(X_te)[:, 1]
model = {"auc": round(roc_auc_score(y_te, p), 4), "accuracy": round(accuracy_score(y_te, p > .5) * 100, 1),
         "baseline": round(max(y_te.mean(), 1 - y_te.mean()) * 100, 1)}
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(8)
model["top"] = [{"var": FR.get(k, k), "imp": round(v, 3)} for k, v in fi.items()]

out = {"kpi": kpi, "scores": scores, "segments": segments, "cross": cross, "delay": delay,
       "age": age, "drivers": drivers, "model": model}
json.dump(out, open(ROOT / "results/p_cx.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ["kpi", "model", "cross"]}, ensure_ascii=False, indent=1))
for d in sorted(drivers, key=lambda d: -d["importance"]): print(d)
print(delay)
