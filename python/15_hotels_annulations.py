"""
Projet : Revenue management hôtelier — anticiper les annulations pour mieux remplir les chambres
Données : Hotel Booking Demand (119 390 réservations, un hôtel urbain et un resort au Portugal, 2015-2017)
          — Antonio, Almeida & Nunes, Data in Brief (2019), via TidyTuesday
Méthode : nettoyage (séjours vides, ADR aberrants) -> KPI hôteliers (taux d'annulation, ADR, RevPAR approché)
          -> facteurs d'annulation -> modèle SANS fuite de données -> prévision du nombre d'annulations par jour
"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/hotel_bookings.csv")
n0 = len(df)

# ---------- 1. Nettoyage ----------
df["nuits"] = df.stays_in_weekend_nights + df.stays_in_week_nights
df["personnes"] = df.adults + df.children.fillna(0) + df.babies
clean = {"brut": n0, "sans_nuit": int((df.nuits == 0).sum()), "sans_personne": int((df.personnes == 0).sum()),
         "adr_aberrant": int(((df.adr <= 0) | (df.adr > 1000)).sum())}
df = df[(df.nuits > 0) & (df.personnes > 0) & (df.adr > 0) & (df.adr <= 1000)].copy()
clean["final"] = len(df)
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
df["mois"] = df.arrival_date_month.map({m: i + 1 for i, m in enumerate(MONTHS)})
df["date"] = pd.to_datetime(dict(year=df.arrival_date_year, month=df.mois, day=df.arrival_date_day_of_month))
df["hotel"] = df.hotel.map({"City Hotel": "Hôtel urbain", "Resort Hotel": "Resort"})
df["revenu"] = df.adr * df.nuits

# ---------- 2. KPI ----------
kpi = {"reservations": len(df), "taux_annulation": round(df.is_canceled.mean() * 100, 1),
       "adr": round(df[df.is_canceled == 0].adr.mean(), 1), "nuits_moy": round(df.nuits.mean(), 2),
       "revenu_annule_m": round(df[df.is_canceled == 1].revenu.sum() / 1e6, 1),
       "revenu_realise_m": round(df[df.is_canceled == 0].revenu.sum() / 1e6, 1),
       "lead_moy": round(df.lead_time.mean())}
by_hotel = df.groupby("hotel").agg(annul=("is_canceled", "mean"), adr=("adr", "mean"), n=("adr", "size"), lead=("lead_time", "median"))
by_hotel["annul"] *= 100
season = df[df.is_canceled == 0].groupby(["hotel", "mois"]).adr.mean().unstack(0).round(1)
annul_m = df.groupby(["hotel", "mois"]).is_canceled.mean().mul(100).unstack(0).round(1)

def rate(col, order=None):
    g = df.groupby(col).is_canceled.agg(["mean", "size"])
    g = g[g["size"] > 300]
    if order: g = g.reindex([o for o in order if o in g.index])
    return {"labels": [str(i) for i in g.index], "taux": (g["mean"] * 100).round(1).tolist(), "n": g["size"].tolist()}
df["lead_band"] = pd.cut(df.lead_time, [-1, 7, 30, 90, 180, 365, 800], labels=["0-7 j", "8-30 j", "1-3 mois", "3-6 mois", "6-12 mois", "> 1 an"])
facteurs = {"Délai de réservation": rate("lead_band", ["0-7 j", "8-30 j", "1-3 mois", "3-6 mois", "6-12 mois", "> 1 an"]),
            "Type d'acompte": rate("deposit_type"), "Canal de vente": rate("market_segment"),
            "Client déjà venu": rate("is_repeated_guest"), "Demandes spéciales": rate(df.total_of_special_requests.clip(upper=3)),
            "Type de client": rate("customer_type")}

# ---------- 3. Modèle sans fuite ----------
# Exclues : reservation_status (c'est la réponse), assigned_room_type et booking_changes (connus après coup),
# country (le pays n'est souvent saisi qu'à l'arrivée, donc absent des annulations : fuite documentée sur ce jeu)
feats = ["hotel", "lead_time", "mois", "arrival_date_week_number", "stays_in_weekend_nights", "stays_in_week_nights", "adults",
         "children", "babies", "meal", "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations",
         "previous_bookings_not_canceled", "reserved_room_type", "deposit_type", "customer_type", "adr",
         "required_car_parking_spaces", "total_of_special_requests", "days_in_waiting_list"]
X = df[feats].copy()
for c in X.select_dtypes(["object", "string"]).columns: X[c] = X[c].astype("category")
tr = df.date < "2017-04-01"; te = ~tr
gb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.06, max_leaf_nodes=48, categorical_features="from_dtype", random_state=0)
gb.fit(X[tr], df.is_canceled[tr]); p = gb.predict_proba(X[te])[:, 1]
auc = roc_auc_score(df.is_canceled[te], p)
fpr, tpr, _ = roc_curve(df.is_canceled[te], p); ix = np.linspace(0, len(fpr) - 1, 40).astype(int)
# Fuite volontaire, pour montrer l'erreur classique : ajouter le pays et le type de chambre attribuée
Xl = X.assign(country=df.country.fillna("?").astype("category"), assigned=df.assigned_room_type.astype("category"))
gbl = HistGradientBoostingClassifier(max_iter=300, categorical_features="from_dtype", random_state=0).fit(Xl[tr], df.is_canceled[tr])
auc_leak = roc_auc_score(df.is_canceled[te], gbl.predict_proba(Xl[te])[:, 1])
NOM = {"lead_time": "Délai de réservation", "deposit_type": "Type d'acompte", "market_segment": "Canal de vente", "total_of_special_requests": "Demandes spéciales",
       "previous_cancellations": "Annulations passées", "required_car_parking_spaces": "Parking demandé", "adr": "Prix par nuit", "customer_type": "Type de client",
       "arrival_date_week_number": "Semaine d'arrivée", "hotel": "Hôtel", "mois": "Mois", "distribution_channel": "Canal de distribution", "stays_in_week_nights": "Nuits en semaine",
       "stays_in_weekend_nights": "Nuits de week-end", "previous_bookings_not_canceled": "Séjours passés honorés", "reserved_room_type": "Type de chambre", "meal": "Formule repas",
       "adults": "Adultes", "children": "Enfants", "babies": "Bébés", "is_repeated_guest": "Client déjà venu", "days_in_waiting_list": "Jours en liste d'attente"}
im = permutation_importance(gb, X[te].iloc[:6000], df.is_canceled[te].iloc[:6000], scoring="roc_auc", n_repeats=3, random_state=0)
imp = pd.Series(im.importances_mean, index=[NOM[f] for f in feats]).sort_values(ascending=False).head(10).round(4)

# ---------- 4. Prévision des annulations par jour d'arrivée (ce qu'utilise un revenue manager pour surréserver) ----------
# Sans la capacité des hôtels (absente des données), on ne peut pas simuler la surréservation elle-même :
# on vérifie que la somme des probabilités prédit bien le nombre d'annulations de chaque jour.
t = df[te].assign(p=p)
day = t.groupby(["hotel", "date"]).agg(resa=("p", "size"), pred=("p", "sum"), reel=("is_canceled", "sum")).reset_index()
day = day[day.resa >= 20]
mae_jour = float((day.pred - day.reel).abs().mean())
corr_jour = float(np.corrcoef(day.pred, day.reel)[0, 1])
naif = day.resa * df.is_canceled[tr].mean()                    # "toujours le taux moyen"
mae_naif = float((naif - day.reel).abs().mean())
ex = day[day.hotel == "Hôtel urbain"].set_index("date").loc["2017-05-01":"2017-06-30"]
jour = {"dates": [d.strftime("%d/%m") for d in ex.index], "pred": ex.pred.round(1).tolist(), "reel": ex.reel.tolist(), "resa": ex.resa.tolist()}
dd = pd.DataFrame({"p": p, "y": df.is_canceled[te].values})
dd["dec"] = pd.qcut(dd.p.rank(method="first"), 10, labels=range(1, 11))
calib = dd.groupby("dec", observed=True).agg(pred=("p", "mean"), reel=("y", "mean")).mul(100).round(1)

out = {"kpi": kpi, "clean": clean, "by_hotel": by_hotel.round(1).reset_index().to_dict("records"),
       "season": {"mois": ["Janv.", "Févr.", "Mars", "Avr.", "Mai", "Juin", "Juil.", "Août", "Sept.", "Oct.", "Nov.", "Déc."],
                  **{h: season[h].tolist() for h in season.columns}},
       "annul_m": {h: annul_m[h].tolist() for h in annul_m.columns}, "facteurs": facteurs,
       "model": {"auc": round(auc, 3), "auc_fuite": round(auc_leak, 3), "roc": {"fpr": fpr[ix].round(3).tolist(), "tpr": tpr[ix].round(3).tolist()},
                 "imp": imp.to_dict(), "n_train": int(tr.sum()), "n_test": int(te.sum())},
       "jour": jour, "jour_stats": {"mae": round(mae_jour, 1), "mae_naif": round(mae_naif, 1), "corr": round(corr_jour, 3), "jours": int(len(day))},
       "calib": {"pred": calib.pred.tolist(), "reel": calib.reel.tolist()}}
json.dump(out, open(ROOT / "results/p_hotel.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(kpi, clean); print(by_hotel); print(facteurs["Type d'acompte"], facteurs["Délai de réservation"]); print(out["model"]["auc"], auc_leak, imp.head(5)); print(out["jour_stats"], calib)
