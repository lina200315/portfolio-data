"""
Préparation des données des 6 dashboards KPI interactifs (e-commerce, hôtellerie, opérations aériennes, RH, vélos, campagnes bancaires)
Principe : comme un modèle Power BI importé, on pré-agrège chaque source en un « cube » (une ligne par combinaison
de dimensions, avec des sommes additives). Le navigateur filtre et additionne ensuite ces lignes instantanément.
Les mesures non additives (moyennes, taux) sont recalculées à partir des sommes : jamais de moyenne de moyennes.
"""
import json, pathlib
import duckdb, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
con = duckdb.connect()
out = {}

def encode(df, dims, meas):
    """Remplace chaque dimension par un index (listes de libellés à part) pour alléger le fichier."""
    lab = {d: sorted(df[d].dropna().unique().tolist(), key=str) for d in dims}
    rows = [[lab[d].index(r[d]) for d in dims] + [round(float(r[m]), 2) for m in meas] for _, r in df.iterrows()]
    return {"dims": dims, "labels": lab, "meas": meas, "rows": rows}

# ---------- 1. E-commerce (Olist) ----------
O = ROOT / "data/olist"
ol = con.execute(f"""
WITH it AS (SELECT order_id, MIN(product_id) AS pid, SUM(price) AS prix, SUM(freight_value) AS port FROM '{O}/olist_order_items.parquet' GROUP BY 1),
     rv AS (SELECT order_id, AVG(review_score) AS note FROM '{O}/olist_order_reviews.parquet' GROUP BY 1)
SELECT strftime(o.order_purchase_timestamp, '%Y-%m') AS mois, c.customer_state AS etat,
       COALESCE(t.product_category_name_english, 'other') AS cat,
       COUNT(*) AS commandes, SUM(it.prix + it.port) AS gmv, SUM(it.port) AS port,
       SUM(CASE WHEN o.order_delivered_customer_date IS NOT NULL THEN 1 ELSE 0 END) AS livrees,
       SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) AS en_retard,
       SUM(date_diff('day', o.order_purchase_timestamp, o.order_delivered_customer_date)) AS jours_livraison,
       SUM(rv.note) AS somme_notes, COUNT(rv.note) AS nb_notes,
       SUM(CASE WHEN rv.note <= 2 THEN 1 ELSE 0 END) AS avis_negatifs
FROM '{O}/olist_orders.parquet' o
JOIN '{O}/olist_customers.parquet' c USING (customer_id)
JOIN it USING (order_id)
JOIN '{O}/olist_products.parquet' p ON p.product_id = it.pid
LEFT JOIN '{O}/product_category_name_translation.parquet' t USING (product_category_name)
LEFT JOIN rv USING (order_id)
WHERE o.order_status NOT IN ('canceled', 'unavailable') AND o.order_purchase_timestamp BETWEEN '2017-01-01' AND '2018-08-31'
GROUP BY 1, 2, 3""").df()
top_cat = ol.groupby("cat").gmv.sum().nlargest(14).index
ol["cat"] = np.where(ol.cat.isin(top_cat), ol.cat, "autres")
etat_top = ol.groupby("etat").commandes.sum().nlargest(12).index
ol["etat"] = np.where(ol.etat.isin(etat_top), ol.etat, "Autres")
ol = ol.groupby(["mois", "etat", "cat"], as_index=False).sum(numeric_only=True)
out["ecom"] = encode(ol, ["mois", "etat", "cat"], ["commandes", "gmv", "port", "livrees", "en_retard", "jours_livraison", "somme_notes", "nb_notes", "avis_negatifs"])

# ---------- 2. Hôtellerie ----------
h = pd.read_csv(ROOT / "data/hotel_bookings.csv")
h = h[(h.stays_in_weekend_nights + h.stays_in_week_nights > 0) & (h.adr > 0) & (h.adr <= 1000)].copy()
M = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
h["mois"] = h.arrival_date_year.astype(str) + "-" + h.arrival_date_month.map({m: f"{i+1:02d}" for i, m in enumerate(M)})
h["nuits"] = h.stays_in_weekend_nights + h.stays_in_week_nights
h["hotel"] = h.hotel.map({"City Hotel": "Hôtel urbain", "Resort Hotel": "Resort"})
h["segment"] = h.market_segment.replace({"Online TA": "Agence en ligne", "Offline TA/TO": "Agence / TO", "Groups": "Groupes", "Direct": "Direct",
                                          "Corporate": "Entreprises", "Complementary": "Autres", "Aviation": "Autres", "Undefined": "Autres"})
h["client"] = h.customer_type.replace({"Transient": "Individuel", "Transient-Party": "Individuel lié", "Contract": "Contrat", "Group": "Groupe"})
h["ok"] = 1 - h.is_canceled
h["nuitees"] = h.nuits * h.ok; h["revenu"] = h.adr * h.nuits * h.ok
h["lead"] = h.lead_time; h["repeat"] = h.is_repeated_guest
hc = h.groupby(["mois", "hotel", "segment", "client"], as_index=False).agg(resa=("ok", "size"), annulees=("is_canceled", "sum"), nuitees=("nuitees", "sum"),
                                                                          revenu=("revenu", "sum"), lead=("lead", "sum"), fideles=("repeat", "sum"))
out["hotel"] = encode(hc, ["mois", "hotel", "segment", "client"], ["resa", "annulees", "nuitees", "revenu", "lead", "fideles"])

# ---------- 3. Opérations aériennes ----------
f = con.execute(f"""
SELECT a.name AS cie, f.origin, f.month AS mois, f.hour AS heure, COUNT(*) AS vols,
       SUM(CASE WHEN f.dep_time IS NULL THEN 1 ELSE 0 END) AS annules,
       SUM(CASE WHEN f.arr_delay IS NOT NULL THEN 1 ELSE 0 END) AS arrives,
       SUM(CASE WHEN f.arr_delay <= 15 THEN 1 ELSE 0 END) AS a_l_heure,
       SUM(COALESCE(f.arr_delay, 0)) AS somme_retard,
       SUM(CASE WHEN f.arr_delay > 60 THEN 1 ELSE 0 END) AS retard_60,
       SUM(f.distance) AS distance
FROM '{ROOT / 'data/nyc_flights.parquet'}' f JOIN read_csv_auto('{ROOT / 'data/nyc_airlines.csv'}') a USING (carrier)
WHERE f.hour BETWEEN 5 AND 23 GROUP BY 1, 2, 3, 4""").df()
f["cie"] = f.cie.str.replace(r" (Inc\.|Co\.|Corporation)$", "", regex=True)
out["air"] = encode(f, ["cie", "origin", "mois", "heure"], ["vols", "annules", "arrives", "a_l_heure", "somme_retard", "retard_60", "distance"])

# ---------- 4. RH ----------
r = pd.read_csv(ROOT / "data/hr_attrition.csv", encoding="utf-8-sig")
r["Departement"] = r.Department.map({"Research & Development": "R&D", "Sales": "Ventes", "Human Resources": "RH"})
r["Genre"] = r.Gender.map({"Male": "Homme", "Female": "Femme"})
r["HeuresSup"] = r.OverTime.map({"Yes": "Oui", "No": "Non"})
r["TrancheAge"] = pd.cut(r.Age, [17, 25, 35, 45, 55, 61], labels=["18-25", "26-35", "36-45", "46-55", "56-60"]).astype(str)
r["Poste"] = r.JobRole
r["depart"] = (r.Attrition == "Yes").astype(int)
rc = r.groupby(["Departement", "Poste", "Genre", "HeuresSup", "TrancheAge"], as_index=False).agg(
    effectif=("depart", "size"), departs=("depart", "sum"), salaire=("MonthlyIncome", "sum"), anciennete=("YearsAtCompany", "sum"),
    satisfaction=("JobSatisfaction", "sum"), equilibre=("WorkLifeBalance", "sum"), formation=("TrainingTimesLastYear", "sum"),
    sans_promo_3ans=("YearsSinceLastPromotion", lambda s: (s >= 3).sum()))
out["rh"] = encode(rc, ["Departement", "Poste", "Genre", "HeuresSup", "TrancheAge"],
                   ["effectif", "departs", "salaire", "anciennete", "satisfaction", "equilibre", "formation", "sans_promo_3ans"])

# ---------- 5. Vélos ----------
b = pd.read_csv(ROOT / "data/bike_hour.csv", parse_dates=["dteday"])
b["mois"] = b.dteday.dt.strftime("%Y-%m")
b["saison"] = b.season.map({1: "Hiver", 2: "Printemps", 3: "Été", 4: "Automne"})
b["meteo"] = b.weathersit.map({1: "Dégagé", 2: "Nuageux", 3: "Pluie / neige", 4: "Pluie / neige"})
b["jour"] = b.workingday.map({1: "Jour ouvré", 0: "Week-end / férié"})
b["temp_c"] = b.temp * 47 - 8
bc = b.groupby(["mois", "saison", "meteo", "jour", "hr"], as_index=False).agg(heures=("cnt", "size"), locations=("cnt", "sum"),
                                                                              occasionnels=("casual", "sum"), abonnes=("registered", "sum"), temp=("temp_c", "sum"))
out["velo"] = encode(bc, ["mois", "saison", "meteo", "jour", "hr"], ["heures", "locations", "occasionnels", "abonnes", "temp"])

# ---------- 6. Campagnes bancaires ----------
bk = pd.read_csv(ROOT / "data/bank_marketing.csv", sep=";")
bk["vente"] = (bk.y == "yes").astype(int)
bk["Age"] = pd.cut(bk.age, [16, 25, 35, 45, 55, 65, 100], labels=["17-25", "26-35", "36-45", "46-55", "56-65", "66+"]).astype(str)
bk["Metier"] = bk.job.map({"admin.": "Administratif", "blue-collar": "Ouvrier", "entrepreneur": "Entrepreneur", "housemaid": "Employé de maison", "management": "Cadre",
                           "retired": "Retraité", "self-employed": "Indépendant", "services": "Services", "student": "Étudiant", "technician": "Technicien",
                           "unemployed": "Sans emploi", "unknown": "Inconnu"})
bk["Canal"] = bk.contact.map({"cellular": "Mobile", "telephone": "Fixe"})
bk["Mois"] = bk.month.map({"mar": "03 Mars", "apr": "04 Avril", "may": "05 Mai", "jun": "06 Juin", "jul": "07 Juillet", "aug": "08 Août", "sep": "09 Septembre",
                           "oct": "10 Octobre", "nov": "11 Novembre", "dec": "12 Décembre"})
bk["Historique"] = bk.poutcome.map({"nonexistent": "Jamais contacté", "failure": "Échec précédent", "success": "Succès précédent"})
bc = bk.groupby(["Metier", "Age", "Canal", "Mois", "Historique"], as_index=False).agg(appels=("vente", "size"), ventes=("vente", "sum"),
                                                                                  duree=("duration", "sum"), tentatives=("campaign", "sum"))
out["banque"] = encode(bc, ["Metier", "Age", "Canal", "Mois", "Historique"], ["appels", "ventes", "duree", "tentatives"])

for k, v in out.items():
    print(k, len(v["rows"]), "lignes", {d: len(l) for d, l in v["labels"].items()})
json.dump(out, open(ROOT / "results/dashboards.json", "w"), ensure_ascii=False, separators=(",", ":"))
print(round((ROOT / "results/dashboards.json").stat().st_size / 1e6, 2), "Mo")
