"""
Projet : Où ouvrir des entrepôts pour rapprocher Olist de ses clients ? (recherche opérationnelle)
Données : commandes, clients et vendeurs Olist + table de géolocalisation des codes postaux (19 010 préfixes géocodés)
Méthode : géocodage des clients et des vendeurs -> distance actuelle vendeur → client (formule de haversine)
          -> problème de la p-médiane résolu en programmation linéaire en nombres entiers (scipy.optimize.milp, solveur HiGHS)
          -> distance moyenne selon le nombre d'entrepôts (1 à 8) -> lien distance / délai / note, pour chiffrer l'enjeu
"""
import json, pathlib
import duckdb, numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

ROOT = pathlib.Path(__file__).resolve().parents[1]
O = ROOT / "data/olist"
con = duckdb.connect()
geo = pd.read_csv(O / "geoloc_zip5.csv", dtype={"geolocation_zip_code_prefix": str})
geo["zip"] = geo.geolocation_zip_code_prefix.str.zfill(5)
con.register("geo", geo)

def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

# ---------- 1. Situation actuelle : distance vendeur -> client ----------
d = con.execute(f"""
SELECT o.order_id, gc.lat AS clat, gc.lng AS clng, gs.lat AS slat, gs.lng AS slng, c.customer_state AS etat,
       date_diff('day', o.order_purchase_timestamp, o.order_delivered_customer_date) AS delai, r.review_score AS note
FROM '{O}/olist_orders.parquet' o
JOIN '{O}/olist_customers.parquet' c USING (customer_id)
JOIN (SELECT order_id, MIN(seller_id) AS seller_id FROM '{O}/olist_order_items.parquet' GROUP BY 1) i USING (order_id)
JOIN '{O}/olist_sellers.parquet' s USING (seller_id)
JOIN geo gc ON gc.zip = lpad(CAST(c.customer_zip_code_prefix AS VARCHAR), 5, '0')
JOIN geo gs ON gs.zip = lpad(CAST(s.seller_zip_code_prefix AS VARCHAR), 5, '0')
LEFT JOIN (SELECT order_id, AVG(review_score) AS review_score FROM '{O}/olist_order_reviews.parquet' GROUP BY 1) r USING (order_id)
WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL""").df()
d["dist"] = hav(d.clat, d.clng, d.slat, d.slng)
geocode_rate = len(d) / con.execute(f"SELECT COUNT(*) FROM '{O}/olist_orders.parquet' WHERE order_status = 'delivered'").fetchone()[0]
d["dbin"] = pd.cut(d.dist, [-1, 100, 500, 1000, 2000, 5000], labels=["< 100 km", "100-500", "500-1 000", "1 000-2 000", "> 2 000 km"])
lien = d.groupby("dbin", observed=True).agg(commandes=("delai", "size"), delai=("delai", "mean"), note=("note", "mean")).round(2)
slope = np.polyfit(d.dist, d.delai, 1)[0] * 1000                     # jours de délai en plus par 1 000 km

# ---------- 2. Points de demande et sites candidats ----------
d["cell_lat"] = (d.clat * 2).round() / 2; d["cell_lng"] = (d.clng * 2).round() / 2      # mailles de 0,5° (~55 km)
dem = d.groupby(["cell_lat", "cell_lng"]).agg(n=("order_id", "size"), lat=("clat", "mean"), lng=("clng", "mean")).reset_index()
dem = dem.sort_values("n", ascending=False)
cover = dem.n.head(300).sum() / dem.n.sum()
dem = dem.head(300).reset_index(drop=True)                               # 300 zones = l'essentiel de la demande
cities = con.execute(f"""
SELECT g.ville AS ville, g.etat AS etat, AVG(g.lat) AS lat, AVG(g.lng) AS lng, COUNT(*) AS n
FROM '{O}/olist_customers.parquet' c JOIN geo g ON g.zip = lpad(CAST(c.customer_zip_code_prefix AS VARCHAR), 5, '0')
GROUP BY 1, 2 ORDER BY n DESC LIMIT 40""").df()                         # 40 villes candidates (les plus grosses en clients)

D = hav(dem.lat.values[:, None], dem.lng.values[:, None], cities.lat.values[None, :], cities.lng.values[None, :])
w = dem.n.values

def p_median(p):
    I, J = D.shape
    nx = I * J; c = np.concatenate([(w[:, None] * D).ravel(), np.zeros(J)])
    A = lil_matrix((I + nx + 1, nx + J)); lb, ub = [], []
    for i in range(I):                                                    # chaque zone est servie par un site
        A[i, i * J:(i + 1) * J] = 1; lb.append(1); ub.append(1)
    r = I
    for i in range(I):                                                    # x_ij <= y_j
        for j in range(J):
            A[r, i * J + j] = 1; A[r, nx + j] = -1; lb.append(-np.inf); ub.append(0); r += 1
    A[r, nx:] = 1; lb.append(p); ub.append(p)                             # exactement p entrepôts
    integ = np.concatenate([np.zeros(nx), np.ones(J)])
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=integ, bounds=Bounds(0, 1), options={"time_limit": 60})
    y = res.x[nx:].round().astype(bool)
    sites = np.where(y)[0]
    assign = sites[D[:, sites].argmin(axis=1)]
    return sites, assign, float((w * D[np.arange(I), assign]).sum() / w.sum()), res.status

curve, sol = [], {}
for p in range(1, 9):
    sites, assign, avg, status = p_median(p)
    curve.append({"p": p, "km": round(avg), "sites": [f"{cities.ville[s].title()} ({cities.etat[s]})" for s in sites], "statut": int(status),
                  "sites_xy": [[round(cities.lat[s], 2), round(cities.lng[s], 2), cities.ville[s].title()] for s in sites]})
    sol[p] = (sites, assign)
actuel = float(np.average(d.dist))
sites3, assign3 = sol[3]
gain_jours = (actuel - curve[2]["km"]) / 1000 * slope

out = {"kpi": {"commandes": len(d), "geocodage": round(geocode_rate * 100, 1), "dist_actuelle": round(actuel), "dist_mediane": round(d.dist.median()),
               "km_3": curve[2]["km"], "km_1": curve[0]["km"], "zones": len(dem), "couverture": round(cover * 100, 1),
               "pente_jours_1000km": round(slope, 2), "gain_jours_3": round(gain_jours, 1)},
       "lien": {"labels": [str(i) for i in lien.index], "commandes": lien.commandes.tolist(), "delai": lien.delai.tolist(), "note": lien.note.tolist()},
       "curve": curve,
       "carte": {"zones": [[round(a, 2), round(b, 2), int(n), int(np.where(sites3 == s)[0][0])] for a, b, n, s in zip(dem.lat, dem.lng, dem.n, assign3)],
                 "sites": [[round(cities.lat[s], 2), round(cities.lng[s], 2), cities.ville[s].title(), cities.etat[s]] for s in sites3],
                 "candidats": [[round(a, 2), round(b, 2)] for a, b in zip(cities.lat, cities.lng)]},
       "taille_probleme": {"variables": int(D.size + D.shape[1]), "contraintes": int(D.shape[0] + D.size + 1)}}
json.dump(out, open(ROOT / "results/p_entrepots.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(out["kpi"]); print(lien); print([(c["p"], c["km"], c["sites"]) for c in curve])
