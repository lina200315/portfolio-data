"""
Projet : Olist, marketplace e-commerce brésilienne — la livraison fait-elle la note client ?
Données : jeu public officiel d'Olist (99 441 commandes, 2016-2018, 8 tables relationnelles : commandes, articles,
          paiements, avis, clients, vendeurs, produits, traduction des catégories)
Méthode : modèle relationnel requêté en SQL (DuckDB sur Parquet) -> KPI e-commerce (GMV, panier, réachat)
          -> délai et retard de livraison -> lien avec la note -> modèle de prédiction d'un avis négatif
"""
import json, pathlib
import duckdb, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

ROOT = pathlib.Path(__file__).resolve().parents[1]
con = duckdb.connect()
for t in ["orders", "order_items", "order_payments", "order_reviews", "customers", "sellers", "products"]:
    con.execute(f"CREATE VIEW {t} AS SELECT * FROM '{ROOT / 'data/olist' / ('olist_' + t + '.parquet')}'")
con.execute(f"CREATE VIEW traduction AS SELECT * FROM '{ROOT / 'data/olist/product_category_name_translation.parquet'}'")

Q = {}
def q(name, sql):
    df = con.execute(sql).df()
    Q[name] = {"sql": sql.strip(), "cols": list(df.columns), "rows": df.round(2).astype(object).where(df.notna(), None).values.tolist()}
    return df

kpi = q("KPI de la marketplace", """
WITH livrees AS (SELECT * FROM orders WHERE order_status = 'delivered'),
     ca AS (SELECT order_id, SUM(price) AS produits, SUM(freight_value) AS port FROM order_items GROUP BY order_id)
SELECT COUNT(DISTINCT l.order_id)                                   AS commandes_livrees,
       COUNT(DISTINCT c.customer_unique_id)                         AS clients_uniques,
       ROUND(SUM(ca.produits + ca.port) / 1e6, 2)                   AS gmv_millions_brl,
       ROUND(AVG(ca.produits + ca.port), 2)                         AS panier_moyen_brl,
       ROUND(100.0 * SUM(ca.port) / SUM(ca.produits + ca.port), 1)  AS part_frais_port,
       (SELECT COUNT(DISTINCT seller_id) FROM order_items)          AS vendeurs
FROM livrees l JOIN customers c USING (customer_id) JOIN ca USING (order_id);""").iloc[0].to_dict()

rachat = q("Taux de réachat : les clients reviennent-ils ?", """
WITH n AS (SELECT c.customer_unique_id, COUNT(*) AS commandes
           FROM orders o JOIN customers c USING (customer_id) WHERE o.order_status = 'delivered' GROUP BY 1)
SELECT CASE WHEN commandes = 1 THEN '1 commande' WHEN commandes = 2 THEN '2 commandes' ELSE '3 et plus' END AS clients,
       COUNT(*) AS nb, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS part
FROM n GROUP BY 1 ORDER BY 1;""")

mois = q("GMV mensuel et croissance", """
SELECT strftime(o.order_purchase_timestamp, '%Y-%m') AS mois,
       COUNT(DISTINCT o.order_id) AS commandes,
       ROUND(SUM(i.price + i.freight_value)) AS gmv,
       ROUND(100.0 * (SUM(i.price + i.freight_value) / LAG(SUM(i.price + i.freight_value)) OVER (ORDER BY 1) - 1), 1) AS croissance_pct
FROM orders o JOIN order_items i USING (order_id)
WHERE o.order_status = 'delivered' AND o.order_purchase_timestamp BETWEEN '2017-01-01' AND '2018-08-31'
GROUP BY 1 ORDER BY 1;""")

cat = q("Catégories : CA, note moyenne et poids du port", """
SELECT COALESCE(t.product_category_name_english, 'inconnue') AS categorie,
       COUNT(DISTINCT i.order_id) AS commandes, ROUND(SUM(i.price)) AS ca_produits,
       ROUND(AVG(r.review_score), 2) AS note_moy,
       ROUND(100.0 * SUM(i.freight_value) / SUM(i.price), 1) AS port_en_pct_du_prix
FROM order_items i
JOIN products p USING (product_id)
LEFT JOIN traduction t USING (product_category_name)
LEFT JOIN order_reviews r USING (order_id)
GROUP BY 1 ORDER BY ca_produits DESC LIMIT 15;""")

livr = q("Retard de livraison et note client", """
WITH d AS (
  SELECT o.order_id, r.review_score,
         date_diff('day', o.order_estimated_delivery_date, o.order_delivered_customer_date) AS retard_j
  FROM orders o JOIN order_reviews r USING (order_id)
  WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL)
SELECT CASE WHEN retard_j <= -10 THEN '1. ≥ 10 j d''avance' WHEN retard_j <= -1 THEN '2. 1-9 j d''avance' WHEN retard_j = 0 THEN '3. Le jour prévu'
            WHEN retard_j <= 3 THEN '4. 1-3 j de retard' WHEN retard_j <= 10 THEN '5. 4-10 j de retard' ELSE '6. > 10 j de retard' END AS livraison,
       COUNT(*) AS commandes, ROUND(AVG(review_score), 2) AS note_moy,
       ROUND(100.0 * AVG(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END), 1) AS pct_avis_negatifs
FROM d GROUP BY 1 ORDER BY 1;""")

etats = q("Délais par État du client", """
SELECT c.customer_state AS etat, COUNT(*) AS commandes,
       ROUND(AVG(date_diff('day', o.order_purchase_timestamp, o.order_delivered_customer_date)), 1) AS delai_moy_j,
       ROUND(100.0 * AVG(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END), 1) AS pct_en_retard
FROM orders o JOIN customers c USING (customer_id)
WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
GROUP BY 1 HAVING COUNT(*) > 300 ORDER BY delai_moy_j;""")

pay = q("Moyens de paiement et paiement en plusieurs fois", """
SELECT payment_type AS moyen, COUNT(DISTINCT order_id) AS commandes,
       ROUND(100.0 * COUNT(DISTINCT order_id) / (SELECT COUNT(DISTINCT order_id) FROM order_payments), 1) AS part,
       ROUND(AVG(payment_value), 2) AS montant_moy, ROUND(AVG(payment_installments), 1) AS mensualites_moy
FROM order_payments WHERE payment_type <> 'not_defined' GROUP BY 1 ORDER BY commandes DESC;""")

vend = q("Concentration des vendeurs (Pareto)", """
WITH s AS (SELECT seller_id, SUM(price) AS ca FROM order_items GROUP BY 1),
     r AS (SELECT ca, ROW_NUMBER() OVER (ORDER BY ca DESC) AS rang, COUNT(*) OVER () AS n,
                  SUM(ca) OVER (ORDER BY ca DESC) / SUM(ca) OVER () AS cumul FROM s)
SELECT CAST(CEIL(100.0 * rang / n / 10) * 10 AS INT) AS top_pct_vendeurs, ROUND(100 * MAX(cumul), 1) AS pct_ca_cumule
FROM r GROUP BY 1 ORDER BY 1;""")

# ---------- Modèle : prédire un avis négatif (1-2 étoiles) avec ce qu'on sait à la livraison ----------
df = con.execute("""
WITH it AS (SELECT order_id, COUNT(*) AS articles, SUM(price) AS prix, SUM(freight_value) AS port,
                   COUNT(DISTINCT seller_id) AS vendeurs FROM order_items GROUP BY 1)
SELECT r.review_score <= 2 AS negatif,
       date_diff('day', o.order_estimated_delivery_date, o.order_delivered_customer_date) AS retard_j,
       date_diff('day', o.order_purchase_timestamp, o.order_delivered_customer_date) AS delai_j,
       date_diff('day', o.order_purchase_timestamp, o.order_estimated_delivery_date) AS delai_promis_j,
       it.articles, it.prix, it.port, it.vendeurs, c.customer_state AS etat,
       month(o.order_purchase_timestamp) AS mois, o.order_purchase_timestamp AS ts
FROM orders o JOIN order_reviews r USING (order_id) JOIN it USING (order_id) JOIN customers c USING (customer_id)
WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL""").df()
df["etat"] = df.etat.astype("category"); df["negatif"] = df.negatif.astype(int)
feats = ["retard_j", "delai_j", "delai_promis_j", "articles", "prix", "port", "vendeurs", "etat"]
tr = df.ts < "2018-04-01"; te = ~tr
gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, categorical_features="from_dtype", random_state=0)
gb.fit(df.loc[tr, feats], df.negatif[tr]); p = gb.predict_proba(df.loc[te, feats])[:, 1]
auc = roc_auc_score(df.negatif[te], p)
auc_retard = roc_auc_score(df.negatif[te], df.retard_j[te])
imp = permutation_importance(gb, df.loc[te, feats].iloc[:8000], df.negatif[te].iloc[:8000], scoring="roc_auc", n_repeats=4, random_state=0)
NOM = {"retard_j": "Retard vs date promise", "delai_j": "Délai réel", "delai_promis_j": "Délai promis", "articles": "Nombre d'articles",
       "prix": "Montant", "port": "Frais de port", "vendeurs": "Nombre de vendeurs", "etat": "État du client"}
imp = pd.Series(imp.importances_mean, index=[NOM[f] for f in feats]).sort_values(ascending=False).round(4)
neg_multi = df.groupby(df.vendeurs.clip(upper=2)).negatif.mean().mul(100).round(1)

out = {"kpi": {k: float(v) for k, v in kpi.items()}, "queries": Q,
       "model": {"auc": round(auc, 3), "auc_retard_seul": round(auc_retard, 3), "base": round(df.negatif.mean() * 100, 1),
                 "n_train": int(tr.sum()), "n_test": int(te.sum()), "imp": imp.to_dict(),
                 "neg_vendeurs": {"1 vendeur": neg_multi.get(1), "2 vendeurs ou plus": neg_multi.get(2)}}}
json.dump(out, open(ROOT / "results/p_olist.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(kpi); print(rachat); print(livr); print(etats.head(3), etats.tail(3)); print(pay); print(vend); print(out["model"])
