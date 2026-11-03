"""
Projet : Du fichier brut à un entrepôt de données fiable (ETL + contrôle qualité + modèle en étoile)
Données : Superstore Sales (8 399 lignes de commande)
Méthode : audit qualité (12 contrôles) -> nettoyage documenté -> modèle en étoile
          (1 table de faits, 4 dimensions) dans SQLite -> requêtes analytiques sur le modèle
"""
import json, pathlib, sqlite3
import pandas as pd, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "data/superstore.csv", encoding="latin1")
df = raw.copy()
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")

# ---------- 1. Audit qualité ----------
checks = []
def check(nom, dimension, n, action):
    checks.append({"controle": nom, "dimension": dimension, "lignes": int(n),
                   "statut": "OK" if n == 0 else "À traiter", "action": action})

check("Lignes entièrement dupliquées", "Unicité", raw.duplicated().sum(), "Aucune")
check("Identifiant de ligne (Row ID) en double", "Unicité", raw["Row ID"].duplicated().sum(), "Aucune")
check("Valeurs manquantes (hors marge produit)", "Complétude", raw.drop(columns="Product Base Margin").isna().any(axis=1).sum(), "Aucune")
check("Marge de base produit manquante", "Complétude", raw["Product Base Margin"].isna().sum(),
      "Imputée par la médiane de la sous-catégorie")
check("Expédition avant la commande", "Cohérence", (df["Ship Date"] < df["Order Date"]).sum(), "Aucune")
check("Quantité ≤ 0", "Validité", (df["Order Quantity"] <= 0).sum(), "Aucune")
check("Ventes ≤ 0", "Validité", (df["Sales"] <= 0).sum(), "Aucune")
check("Remise hors [0 ; 100 %]", "Validité", (~df["Discount"].between(0, 1)).sum(), "Aucune")
orders_multi_cust = df.groupby("Order ID")["Customer Name"].nunique()
check("Commande rattachée à plusieurs clients", "Cohérence", (orders_multi_cust > 1).sum(),
      "Clé client = nom + province (le nom seul ne suffit pas)")
cust_multi_prov = df.groupby("Customer Name")["Province"].nunique()
check("Même nom de client dans plusieurs provinces", "Cohérence", (cust_multi_prov > 1).sum(),
      "Clé client = nom + province")
prod_multi_cat = df.groupby("Product Name")["Product Sub-Category"].nunique()
check("Produit classé dans plusieurs sous-catégories", "Cohérence", (prod_multi_cat > 1).sum(), "Aucune")
q1, q3 = df.Sales.quantile([.25, .75]); out = df.Sales > q3 + 3 * (q3 - q1)
check("Ventes extrêmes (> Q3 + 3 × IQR)", "Plausibilité", out.sum(),
      "Conservées (grosses commandes B2B réelles), signalées par un indicateur")
prix = (df.Sales / df["Order Quantity"]) / df["Unit Price"]
check("Ventes incohérentes avec prix × quantité (écart > 50 %)", "Cohérence", ((prix < .5) | (prix > 1.5)).sum(),
      "Conservées : le prix unitaire est un prix catalogue, la vente intègre remise et options")

# ---------- 2. Nettoyage ----------
df["Product Base Margin"] = df.groupby("Product Sub-Category")["Product Base Margin"].transform(lambda s: s.fillna(s.median()))
df["vente_extreme"] = out.astype(int)
df["client_key"] = df["Customer Name"] + " | " + df["Province"]

# ---------- 3. Modèle en étoile ----------
def dim(cols, name):
    d = df[cols].drop_duplicates().reset_index(drop=True)
    d.insert(0, name, range(1, len(d) + 1))
    return d
dim_client = dim(["client_key", "Customer Name", "Province", "Customer Segment"], "id_client")
dim_produit = dim(["Product Name", "Product Category", "Product Sub-Category", "Product Container"], "id_produit")
dim_geo = dim(["Province", "Region"], "id_geo")
dates = pd.date_range(df["Order Date"].min(), df["Ship Date"].max(), freq="D")
dim_date = pd.DataFrame({"id_date": dates.strftime("%Y%m%d").astype(int), "date": dates.strftime("%Y-%m-%d"),
                         "annee": dates.year, "trimestre": dates.quarter, "mois": dates.month,
                         "jour_semaine": dates.dayofweek + 1, "week_end": (dates.dayofweek >= 5).astype(int)})
fact = (df.merge(dim_client[["id_client", "client_key", "Customer Segment"]], on=["client_key", "Customer Segment"])
          .merge(dim_produit, on=["Product Name", "Product Category", "Product Sub-Category", "Product Container"])
          .merge(dim_geo, on=["Province", "Region"]))
fact = pd.DataFrame({"id_ligne": fact["Row ID"], "num_commande": fact["Order ID"],
                     "id_date_commande": fact["Order Date"].dt.strftime("%Y%m%d").astype(int),
                     "id_date_expedition": fact["Ship Date"].dt.strftime("%Y%m%d").astype(int),
                     "id_client": fact.id_client, "id_produit": fact.id_produit, "id_geo": fact.id_geo,
                     "mode_expedition": fact["Ship Mode"], "priorite": fact["Order Priority"],
                     "quantite": fact["Order Quantity"], "ventes": fact.Sales, "remise": fact.Discount,
                     "profit": fact.Profit, "frais_port": fact["Shipping Cost"], "vente_extreme": fact.vente_extreme})
assert len(fact) == len(df), "La jointure a perdu ou dupliqué des lignes"
assert abs(fact.ventes.sum() - raw.Sales.sum()) < 0.01, "Le total des ventes ne correspond plus"

db = ROOT / "results/entrepot_superstore.sqlite"
db.unlink(missing_ok=True)
con = sqlite3.connect(db)
for name, t in [("dim_client", dim_client), ("dim_produit", dim_produit), ("dim_geo", dim_geo),
                ("dim_date", dim_date), ("fait_ventes", fact)]:
    t.to_sql(name, con, index=False)
con.executescript("""
CREATE INDEX ix_f_date ON fait_ventes(id_date_commande);
CREATE INDEX ix_f_client ON fait_ventes(id_client);
CREATE INDEX ix_f_produit ON fait_ventes(id_produit);
""")

# ---------- 4. Requêtes analytiques sur le modèle ----------
queries = {
 "CA par trimestre et catégorie": """
SELECT d.annee || '-T' || d.trimestre AS trimestre, p."Product Category" AS categorie,
       ROUND(SUM(f.ventes)) AS ca, ROUND(SUM(f.profit)) AS profit
FROM fait_ventes f
JOIN dim_date d    ON d.id_date = f.id_date_commande
JOIN dim_produit p ON p.id_produit = f.id_produit
GROUP BY 1, 2 ORDER BY 1, 2;""",
 "Commandes passées le week-end vs en semaine": """
SELECT CASE d.week_end WHEN 1 THEN 'Week-end' ELSE 'Semaine' END AS jour,
       COUNT(DISTINCT f.num_commande) AS commandes,
       ROUND(AVG(f.ventes), 1) AS vente_moy_ligne,
       ROUND(100.0 * SUM(f.profit) / SUM(f.ventes), 1) AS marge_pct
FROM fait_ventes f JOIN dim_date d ON d.id_date = f.id_date_commande
GROUP BY 1;""",
 "Délai d'expédition par priorité": """
SELECT f.priorite,
       COUNT(*) AS lignes,
       ROUND(AVG(julianday(de.date) - julianday(dc.date)), 2) AS delai_moyen_j,
       ROUND(100.0 * SUM(CASE WHEN julianday(de.date) - julianday(dc.date) > 2 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_plus_2j
FROM fait_ventes f
JOIN dim_date dc ON dc.id_date = f.id_date_commande
JOIN dim_date de ON de.id_date = f.id_date_expedition
GROUP BY 1 ORDER BY delai_moyen_j;""",
 "Top 5 clients par région (fenêtre)": """
WITH c AS (
  SELECT g.Region AS region, cl."Customer Name" AS client, ROUND(SUM(f.ventes)) AS ca,
         ROW_NUMBER() OVER (PARTITION BY g.Region ORDER BY SUM(f.ventes) DESC) AS rang
  FROM fait_ventes f JOIN dim_client cl ON cl.id_client = f.id_client JOIN dim_geo g ON g.id_geo = f.id_geo
  GROUP BY 1, 2)
SELECT * FROM c WHERE rang <= 5 AND region IN ('West', 'Ontario') ORDER BY region, rang;""",
}
res = {}
for k, q in queries.items():
    cur = con.execute(q); res[k] = {"sql": q.strip(), "cols": [c[0] for c in cur.description], "rows": cur.fetchall()}
schema = {n: {"lignes": int(con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0]),
              "colonnes": [c[1] for c in con.execute(f"PRAGMA table_info({n})")]}
          for n in ["fait_ventes", "dim_client", "dim_produit", "dim_geo", "dim_date"]}
con.close()

out = {"checks": checks, "schema": schema, "queries": res,
       "kpi": {"lignes": len(df), "controles": len(checks), "a_traiter": sum(c["statut"] != "OK" for c in checks),
               "clients_nom": int(df["Customer Name"].nunique()), "clients_cle": int(len(dim_client)),
               "produits": int(len(dim_produit)), "taille_ko": round(db.stat().st_size / 1024)}}
json.dump(out, open(ROOT / "results/p_etl.json", "w"), ensure_ascii=False, indent=1, default=float)
for c in checks: print(c)
print(out["kpi"]); print({k: v["lignes"] for k, v in schema.items()})
for k, v in res.items(): print(k, v["rows"][:4])
