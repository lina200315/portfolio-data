"""
Projet : Analyse de panier — quels produits sont achetés ensemble ?
Données : Superstore Sales, 5 496 commandes dont 2 156 contiennent plusieurs lignes
Méthode : matrice commandes × sous-catégories -> règles d'association (support, confiance, lift)
          codées à la main avec pandas -> test de significativité (khi-deux) -> recommandations de vente croisée
"""
import json, pathlib, itertools
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/superstore.csv", encoding="latin1")
FR = {"Appliances": "Électroménager", "Binders and Binder Accessories": "Classeurs", "Bookcases": "Bibliothèques",
      "Chairs & Chairmats": "Chaises", "Computer Peripherals": "Périphériques info.", "Copiers and Fax": "Copieurs et fax",
      "Envelopes": "Enveloppes", "Labels": "Étiquettes", "Office Furnishings": "Déco de bureau",
      "Office Machines": "Machines de bureau", "Paper": "Papier", "Pens & Art Supplies": "Stylos",
      "Rubber Bands": "Élastiques", "Scissors, Rulers and Trimmers": "Ciseaux et règles",
      "Storage & Organization": "Rangement", "Tables": "Tables", "Telephones and Communication": "Téléphonie"}
df["item"] = df["Product Sub-Category"].map(FR)

basket = pd.crosstab(df["Order ID"], df["item"]).clip(upper=1)        # 1 = la commande contient l'article
n_all = len(basket)
size = basket.sum(axis=1).value_counts().sort_index()
multi = basket[basket.sum(axis=1) >= 2]
# Les règles sont calculées sur les commandes multi-articles : sinon les 63 % de commandes
# à un seul article font mécaniquement paraître toutes les paires "rares" (lift < 1).
basket = multi
n = len(basket)
support = basket.mean()

rules = []
for a, b in itertools.permutations(basket.columns, 2):
    both = (basket[a] & basket[b]).sum()
    if both < 15:                                                     # support minimal : 15 commandes
        continue
    sup = both / n; conf = both / basket[a].sum(); lift = conf / support[b]
    tab = pd.crosstab(basket[a], basket[b])
    p = chi2_contingency(tab)[1]
    rules.append({"si": a, "alors": b, "commandes": int(both), "support": round(sup * 100, 2),
                  "confiance": round(conf * 100, 1), "lift": round(lift, 2), "p": float(p)})
rules = pd.DataFrame(rules).sort_values("lift", ascending=False)
rules["force"] = np.where(rules.lift >= 1.2, "Associés", np.where(rules.lift <= 0.8, "Se substituent", "Indépendants"))
sig = rules[rules.p < 0.05]

# Matrice de lift (pour la carte de chaleur) : top 10 sous-catégories par fréquence
top = support.sort_values(ascending=False).head(10).index.tolist()
lift_m = []
for a in top:
    row = []
    for b in top:
        if a == b: row.append(None); continue
        both = (basket[a] & basket[b]).mean()
        row.append(round(both / (support[a] * support[b]), 2))
    lift_m.append(row)

# Valeur des commandes mixtes vs simples
ov = df.groupby("Order ID").agg(ca=("Sales", "sum"), n=("item", "nunique"))
val = {"simple": round(ov[ov.n == 1].ca.mean()), "multi": round(ov[ov.n >= 2].ca.mean())}

kpi = {"commandes": n_all, "multi": int(len(multi)), "part_multi": round(len(multi) / n_all * 100, 1),
       "articles_moy": round(sum(k * v for k, v in size.items()) / n_all, 2), "lift_max": round(rules.lift.max(), 2), "lift_min": round(rules.lift.min(), 2), "regles": int(len(rules)), "regles_sig": int(len(sig)),
       "panier_simple": val["simple"], "panier_multi": val["multi"]}
out = {"kpi": kpi, "support": {k: round(v * 100, 1) for k, v in support.sort_values(ascending=False).items()},
       "taille": {int(k): int(v) for k, v in size.items()},
       "regles": rules.head(15).round(4).to_dict("records"), "regles_sig": sig.head(10).round(4).to_dict("records"), "bas": rules.sort_values("lift").head(6).round(4).to_dict("records"),
       "lift": {"items": top, "valeurs": lift_m}}
json.dump(out, open(ROOT / "results/p_basket.json", "w"), ensure_ascii=False, indent=1, default=float)
print(kpi); print(rules.head(12).to_string()); print(len(sig))
