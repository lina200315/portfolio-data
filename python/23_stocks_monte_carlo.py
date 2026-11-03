"""
Projet : Gestion des stocks — quel point de commande et quelle quantité pour servir 95 % de la demande ? (simulation Monte Carlo)
Données : demande quotidienne réelle de Superstore (unités commandées par jour, 2009-2012) pour 3 sous-catégories
Méthode : demande journalière observée (beaucoup de jours à zéro, quelques gros pics) -> politique (s, Q) :
          quand le stock passe sous s, on commande Q unités, livrées après un délai aléatoire de 3 à 7 jours
          -> 500 années simulées par tirage (bootstrap) dans l'historique -> taux de service et stock moyen
          -> comparaison avec la règle « manuel » (loi normale) qui suppose une demande régulière
"""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/superstore.csv", encoding="latin1")
df["date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
days = pd.date_range(df.date.min(), df.date.max(), freq="D")
SUBS = {"Paper": "Papier", "Binders and Binder Accessories": "Classeurs", "Telephones and Communication": "Téléphonie"}
rng = np.random.default_rng(7)
N, T = 500, 365                                                # 500 années simulées

def simulate(dem, s, Q, reps=N):
    """Simulation vectorisée : chaque ligne est une année simulée. Retourne taux de service (fill rate) et stock moyen."""
    stock = np.full(reps, s + Q, dtype=float); pipeline = []      # commandes en cours : (jour d'arrivée, quantité, masque)
    served = np.zeros(reps); asked = np.zeros(reps); stock_sum = np.zeros(reps); orders = np.zeros(reps)
    on_order = np.zeros(reps, dtype=bool)
    D = rng.choice(dem, size=(reps, T))                            # bootstrap de la demande historique
    L = rng.integers(3, 8, size=(reps, T))                         # délai fournisseur 3 à 7 jours
    arrivals = np.zeros((reps, T + 10))
    for t in range(T):
        stock += arrivals[:, t]; on_order &= ~(arrivals[:, t] > 0)
        d = D[:, t]; s_ok = np.minimum(stock, d)
        served += s_ok; asked += d; stock -= s_ok                  # ventes perdues si rupture
        need = (stock <= s) & ~on_order
        idx = np.where(need)[0]
        arrivals[idx, t + L[idx, t]] += Q; on_order[idx] = True; orders[idx] += 1
        stock_sum += stock
    return served.sum() / asked.sum() * 100, float((stock_sum / T).mean()), float(orders.mean())

res = {}
for sub, fr in SUBS.items():
    dem = df[df["Product Sub-Category"] == sub].groupby("date")["Order Quantity"].sum().reindex(days, fill_value=0).values
    mu, sd = dem.mean(), dem.std()
    price = df.loc[df["Product Sub-Category"] == sub, "Unit Price"].median()
    # Règle de manuel : s = demande moyenne pendant le délai + 1,65 écart-type (95 %), Q = quantité économique (EOQ, coûts supposés)
    Lm = 5; s_rule = mu * Lm + 1.65 * sd * np.sqrt(Lm)
    K, h = 50, 0.25 * price / 365                                  # coût fixe de commande 50 $, coût de stockage 25 %/an de la valeur
    Q_eoq = np.sqrt(2 * mu * K / h)
    fr_rule, inv_rule, ord_rule = simulate(dem, s_rule, Q_eoq)
    # Recherche du plus petit stock moyen garantissant 95 % de service
    grid = []
    for s in np.linspace(0, s_rule * 3, 16):
        for Qm in [0.5, 1, 1.5, 2]:
            f_, inv, o = simulate(dem, s, Q_eoq * Qm, reps=200)
            grid.append({"s": round(s), "Q": round(Q_eoq * Qm), "service": round(f_, 2), "stock": round(inv, 1), "commandes": round(o, 1)})
    ok = [g for g in grid if g["service"] >= 95]
    best = min(ok, key=lambda g: g["stock"] * h * 365 + g["commandes"] * K) if ok else max(grid, key=lambda g: g["service"])
    fr_best, inv_best, ord_best = simulate(dem, best["s"], best["Q"])
    hist = np.histogram(np.minimum(dem, np.percentile(dem, 99)), bins=20)
    res[fr] = {"moy": round(mu, 1), "ecart_type": round(sd, 1), "cv": round(sd / mu, 2), "jours_zero": round((dem == 0).mean() * 100, 1),
               "p99": round(float(np.percentile(dem, 99))), "prix": round(price, 2),
               "regle": {"s": round(s_rule), "Q": round(Q_eoq), "service": round(fr_rule, 1), "stock": round(inv_rule), "commandes": round(ord_rule, 1)},
               "optimum": {"s": best["s"], "Q": best["Q"], "service": round(fr_best, 1), "stock": round(inv_best), "commandes": round(ord_best, 1),
                           "cout_annuel": round(inv_best * h * 365 + ord_best * K)},
               "cout_regle": round(inv_rule * h * 365 + ord_rule * K),
               "grille": grid, "hist": {"x": hist[1][:-1].round(0).tolist(), "y": hist[0].tolist()},
               "serie": dem[-120:].tolist()}
    print(fr, res[fr]["moy"], res[fr]["cv"], res[fr]["jours_zero"], res[fr]["regle"], res[fr]["optimum"])

out = {"kpi": {"simulations": N, "jours": T, "produits": len(SUBS), "delai": "3 à 7 jours", "jours_historique": len(days)},
       "produits": res, "hyp": {"cout_commande": 50, "stockage_annuel": 25, "delai_min": 3, "delai_max": 7}}
json.dump(out, open(ROOT / "results/p_stocks.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
