"""
Projet : Émissions de CO₂ dans le monde — qui émet, qui progresse, qui découple croissance et carbone ?
Données : Our World in Data, CO₂ and Greenhouse Gas Emissions (pays × année, 1750-2023, 79 variables)
Méthode : séparer pays et agrégats régionaux -> émissions totales, par habitant, cumulées
          -> découplage PIB / CO₂ (2005-2022) -> clustering des trajectoires par habitant (k-means sur séries normalisées)
"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/owid_co2.csv")
countries = df[df.iso_code.notna() & ~df.iso_code.str.startswith("OWID", na=True)]      # retire "World", "Europe"...
last = int(df[(df.country == "World") & df.co2.notna()].year.max())
world = df[df.country == "World"].set_index("year")

# ---------- KPI ----------
w = world.loc[last]
kpi = {"annee": last, "co2_monde": round(w.co2 / 1000, 1), "par_hab_monde": round(w.co2_per_capita, 2),
       "x_depuis_1950": round(w.co2 / world.loc[1950].co2, 1), "pays": int(countries.country.nunique()),
       "variables": df.shape[1]}

# ---------- Séries mondiales et grandes régions ----------
regions = ["China", "United States", "European Union (27)", "India", "Russia", "Japan"]
reg = df[df.country.isin(regions) & (df.year >= 1950)].pivot(index="year", columns="country", values="co2")
serie = {"annees": reg.index.tolist(), **{c: (reg[c] / 1000).round(2).tolist() for c in regions},
         "Monde": (world.loc[1950:last, "co2"] / 1000).round(2).tolist()}

c = countries[countries.year == last].dropna(subset=["co2"])
top = c.nlargest(15, "co2")[["country", "co2", "co2_per_capita", "share_global_co2", "cumulative_co2", "share_global_cumulative_co2"]]
cum = countries.groupby("country").cumulative_co2.max().nlargest(10)

# ---------- Découplage 2005 -> 2022 (PIB réel disponible jusqu'en 2022) ----------
a, b = 2005, 2022
d = countries[countries.year.isin([a, b])].pivot(index="country", columns="year", values=["gdp", "co2", "population"]).dropna()
d = d[d[("co2", a)] > 20]                                                 # pays émettant > 20 Mt en 2005
dec = pd.DataFrame({"pib": (d[("gdp", b)] / d[("gdp", a)] - 1) * 100, "co2": (d[("co2", b)] / d[("co2", a)] - 1) * 100,
                    "co2_2022": d[("co2", b)]}).round(1)
dec["groupe"] = np.select([dec.pib <= 0, dec.co2 < 0, dec.co2 < dec.pib],
                          ["Économie en recul", "Découplage absolu", "Découplage relatif"], "Pas de découplage")
dec_counts = dec.groupe.value_counts().to_dict()

# ---------- Clustering des trajectoires par habitant 1990-2022 ----------
pc = countries[(countries.year >= 1990) & (countries.year <= 2022)].pivot(index="country", columns="year", values="co2_per_capita")
pc = pc.dropna()
pc = pc[countries[countries.year == 2022].set_index("country").population.reindex(pc.index) > 5e6]  # pays > 5 M hab.
norm = pc.div(pc.mean(axis=1), axis=0)                                    # forme de la trajectoire, pas le niveau
km = KMeans(4, n_init=20, random_state=0).fit(norm)
pc["cluster"] = km.labels_
traj = []
for k in range(4):
    g = pc[pc.cluster == k].drop(columns="cluster")
    ch = (g[2022] / g[1990] - 1).median() * 100
    traj.append({"k": k, "n": len(g), "moy": g.median().round(2).tolist(), "evol": round(ch, 0),
                 "exemples": g.assign(pop=countries[countries.year == 2022].set_index("country").population.reindex(g.index))
                               .nlargest(6, "pop").index.tolist()})
traj = sorted(traj, key=lambda t: t["evol"])
names = ["Forte baisse", "Baisse", "Hausse", "Très forte hausse"]
for t, n in zip(traj, names): t["nom"] = n

# Sources d'émission mondiales
src = ["coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2", "other_industry_co2"]
mix = (world.loc[last, src] / world.loc[last, src].sum() * 100).round(1)
mix_hist = world.loc[1950:last:5, src].div(1000).round(2)

out = {"kpi": kpi, "serie": serie,
       "top": top.round(2).to_dict("records"), "cumul": {k: round(v / 1000, 1) for k, v in cum.items()},
       "decouplage": dec.reset_index().rename(columns={"country": "pays"}).to_dict("records"), "dec_counts": dec_counts,
       "traj": {"annees": list(range(1990, 2023)), "groupes": traj},
       "mix": {"labels": ["Charbon", "Pétrole", "Gaz", "Ciment", "Torchage", "Autres industries"], "valeurs": mix.tolist(),
               "hist_annees": mix_hist.index.tolist(), "hist": {k: mix_hist[s].tolist() for k, s in zip(["Charbon", "Pétrole", "Gaz", "Ciment", "Torchage", "Autres industries"], src)}}}
json.dump(out, open(ROOT / "results/p_co2.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else None)
print(kpi, dec_counts); print(top.head(8)); print([(t["nom"], t["n"], t["evol"], t["exemples"][:4]) for t in traj]); print(mix)
print(dec.sort_values("co2").head(8)); print(dec.loc[dec.index.isin(["France", "China", "India", "United States", "Germany", "United Kingdom"])])
