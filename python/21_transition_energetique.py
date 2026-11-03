"""
Projet : La transition énergétique mondiale, pays par pays — carte interactive et rythme du solaire et de l'éolien
Données : Our World in Data, Energy dataset (pays × année, 1900-2025, 130 variables : consommation par source,
          mix électrique, intensité carbone de l'électricité) + fond de carte Natural Earth (world-atlas)
Méthode : jointure pays OWID (code ISO alpha-3) ↔ carte (code ISO numérique) -> part des énergies bas carbone
          dans l'électricité par pays et par année -> mix primaire mondial -> croissance exponentielle du solaire
          (temps de doublement par régression log-linéaire) -> profils de pays (électricité 2024)
"""
import json, pathlib
import numpy as np, pandas as pd, pycountry

ROOT = pathlib.Path(__file__).resolve().parents[1]
e = pd.read_csv(ROOT / "data/owid_energy.csv")
pays = e[e.iso_code.notna() & ~e.iso_code.str.startswith("OWID", na=True)].copy()
world = e[e.country == "World"].set_index("year")

# ---------- 1. Jointure avec la carte ----------
topo = json.load(open(ROOT / "data/carte/countries-110m.json"))
geoms = topo["objects"]["countries"]["geometries"]
num2a3 = {}
for g in geoms:
    c = pycountry.countries.get(numeric=str(g.get("id", "")).zfill(3)) if g.get("id") else None
    if c: num2a3[g["id"]] = c.alpha_3
a3_in_data = set(pays.iso_code)
matched = sum(1 for v in num2a3.values() if v in a3_in_data)
jointure = {"formes_carte": len(geoms), "avec_code_iso": len(num2a3), "trouvees_dans_owid": matched}

# Valeurs par pays et par année pour la carte (2000-2024)
years = list(range(2000, 2025))
cols = {"low_carbon_share_elec": "bas_carbone", "renewables_share_elec": "renouvelables", "fossil_share_elec": "fossiles",
        "solar_share_elec": "solaire", "wind_share_elec": "eolien", "nuclear_share_elec": "nucleaire", "carbon_intensity_elec": "intensite"}
sub = pays[pays.year.isin(years)].set_index(["iso_code", "year"])
carte = {}
for num, a3 in num2a3.items():
    if a3 not in a3_in_data: continue
    d = {}
    for c, short in cols.items():
        s = sub[c].xs(a3, level=0) if a3 in sub.index.get_level_values(0) else pd.Series(dtype=float)
        d[short] = [None if (y not in s.index or pd.isna(s[y])) else round(float(s[y]), 1) for y in years]
    d["nom"] = pays.loc[pays.iso_code == a3, "country"].iloc[0]
    carte[num] = d

# ---------- 2. Mix mondial d'énergie primaire ----------
src = ["coal_consumption", "oil_consumption", "gas_consumption", "nuclear_consumption", "hydro_consumption", "wind_consumption", "solar_consumption", "biofuel_consumption", "other_renewable_consumption"]
mix = world.loc[1965:2024, src].fillna(0)
mix_share = mix.div(mix.sum(axis=1), axis=0) * 100
last = int(mix.index.max())

# ---------- 3. Le solaire et l'éolien : croissance exponentielle ----------
def doubling(series, a, b):
    s = series.loc[a:b].dropna(); s = s[s > 0]
    slope = np.polyfit(s.index, np.log(s.values), 1)[0]
    return round(np.log(2) / slope, 1), round((np.exp(slope) - 1) * 100, 1)
dbl = {"Solaire (2010-2024)": doubling(world.solar_consumption, 2010, 2024), "Éolien (2010-2024)": doubling(world.wind_consumption, 2010, 2024)}
elec_w = world.loc[2000:2024, ["solar_share_elec", "wind_share_elec", "nuclear_share_elec", "hydro_share_elec", "coal_share_elec", "gas_share_elec"]].round(2)

# ---------- 4. Classements et profils 2024 ----------
p24 = pays[(pays.year == 2024) & (pays.electricity_generation > 20)].set_index("country")      # pays produisant > 20 TWh
top_bas_carbone = p24.low_carbon_share_elec.dropna().nlargest(12).round(1)
top_solaire = p24.solar_share_elec.dropna().nlargest(10).round(1)
prog = pays[pays.year.isin([2015, 2024])].pivot(index="country", columns="year", values="low_carbon_share_elec").dropna()
prog = prog[prog.index.isin(p24.index)]
prog["gain"] = prog[2024] - prog[2015]
top_prog = prog.gain.nlargest(10).round(1)
focus = ["France", "Germany", "Spain", "United Kingdom", "United States", "China", "India", "Brazil", "Morocco", "Algeria"]
prof = p24.loc[[f for f in focus if f in p24.index], ["coal_share_elec", "gas_share_elec", "oil_share_elec", "nuclear_share_elec", "hydro_share_elec", "wind_share_elec", "solar_share_elec", "biofuel_share_elec", "carbon_intensity_elec"]].round(1)

out = {"kpi": {"annee": last, "fossile_primaire": round(mix_share.loc[last, ["coal_consumption", "oil_consumption", "gas_consumption"]].sum(), 1),
               "fossile_1965": round(mix_share.loc[1965, ["coal_consumption", "oil_consumption", "gas_consumption"]].sum(), 1),
               "bas_carbone_elec": round(world.loc[2024, "low_carbon_share_elec"], 1), "solaire_elec": round(world.loc[2024, "solar_share_elec"], 1),
               "doublement_solaire": dbl["Solaire (2010-2024)"][0], "pays_carte": len(carte)},
       "jointure": jointure, "annees": years, "carte": carte, "topo": topo,
       "mix": {"annees": mix.index.tolist(), "labels": ["Charbon", "Pétrole", "Gaz", "Nucléaire", "Hydraulique", "Éolien", "Solaire", "Biocarburants", "Autres renouvelables"],
               "twh": {s: mix[s].round(0).tolist() for s in src}},
       "doublement": {k: {"annees": v[0], "croissance": v[1]} for k, v in dbl.items()},
       "elec_monde": {"annees": elec_w.index.tolist(), **{c: elec_w[c].tolist() for c in elec_w.columns}},
       "top_bas_carbone": top_bas_carbone.to_dict(), "top_solaire": top_solaire.to_dict(), "top_prog": top_prog.to_dict(),
       "profils": prof.reset_index().rename(columns={"country": "pays"}).to_dict("records")}
json.dump(out, open(ROOT / "results/p_energie_monde.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else None)
print(out["kpi"], jointure); print(dbl); print(top_bas_carbone.head(5)); print(top_prog.head(5)); print(prof)
