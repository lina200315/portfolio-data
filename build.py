"""Assemble les résultats de chaque projet dans la page index.html (site statique, sans serveur)."""
import base64, json, pathlib, sqlite3
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
J = lambda f: json.load(open(ROOT / "results" / f, encoding="utf-8"))

# ---- Synthèse Excel (mêmes calculs que le classeur, en pandas, pour les graphiques du site) ----
s = pd.read_csv(ROOT / "data/superstore.csv", encoding="latin1")
s["d0"] = pd.to_datetime(s["Order Date"], format="%m/%d/%Y"); s["d1"] = pd.to_datetime(s["Ship Date"], format="%m/%d/%Y")
g = lambda col: [{"label": k, "ca": round(v.Sales.sum()), "profit": round(v.Profit.sum()),
                  "marge": round(v.Profit.sum() / v.Sales.sum() * 100, 1)} for k, v in s.groupby(col)]
s["rem"] = pd.cut(s.Discount, [-.01, .02, .05, .08, 1], labels=["0-2 %", "3-5 %", "6-8 %", "> 8 %"])
excel = {"kpi": {"ca": s.Sales.sum(), "profit": s.Profit.sum(), "marge": s.Profit.sum() / s.Sales.sum() * 100,
                 "lignes": len(s), "panier": s.Sales.mean(), "delai": (s.d1 - s.d0).dt.days.mean()},
         "cat": g("Product Category"), "sub": g("Product Sub-Category"), "disc": g("rem"),
         "region": sorted(g("Region"), key=lambda r: -r["ca"])}

# ---- Données du dashboard interactif : une ligne = [année, mois, région, catégorie, sous-cat., segment, expédition, CA, profit, commande] ----
dims = {"year": sorted(s.d0.dt.year.unique().tolist()), "reg": sorted(s.Region.unique()), "cat": sorted(s["Product Category"].unique()),
        "sub": sorted(s["Product Sub-Category"].unique()), "seg": sorted(s["Customer Segment"].unique()), "ship": sorted(s["Ship Mode"].unique())}
code_of = lambda col, k: s[col].map({v: i for i, v in enumerate(dims[k])})
ordc = {v: i for i, v in enumerate(s["Order ID"].unique())}
dash = {"dims": {k: [int(x) if k == "year" else x for x in v] for k, v in dims.items()},
        "rows": [list(r) for r in zip(s.d0.dt.year.map({y: i for i, y in enumerate(dims["year"])}), s.d0.dt.month,
                 code_of("Region", "reg"), code_of("Product Category", "cat"), code_of("Product Sub-Category", "sub"),
                 code_of("Customer Segment", "seg"), code_of("Ship Mode", "ship"), s.Sales.round(2), s.Profit.round(2), s["Order ID"].map(ordc))]}

# ---- Aperçus des jeux de données ----
def sample(name, file, desc, project, source, n=150, cols=None, **kw):
    df = pd.read_csv(ROOT / "data" / file, **kw)
    if cols: df = df[cols]
    return {"name": name, "file": "data/" + file, "desc": desc, "project": project, "source": source, "n": int(len(df)),
            "cols": list(df.columns), "rows": df.head(n).astype(object).where(df.head(n).notna(), "").values.tolist()}

samples = {
    "telco": sample("Telco Customer Churn", "telco.csv", "Un abonné par ligne : contrat, services souscrits, facturation et résiliation (Churn).", "Churn", "IBM (GitHub IBM/telco-customer-churn-on-icp4d)"),
    "cx": sample("Airline Passenger Satisfaction", "airline_satisfaction.csv", "Un passager par ligne : profil, vol, retards, 14 notes de 0 à 5 et satisfaction globale.", "Expérience client", "Maven Analytics / Kaggle (miroir GitHub)"),
    "tweets": sample("Twitter US Airline Sentiment", "airline_tweets.csv", "Un tweet par ligne, avec le sentiment et le motif de plainte annotés à la main.", "Voix du client", "Crowdflower / Kaggle (miroir GitHub)",
                     cols=["airline", "airline_sentiment", "negativereason", "text", "tweet_created"]),
    "superstore": sample("Superstore Sales", "superstore.csv", "Une ligne de commande par ligne : client, région, produit, ventes, remise, profit, dates.", "Excel et Prévision", "curran/data (GitHub)", encoding="latin1"),
    "marketing": sample("Marketing Campaign (iFood)", "marketing.csv", "Un client par ligne : revenu, foyer, dépenses par rayon, achats par canal, réponses aux campagnes.", "Segmentation", "Test business analyst iFood (GitHub)"),
    "ab": sample("A/B Test e-commerce", "ab_test.csv", "Une visite par ligne : groupe (contrôle ou traitement), page vue et conversion.", "Test A/B", "Projet Udacity, miroir GitHub"),
}
import duckdb
fl = duckdb.sql(f"SELECT * FROM '{ROOT / 'data/nyc_flights.parquet'}'").df()
samples["vols"] = {"name": "nycflights13 (vols New York 2013)", "file": "data/nyc_flights.parquet", "desc": "Un vol par ligne : horaires prévus et réels, retards, compagnie, avion, origine, destination, distance.",
                   "project": "Ponctualité des vols", "source": "Bureau of Transportation Statistics, via le package R nycflights13", "n": int(len(fl)), "cols": list(fl.columns),
                   "rows": fl.head(150).astype(object).where(fl.head(150).notna(), "").astype(str).values.tolist()}
OF = ["id_cie", "compagnie_id", "depart", "depart_id", "arrivee", "arrivee_id", "codeshare", "escales", "avions"]
samples["routes"] = sample("OpenFlights : routes aériennes", "openflights_routes.dat", "Une liaison par ligne : compagnie, aéroports de départ et d'arrivée, escales, types d'avion. Fichier brut sans en-tête.",
                           "Réseau aérien mondial", "OpenFlights (GitHub jpatokal/openflights)", header=None, names=OF)
samples["airports"] = sample("OpenFlights : aéroports", "openflights_airports.dat", "Un aéroport par ligne : nom, ville, pays, codes IATA/ICAO, coordonnées, altitude.",
                             "Réseau aérien mondial", "OpenFlights", header=None, names=["id", "nom", "ville", "pays", "iata", "icao", "lat", "lon", "alt", "tz", "dst", "tzdb", "type", "source"])
samples["bike"] = sample("Capital Bikeshare (horaire)", "bike_hour.csv", "Une heure par ligne : calendrier, météo normalisée, locations des occasionnels et des abonnés.", "Demande de vélos", "UCI Machine Learning Repository")
samples["energy"] = sample("PJM East (électricité)", "pjm_energy.csv", "Un relevé horaire de consommation électrique en mégawatts.", "Consommation électrique", "PJM Interconnection, via Kaggle (miroir GitHub)")
samples["ames"] = sample("Ames Housing", "ames_housing.csv", "Une vente par ligne : 82 caractéristiques du bien et prix de vente.", "Prix immobilier", "Dean De Cock, Journal of Statistics Education (miroir GitHub)", index_col=0)
samples["hr"] = sample("IBM HR Attrition", "hr_attrition.csv", "Un salarié par ligne : poste, salaire, satisfaction, heures sup., départ.", "People analytics", "IBM (GitHub IBM/employee-attrition-aif360)", encoding="utf-8-sig")
samples["co2"] = sample("Our World in Data : CO₂", "owid_co2.csv", "Un pays-année par ligne, de 1750 à 2024 : émissions par source, par habitant, cumulées, PIB, population.", "Émissions de CO₂", "Our World in Data (GitHub owid/co2-data)",
                        cols=["country", "year", "iso_code", "population", "gdp", "co2", "co2_per_capita", "coal_co2", "oil_co2", "gas_co2", "share_global_co2"])
samples["co2"]["rows"] = [r for r in samples["co2"]["rows"]]
ol = duckdb.sql(f"SELECT * FROM '{ROOT / 'data/olist/olist_orders.parquet'}'").df()
samples["olist"] = {"name": "Olist : commandes (1 table sur 8)", "file": "data/olist/olist_orders.parquet", "desc": "Une commande par ligne : statut et horodatages (achat, validation, expédition, livraison, date promise). Les 7 autres tables sont dans data/olist/.",
                    "project": "E-commerce Olist", "source": "Olist (GitHub olist/work-at-olist-data)", "n": int(len(ol)), "cols": list(ol.columns),
                    "rows": ol.head(150).astype(object).where(ol.head(150).notna(), "").astype(str).values.tolist()}
samples["hotels"] = sample("Hotel Booking Demand", "hotel_bookings.csv", "Une réservation par ligne : hôtel, délai, dates, séjour, canal, acompte, prix, annulation.", "Annulations d'hôtel", "Antonio, Almeida et Nunes (2019), via TidyTuesday")
samples["spotify"] = sample("Spotify Songs", "spotify_songs.csv", "Un titre de playlist par ligne : artiste, genre, sous-genre, popularité et 12 mesures audio.", "Genres musicaux", "TidyTuesday (spotifyr)",
                            cols=["track_name", "track_artist", "track_popularity", "playlist_genre", "playlist_subgenre", "danceability", "energy", "loudness", "speechiness", "acousticness", "valence", "tempo"])
samples["gapminder"] = sample("Gapminder", "gapminder.tsv", "Un pays-année par ligne (tous les 5 ans, 1952-2007) : continent, espérance de vie, population, PIB par habitant.", "Richesse et espérance de vie", "Package R gapminder (Jenny Bryan)", sep="\t")
samples["netflix"] = sample("Netflix Titles", "netflix_titles.csv", "Un titre par ligne : type, titre, réalisateur, pays, date d'ajout, année, classification, durée, genres, résumé.", "Catalogue Netflix", "TidyTuesday (Shivam Bansal, Kaggle)",
                            cols=["type", "title", "country", "date_added", "release_year", "rating", "duration", "listed_in", "description"])
samples["football"] = sample("Football international", "football_results.csv", "Un match par ligne : date, équipes, score, compétition, ville, pays, terrain neutre.", "Football et Elo", "Mart Jürisoo (GitHub martj42/international_results)")
samples["bank"] = sample("UCI Bank Marketing", "bank_marketing.csv", "Un appel par ligne : profil du client, contact, historique de campagne, contexte économique et souscription (y).", "Marketing bancaire", "UCI Machine Learning Repository (Moro et al., 2014)", sep=";")
samples["insurance"] = sample("Medical Cost Personal", "insurance.csv", "Un assuré par ligne : âge, sexe, IMC, enfants, tabac, région et frais médicaux annuels.", "Tarification assurance", "Machine Learning with R (Brett Lantz), GitHub stedy")
samples["energie"] = sample("Our World in Data : énergie", "owid_energy.csv", "Un pays-année par ligne : consommation par source (TWh), mix électrique (%), intensité carbone.", "Transition énergétique", "Our World in Data (GitHub owid/energy-data)",
                            cols=["country", "year", "iso_code", "electricity_generation", "low_carbon_share_elec", "renewables_share_elec", "solar_share_elec", "wind_share_elec", "nuclear_share_elec", "coal_share_elec", "carbon_intensity_elec"])
samples["lalonde"] = sample("LaLonde (NSW + PSID)", "lalonde_observationnel.tsv", "Une personne par ligne : bénéficiaire ou non de la formation, profil, revenus 1974, 1975 et 1978.", "Évaluation d'impact", "Package R MatchIt (Dehejia et Wahba, LaLonde 1986)", sep=r"\s+")
con = sqlite3.connect(ROOT / "data/chinook.sqlite")
tabs = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite%' ORDER BY name")]
samples["chinook"] = {"name": "Chinook (base SQLite)", "file": "data/chinook.sqlite", "project": "SQL", "source": "lerocha/chinook-database (GitHub)",
                      "desc": "Base relationnelle d'une boutique de musique. Le tableau liste ses tables et leur nombre de lignes.",
                      "n": len(tabs), "cols": ["Table", "Lignes", "Colonnes"],
                      "rows": [[t, con.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0],
                                ", ".join(c[1] for c in con.execute(f"PRAGMA table_info([{t}])"))] for t in tabs]}
total_rows = sum(v["n"] for k, v in samples.items() if k != "chinook") + sum(r[1] for r in samples["chinook"]["rows"])

code_files = ["excel/build_dashboard.py", "sql/analyses_chinook.sql", "python/01_churn_prediction.py", "python/02_experience_client.py",
              "python/03_voix_du_client.py", "python/04_prevision_ventes.py", "python/05_etl_modele_etoile.py", "python/06_valeur_client.py",
              "python/07_analyse_panier.py", "python/08_retards_aeriens.py", "python/09_reseau_aerien.py", "python/10_co2_monde.py",
              "python/11_velos_demande.py", "python/12_prix_immobilier.py", "python/13_electricite.py",
              "python/14_olist_ecommerce.py", "python/15_hotels_annulations.py", "python/16_spotify_genres.py", "python/17_netflix_catalogue.py",
              "python/18_football_elo.py", "python/19_banque_marketing.py", "python/20_dashboards_donnees.py", "R/06_assurance_sante.R",
              "python/21_transition_energetique.py", "python/22_optimisation_entrepots.py", "python/23_stocks_monte_carlo.py", "R/07_evaluation_impact.R",
              "R/05_gapminder_developpement.R", "build.py", "make_notebooks.py", "sql/run_sql.py",
              "R/01_segmentation_clients.R", "R/02_ab_test.R", "R/03_survie_clients.R", "R/04_attrition_rh.R"]
D = {"totalRows": total_rows, "excel": excel, "sql": J("sql_chinook.json"), "cx": J("p_cx.json"), "voc": J("p_voc.json"),
     "fc": J("p_forecast.json"), "dash": dash, "etl": J("p_etl.json"), "basket": J("p_basket.json"), "clv": J("p_clv.json"), "surv": J("r_survie.json"), "flights": J("p_flights.json"), "network": J("p_network.json"), "co2": J("p_co2.json"), "bike": J("p_bike.json"), "house": J("p_house.json"), "energy": J("p_energy.json"), "rh": J("r_rh.json"), "olist": J("p_olist.json"), "hotel": J("p_hotel.json"), "gap": J("r_gapminder.json"), "spotify": J("p_spotify.json"), "netflix": J("p_netflix.json"), "dash5": J("dashboards.json"), "foot": J("p_football.json"), "bank": J("p_bank.json"), "ins": J("r_assurance.json"), "emonde": J("p_energie_monde.json"), "impact": J("r_impact.json"), "wh": J("p_entrepots.json"), "stocks": J("p_stocks.json"), "seg": J("r_segmentation.json"), "ab": J("r_abtest.json"), "churn": J("p_churn.json"),
     "code": {f: (ROOT / f).read_text(encoding="utf-8") for f in code_files}, "samples": samples,
     "shot": "data:image/jpeg;base64," + base64.b64encode((ROOT / "excel/apercu_dashboard.jpg").read_bytes()).decode()}

payload = json.dumps(D, ensure_ascii=False, allow_nan=False, default=lambda o: o.item() if hasattr(o, "item") else float(o), separators=(",", ":")).replace("</", "<\\/")
html = (ROOT / "site/template.html").read_text(encoding="utf-8").replace("__DATA__", payload)
(ROOT / "site/artifact.html").write_text(html, encoding="utf-8")       # version publiée sur claude.ai
full = ('<!doctype html>\n<html lang="fr">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n'
        + html + "\n</body>\n</html>\n")
(ROOT / "index.html").write_text(full, encoding="utf-8")                # version GitHub Pages
print("index.html", round(len(html.encode()) / 1e6, 2), "Mo ; lignes de données :", total_rows)
