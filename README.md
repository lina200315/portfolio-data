# Lina Benslimane · Portfolio Data Analyst

Portfolio d'analyse de données réalisé dans le cadre de ma recherche d'alternance : 32 études et 6 dashboards KPI interactifs sur 25 jeux de données publics, en Excel, SQL, Python et R.

**Site en ligne : https://lina200315.github.io/portfolio-data/**

Chaque étude suit le même déroulé : la question métier, les données, la méthode, les résultats, puis une recommandation et les questions qu'on pourrait me poser en entretien.

## Thèmes

- **BI, reporting et données** : dashboard Excel en formules, dashboard à filtres croisés, reporting 100 % SQL, entrepôt de données en étoile avec contrôles qualité
- **Aérien et expérience client** : ponctualité de 336 776 vols, réseau aérien mondial (graphes), satisfaction des passagers, analyse de tweets
- **Clients et marketing** : e-commerce Olist, annulations d'hôtel, marketing bancaire, segmentation, analyse du panier, valeur client, test A/B, churn, analyse de survie
- **Prévision et machine learning** : ventes, demande de vélos, consommation électrique, prix immobiliers, assurance santé, attrition RH
- **Société et médias** : CO₂, transition énergétique, développement (Gapminder), Spotify, Netflix, football (Elo)
- **Décision et optimisation** : localisation d'entrepôts, simulation Monte Carlo, évaluation d'impact

## Organisation du dépôt

| Dossier | Contenu |
|---|---|
| `index.html` | Le site : fiches projets, graphiques, dashboards, explorateur de données et de code |
| `python/` | Scripts Python des études (pandas, scikit-learn, statsmodels, DuckDB, networkx, SciPy) |
| `R/` | Scripts R (segmentation, test A/B, survie, attrition, Gapminder, assurance, évaluation d'impact) |
| `sql/` | Requêtes SQL du reporting Chinook |
| `excel/` | Dashboard Excel des ventes (formules, filtres) et le script qui le génère |
| `results/` | Entrepôt de données Superstore (SQLite, modèle en étoile) |
| `data/` | Jeux de données utilisés |

## Données

Tous les jeux sont publics. La plupart sont fournis en entier dans `data/`. Les deux plus lourds sont compressés en .zip. Pour ceux que je n'ai pas pu joindre en entier, le dossier `data/extraits/` contient les 150 premières lignes et la source d'origine est indiquée ci-dessous. Certaines sources (Our World in Data, résultats de football) sont mises à jour régulièrement : la version du dépôt peut différer légèrement de celle utilisée pour les analyses.

| Jeu de données | Fichier | Lignes | Source |
|---|---|---|---|
| Telco Customer Churn | `data/telco.csv` | 7 043 | IBM (GitHub IBM/telco-customer-churn-on-icp4d) |
| Airline Passenger Satisfaction | extrait : `data/extraits/airline_satisfaction_extrait.csv` | 129 880 | Maven Analytics / Kaggle (miroir GitHub) |
| Twitter US Airline Sentiment | extrait : `data/extraits/airline_tweets_extrait.csv` | 14 640 | Crowdflower / Kaggle (miroir GitHub) |
| Superstore Sales | `data/superstore.csv` | 8 399 | curran/data (GitHub) |
| Marketing Campaign (iFood) | `data/marketing.csv` | 2 240 | Test business analyst iFood (GitHub) |
| A/B Test e-commerce | extrait : `data/extraits/ab_test_extrait.csv` | 294 478 | Projet Udacity  miroir GitHub |
| nycflights13 (vols New York 2013) | extrait : `data/extraits/nyc_flights_extrait.csv` | 336 776 | Bureau of Transportation Statistics  via le package R nycflights13 |
| OpenFlights : routes aériennes | `data/openflights_routes.dat` | 67 663 | OpenFlights (GitHub jpatokal/openflights) |
| OpenFlights : aéroports | `data/openflights_airports.dat` | 7 698 | OpenFlights |
| Capital Bikeshare (horaire) | extrait : `data/extraits/bike_hour_extrait.csv` | 17 379 | UCI Machine Learning Repository |
| PJM East (électricité) | `data/pjm_energy.csv` | 145 366 | PJM Interconnection  via Kaggle (miroir GitHub) |
| Ames Housing | `data/ames_housing.csv` | 2 930 | Dean De Cock  Journal of Statistics Education (miroir GitHub) |
| IBM HR Attrition | `data/hr_attrition.csv` | 1 470 | IBM (GitHub IBM/employee-attrition-aif360) |
| Our World in Data : CO₂ | `data/owid_co2.csv.zip` (à décompresser) | 50 411 | Our World in Data (GitHub owid/co2-data) |
| Olist : commandes (1 table sur 8) | extrait : `data/extraits/olist_orders_extrait.csv` | 99 441 | Olist (GitHub olist/work-at-olist-data) |
| Hotel Booking Demand | `data/hotel_bookings.csv.zip` (à décompresser) | 119 390 | Antonio  Almeida et Nunes (2019)  via TidyTuesday |
| Spotify Songs | `data/spotify_songs.csv` | 32 833 | TidyTuesday (spotifyr) |
| Gapminder | `data/gapminder.tsv` | 1 704 | Package R gapminder (Jenny Bryan) |
| Netflix Titles | `data/netflix_titles.csv` | 7 787 | TidyTuesday (Shivam Bansal  Kaggle) |
| Football international | `data/football_results.csv` | 49 945 | Mart Jürisoo (GitHub martj42/international_results) |
| UCI Bank Marketing | extrait : `data/extraits/bank_marketing_extrait.csv` | 41 188 | UCI Machine Learning Repository (Moro et al.  2014) |
| Medical Cost Personal | `data/insurance.csv` | 1 338 | Machine Learning with R (Brett Lantz)  GitHub stedy |
| Our World in Data : énergie | `data/owid_energy.csv` | 23 377 | Our World in Data (GitHub owid/energy-data) |
| LaLonde (NSW + PSID) | `data/lalonde_observationnel.tsv` | 614 | Package R MatchIt (Dehejia et Wahba  LaLonde 1986) |
| Chinook (base SQLite) | `data/chinook.sqlite` | 11 | lerocha/chinook-database (GitHub) |

## Contact

lina.benslimane.pro@hotmail.com
