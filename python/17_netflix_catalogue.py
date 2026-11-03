"""
Projet : Le catalogue Netflix — stratégie de contenu et thèmes cachés dans les résumés
Données : 7 787 titres (films et séries) du catalogue Netflix au début de 2021, via TidyTuesday (Shivam Bansal, Kaggle)
Méthode : nettoyage des champs texte multi-valeurs (pays, genres, acteurs) -> croissance du catalogue
          -> internationalisation -> durées et publics -> thèmes des résumés par factorisation de matrice (NMF sur TF-IDF)
"""
import json, pathlib, re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/netflix_titles.csv")
df["date_added"] = pd.to_datetime(df.date_added.str.strip(), format="%B %d, %Y", errors="coerce")
df["annee_ajout"] = df.date_added.dt.year
miss = (df[["director", "cast", "country", "date_added", "rating"]].isna().mean() * 100).round(1).to_dict()
explode = lambda col: df.assign(**{col: df[col].str.split(", ")}).explode(col).assign(**{col: lambda x: x[col].str.strip()})

# ---------- Croissance ----------
growth = df.dropna(subset=["annee_ajout"]).groupby(["annee_ajout", "type"]).size().unstack(fill_value=0)
growth = growth.loc[2015:2020]
age = (df.annee_ajout - df.release_year).dropna()
age_type = df.assign(age=df.annee_ajout - df.release_year).groupby("type").age.median()
recent = df.assign(age=df.annee_ajout - df.release_year)
part_recent = recent[recent.annee_ajout >= 2019].age.le(1).mean() * 100

# ---------- International ----------
c = explode("country").dropna(subset=["country"])
c = c[c.country != ""]
top_c = c.country.value_counts().head(12)
us = c.assign(us=c.country.eq("United States")).groupby("annee_ajout").us.mean().loc[2016:2020].mul(100).round(1)
pays_par_an = c.groupby("annee_ajout").country.nunique().loc[2015:2020]

# ---------- Genres ----------
gnr = explode("listed_in")
top_g = {t: [{"genre": g, "n": int(n)} for g, n in gnr[gnr.type == t].listed_in.value_counts().head(8).items()]
         for t in ["Movie", "TV Show"]}
inter = gnr[gnr.listed_in.isin(["International Movies", "International TV Shows"])].show_id.nunique() / len(df) * 100

# ---------- Durées et publics ----------
mv = df[df.type == "Movie"].copy(); mv["minutes"] = mv.duration.str.extract(r"(\d+)").astype(float)
tv = df[df.type == "TV Show"].copy(); tv["saisons"] = tv.duration.str.extract(r"(\d+)").astype(float)
hist_min = np.histogram(mv.minutes.dropna().clip(upper=200), bins=range(0, 210, 10))
saisons = tv.saisons.value_counts().sort_index().head(8)
PUB = {"TV-MA": "Adultes", "R": "Adultes", "NC-17": "Adultes", "TV-14": "Ados", "PG-13": "Ados", "TV-PG": "Famille", "PG": "Famille",
       "TV-Y7": "Enfants", "TV-Y": "Enfants", "TV-G": "Tous publics", "G": "Tous publics", "TV-Y7-FV": "Enfants"}
public = df.rating.map(PUB).fillna("Non classé").value_counts()

# ---------- Thèmes des résumés (NMF) ----------
tf = TfidfVectorizer(stop_words="english", max_df=0.05, min_df=8, ngram_range=(1, 2), sublinear_tf=True)
X = tf.fit_transform(df.description.fillna(""))
nmf = NMF(n_components=12, random_state=1, init="nndsvda", max_iter=600).fit(X)
words = np.array(tf.get_feature_names_out())
W = nmf.transform(X)
df["theme"] = W.argmax(axis=1)
themes = []
for k, comp in enumerate(nmf.components_):
    top = words[np.argsort(comp)[::-1][:8]].tolist()
    ex = df[df.theme == k].assign(w=W[df.theme == k, k]).nlargest(3, "w").title.tolist()
    themes.append({"k": k, "mots": top, "n": int((df.theme == k).sum()), "exemples": ex,
                   "part_series": round((df[df.theme == k].type == "TV Show").mean() * 100, 1)})
themes = sorted(themes, key=lambda t: -t["n"])

out = {"kpi": {"titres": len(df), "films": int((df.type == "Movie").sum()), "series": int((df.type == "TV Show").sum()),
               "pays": int(c.country.nunique()), "part_inter": round(inter, 1), "duree_med": int(mv.minutes.median()),
               "part_recent": round(part_recent, 1)},
       "miss": miss, "growth": {"annees": [int(a) for a in growth.index], "films": growth["Movie"].tolist(), "series": growth["TV Show"].tolist()},
       "age_med": {k: float(v) for k, v in age_type.items()}, "top_pays": top_c.to_dict(), "us": us.to_dict(),
       "pays_par_an": {int(k): int(v) for k, v in pays_par_an.items()}, "genres": top_g,
       "minutes": {"x": hist_min[1][:-1].tolist(), "y": hist_min[0].tolist()}, "saisons": {int(k): int(v) for k, v in saisons.items()},
       "public": public.to_dict(), "themes": themes}
json.dump(out, open(ROOT / "results/p_netflix.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(out["kpi"], miss); print(growth); print(us); print(top_c.head(6)); print(public)
for t in themes: print(t["n"], t["mots"][:6], t["exemples"][:2])
