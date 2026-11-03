"""
Projet : Peut-on reconnaître le genre d'un morceau à partir de son « ADN » audio ?
Données : 32 833 titres de playlists Spotify, 6 genres, 12 caractéristiques audio calculées par Spotify (TidyTuesday 2020)
Méthode : dédoublonnage (un titre peut être dans plusieurs playlists) -> profils audio par genre
          -> classification multi-classes (random forest) -> matrice de confusion
          -> évolution du son par décennie -> la popularité est-elle prévisible ?
"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.preprocessing import MinMaxScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "data/spotify_songs.csv")
df = raw.drop_duplicates("track_id").copy()                      # même titre, plusieurs playlists
dups = len(raw) - len(df)
df["annee"] = pd.to_numeric(df.track_album_release_date.str[:4], errors="coerce")
df["duree_min"] = df.duration_ms / 60000
GEN = {"edm": "EDM", "latin": "Latin", "pop": "Pop", "r&b": "R&B", "rap": "Rap", "rock": "Rock"}
df["genre"] = df.playlist_genre.map(GEN)
AUD = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duree_min"]
FR = {"danceability": "Dansabilité", "energy": "Énergie", "loudness": "Volume (dB)", "speechiness": "Parole", "acousticness": "Acoustique",
      "instrumentalness": "Instrumental", "liveness": "Live", "valence": "Gaieté", "tempo": "Tempo (BPM)", "duree_min": "Durée (min)", "key": "Tonalité", "mode": "Mode majeur"}

# ---------- Profils (normalisés 0-1 pour le radar) ----------
sc = MinMaxScaler().fit(df[AUD].clip(df[AUD].quantile(.01), df[AUD].quantile(.99), axis=1))
norm = pd.DataFrame(sc.transform(df[AUD].clip(df[AUD].quantile(.01), df[AUD].quantile(.99), axis=1)), columns=AUD, index=df.index)
radar = norm.groupby(df.genre).mean().round(3)
brut = df.groupby("genre")[AUD].mean().round(2)

# ---------- Classification ----------
X = df[AUD + ["key", "mode"]]; y = df.genre
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
pred = rf.predict(X_te)
labels = sorted(GEN.values())
cm = confusion_matrix(y_te, pred, labels=labels)
cm_pct = (cm / cm.sum(axis=1, keepdims=True) * 100).round(1)
acc = accuracy_score(y_te, pred)
per = {l: round(cm_pct[i, i], 1) for i, l in enumerate(labels)}
imp = pd.Series(rf.feature_importances_, index=[FR[c] for c in X.columns]).sort_values(ascending=False).round(3)
# Sous-genres : la confusion vient-elle des frontières floues ?
sub = df.loc[X_te.index].assign(ok=(pred == y_te.values))
sub_acc = sub.groupby(["genre", "playlist_subgenre"]).ok.mean().mul(100).round(1).sort_values()

# ---------- Le son par décennie ----------
d = df[(df.annee >= 1960) & (df.annee <= 2019)].copy()
d["decennie"] = (d.annee // 10 * 10).astype(int)
dec = d.groupby("decennie")[["energy", "acousticness", "valence", "danceability", "duree_min", "loudness"]].mean().round(3)
dec_n = d.decennie.value_counts().sort_index()

# ---------- Popularité ----------
pop = df[df.track_popularity > 0]
Xp = pop[AUD + ["key", "mode", "annee"]].fillna(2000); yp = pop.track_popularity
r2 = cross_val_score(HistGradientBoostingRegressor(max_iter=300, random_state=0), Xp, yp, cv=5, scoring="r2").mean()

out = {"kpi": {"titres": len(df), "doublons": int(dups), "artistes": int(df.track_artist.nunique()), "genres": 6, "sous_genres": int(df.playlist_subgenre.nunique()),
               "accuracy": round(acc * 100, 1), "hasard": round(100 / 6, 1), "r2_popularite": round(r2, 3)},
       "radar": {"axes": [FR[a] for a in AUD], "genres": {g: radar.loc[g].tolist() for g in radar.index}},
       "brut": brut.reset_index().to_dict("records"), "cm": {"labels": labels, "pct": cm_pct.tolist()}, "per": per,
       "imp": imp.to_dict(), "sub_worst": [{"genre": g, "sous_genre": s, "acc": v} for (g, s), v in sub_acc.head(6).items()],
       "sub_best": [{"genre": g, "sous_genre": s, "acc": v} for (g, s), v in sub_acc.tail(6).items()],
       "decennies": {"labels": [f"{x}s" for x in dec.index], "n": dec_n.tolist(), **{c: dec[c].tolist() for c in dec.columns}}}
json.dump(out, open(ROOT / "results/p_spotify.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(out["kpi"]); print(per); print(imp.head(6)); print(sub_acc.head(4), sub_acc.tail(4)); print(dec, dec_n)
