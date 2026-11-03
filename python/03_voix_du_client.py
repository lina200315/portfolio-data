"""
Projet : Voix du client — analyse de 14 640 tweets adressés à 6 compagnies aériennes américaines
Données : Twitter US Airline Sentiment (février 2015, sentiment annoté à la main)
Méthode : nettoyage du texte -> KPI de sentiment par compagnie -> motifs de plainte
          -> classifieur TF-IDF + régression logistique -> mots qui pèsent le plus
"""
import json, pathlib, re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/airline_tweets.csv")
df["date"] = pd.to_datetime(df.tweet_created.str[:10])

def clean(t):
    t = t.lower()
    t = re.sub(r"@\w+", " ", t)              # mentions (nom de la compagnie = fuite d'info)
    t = re.sub(r"http\S+", " ", t)           # liens
    t = re.sub(r"&amp;", " and ", t)
    t = re.sub(r"[^a-z' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
df["clean"] = df.text.map(clean)

# ---------- KPI ----------
share = pd.crosstab(df.airline, df.airline_sentiment, normalize="index") * 100
share["volume"] = df.airline.value_counts()
# Net sentiment = % positif - % négatif (même logique qu'un NPS)
share["net"] = share["positive"] - share["negative"]
share = share.sort_values("net", ascending=False).round(1)
airlines = [{"airline": a, "pos": r.positive, "neu": r.neutral, "neg": r.negative,
             "net": r.net, "volume": int(r.volume)} for a, r in share.iterrows()]
kpi = {"tweets": len(df), "neg": round((df.airline_sentiment == "negative").mean() * 100, 1),
       "pos": round((df.airline_sentiment == "positive").mean() * 100, 1),
       "net": round(((df.airline_sentiment == "positive").mean() - (df.airline_sentiment == "negative").mean()) * 100, 1),
       "periode": f"{df.date.min():%d/%m/%Y} – {df.date.max():%d/%m/%Y}"}

TR = {"Customer Service Issue": "Service client", "Late Flight": "Retard", "Can't Tell": "Indéterminé",
      "Cancelled Flight": "Vol annulé", "Lost Luggage": "Bagage perdu", "Bad Flight": "Mauvais vol",
      "Flight Booking Problems": "Réservation", "Flight Attendant Complaints": "Personnel navigant",
      "longlines": "Files d'attente", "Damaged Luggage": "Bagage abîmé"}
neg = df[df.airline_sentiment == "negative"].copy()
neg["motif"] = neg.negativereason.map(TR)
reasons = neg.motif.value_counts()
reasons = [{"motif": k, "n": int(v), "part": round(v / len(neg) * 100, 1)} for k, v in reasons.items()]
heat = (pd.crosstab(neg.airline, neg.motif, normalize="index") * 100).round(1)
top_motifs = [r["motif"] for r in reasons[:6]]
heat = {"lignes": heat.index.tolist(), "colonnes": top_motifs, "valeurs": heat[top_motifs].values.tolist()}

daily = df.groupby([df.date.dt.strftime("%d/%m"), "airline_sentiment"]).size().unstack(fill_value=0)
daily = {"jours": daily.index.tolist(), "neg": daily["negative"].tolist(),
         "neu": daily["neutral"].tolist(), "pos": daily["positive"].tolist()}

# ---------- Classifieur ----------
X_tr, X_te, y_tr, y_te = train_test_split(df.clean, df.airline_sentiment, test_size=0.2,
                                          stratify=df.airline_sentiment, random_state=7)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
Xtr = vec.fit_transform(X_tr); Xte = vec.transform(X_te)
clf = LogisticRegression(max_iter=3000, C=4, class_weight="balanced").fit(Xtr, y_tr)
pred = clf.predict(Xte)
labels = ["negative", "neutral", "positive"]
model = {"accuracy": round(accuracy_score(y_te, pred) * 100, 1),
         "f1_macro": round(f1_score(y_te, pred, average="macro"), 3),
         "baseline": round(y_te.value_counts(normalize=True).max() * 100, 1),
         "confusion": confusion_matrix(y_te, pred, labels=labels).tolist(),
         "vocab": len(vec.vocabulary_)}
names = np.array(vec.get_feature_names_out())
words = {}
for i, lab in enumerate(clf.classes_):
    top = np.argsort(clf.coef_[i])[::-1][:12]
    words[lab] = [{"mot": names[j], "poids": round(clf.coef_[i][j], 2)} for j in top]

# Exemples de prédictions sur des phrases nouvelles
demo = ["thanks for the quick rebooking, great crew", "flight delayed 3 hours and nobody tells us anything",
        "what time does boarding start for flight 123", "lost my bag again, worst customer service ever"]
demo = [{"texte": t, "pred": clf.predict(vec.transform([t]))[0],
         "proba": round(clf.predict_proba(vec.transform([t])).max() * 100)} for t in demo]

out = {"kpi": kpi, "airlines": airlines, "reasons": reasons, "heat": heat, "daily": daily,
       "model": model, "words": words, "demo": demo}
json.dump(out, open(ROOT / "results/p_voc.json", "w"), ensure_ascii=False, indent=1)
print(kpi, airlines, reasons[:5], model, demo, sep="\n")
print({k: [w["mot"] for w in v[:8]] for k, v in words.items()})
