"""
Projet : 150 ans de football international — classement Elo, avantage du terrain et prédiction des matchs
Données : 49 945 matchs internationaux masculins de 1872 à 2026 (résultats et tirs au but), base de Mart Jürisoo
Méthode : calcul d'un classement Elo match par match (comme le World Football Elo Ratings)
          -> évolution des grandes nations -> avantage du terrain dans le temps -> buts par match
          -> qualité de prédiction de l'Elo, testée sur les matchs de 2024 à 2026
"""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/football_results.csv", parse_dates=["date"])
df = df.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
df["buts"] = df.home_score + df.away_score
df["res"] = np.sign(df.home_score - df.away_score)              # 1 domicile, 0 nul, -1 extérieur

# ---------- 1. Elo ----------
# K dépend de l'enjeu ; l'écart de buts augmente l'ajustement ; +100 points pour l'équipe qui reçoit (hors terrain neutre)
def k_of(t):
    if t == "FIFA World Cup": return 60
    if any(s in t for s in ["Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup", "Confederations"]) and "qualification" not in t: return 50
    if "qualification" in t or "Nations League" in t: return 40
    if t == "Friendly": return 20
    return 30
HOME = 100
elo, hist = {}, []
pred_rows = []
for r in df.itertuples():
    ra, rb = elo.get(r.home_team, 1500), elo.get(r.away_team, 1500)
    adv = 0 if r.neutral else HOME
    we = 1 / (1 + 10 ** (-(ra + adv - rb) / 400))
    w = 1 if r.res > 0 else 0.5 if r.res == 0 else 0
    g = abs(r.home_score - r.away_score)
    mult = 1 if g <= 1 else 1.5 if g == 2 else (11 + g) / 8
    delta = k_of(r.tournament) * mult * (w - we)
    elo[r.home_team] = ra + delta; elo[r.away_team] = rb - delta
    if r.date.year >= 2024:
        pred_rows.append((r.date, r.tournament, r.home_team, r.away_team, we, r.res, ra + adv - rb))
    hist.append((r.date.year, r.home_team, elo[r.home_team])); hist.append((r.date.year, r.away_team, elo[r.away_team]))

h = pd.DataFrame(hist, columns=["annee", "equipe", "elo"]).groupby(["annee", "equipe"]).elo.last().unstack()
h = h.ffill()
recent = df[df.date >= "2022-01-01"]
actives = set(recent.home_team) | set(recent.away_team)
top = sorted([(t, e) for t, e in elo.items() if t in actives], key=lambda x: -x[1])[:20]
nations = ["Brazil", "Argentina", "Germany", "France", "Spain", "England", "Italy", "Netherlands", "Morocco"]
traj = {n: [None if pd.isna(v) else v for v in h[n].loc[1950::2].round(0).tolist()] for n in nations}   # None avant le 1er match
# Meilleur pic de chaque nation
peaks = {n: {"elo": round(h[n].max()), "annee": int(h[n].idxmax())} for n in nations}

# ---------- 2. Avantage du terrain ----------
nn = df[~df.neutral].copy(); nn["decennie"] = (nn.date.dt.year // 10 * 10)
home = nn[nn.decennie >= 1900].groupby("decennie").res.agg(lambda s: (s == 1).mean() * 100).round(1)
away = nn[nn.decennie >= 1900].groupby("decennie").res.agg(lambda s: (s == -1).mean() * 100).round(1)
draw = nn[nn.decennie >= 1900].groupby("decennie").res.agg(lambda s: (s == 0).mean() * 100).round(1)
# 2020 : matchs à huis clos (COVID) — l'avantage du terrain a-t-il baissé ?
y = nn[nn.date.dt.year.between(2015, 2023)].groupby(nn.date.dt.year).res.agg(lambda s: (s == 1).mean() * 100).round(1)

# ---------- 3. Buts ----------
df["decennie"] = df.date.dt.year // 10 * 10
goals = df[df.decennie >= 1900].groupby("decennie").buts.mean().round(2)
wc = df[df.tournament == "FIFA World Cup"]
wc_goals = wc.groupby(wc.date.dt.year).buts.mean().round(2)

# ---------- 4. Qualité de la prédiction (2024-2026) ----------
p = pd.DataFrame(pred_rows, columns=["date", "tournoi", "dom", "ext", "we", "res", "ecart"])
dec = p[p.res != 0]                                          # matchs avec un vainqueur
acc = ((dec.we > 0.5) == (dec.res == 1)).mean() * 100
p["bin"] = pd.cut(p.we, [0, .2, .35, .5, .65, .8, 1])
calib = p.groupby("bin", observed=True).agg(attendu=("we", "mean"), n=("we", "size"),
                                             victoires=("res", lambda s: (s == 1).mean()), nuls=("res", lambda s: (s == 0).mean()))
brier = float(((p.we - p.res.map({1: 1, 0: .5, -1: 0})) ** 2).mean())
wc26 = p[p.tournoi == "FIFA World Cup"]
wc26_acc = float(((wc26[wc26.res != 0].we > .5) == (wc26[wc26.res != 0].res == 1)).mean() * 100) if len(wc26) else None
surprises = p[p.tournoi.isin(["FIFA World Cup", "UEFA Euro", "Copa América", "African Cup of Nations"])].assign(
    proba_vainqueur=lambda d: np.where(d.res == 1, d.we, 1 - d.we))
surprises = surprises[surprises.res != 0].nsmallest(8, "proba_vainqueur")
final26 = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == 2026)].tail(1)

out = {"kpi": {"matchs": len(df), "equipes": int(pd.concat([df.home_team, df.away_team]).nunique()), "debut": int(df.date.dt.year.min()),
               "fin": df.date.max().strftime("%d/%m/%Y"), "buts_moy": round(df.buts.mean(), 2), "acc": round(acc, 1), "brier": round(brier, 3),
               "victoire_dom": round((nn.res == 1).mean() * 100, 1), "wc26_acc": round(wc26_acc, 1) if wc26_acc else None, "wc26_n": int(len(wc26))},
       "top": [{"equipe": t, "elo": round(e)} for t, e in top],
       "traj": {"annees": list(range(1950, int(h.index.max()) + 1, 2))[:len(traj[nations[0]])], **traj}, "peaks": peaks,
       "home": {"decennies": [int(d) for d in home.index], "dom": home.tolist(), "nul": draw.tolist(), "ext": away.tolist()},
       "home_years": {int(k): v for k, v in y.items()},
       "goals": {"decennies": [int(d) for d in goals.index], "v": goals.tolist()},
       "wc_goals": {int(k): v for k, v in wc_goals.items()},
       "calib": {"labels": [f"{int(i.left*100)}-{int(i.right*100)} %" for i in calib.index], "attendu": (calib.attendu * 100).round(1).tolist(),
                 "victoires": (calib.victoires * 100).round(1).tolist(), "nuls": (calib.nuls * 100).round(1).tolist(), "n": calib.n.tolist()},
       "surprises": [{"date": r.date.strftime("%d/%m/%Y"), "tournoi": r.tournoi, "match": f"{r.dom} - {r.ext}",
                      "vainqueur": r.dom if r.res == 1 else r.ext, "proba": round(r.proba_vainqueur * 100, 1)} for r in surprises.itertuples()],
       "final26": final26[["date", "home_team", "away_team", "home_score", "away_score"]].astype(str).to_dict("records")}
json.dump(out, open(ROOT / "results/p_football.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(out["kpi"]); print(out["top"][:10]); print(peaks); print(out["home_years"]); print(out["calib"]); print(out["surprises"][:4]); print(out["final26"])
