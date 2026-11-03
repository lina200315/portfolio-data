"""
Projet : Prévoir les ventes mensuelles d'un distributeur de fournitures de bureau
Données : Superstore Sales (8 399 lignes de commande, Canada, 2009-2012)
Méthode : agrégation mensuelle -> décomposition tendance / saisonnalité
          -> 3 modèles (naïf saisonnier, Holt-Winters, SARIMA) testés sur les 9 derniers mois
          -> prévision 6 mois avec intervalle de confiance
"""
import json, pathlib, warnings
import numpy as np, pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/superstore.csv", encoding="latin1")
df["date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
ts = df.set_index("date").Sales.resample("MS").sum()
H = 9                                            # mois mis de côté pour le test
train, test = ts[:-H], ts[-H:]

def mape(a, f): return float(np.mean(np.abs((a - f) / a)) * 100)

fc = {}
fc["Naïf saisonnier"] = ts.shift(12)[-H:].values                 # même mois l'an dernier
hw = ExponentialSmoothing(np.log(train), trend="add", seasonal="add", seasonal_periods=12).fit()
fc["Holt-Winters"] = np.exp(hw.forecast(H)).values
sar = SARIMAX(np.log(train), order=(1, 0, 0), seasonal_order=(0, 1, 1, 12), trend="c").fit(disp=False)
fc["SARIMA"] = np.exp(sar.forecast(H)).values
fc["Moyenne mobile 3 mois"] = np.repeat(train[-3:].mean(), H)
scores = {k: round(mape(test.values, v), 1) for k, v in fc.items()}
best = min(scores, key=scores.get)

# Prévision finale : réentraînement sur tout l'historique
if best == "SARIMA":
    final = SARIMAX(np.log(ts), order=(1, 0, 0), seasonal_order=(0, 1, 1, 12), trend="c").fit(disp=False)
    f = final.get_forecast(6); mean = np.exp(f.predicted_mean); ci = np.exp(f.conf_int(alpha=0.2))
    lo, hi = ci.iloc[:, 0], ci.iloc[:, 1]
else:
    final = ExponentialSmoothing(np.log(ts), trend="add", seasonal="add", seasonal_periods=12).fit()
    mean = np.exp(final.forecast(6))
    resid = np.log(ts) - final.fittedvalues; sd = resid.std()
    lo, hi = mean * np.exp(-1.28 * sd), mean * np.exp(1.28 * sd)
fut = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(), periods=6, freq="MS")

dec = seasonal_decompose(np.log(ts), model="additive", period=12)
season = (np.exp(dec.seasonal[:12]) - 1) * 100
season = {pd.Timestamp(d).strftime("%b"): round(v, 1) for d, v in season.items()}

yearly = ts.groupby(ts.index.year).sum()
kpi = {"ca_total": round(ts.sum()), "mois": len(ts), "meilleur_modele": best,
       "mape": scores[best], "prev_6m": round(mean.sum()),
       "croissance_2012": round((yearly[2012] / yearly[2011] - 1) * 100, 1) if 2012 in yearly else None,
       "cv_mensuel": round(ts.std() / ts.mean() * 100, 1)}
out = {"kpi": kpi, "scores": scores,
       "serie": {"mois": [d.strftime("%m/%Y") for d in ts.index], "ventes": ts.round(0).tolist()},
       "test": {"mois": [d.strftime("%m/%Y") for d in test.index], "reel": test.round(0).tolist(),
                **{k: np.round(v, 0).tolist() for k, v in fc.items()}},
       "prevision": {"mois": [d.strftime("%m/%Y") for d in fut], "moy": np.round(mean.values, 0).tolist(),
                     "bas": np.round(np.asarray(lo), 0).tolist(), "haut": np.round(np.asarray(hi), 0).tolist()},
       "saison": season, "annuel": {int(k): round(v) for k, v in yearly.items()}}
json.dump(out, open(ROOT / "results/p_forecast.json", "w"), ensure_ascii=False, indent=1)
print(kpi, scores, season, out["annuel"], out["prevision"], sep="\n")
