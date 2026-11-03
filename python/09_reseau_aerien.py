"""
Projet : Le réseau aérien mondial vu comme un graphe — hubs, connectivité et communautés
Données : OpenFlights (67 663 routes, 7 698 aéroports, 6 162 compagnies)
Méthode : nettoyage des fichiers bruts (sans en-têtes, valeurs \\N) -> graphe orienté aéroports/routes (networkx)
          -> centralités (degré, intermédiarité) -> détection de communautés (Louvain)
          -> plus courts chemins depuis Marseille
"""
import json, pathlib
import networkx as nx, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = pd.read_csv(ROOT / "data/openflights_airports.dat", header=None, na_values="\\N",
                 names=["id", "nom", "ville", "pays", "iata", "icao", "lat", "lon", "alt", "tz", "dst", "tzdb", "type", "source"])
rt = pd.read_csv(ROOT / "data/openflights_routes.dat", header=None, na_values="\\N",
                 names=["cie", "cie_id", "src", "src_id", "dst", "dst_id", "codeshare", "escales", "avions"])
al = pd.read_csv(ROOT / "data/openflights_airlines.dat", header=None, na_values="\\N",
                 names=["id", "nom", "alias", "iata", "icao", "callsign", "pays", "active"])

# ---------- Nettoyage ----------
n_rt = len(rt)
rt = rt.dropna(subset=["src_id", "dst_id"]).astype({"src_id": int, "dst_id": int})
rt = rt[rt.escales == 0]                                           # vols directs uniquement
ap = ap.dropna(subset=["iata"]).set_index("id")
rt = rt[rt.src_id.isin(ap.index) & rt.dst_id.isin(ap.index)]
quality = {"routes_brutes": n_rt, "routes_gardees": len(rt), "aeroports_avec_iata": len(ap)}

# ---------- Graphe ----------
pairs = rt.groupby(["src_id", "dst_id"]).size().reset_index(name="cies")  # nb de compagnies sur la liaison
G = nx.DiGraph()
for r in pairs.itertuples():
    G.add_edge(r.src_id, r.dst_id, weight=r.cies)
U = G.to_undirected()
lcc = max(nx.connected_components(U), key=len)
H = U.subgraph(lcc).copy()

deg = dict(G.out_degree())
btw = nx.betweenness_centrality(H, k=600, seed=1, normalized=True)   # approximation sur 600 sources
lab = lambda i: f"{ap.loc[i, 'iata']} · {ap.loc[i, 'ville']}"
top_deg = sorted(deg, key=deg.get, reverse=True)[:15]
top_btw = sorted(btw, key=btw.get, reverse=True)[:15]
hubs = [{"code": ap.loc[i, "iata"], "ville": ap.loc[i, "ville"], "pays": ap.loc[i, "pays"],
         "destinations": deg[i], "intermediarite": round(btw.get(i, 0), 4)} for i in top_deg]
brokers = [{"code": ap.loc[i, "iata"], "ville": ap.loc[i, "ville"], "pays": ap.loc[i, "pays"],
            "destinations": deg.get(i, 0), "intermediarite": round(btw[i], 4)} for i in top_btw]

# Distribution des degrés (réseau "sans échelle" : quelques hubs, beaucoup de petits aéroports)
dv = np.array(list(dict(U.degree()).values()))
dist = {"1": int((dv == 1).sum()), "2-5": int(((dv >= 2) & (dv <= 5)).sum()), "6-20": int(((dv >= 6) & (dv <= 20)).sum()),
        "21-50": int(((dv >= 21) & (dv <= 50)).sum()), "51-100": int(((dv >= 51) & (dv <= 100)).sum()), "> 100": int((dv > 100).sum())}

# ---------- Communautés ----------
comms = nx.community.louvain_communities(H, seed=2, resolution=1.0)
comms = sorted(comms, key=len, reverse=True)
mod = nx.community.modularity(H, comms)
cinfo = []
for c in comms[:8]:
    sub = ap.loc[list(c)]
    top = sorted(c, key=lambda i: deg.get(i, 0), reverse=True)[:4]
    cinfo.append({"taille": len(c), "pays": sub.pays.value_counts().head(3).index.tolist(), "hubs": [ap.loc[i, "iata"] for i in top]})
cid = {n: k for k, c in enumerate(comms[:8]) for n in c}

# ---------- Marseille ----------
mrs = ap.index[ap.iata == "MRS"][0]
dests = [ap.loc[d, "pays"] for d in G.successors(mrs)]
sp = nx.single_source_shortest_path_length(U, mrs)
reach = pd.Series(sp).value_counts().sort_index()
paths = {}
for code in ["SYD", "YUL", "NRT", "GIG", "ALG", "JNB"]:
    t = ap.index[ap.iata == code][0]
    paths[code] = [lab(i) for i in nx.shortest_path(U, mrs, t)]
marseille = {"destinations": G.out_degree(mrs), "pays": len(set(dests)),
             "top_pays": pd.Series(dests).value_counts().head(8).to_dict(),
             "accessibles": {f"{int(k)} vol(s)": int(v) for k, v in reach.items() if k > 0},
             "chemins": paths}

# ---------- Compagnies ----------
cie = rt.groupby("cie").agg(routes=("src", "size"), aeroports=("src_id", "nunique")).sort_values("routes", ascending=False).head(12)
names = al.dropna(subset=["iata"]).drop_duplicates("iata").set_index("iata").nom
cie["nom"] = [names.get(c, c) for c in cie.index]

# ---------- Données pour la carte (aéroports du composant principal + routes des 60 premiers hubs) ----------
pts = [[round(ap.loc[i, "lon"], 2), round(ap.loc[i, "lat"], 2), int(U.degree(i)), cid.get(i, -1)] for i in H.nodes]
big = set(sorted(H.nodes, key=lambda i: U.degree(i), reverse=True)[:60])
edges = [[round(ap.loc[a, "lon"], 1), round(ap.loc[a, "lat"], 1), round(ap.loc[b, "lon"], 1), round(ap.loc[b, "lat"], 1)]
         for a, b in H.edges if a in big and b in big]

out = {"kpi": {"aeroports": H.number_of_nodes(), "liaisons": G.number_of_edges(), "pays": int(ap.loc[list(H.nodes)].pays.nunique()),
               "diametre_moy": round(nx.average_shortest_path_length(H.subgraph(sorted(H.nodes, key=lambda i: U.degree(i), reverse=True)[:800])), 2),
               "modularite": round(mod, 3), "communautes": len(comms), "part_1_route": round((dv == 1).mean() * 100, 1)},
       "quality": quality, "hubs": hubs, "brokers": brokers, "dist": dist, "communautes": cinfo, "marseille": marseille,
       "cies": [{"code": k, "nom": r.nom, "routes": int(r.routes), "aeroports": int(r.aeroports)} for k, r in cie.iterrows()],
       "map": {"pts": pts, "edges": edges}}
json.dump(out, open(ROOT / "results/p_network.json", "w"), ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(out["kpi"], quality); print(pd.DataFrame(hubs).head(8)); print(pd.DataFrame(brokers).head(8)); print(cinfo); print(marseille)
