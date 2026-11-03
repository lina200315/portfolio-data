"""Exécute chaque requête de analyses_chinook.sql sur la base SQLite et enregistre les résultats."""
import json, pathlib, re, sqlite3

ROOT = pathlib.Path(__file__).resolve().parents[1]
sql = (ROOT / "sql/analyses_chinook.sql").read_text(encoding="utf-8")
con = sqlite3.connect(ROOT / "data/chinook.sqlite")
blocks = re.split(r"\n(?=-- @name:)", sql)
out = []
for b in blocks:
    m = re.search(r"-- @name: (\S+)\n-- @title: (.+)\n", b)
    if not m:
        continue
    query = b[m.end():].strip()
    cur = con.execute(query)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    out.append({"id": m.group(1), "title": m.group(2), "sql": query, "cols": cols, "rows": rows})
    print(f"{m.group(1)}: {len(rows)} lignes | {cols}\n   {rows[:3]}")
print("SQLite", sqlite3.sqlite_version)
json.dump(out, open(ROOT / "results/sql_chinook.json", "w"), ensure_ascii=False, indent=1)
