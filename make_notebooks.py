"""Transforme chaque script Python d'analyse en notebook Jupyter commenté, puis l'exécute pour enregistrer les sorties."""
import pathlib, re, sys
import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "notebooks"; OUT.mkdir(exist_ok=True)

def to_nb(path):
    src = path.read_text(encoding="utf-8")
    m = re.match(r'"""(.*?)"""\s*', src, re.S)
    doc, body = (m.group(1).strip(), src[m.end():]) if m else ("", src)
    lines = doc.splitlines()
    title = lines[0].replace("Projet : ", "") if lines else path.stem
    md = [f"# {title}", ""] + [l.strip() for l in lines[1:]] + ["", f"*Script source : `{path.relative_to(ROOT)}`. Ce notebook est généré à partir du script et exécuté tel quel.*"]
    cells = [nbformat.v4.new_markdown_cell("\n".join(md))]
    body = body.replace("pathlib.Path(__file__).resolve().parents[1]", "pathlib.Path.cwd().parent")
    parts = re.split(r"\n(?=# -{5,} .*? -{5,}\n)", body)
    for part in parts:
        h = re.match(r"# -{5,} (.*?) -{5,}\n", part)
        if h:
            cells.append(nbformat.v4.new_markdown_cell("## " + h.group(1)))
            part = part[h.end():]
        if part.strip():
            cells.append(nbformat.v4.new_code_cell(part.strip("\n")))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb

files = sorted((ROOT / "python").glob("*.py")) if len(sys.argv) == 1 else [ROOT / f for f in sys.argv[1:]]
for f in files:
    nb = to_nb(f)
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(OUT)}}).execute()
    nbformat.write(nb, OUT / (f.stem + ".ipynb"))
    print("ok", f.name, sum(len(c.get("outputs", [])) for c in nb.cells), "sorties")
