"""
Construit le tableau de bord Excel des ventes Superstore (100 % formules Excel, filtres par listes déroulantes).
Feuilles : Dashboard | Analyse régions | Données | Méthode
"""
import pathlib
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.chart import LineChart, BarChart, DoughnutChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule
from openpyxl.utils import get_column_letter as L

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/superstore.csv", encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")
cols = ["Order ID", "Order Date", "Ship Date", "Customer Segment", "Region", "Product Category",
        "Product Sub-Category", "Ship Mode", "Order Priority", "Order Quantity", "Sales", "Discount",
        "Profit", "Shipping Cost"]
df = df[cols].sort_values("Order Date")
FR = ["N° commande", "Date commande", "Date expédition", "Segment", "Région", "Catégorie",
      "Sous-catégorie", "Mode d'expédition", "Priorité", "Quantité", "Ventes", "Remise", "Profit",
      "Frais de port", "Année", "Mois", "Délai (j)"]

# Palette sombre / pastel
BG, CARD, INK, MUTE = "1B1B2F", "26263F", "F2F0FF", "A9A6C9"
PASTELS = ["B8A9FF", "8FD3FF", "FFB5C8", "A8E6CF", "FFD59E", "C3B1E1"]
F = lambda **k: Font(name="Arial", **k)
fill = lambda c: PatternFill("solid", fgColor=c)
thin = Side(style="thin", color="3A3A5C")

wb = Workbook()
dash = wb.active; dash.title = "Dashboard"
reg = wb.create_sheet("Analyse régions")
data = wb.create_sheet("Données")
meth = wb.create_sheet("Méthode")

# ---------------- Données ----------------
data.append(FR)
for i, r in enumerate(df.itertuples(index=False), start=2):
    data.append(list(r) + [f"=YEAR(B{i})", f"=MONTH(B{i})", f"=C{i}-B{i}"])
N = len(df) + 1
for c in range(1, 18):
    data.cell(1, c).font = F(bold=True, color="FFFFFF")
for row in data.iter_rows(min_row=2, max_row=N):
    row[1].number_format = row[2].number_format = "dd/mm/yyyy"
    row[10].number_format = row[12].number_format = row[13].number_format = '#,##0.00 "$"'
    row[11].number_format = "0%"
tab = Table(displayName="Ventes", ref=f"A1:Q{N}")
tab.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
data.add_table(tab)
for c, w in zip(range(1, 18), [11, 13, 13, 15, 22, 16, 30, 16, 13, 9, 12, 8, 12, 11, 8, 7, 9]):
    data.column_dimensions[L(c)].width = w
data.freeze_panes = "A2"

R = lambda col: f"Données!${col}$2:${col}${N}"   # plages absolues
SALES, PROFIT, YEAR, REGION, CAT, SUB, SEG, MONTH, DELAY, QTY, SHIP = (
    R("K"), R("M"), R("O"), R("E"), R("F"), R("G"), R("D"), R("P"), R("Q"), R("J"), R("H"))

# ---------------- Dashboard ----------------
ws = dash
ws.sheet_view.showGridLines = False
for c in range(1, 30):
    ws.column_dimensions[L(c)].width = 13
for r in range(1, 120):
    for c in range(1, 30):
        ws.cell(r, c).fill = fill(BG)
ws.column_dimensions["A"].width = 3
ws.column_dimensions["B"].width = 17
ws["B2"] = "Tableau de bord des ventes — Superstore Canada"
ws["B2"].font = F(size=20, bold=True, color=INK)
ws["B3"] = "Toutes les valeurs sont calculées par formules à partir de la feuille Données. Changez les filtres en jaune."
ws["B3"].font = F(size=10, color=MUTE, italic=True)

# Filtres
years = sorted(df["Order Date"].dt.year.unique())
regions = sorted(df.Region.unique())
lists = wb.create_sheet("Listes"); lists.sheet_state = "hidden"
lists["A1"] = "Toutes"; lists["B1"] = "Toutes"
for i, y in enumerate(years, 2): lists.cell(i, 1, int(y))
for i, rg in enumerate(regions, 2): lists.cell(i, 2, rg)
ws["B5"], ws["D5"] = "Année", "Région"
for c in ("B5", "D5"): ws[c].font = F(bold=True, color=MUTE)
ws["C5"], ws["E5"] = "Toutes", "Toutes"
for c in ("C5", "E5"):
    ws[c].fill = fill("FFE9A8"); ws[c].font = F(bold=True, color="1B1B2F")
    ws[c].alignment = Alignment(horizontal="center")
ws.merge_cells("E5:F5")
dv1 = DataValidation(type="list", formula1=f"=Listes!$A$1:$A${len(years)+1}", allow_blank=False)
dv2 = DataValidation(type="list", formula1=f"=Listes!$B$1:$B${len(regions)+1}", allow_blank=False)
ws.add_data_validation(dv1); ws.add_data_validation(dv2); dv1.add("C5"); dv2.add("E5")
ws["H5"] = '=IF(AND(C5="Toutes",E5="Toutes"),"Vue : ensemble du périmètre","Vue filtrée : "&C5&" · "&E5)'
ws["H5"].font = F(color=PASTELS[1], italic=True)

# Critères réutilisables (cellules cachées) : "<>" = tout sélectionner
ws["Z1"] = '=IF($C$5="Toutes","<>",$C$5)'
ws["Z2"] = '=IF($E$5="Toutes","<>",$E$5)'
ws["Z3"] = '=IF($C$5="Toutes","<>",$C$5-1)'
ws.column_dimensions["Z"].hidden = True
FY, FR_ = "$Z$1", "$Z$2"
flt = f"{YEAR},{FY},{REGION},{FR_}"

# Cartes KPI
cards = [
    ("Chiffre d'affaires", f"=SUMIFS({SALES},{flt})", '#,##0 "$"'),
    ("Profit", f"=SUMIFS({PROFIT},{flt})", '#,##0 "$"'),
    ("Marge nette", "=IFERROR(E9/C9,0)", "0.0%"),
    ("Lignes de commande", f"=COUNTIFS({flt})", "#,##0"),
    ("Panier moyen", "=IFERROR(C9/I9,0)", '#,##0 "$"'),
    ("Délai d'expédition", f"=IFERROR(AVERAGEIFS({DELAY},{flt}),0)", '0.0 "j"'),
]
for k, (lab, f, fmt) in enumerate(cards):
    c = 3 + 2 * k
    ws.merge_cells(start_row=8, start_column=c - 1, end_row=8, end_column=c)
    ws.merge_cells(start_row=9, start_column=c - 1, end_row=9, end_column=c)
    ws.merge_cells(start_row=10, start_column=c - 1, end_row=10, end_column=c)
    for r in (8, 9, 10):
        for cc in (c - 1, c):
            ws.cell(r, cc).fill = fill(CARD)
    t = ws.cell(8, c - 1, lab); t.font = F(size=10, color=MUTE); t.alignment = Alignment(horizontal="center")
    v = ws.cell(9, c - 1, f); v.font = F(size=18, bold=True, color=PASTELS[k]); v.number_format = fmt
    v.alignment = Alignment(horizontal="center")
# Les formules de marge/panier pointent sur la cellule ancre (B9, D9, ...) : on corrige les références
ws["F9"] = "=IFERROR(D9/B9,0)"; ws["J9"] = "=IFERROR(B9/H9,0)"
# Variation vs année précédente (si une année est choisie)
for col, rng in (("B", SALES), ("D", PROFIT)):
    ws[f"{col}10"] = (f'=IF($C$5="Toutes","Choisir une année pour N-1",'
                      f'IFERROR(TEXT({col}9/SUMIFS({rng},{YEAR},$Z$3,{REGION},{FR_})-1,"+0.0%;-0.0%")&" vs N-1","—"))')
    ws[f"{col}10"].font = F(size=9, color=PASTELS[3]); ws[f"{col}10"].alignment = Alignment(horizontal="center")

def header(ws, r, c, labels, color=PASTELS[0]):
    for i, h in enumerate(labels):
        x = ws.cell(r, c + i, h); x.font = F(bold=True, color="1B1B2F"); x.fill = fill(color)
        x.alignment = Alignment(horizontal="center")

def body(cell, fmt=None, color=INK):
    cell.font = F(color=color); cell.fill = fill(CARD); cell.border = Border(bottom=thin)
    if fmt: cell.number_format = fmt

# Tableau 1 : CA mensuel
r0 = 13
ws.cell(r0 - 1, 2, "CA et profit par mois").font = F(bold=True, size=12, color=INK)
header(ws, r0, 2, ["Mois", "CA", "Profit", "Marge"])
mois = ["Janv.", "Févr.", "Mars", "Avr.", "Mai", "Juin", "Juil.", "Août", "Sept.", "Oct.", "Nov.", "Déc."]
for i, m in enumerate(mois, 1):
    r = r0 + i
    body(ws.cell(r, 2, m))
    body(ws.cell(r, 3, f"=SUMIFS({SALES},{flt},{MONTH},{i})"), '#,##0 "$"')
    body(ws.cell(r, 4, f"=SUMIFS({PROFIT},{flt},{MONTH},{i})"), '#,##0 "$"')
    body(ws.cell(r, 5, f"=IFERROR(D{r}/C{r},0)"), "0.0%")
ch = LineChart(); ch.title = "CA mensuel"; ch.height, ch.width = 7.5, 15
ch.add_data(Reference(ws, min_col=3, min_row=r0, max_row=r0 + 12), titles_from_data=True)
ch.set_categories(Reference(ws, min_col=2, min_row=r0 + 1, max_row=r0 + 12))
ch.series[0].graphicalProperties.line.solidFill = PASTELS[0]; ch.series[0].graphicalProperties.line.width = 32000
ch.series[0].smooth = True; ch.legend = None
ws.add_chart(ch, "G12")

# Tableau 2 : catégories
r1 = 29
ws.cell(r1 - 1, 2, "Par catégorie de produits").font = F(bold=True, size=12, color=INK)
header(ws, r1, 2, ["Catégorie", "CA", "Profit", "Marge", "Part CA"], PASTELS[1])
cats = sorted(df["Product Category"].unique())
for i, cname in enumerate(cats, 1):
    r = r1 + i
    body(ws.cell(r, 2, cname))
    body(ws.cell(r, 3, f'=SUMIFS({SALES},{flt},{CAT},B{r})'), '#,##0 "$"')
    body(ws.cell(r, 4, f'=SUMIFS({PROFIT},{flt},{CAT},B{r})'), '#,##0 "$"')
    body(ws.cell(r, 5, f"=IFERROR(D{r}/C{r},0)"), "0.0%")
    body(ws.cell(r, 6, f"=IFERROR(C{r}/SUM($C${r1+1}:$C${r1+len(cats)}),0)"), "0.0%")
dn = DoughnutChart(); dn.title = "Répartition du CA"; dn.height, dn.width = 7.5, 9
dn.add_data(Reference(ws, min_col=3, min_row=r1, max_row=r1 + len(cats)), titles_from_data=True)
dn.set_categories(Reference(ws, min_col=2, min_row=r1 + 1, max_row=r1 + len(cats)))
from openpyxl.chart.series import DataPoint
for i in range(len(cats)):
    pt = DataPoint(idx=i); pt.graphicalProperties.solidFill = PASTELS[i]; dn.series[0].dPt.append(pt)
dn.dataLabels = DataLabelList(); dn.dataLabels.showPercent = True
for a in ("showVal","showCatName","showSerName","showLegendKey"): setattr(dn.dataLabels, a, False)
ws.add_chart(dn, "H28")

# Tableau 3 : segments clients
r2 = r1 + len(cats) + 3
ws.cell(r2 - 1, 2, "Par segment client").font = F(bold=True, size=12, color=INK)
header(ws, r2, 2, ["Segment", "CA", "Profit", "Marge", "Panier moy."], PASTELS[2])
segs = sorted(df["Customer Segment"].unique())
for i, s in enumerate(segs, 1):
    r = r2 + i
    body(ws.cell(r, 2, s))
    body(ws.cell(r, 3, f'=SUMIFS({SALES},{flt},{SEG},B{r})'), '#,##0 "$"')
    body(ws.cell(r, 4, f'=SUMIFS({PROFIT},{flt},{SEG},B{r})'), '#,##0 "$"')
    body(ws.cell(r, 5, f"=IFERROR(D{r}/C{r},0)"), "0.0%")
    body(ws.cell(r, 6, f'=IFERROR(C{r}/COUNTIFS({flt},{SEG},B{r}),0)'), '#,##0 "$"')

# Tableau 4 : sous-catégories classées par profit
r3 = r2 + len(segs) + 3
ws.cell(r3 - 1, 2, "Sous-catégories : qui crée (ou détruit) la marge ?").font = F(bold=True, size=12, color=INK)
header(ws, r3, 2, ["Sous-catégorie", "", "CA", "Profit", "Marge", "Rang profit"], PASTELS[3])
subs = sorted(df["Product Sub-Category"].unique())
for i, s in enumerate(subs, 1):
    r = r3 + i
    ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=3)
    body(ws.cell(r, 2, s)); body(ws.cell(r, 3))
    body(ws.cell(r, 4, f'=SUMIFS({SALES},{flt},{SUB},B{r})'), '#,##0 "$"')
    body(ws.cell(r, 5, f'=SUMIFS({PROFIT},{flt},{SUB},B{r})'), '#,##0 "$";[Red]-#,##0 "$"')
    body(ws.cell(r, 6, f"=IFERROR(E{r}/D{r},0)"), "0.0%")
    body(ws.cell(r, 7, f"=RANK(E{r},$E${r3+1}:$E${r3+len(subs)})"), "0")
ws.cell(r3 - 1, 2).font = F(bold=True, size=12, color=INK)
last = r3 + len(subs)
ws.conditional_formatting.add(f"E{r3+1}:E{last}",
    ColorScaleRule(start_type="min", start_color="FF8FA3", mid_type="num", mid_value=0, mid_color="26263F",
                   end_type="max", end_color="7FD8B0"))
ws.conditional_formatting.add(f"D{r3+1}:D{last}", DataBarRule(start_type="min", end_type="max", color="8FD3FF"))
bc = BarChart(); bc.type = "bar"; bc.title = "Profit par sous-catégorie"; bc.height, bc.width = 11, 15
bc.add_data(Reference(ws, min_col=5, min_row=r3, max_row=last), titles_from_data=True)
bc.set_categories(Reference(ws, min_col=2, min_row=r3 + 1, max_row=last))
bc.series[0].graphicalProperties.solidFill = PASTELS[0]; bc.legend = None
ws.add_chart(bc, f"I{r3-1}")

# Tableau 5 : mode d'expédition
r4 = last + 3
ws.cell(r4 - 1, 2, "Logistique : mode d'expédition").font = F(bold=True, size=12, color=INK)
header(ws, r4, 2, ["Mode", "Lignes", "Délai moy. (j)", "CA", "Profit"], PASTELS[4])
for i, s in enumerate(sorted(df["Ship Mode"].unique()), 1):
    r = r4 + i
    body(ws.cell(r, 2, s))
    body(ws.cell(r, 3, f'=COUNTIFS({flt},{SHIP},B{r})'), "#,##0")
    body(ws.cell(r, 4, f'=IFERROR(AVERAGEIFS({DELAY},{flt},{SHIP},B{r}),0)'), "0.0")
    body(ws.cell(r, 5, f'=SUMIFS({SALES},{flt},{SHIP},B{r})'), '#,##0 "$"')
    body(ws.cell(r, 6, f'=SUMIFS({PROFIT},{flt},{SHIP},B{r})'), '#,##0 "$"')
ws.freeze_panes = "A7"
ws.sheet_properties.pageSetUpPr.fitToPage = True
ws.page_setup.fitToWidth = 1
ws.page_setup.fitToHeight = 0
ws.print_area = "A1:P75"
ws.page_setup.orientation = "landscape"

# ---------------- Analyse régions ----------------
w = reg; w.sheet_view.showGridLines = False
w["B2"] = "CA par région et par année (matrice) — formules SUMIFS"; w["B2"].font = F(size=14, bold=True)
w["B3"] = "La couleur montre l'intensité ; la colonne Évolution compare 2012 à 2009."
w["B3"].font = F(size=9, italic=True, color="666666")
hdr = ["Région"] + [str(y) for y in years] + ["Total", "Évolution 2009→2012", "Profit total", "Marge"]
for i, h in enumerate(hdr):
    x = w.cell(5, 2 + i, h); x.font = F(bold=True, color="FFFFFF"); x.fill = fill("3B3B63")
    x.alignment = Alignment(horizontal="center", wrap_text=True)
for i, rg in enumerate(regions, 6):
    w.cell(i, 2, rg).font = F()
    for j, y in enumerate(years):
        c = w.cell(i, 3 + j, f"=SUMIFS({SALES},{REGION},$B{i},{YEAR},{int(y)})"); c.number_format = '#,##0'; c.font = F()
    tc = 3 + len(years)
    w.cell(i, tc, f"=SUM({L(3)}{i}:{L(tc-1)}{i})").number_format = '#,##0'
    w.cell(i, tc + 1, f"=IFERROR({L(tc-1)}{i}/{L(3)}{i}-1,0)").number_format = "+0.0%;-0.0%"
    w.cell(i, tc + 2, f"=SUMIFS({PROFIT},{REGION},$B{i})").number_format = '#,##0'
    w.cell(i, tc + 3, f"=IFERROR({L(tc+2)}{i}/{L(tc)}{i},0)").number_format = "0.0%"
    for c in range(tc, tc + 4): w.cell(i, c).font = F()
lr = 5 + len(regions)
tr = lr + 1
w.cell(tr, 2, "Total").font = F(bold=True)
for c in range(3, 3 + len(years) + 1):
    x = w.cell(tr, c, f"=SUM({L(c)}6:{L(c)}{lr})"); x.number_format = '#,##0'; x.font = F(bold=True)
w.conditional_formatting.add(f"C6:{L(2+len(years))}{lr}",
    ColorScaleRule(start_type="min", start_color="F3F0FF", end_type="max", end_color="8E7CFF"))
w.column_dimensions["B"].width = 26
for c in range(3, 12): w.column_dimensions[L(c)].width = 14
w.row_dimensions[5].height = 32

# ---------------- Méthode ----------------
notes = [
    ("Tableau de bord des ventes Superstore", True),
    ("Source : jeu de données public « Superstore Sales » (8 399 lignes de commande, Canada, 2009-2012).", False),
    ("", False),
    ("Comment l'utiliser", True),
    ("1. Dans l'onglet Dashboard, choisissez une année et une région dans les cellules jaunes.", False),
    ("2. Toutes les cartes, tableaux et graphiques se recalculent instantanément.", False),
    ("3. L'onglet Analyse régions croise région × année ; l'onglet Données contient la table source.", False),
    ("", False),
    ("Comment c'est construit", True),
    ("• La table source est un tableau Excel (« Ventes ») avec 3 colonnes calculées : Année, Mois, Délai.", False),
    ("• Chaque indicateur utilise SUMIFS / COUNTIFS / AVERAGEIFS avec les filtres en critères.", False),
    ("• Astuce « Toutes » : le critère devient \"<>\" (= toute cellule non vide), donc aucun filtre.", False),
    ("• Variation N-1 : même formule avec l'année choisie moins 1.", False),
    ("• Mise en forme conditionnelle : échelle rouge→vert sur le profit, barres de données sur le CA.", False),
    ("", False),
    ("Définitions des KPI", True),
    ("• Marge nette = Profit / CA.   • Panier moyen = CA / nombre de lignes de commande.", False),
    ("• Délai d'expédition = Date d'expédition − Date de commande, en jours.", False),
]
for i, (t, b) in enumerate(notes, 2):
    meth.cell(i, 2, t).font = F(bold=b, size=13 if b else 11)
meth.column_dimensions["B"].width = 110
meth.sheet_view.showGridLines = False

out = ROOT / "excel/Dashboard_Ventes_Superstore.xlsx"
wb.save(out)
print("saved", out, N)
