"""Compara o results.csv com o gabarito embutido no nome do arquivo
(real_* = real, fake_<gerador>_* = IA), usando limiar 0,5."""

import csv
import sys
from collections import defaultdict

from sklearn.metrics import average_precision_score, f1_score

LIMIAR = 0.5
linhas = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
colunas = [c for c in linhas[0] if c != "Imagem"]


def grupo(nome):
    return "real" if nome.startswith("real_") else nome.rsplit("_", 1)[0].removeprefix("fake_")


y = [0 if grupo(l["Imagem"]) == "real" else 1 for l in linhas]
grupos = sorted({grupo(l["Imagem"]) for l in linhas}, key=lambda g: (g != "real", g))
print(f"{len(linhas)} imagens: {y.count(0)} reais, {y.count(1)} de IA | limiar {LIMIAR}\n")

cab = ["Score", "Acurácia", "F1", "AP"] + [f"acerto {g}" for g in grupos]
print(" | ".join(cab))
for c in colunas:
    s = [float(l[c]) for l in linhas]
    pred = [int(v >= LIMIAR) for v in s]
    acc = sum(p == t for p, t in zip(pred, y)) / len(y)
    por_grupo = defaultdict(list)
    for l, p, t in zip(linhas, pred, y):
        por_grupo[grupo(l["Imagem"])].append(p == t)
    cel = [c, f"{acc:.0%}", f"{f1_score(y, pred):.2f}", f"{average_precision_score(y, s):.2f}"]
    cel += [f"{sum(por_grupo[g])}/{len(por_grupo[g])}" for g in grupos]
    print(" | ".join(cel))
