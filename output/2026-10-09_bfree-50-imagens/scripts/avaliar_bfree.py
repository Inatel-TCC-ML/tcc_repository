"""Resume o resultado do B-Free gerado por rodar_bfree.py.

Rodar da raiz do projeto:
    python output/2026-10-09_bfree-50-imagens/scripts/avaliar_bfree.py <resultados_bfree.csv>
"""

import csv
import sys
from collections import defaultdict
from statistics import mean

from sklearn.metrics import average_precision_score, roc_auc_score

LIMIAR = 0.5


def metricas(linhas):
    y = [1 if l["classe"] == "ia" else 0 for l in linhas]
    s = [float(l["score"]) for l in linhas]
    acertos = sum((sc >= LIMIAR) == bool(rot) for sc, rot in zip(s, y))
    return acertos, len(linhas), average_precision_score(y, s), roc_auc_score(y, s)


def linha_grupo(nome, linhas, real=False):
    scores = [float(l["score"]) for l in linhas]
    acertos = sum((s < LIMIAR) if real else (s >= LIMIAR) for s in scores)
    return f"| {nome} | {acertos}/{len(scores)} | {mean(scores):.3f} | {min(scores):.3f} | {max(scores):.3f} |"


def main():
    with open(sys.argv[1], encoding="utf-8") as arquivo:
        dados = list(csv.DictReader(arquivo))

    original = [l for l in dados if l["variante"] == "original"]
    reais = [l for l in original if l["classe"] == "real"]
    jpeg95 = [l for l in dados if l["variante"] == "jpeg95"]
    firefly = [l for l in original if l["gerador"] == "Firefly_Image3"]

    print(f"Limiar: {LIMIAR} (score >= {LIMIAR} = IA)\n")

    print("## 1. Formato original (50 imagens)\n")
    print("| Grupo | Acertos | Score médio | Mín | Máx |")
    print("|---|---|---|---|---|")
    print(linha_grupo("Reais (LAION, JPEG)", reais, real=True))
    por_gerador = defaultdict(list)
    for l in original:
        if l["classe"] == "ia":
            por_gerador[(l["arquitetura"], l["gerador"], l["formato_testado"])].append(l)
    for (arq, gerador, formato), linhas in por_gerador.items():
        print(linha_grupo(f"{gerador} ({arq}, {formato})", linhas))
    acertos, total, ap, auc = metricas(original)
    print(f"\nAcurácia: {acertos}/{total} = {acertos / total:.0%} | AP: {ap:.3f} | AUC: {auc:.3f}\n")

    print("## 2. Tudo em JPEG (reais + IA convertidas para JPEG 95 + Firefly 3 nativo)\n")
    cenario_jpeg = reais + jpeg95 + firefly
    print("| Grupo | Acertos | Score médio | Mín | Máx |")
    print("|---|---|---|---|---|")
    print(linha_grupo("Reais (LAION, JPEG)", reais, real=True))
    por_gerador_jpeg = defaultdict(list)
    for l in jpeg95 + firefly:
        por_gerador_jpeg[(l["arquitetura"], l["gerador"])].append(l)
    for (arq, gerador), linhas in por_gerador_jpeg.items():
        print(linha_grupo(f"{gerador} ({arq})", linhas))
    acertos, total, ap, auc = metricas(cenario_jpeg)
    print(f"\nAcurácia: {acertos}/{total} = {acertos / total:.0%} | AP: {ap:.3f} | AUC: {auc:.3f}\n")

    print("## 3. Mesmas imagens de IA: PNG original x JPEG 95\n")
    print("| Gerador | Acertos PNG | Acertos JPEG 95 | Score médio PNG | Score médio JPEG 95 |")
    print("|---|---|---|---|---|")
    png = {l["arquivo"].rsplit(".", 1)[0]: l for l in original if l["formato_testado"] == "PNG"}
    pares = defaultdict(list)
    for l in jpeg95:
        base = l["arquivo"].rsplit(".", 1)[0]
        pares[l["gerador"]].append((float(png[base]["score"]), float(l["score"])))
    for gerador, valores in pares.items():
        a_png = sum(p >= LIMIAR for p, _ in valores)
        a_jpg = sum(j >= LIMIAR for _, j in valores)
        print(f"| {gerador} | {a_png}/{len(valores)} | {a_jpg}/{len(valores)} | "
              f"{mean(p for p, _ in valores):.3f} | {mean(j for _, j in valores):.3f} |")

    tempos = [float(l["segundos"]) for l in dados]
    print(f"\nTempo por imagem (CPU): média {mean(tempos):.1f}s, "
          f"mín {min(tempos):.1f}s, máx {max(tempos):.1f}s, total {sum(tempos) / 60:.1f} min")


if __name__ == "__main__":
    main()
