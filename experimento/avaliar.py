"""Gera as tabelas do experimento oficial a partir do CSV do rodar.py.

Uso (da raiz do projeto):
    python experimento/avaliar.py output/AAAA-MM-DD_experimento-oficial/resultados.csv

Use o mesmo --por-grupo do rodar.py (padrão 50). Gera, na pasta do CSV:
- avaliacao.md: as tabelas prontas para ler e copiar;
- metricas.csv: todas as métricas, uma linha por condição, score e grupo.

Métricas, calculadas em cada grupo (IA do gerador + reais pareadas):
- AP (average precision): não depende de limiar;
- Acc: acerto com limiar 0,5 (score >= 0,5 = IA), com Acc real e Acc IA.
mAP e mAcc são as médias nos 22 grupos do Community Forensics, como no artigo
do dataset (Park e Owens, CVPR 2025).
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from sklearn.metrics import average_precision_score

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.aggregation import (  # noqa: E402
    MaxScoreAggregator,
    MeanScoreAggregator,
    MedianScoreAggregator,
    PercentileScoreAggregator,
)

LIMIAR = 0.5
DETECTORES = {
    "cnndetection": "CNNDetection",
    "univfd": "UnivFD",
    "npr": "NPR",
    "bfree": "B-Free",
}
AGREGADORES = {
    "média": MeanScoreAggregator(),
    "mediana": MedianScoreAggregator(),
    "P75": PercentileScoreAggregator(75),
    "P90": PercentileScoreAggregator(90),
    "máximo": MaxScoreAggregator(),
}
# Resultados publicados no CF-Eval (mAP, mAcc), README do dataset no Hugging Face
PUBLICADOS = {"CNNDetection": (0.543, 0.515), "UnivFD": (0.583, 0.540)}
# Grupos em que reais e IA já estão no mesmo tipo de formato (ambos sem perda)
SEM_PERDA = {"ProGAN", "SD2.1", "Hourglass_ffhq", "DeciDiffusionV2", "GALIP", "LCM_lora_sdv15",
             "LCM_lora_sdxl", "LCM_lora_ssd1b", "kandinsky_2_2", "kvikontent_midjourney_v6",
             "stable_cascade"}


def metricas(rotulos, scores):
    acertos = [(s >= LIMIAR) == bool(r) for r, s in zip(rotulos, scores)]
    reais = [a for a, r in zip(acertos, rotulos) if not r]
    ias = [a for a, r in zip(acertos, rotulos) if r]
    return {
        "AP": average_precision_score(rotulos, scores),
        "Acc": fmean(acertos),
        "Acc_real": fmean(reais),
        "Acc_IA": fmean(ias),
        "n": len(rotulos),
    }


def tabela(cabecalho, linhas):
    saida = ["| " + " | ".join(cabecalho) + " |", "|" + "---|" * len(cabecalho)]
    saida += ["| " + " | ".join(str(c) for c in linha) + " |" for linha in linhas]
    return "\n".join(saida)


def fmt(valor):
    return "-" if valor is None else f"{valor:.3f}".replace(".", ",")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("resultados", nargs="+", help="um ou mais CSVs do rodar.py (são juntados)")
    parser.add_argument("--por-grupo", type=int, default=50)
    args = parser.parse_args()

    scores = defaultdict(dict)  # (id, condição) -> {detector: score}
    for caminho in args.resultados:
        with open(caminho, encoding="utf-8") as arquivo:
            for l in csv.DictReader(arquivo):
                scores[(l["id"], l["condicao"])][DETECTORES[l["detector"]]] = float(l["score"])

    with open(RAIZ / "experimento" / "amostra.csv", encoding="utf-8") as arquivo:
        amostra = [l for l in csv.DictReader(arquivo) if int(l["ordem"]) < args.por_grupo]

    condicoes = sorted({c for _, c in scores}, key=lambda c: c != "original")
    detectores = [d for d in DETECTORES.values() if any(d in s for s in scores.values())]
    completos = len(detectores) == len(DETECTORES)
    nomes_scores = detectores + (list(AGREGADORES) if completos else [])

    grupos = defaultdict(list)
    visto_por, arquitetura, base = {}, {}, {}
    for l in amostra:
        grupos[l["grupo"]].append(l)
        if l["classe"] == "ia":
            visto_por[l["grupo"]] = l["visto_por"]
            arquitetura[l["grupo"]] = l["arquitetura"]
            base[l["grupo"]] = l["base"]
    grupos_cf = sorted(g for g in grupos if base[g] == "CF-Eval")
    ordem_grupos = ["ProGAN", "SD2.1"] + grupos_cf

    # resultado[condição][score][grupo] = métricas
    resultado = defaultdict(lambda: defaultdict(dict))
    faltando = 0
    for condicao in condicoes:
        for grupo in ordem_grupos:
            valores = defaultdict(list)
            rotulos = []
            for l in grupos[grupo]:
                s = scores.get((l["id"], condicao), {})
                if not all(d in s for d in detectores):
                    faltando += 1
                    continue
                rotulos.append(1 if l["classe"] == "ia" else 0)
                for d in detectores:
                    valores[d].append(s[d])
                if completos:
                    lista = [s[d] for d in detectores]
                    for nome, agregador in AGREGADORES.items():
                        valores[nome].append(agregador.aggregate(lista))
            if len(set(rotulos)) < 2:
                continue
            for nome in nomes_scores:
                resultado[condicao][nome][grupo] = metricas(rotulos, valores[nome])

    def media(condicao, nome, chave, lista_grupos):
        vals = [resultado[condicao][nome][g][chave] for g in lista_grupos if g in resultado[condicao][nome]]
        return fmean(vals) if vals else None

    md = ["# Avaliação do experimento oficial", ""]
    md.append(f"Arquivos: {', '.join(args.resultados)}. Imagens por classe em cada grupo: até "
              f"{args.por_grupo}. Limiar da acurácia: {LIMIAR}.")
    if faltando:
        md.append(f"\n**Atenção:** {faltando} pares (imagem, condição) sem score de todos os "
                  "detectores ficaram de fora. Rode o rodar.py até o fim.")
    if not completos:
        md.append("\nAs agregações só são calculadas quando os quatro detectores rodaram.")

    md += ["", "## 1. Resumo: mAP e mAcc nos 22 grupos do Community Forensics", ""]
    linhas = []
    for nome in nomes_scores:
        linha = [nome]
        for condicao in condicoes:
            linha += [fmt(media(condicao, nome, "AP", grupos_cf)), fmt(media(condicao, nome, "Acc", grupos_cf))]
        publicado = PUBLICADOS.get(nome)
        linha.append(f"{fmt(publicado[0])} / {fmt(publicado[1])}" if publicado else "-")
        linhas.append(linha)
    cab = ["Score"] + [f"{m} ({c})" for c in condicoes for m in ("mAP", "mAcc")] + ["Publicado (mAP / mAcc)"]
    md += [tabela(cab, linhas), ""]

    md += ["## 2. Gerador visto no treino x geradores não vistos (AP)", "",
           "Visto: ProGAN para CNNDetection, UnivFD e NPR; SD 2.1 para o B-Free. "
           "Não vistos: média dos 22 grupos do Community Forensics.", ""]
    linhas = []
    for condicao in condicoes:
        for d in detectores:
            visto = "SD2.1" if d == "B-Free" else "ProGAN"
            ap_visto = resultado[condicao][d].get(visto, {}).get("AP")
            ap_nao = media(condicao, d, "AP", grupos_cf)
            queda = None if ap_visto is None or ap_nao is None else ap_visto - ap_nao
            linhas.append([condicao, d, visto, fmt(ap_visto), fmt(ap_nao), fmt(queda)])
    md += [tabela(["Condição", "Detector", "Visto", "AP visto", "mAP não vistos", "Diferença"], linhas), ""]

    md += ["## 3. mAP por arquitetura do gerador (grupos do Community Forensics)", ""]
    arquiteturas = sorted({arquitetura[g] for g in grupos_cf})
    for condicao in condicoes:
        linhas = []
        for nome in nomes_scores:
            linhas.append([nome] + [fmt(media(condicao, nome, "AP", [g for g in grupos_cf if arquitetura[g] == a]))
                                    for a in arquiteturas])
        cab = ["Score"] + [f"{a} ({sum(arquitetura[g] == a for g in grupos_cf)})" for a in arquiteturas]
        md += [f"Condição: {condicao}", "", tabela(cab, linhas), ""]

    md += ["## 4. Efeito do formato", "",
           "Original: cada imagem como veio da base. Nos grupos 'sem perda', reais e IA já estão "
           "sem compressão (PNG/TIFF); nos demais, pelo menos um dos lados tem compressão com perda "
           "(reais JPEG/WEBP e IA PNG, ou as duas JPEG no Firefly). "
           "JPEG 95: todas recomprimidas igual.", ""]
    sem_perda = [g for g in grupos_cf if g in SEM_PERDA]
    com_diferenca = [g for g in grupos_cf if g not in SEM_PERDA]
    linhas = []
    for nome in nomes_scores:
        linha = [nome]
        for condicao in condicoes:
            linha += [fmt(media(condicao, nome, "AP", sem_perda)), fmt(media(condicao, nome, "AP", com_diferenca))]
        linhas.append(linha)
    cab = ["Score"] + [f"mAP {t} ({c})" for c in condicoes
                       for t in (f"sem perda, {len(sem_perda)}", f"demais, {len(com_diferenca)}")]
    md += [tabela(cab, linhas), ""]

    for condicao in condicoes:
        md += [f"## 5. AP por grupo ({condicao})", ""]
        linhas = [[g + (" (visto)" if visto_por[g] else ""), arquitetura[g]]
                  + [fmt(resultado[condicao][n].get(g, {}).get("AP")) for n in nomes_scores]
                  for g in ordem_grupos]
        md += [tabela(["Grupo", "Arquitetura"] + nomes_scores, linhas), ""]
        md += [f"## 6. Acurácia por grupo ({condicao}): Acc (Acc real / Acc IA)", ""]
        linhas = []
        for g in ordem_grupos:
            linha = [g + (" (visto)" if visto_por[g] else "")]
            for n in nomes_scores:
                m = resultado[condicao][n].get(g)
                linha.append(f"{fmt(m['Acc'])} ({fmt(m['Acc_real'])} / {fmt(m['Acc_IA'])})" if m else "-")
            linhas.append(linha)
        md += [tabela(["Grupo"] + nomes_scores, linhas), ""]

    pasta = Path(args.resultados[0]).parent
    (pasta / "avaliacao.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(pasta / "metricas.csv", "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["condicao", "score", "grupo", "base", "arquitetura", "visto_por",
                           "AP", "Acc", "Acc_real", "Acc_IA", "n"])
        for condicao in condicoes:
            for nome in nomes_scores:
                for g in ordem_grupos:
                    m = resultado[condicao][nome].get(g)
                    if m:
                        escritor.writerow([condicao, nome, g, base[g], arquitetura[g], visto_por[g],
                                           f"{m['AP']:.4f}", f"{m['Acc']:.4f}", f"{m['Acc_real']:.4f}",
                                           f"{m['Acc_IA']:.4f}", m["n"]])
    print(f"Tabelas em {pasta / 'avaliacao.md'} e {pasta / 'metricas.csv'}")


if __name__ == "__main__":
    main()
