"""Roda os detectores nas imagens do experimento oficial.

Uso (da raiz do projeto, depois do baixar_dados.py):
    python experimento/rodar.py --saida output/AAAA-MM-DD_experimento-oficial/resultados.csv

Opções úteis:
    --por-grupo 25            usa só as 25 primeiras imagens de cada classe em cada grupo
    --detectores bfree npr    roda só esses detectores (padrão: os quatro)
    --condicoes original      roda só uma condição (padrão: original e jpeg95)

O CSV de saída tem uma linha por imagem, condição e detector. Cada linha é
gravada assim que termina: se o processo cair, basta rodar o mesmo comando de
novo que ele continua de onde parou. Dá também para dividir o trabalho entre
máquinas (ex.: uma roda --detectores bfree, outra os outros três) e juntar os
CSVs depois.
"""

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.detection import (  # noqa: E402
    BFreeDetector,
    CNNDetectionDetector,
    NPRDetector,
    UnivFDDetector,
)

# As fotos TIFF do RAISE têm metadados EXIF malformados; o Pillow avisa
# "Truncated File Read", mas os pixels carregam inteiros (conferido).
warnings.filterwarnings("ignore", message="Truncated File Read")

DETECTORES = {
    "cnndetection": CNNDetectionDetector,
    "univfd": UnivFDDetector,
    "npr": NPRDetector,
    "bfree": BFreeDetector,
}
EXTENSOES = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "WEBP": ".webp"}
COLUNAS = ["id", "condicao", "detector", "score", "segundos"]


def caminho_imagem(dados, condicao, linha):
    if condicao == "original":
        return dados / "original" / (linha["id"] + EXTENSOES[linha["formato"]])
    return dados / condicao / (linha["id"] + ".jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saida", required=True, help="CSV de resultados (criado ou continuado)")
    parser.add_argument("--por-grupo", type=int, default=50)
    parser.add_argument("--detectores", nargs="+", choices=list(DETECTORES), default=list(DETECTORES))
    parser.add_argument("--condicoes", nargs="+", choices=["original", "jpeg95"], default=["original", "jpeg95"])
    parser.add_argument("--dados", default="datasets/experimento")
    parser.add_argument("--limite", type=int, help="só as N primeiras imagens (para testar)")
    args = parser.parse_args()

    with open(RAIZ / "experimento" / "amostra.csv", encoding="utf-8") as arquivo:
        selecionadas = [l for l in csv.DictReader(arquivo) if int(l["ordem"]) < args.por_grupo]
    imagens = list({l["id"]: l for l in selecionadas}.values())[:args.limite]

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    feitas = set()
    if saida.exists():
        with open(saida, encoding="utf-8") as arquivo:
            feitas = {(l["id"], l["condicao"], l["detector"]) for l in csv.DictReader(arquivo)}
    novo = not saida.exists()

    dados = Path(args.dados)
    with open(saida, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        if novo:
            escritor.writerow(COLUNAS)

        for chave in args.detectores:
            tarefas = [(c, l) for c in args.condicoes for l in imagens if (l["id"], c, chave) not in feitas]
            if not tarefas:
                print(f"{chave}: nada a fazer", flush=True)
                continue
            inicio = time.time()
            detector = DETECTORES[chave]()
            print(f"{chave}: {detector.name} carregado em {time.time() - inicio:.0f}s, "
                  f"{len(tarefas)} imagens", flush=True)

            inicio = time.time()
            for n, (condicao, linha) in enumerate(tarefas, 1):
                caminho = caminho_imagem(dados, condicao, linha)
                t = time.time()
                try:
                    score = detector.predict(str(caminho))
                except Exception as erro:  # registra e segue; rodar de novo tenta outra vez
                    print(f"  ERRO em {caminho}: {erro}", flush=True)
                    continue
                escritor.writerow([linha["id"], condicao, chave, f"{score:.8g}", f"{time.time() - t:.2f}"])
                arquivo.flush()
                if n % 50 == 0 or n == len(tarefas):
                    decorrido = time.time() - inicio
                    restante = decorrido / n * (len(tarefas) - n)
                    print(f"  [{n}/{len(tarefas)}] {decorrido / n:.1f}s por imagem, "
                          f"faltam ~{restante / 60:.0f} min", flush=True)
            del detector

    print(f"Resultados em {saida}")


if __name__ == "__main__":
    main()
