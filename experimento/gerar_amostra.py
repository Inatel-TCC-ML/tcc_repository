"""Sorteia as imagens do experimento oficial e grava experimento/amostra.csv.

Não precisa rodar de novo: o amostra.csv já está no repositório. Este script
fica para mostrar como a amostra foi feita (semente fixa, dá o mesmo resultado).

Uso (da raiz do projeto):
    python experimento/gerar_amostra.py <metadados_cf_eval.csv>

O CSV de metadados do Community Forensics-Eval é gerado por
experimento/extrair_metadados.py (lê só as colunas de texto dos parquets).
"""

import csv
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zip_remoto import abrir_zip  # noqa: E402

SEMENTE = 42
POR_GRUPO = 50  # imagens por classe (real e IA) em cada grupo
TAMANHO_MINIMO = 128  # menor lado, em pixels
FORMATOS_REAIS = {"JPEG", "PNG", "WEBP", "TIFF"}
LIMITE_BLOCO = 280_000_000  # a API do Hugging Face recusa blocos acima de 300 MB

PROGAN_ZIP = "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_testset.zip"
BFREE_URL = "https://www.grip.unina.it/download/prog/B-Free/training_data/"

SAIDA = Path(__file__).resolve().parent / "amostra.csv"
COLUNAS = ["id", "grupo", "classe", "ordem", "base", "gerador", "arquitetura",
           "origem_real", "formato", "visto_por", "fonte"]


def menor_lado(resolucao):
    largura, altura = (int(x) for x in resolucao.strip("[]").split(","))
    return min(largura, altura)


def amostra_cf_eval(caminho_metadados, rng):
    grupos = defaultdict(lambda: {"real": [], "ia": []})
    reais_raise = []
    with open(caminho_metadados, encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            if linha["mode"] != "RGB" or menor_lado(linha["resolution"]) < TAMANHO_MINIMO:
                continue
            if int(linha["tamanho_bloco"]) >= LIMITE_BLOCO:
                continue
            classe = "ia" if linha["label"] == "1" else "real"
            if classe == "real" and linha["format"] not in FORMATOS_REAIS:
                continue
            gerador = linha["model_name"]
            grupo = f"{gerador}_{linha['real_source']}" if gerador == "Hourglass" else gerador
            if classe == "real" and linha["real_source"] == "RAISE":
                reais_raise.append(linha)
            grupos[grupo][classe].append(linha)

    # Poucas fotos do RAISE ficam em blocos acessíveis pela API (TIFF de ~22 MB,
    # blocos de até 600 MB). Os 8 grupos do RAISE usam o mesmo conjunto de 50
    # fotos reais do RAISE, sorteado entre todas as acessíveis.
    reais_raise = rng.sample(sorted(reais_raise, key=lambda l: int(l["idx"])), POR_GRUPO)

    linhas = []
    for grupo in sorted(grupos):
        for classe in ("real", "ia"):
            if classe == "real" and grupos[grupo]["real"][0]["real_source"] == "RAISE":
                sorteadas = reais_raise
            else:
                candidatas = sorted(grupos[grupo][classe], key=lambda l: int(l["idx"]))
                sorteadas = rng.sample(candidatas, POR_GRUPO)
            for ordem, l in enumerate(sorteadas):
                linhas.append({
                    "id": f"cf_{int(l['idx']):05d}",
                    "grupo": grupo,
                    "classe": classe,
                    "ordem": ordem,
                    "base": "CF-Eval",
                    "gerador": l["model_name"] if classe == "ia" else l["real_source"],
                    "arquitetura": "Real" if classe == "real" else l["architecture"],
                    "origem_real": l["real_source"],
                    "formato": l["format"],
                    "visto_por": "",
                    "fonte": l["idx"],
                })
    return linhas


def amostra_progan(rng):
    # 20 categorias com 200 reais (LSUN) e 200 ProGAN cada, todas em PNG.
    # O sorteio alterna as categorias para que os primeiros N cubram todas.
    nomes = [i.filename for i in abrir_zip(PROGAN_ZIP).infolist() if not i.is_dir()]
    por_categoria = defaultdict(lambda: {"real": [], "ia": []})
    for nome in sorted(nomes):
        _, categoria, pasta, _ = nome.split("/")
        por_categoria[categoria]["real" if pasta == "0_real" else "ia"].append(nome)

    linhas = []
    categorias = sorted(por_categoria)
    for classe in ("real", "ia"):
        sorteadas = {c: rng.sample(por_categoria[c][classe], POR_GRUPO) for c in categorias}
        escolhidas = [sorteadas[c][k] for k in range(POR_GRUPO) for c in categorias][:POR_GRUPO]
        for ordem, nome in enumerate(escolhidas):
            _, categoria, _, arquivo = nome.split("/")
            linhas.append({
                "id": f"progan_{classe}_{categoria}_{Path(arquivo).stem}",
                "grupo": "ProGAN",
                "classe": classe,
                "ordem": ordem,
                "base": "ForenSynths (teste ProGAN)",
                "gerador": "ProGAN" if classe == "ia" else "LSUN",
                "arquitetura": "Real" if classe == "real" else "GAN",
                "origem_real": "LSUN",
                "formato": "PNG",
                "visto_por": "CNNDetection;UnivFD;NPR",
                "fonte": nome,
            })
    return linhas


def amostra_sd21(rng):
    # Validação do B-Free: reais do COCO e as mesmas cenas regeneradas pelo
    # SD 2.1 (self-conditioned), ambas PNG 512x512 e fora do treino.
    with urllib.request.urlopen(BFREE_URL + "valid_list.csv", timeout=60) as resposta:
        ids = [l["id"] for l in csv.DictReader(resposta.read().decode().splitlines())]
    escolhidos = rng.sample(sorted(ids), POR_GRUPO)
    linhas = []
    for classe, pasta in (("real", "COCO_real_512"), ("ia", "SD2.1_selfconditioned")):
        for ordem, ident in enumerate(escolhidos):
            linhas.append({
                "id": f"sd21_{classe}_{ident}",
                "grupo": "SD2.1",
                "classe": classe,
                "ordem": ordem,
                "base": "B-Free (validação)",
                "gerador": "SD2.1" if classe == "ia" else "COCO",
                "arquitetura": "Real" if classe == "real" else "LatDiff",
                "origem_real": "COCO",
                "formato": "PNG",
                "visto_por": "B-Free",
                "fonte": f"{pasta}.zip:{pasta}/{ident}.png",
            })
    return linhas


def main():
    rng = random.Random(SEMENTE)
    linhas = amostra_progan(rng) + amostra_sd21(rng) + amostra_cf_eval(sys.argv[1], rng)
    with open(SAIDA, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS)
        escritor.writeheader()
        escritor.writerows(linhas)
    grupos = sorted({l["grupo"] for l in linhas})
    print(f"{len(linhas)} imagens, {len(grupos)} grupos -> {SAIDA}")


if __name__ == "__main__":
    main()
