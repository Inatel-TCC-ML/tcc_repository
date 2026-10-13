"""Baixa as imagens do experimento oficial listadas em experimento/amostra.csv.

Uso (da raiz do projeto):
    python experimento/baixar_dados.py --por-grupo 50

Saída em datasets/experimento/:
- original/: as imagens como vêm de cada base (formato original);
- jpeg95/: as mesmas imagens recomprimidas em JPEG qualidade 95, para que reais
  e IA tenham o mesmo formato.

Pode ser interrompido e rodado de novo: o que já foi baixado é pulado.
"""

import argparse
import base64
import csv
import json
import sys
import time
import urllib.error
import urllib.request
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

# As fotos TIFF do RAISE têm metadados EXIF malformados; o Pillow avisa
# "Truncated File Read", mas os pixels carregam inteiros (conferido).
warnings.filterwarnings("ignore", message="Truncated File Read")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zip_remoto import abrir_zip  # noqa: E402

AMOSTRA = Path(__file__).resolve().parent / "amostra.csv"
CF_ROWS = (
    "https://datasets-server.huggingface.co/rows?"
    "dataset=OwensLab/CommunityForensics-Eval&config=default&split=CompEval"
)
PROGAN_ZIP = "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_testset.zip"
BFREE_URL = "https://www.grip.unina.it/download/prog/B-Free/training_data/"
EXTENSOES = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "WEBP": ".webp"}
QUALIDADE_JPEG = 95


def caminho_original(destino, linha):
    return destino / "original" / (linha["id"] + EXTENSOES[linha["formato"]])


def baixar_cf(idx):
    for tentativa in range(8):
        try:
            url = f"{CF_ROWS}&offset={idx}&length=1"
            with urllib.request.urlopen(url, timeout=120) as resposta:
                linha = json.load(resposta)["rows"][0]["row"]
            return base64.b64decode(linha["image_data"])
        except urllib.error.HTTPError as erro:
            # A API do Hugging Face falha de vez em quando (429, 500, 502);
            # na segunda ou terceira tentativa costuma funcionar.
            espera = 30 * (tentativa + 1) if erro.code == 429 else 5 * (tentativa + 1)
            print(f"  idx {idx}: HTTP {erro.code}, tentando de novo em {espera}s", flush=True)
            time.sleep(espera)
        except OSError as erro:  # conexão derrubada, DNS, timeout
            print(f"  idx {idx}: {erro}, tentando de novo", flush=True)
            time.sleep(10 * (tentativa + 1))
    raise RuntimeError(f"Falha ao baixar a linha {idx} do CF-Eval")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--por-grupo", type=int, default=50,
                        help="imagens por classe em cada grupo (máx. 50)")
    parser.add_argument("--destino", default="datasets/experimento")
    parser.add_argument("--paralelo", type=int, default=4,
                        help="downloads simultâneos do Community Forensics")
    args = parser.parse_args()

    destino = Path(args.destino)
    (destino / "original").mkdir(parents=True, exist_ok=True)
    (destino / "jpeg95").mkdir(parents=True, exist_ok=True)

    with open(AMOSTRA, encoding="utf-8") as arquivo:
        selecionadas = [l for l in csv.DictReader(arquivo) if int(l["ordem"]) < args.por_grupo]
    # Uma imagem pode aparecer em mais de um grupo (as reais do RAISE)
    linhas = list({l["id"]: l for l in selecionadas}.values())
    faltando = [l for l in linhas if not caminho_original(destino, l).exists()]
    print(f"{len(linhas)} imagens únicas na amostra, {len(faltando)} para baixar", flush=True)

    def salvar(linha, dados):
        caminho = caminho_original(destino, linha)
        caminho.with_suffix(".tmp").write_bytes(dados)
        caminho.with_suffix(".tmp").replace(caminho)

    # Imagens que vêm de zips (ProGAN e B-Free): abre cada zip uma vez só
    zips = {}
    de_zip = [l for l in faltando if l["base"] != "CF-Eval"]
    for n, linha in enumerate(de_zip, 1):
        if linha["grupo"] == "ProGAN":
            url, membro = PROGAN_ZIP, linha["fonte"]
        else:
            nome_zip, membro = linha["fonte"].split(":")
            url = BFREE_URL + nome_zip
        if url not in zips:
            zips[url] = abrir_zip(url)
        salvar(linha, zips[url].read(membro))
        if n % 25 == 0 or n == len(de_zip):
            print(f"[zips {n}/{len(de_zip)}]", flush=True)

    # Imagens do Community Forensics: a API demora alguns segundos por imagem,
    # então baixa várias ao mesmo tempo
    do_cf = [l for l in faltando if l["base"] == "CF-Eval"]
    inicio = time.time()
    with ThreadPoolExecutor(max_workers=args.paralelo) as executor:
        futuros = {executor.submit(baixar_cf, l["fonte"]): l for l in do_cf}
        for n, futuro in enumerate(as_completed(futuros), 1):
            salvar(futuros[futuro], futuro.result())
            if n % 25 == 0 or n == len(do_cf):
                restante = (time.time() - inicio) / n * (len(do_cf) - n)
                print(f"[Community Forensics {n}/{len(do_cf)}] faltam ~{restante / 60:.0f} min", flush=True)

    # Versão JPEG 95 de todas as imagens (reais e IA)
    convertidas = 0
    for linha in linhas:
        saida = destino / "jpeg95" / (linha["id"] + ".jpg")
        if saida.exists():
            continue
        with Image.open(caminho_original(destino, linha)) as imagem:
            imagem.convert("RGB").save(saida, "JPEG", quality=QUALIDADE_JPEG)
        convertidas += 1
    print(f"{convertidas} imagens convertidas para JPEG {QUALIDADE_JPEG}")

    tamanho = sum(p.stat().st_size for p in destino.rglob("*") if p.is_file())
    formatos = Counter(l["formato"] for l in linhas)
    print(f"Pronto: {len(linhas)} imagens ({dict(formatos)}), {tamanho / 1e9:.2f} GB em {destino}")


if __name__ == "__main__":
    main()
