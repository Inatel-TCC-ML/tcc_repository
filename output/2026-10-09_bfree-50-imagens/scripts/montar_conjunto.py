"""Monta o conjunto de 50 imagens do teste do B-Free.

- 20 reais (as 20 primeiras em ordem de nome) e 16 de difusão (4 por gerador) vêm da amostra de 89 imagens de
  output/2026-10-08_cf-89-imagens (precisa estar em datasets/input/).
- 4 do Firefly 3 (JPEG nativo) e 10 de GAN (5 DFGAN + 5 GALIP) são baixadas do
  Community Forensics-Eval pela API do dataset viewer do Hugging Face.
- As imagens de IA em PNG também são salvas convertidas para JPEG qualidade 95,
  para checar o viés de formato.

Rodar da raiz do projeto:
    python output/2026-10-09_bfree-50-imagens/scripts/montar_conjunto.py
Saída: datasets/bfree_50/original/, datasets/bfree_50/jpeg95/ e
datasets/bfree_50/gabarito.csv.
"""

import base64
import csv
import io
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

BASE = (
    "https://datasets-server.huggingface.co/rows?"
    "dataset=OwensLab/CommunityForensics-Eval&config=default&split=CompEval"
)
ORIGEM_89 = Path("datasets/input")
SAIDA = Path("datasets/bfree_50")
QUALIDADE_JPEG = 95

# Arquitetura de cada gerador já presente na amostra de 89 imagens
DIFUSAO_89 = {
    "FLUX-schnell": "Commercial",
    "IdeogramV2": "Commercial",
    "LCM_lora_sdxl": "LatDiff",
    "MidjourneyV6_1": "Commercial",
}
# Gerador a baixar: (offset inicial no split CompEval, quantidade)
DOWNLOADS = {
    "DFGAN": (100, 5),
    "GALIP": (15000, 5),
    "Firefly_Image3": (41000, 4),
}


def busca(offset, tamanho=5):
    for tentativa in range(6):
        try:
            url = f"{BASE}&offset={offset}&length={tamanho}"
            return json.load(urllib.request.urlopen(url, timeout=120))["rows"]
        except urllib.error.HTTPError as erro:
            espera = 30 * (tentativa + 1) if erro.code == 429 else 5
            print(f"  offset {offset}: HTTP {erro.code}, esperando {espera}s", flush=True)
            time.sleep(espera)
    raise RuntimeError(f"Falha ao baixar offset {offset}")


def main():
    original = SAIDA / "original"
    jpeg95 = SAIDA / "jpeg95"
    original.mkdir(parents=True, exist_ok=True)
    jpeg95.mkdir(parents=True, exist_ok=True)
    linhas = []

    # As 20 primeiras reais em ordem (a numeração tem um buraco: uma imagem
    # corrompida foi descartada no teste das 89)
    for caminho in sorted(ORIGEM_89.glob("real_*.jpg"))[:20]:
        nome = caminho.name
        shutil.copy2(caminho, original / nome)
        linhas.append([nome, "real", "LAION", "Real", "JPEG", "CompEval (amostra 89)"])

    for gerador, arquitetura in DIFUSAO_89.items():
        for i in range(1, 5):
            nome = f"fake_{gerador}_{i:02d}.png"
            shutil.copy2(ORIGEM_89 / nome, original / nome)
            linhas.append([nome, "ia", gerador, arquitetura, "PNG", "CompEval (amostra 89)"])

    for gerador, (offset, quantidade) in DOWNLOADS.items():
        baixadas = 0
        while baixadas < quantidade:
            for item in busca(offset):
                linha = item["row"]
                if linha["label"] != 1 or linha["model_name"] != gerador:
                    continue
                if baixadas >= quantidade:
                    break
                baixadas += 1
                ext = ".png" if linha["format"] == "PNG" else ".jpg"
                nome = f"fake_{gerador}_{baixadas:02d}{ext}"
                (original / nome).write_bytes(base64.b64decode(linha["image_data"]))
                linhas.append([nome, "ia", gerador, linha["architecture"], linha["format"],
                               f"CompEval idx {item['row_idx']} ({linha['image_name']})"])
            offset += 5
            time.sleep(2)
        print(f"{gerador}: {baixadas} imagens", flush=True)

    # Versão JPEG 95 das imagens de IA que estão em PNG
    for nome, classe, *_ in linhas:
        if classe == "ia" and nome.endswith(".png"):
            with Image.open(original / nome) as imagem:
                destino = jpeg95 / nome.replace(".png", ".jpg")
                imagem.convert("RGB").save(destino, "JPEG", quality=QUALIDADE_JPEG)

    with open(SAIDA / "gabarito.csv", "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["arquivo", "classe", "gerador", "arquitetura", "formato", "origem"])
        escritor.writerows(linhas)

    print(f"Conjunto: {len(linhas)} imagens em {original}")
    print(f"Versões JPEG {QUALIDADE_JPEG}: {len(list(jpeg95.iterdir()))} imagens em {jpeg95}")


if __name__ == "__main__":
    main()
