"""Baixa uma amostra rotulada (50 reais + 50 de IA) do Community Forensics-Eval
pela API do dataset viewer do Hugging Face, sem baixar os parquets inteiros."""

import base64
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = (
    "https://datasets-server.huggingface.co/rows?"
    "dataset=OwensLab/CommunityForensics-Eval&config=default&split=CompEval"
)
OUT = Path(sys.argv[1])
POR_GERADOR = 10  # 5 geradores x 10 = 50 imagens de IA
TOTAL_REAIS = 50
# Regiões do dataset onde a amostragem achou esses geradores
OFFSETS = [17600, 9600, 18400, 12000, 7200, 16000, 2400, 3200, 21600, 23200, 0, 13600]
PAGINA = 20

OUT.mkdir(parents=True, exist_ok=True)
fakes, reais, linhas = {}, 0, []


def busca(offset):
    for tentativa in range(6):
        try:
            url = f"{BASE}&offset={offset}&length={PAGINA}"
            return json.load(urllib.request.urlopen(url, timeout=120))["rows"]
        except urllib.error.HTTPError as e:
            espera = 30 * (tentativa + 1) if e.code == 429 else 5
            print(f"  offset {offset}: HTTP {e.code}, esperando {espera}s", flush=True)
            time.sleep(espera)
        except Exception as e:
            print(f"  offset {offset}: {e}", flush=True)
            time.sleep(5)
    return []


def completo():
    return reais >= TOTAL_REAIS and len(fakes) >= 5 and all(n >= POR_GERADOR for n in fakes.values())


for base in OFFSETS:
    for offset in range(base, base + 200, PAGINA):
        if completo():
            break
        for item in busca(offset):
            r = item["row"]
            ext = ".png" if r["format"] == "PNG" else ".jpg"
            if r["label"] == 0 and reais < TOTAL_REAIS:
                reais += 1
                nome = f"real_{reais:03d}{ext}"
            elif r["label"] == 1 and r["architecture"] != "GAN":
                modelo = r["model_name"]
                if modelo not in fakes and len(fakes) >= 5:
                    continue
                if fakes.get(modelo, 0) >= POR_GERADOR:
                    continue
                fakes[modelo] = fakes.get(modelo, 0) + 1
                nome = f"fake_{modelo}_{fakes[modelo]:02d}{ext}"
            else:
                continue
            (OUT / nome).write_bytes(base64.b64decode(r["image_data"]))
            linhas.append([nome, "real" if r["label"] == 0 else "ia", r["model_name"],
                           r["architecture"], r["real_source"], r["image_name"]])
        print(f"offset {offset}: reais={reais} ia={dict(fakes)}", flush=True)
        time.sleep(3)
    if completo():
        break

with open(OUT / "gabarito.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["arquivo", "classe", "modelo_pareado", "arquitetura", "origem_real", "nome_original"])
    w.writerows(linhas)
print("FIM", reais, dict(fakes))
