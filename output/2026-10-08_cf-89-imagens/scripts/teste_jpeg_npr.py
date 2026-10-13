"""Testa se o NPR reconhece IA ou só o formato: roda o NPR nas imagens de IA
originais (PNG) e nas mesmas imagens convertidas para JPEG."""

import io
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROJETO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(PROJETO))
from src.detection import NPRDetector  # noqa: E402

entrada = PROJETO / "datasets" / "input"
fakes = sorted(entrada.glob("fake_*.png"))
reais = sorted(entrada.glob("real_*"))
npr = NPRDetector()


def gerador(p):
    return p.stem.rsplit("_", 1)[0].removeprefix("fake_")


def avaliar(rotulo, arquivos, transforma=None):
    acertos = defaultdict(int)
    total = defaultdict(int)
    with tempfile.TemporaryDirectory() as tmp:
        for f in arquivos:
            alvo = f
            if transforma:
                alvo = Path(tmp) / (f.stem + ".jpg")
                transforma(f, alvo)
            g = "real" if f.name.startswith("real_") else gerador(f)
            pred_ia = npr.predict(str(alvo)) >= 0.5
            acertos[g] += pred_ia if g != "real" else not pred_ia
            total[g] += 1
    partes = [f"{g} {acertos[g]}/{total[g]}" for g in sorted(total)]
    soma = sum(acertos.values())
    print(f"{rotulo:<28} {soma}/{sum(total.values())}  |  " + "  ".join(partes), flush=True)


def jpeg(qualidade):
    def converte(origem, destino):
        with Image.open(origem) as im:
            im.convert("RGB").save(destino, "JPEG", quality=qualidade)
    return converte


avaliar("Reais (JPEG original)", reais)
avaliar("IA em PNG (original)", fakes)
avaliar("IA convertida p/ JPEG q95", fakes, jpeg(95))
avaliar("IA convertida p/ JPEG q75", fakes, jpeg(75))
