"""Roda só o B-Free nas imagens de datasets/bfree_50/ (original e jpeg95).

Rodar da raiz do projeto:
    python output/2026-10-09_bfree-50-imagens/scripts/rodar_bfree.py <arquivo_saida.csv>
Cada linha é gravada assim que a imagem termina, então dá para acompanhar o
progresso e não se perde o que já rodou se o processo for interrompido.
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.detection import BFreeDetector  # noqa: E402

PASTA = Path("datasets/bfree_50")


def main():
    saida = Path(sys.argv[1])
    with open(PASTA / "gabarito.csv", encoding="utf-8") as arquivo:
        gabarito = {linha["arquivo"]: linha for linha in csv.DictReader(arquivo)}

    tarefas = [("original", nome, PASTA / "original" / nome) for nome in gabarito]
    for caminho in sorted((PASTA / "jpeg95").iterdir()):
        tarefas.append(("jpeg95", caminho.name.replace(".jpg", ".png"), caminho))

    inicio = time.time()
    detector = BFreeDetector()
    print(f"Modelo carregado em {time.time() - inicio:.1f}s ({detector.device})", flush=True)

    with open(saida, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["arquivo", "variante", "classe", "gerador", "arquitetura",
                           "formato_testado", "score", "segundos"])
        for n, (variante, nome, caminho) in enumerate(tarefas, 1):
            info = gabarito[nome]
            formato = "JPEG" if caminho.suffix.lower() == ".jpg" else "PNG"
            t = time.time()
            score = detector.predict(str(caminho))
            segundos = time.time() - t
            escritor.writerow([caminho.name, variante, info["classe"], info["gerador"],
                               info["arquitetura"], formato, f"{score:.6f}", f"{segundos:.1f}"])
            arquivo.flush()
            print(f"[{n}/{len(tarefas)}] {variante}/{caminho.name}: {score:.4f} ({segundos:.1f}s)",
                  flush=True)

    print(f"Total: {(time.time() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
