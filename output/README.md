# Experimentos

Registro de todos os experimentos rodados, para servir de material ao TCC. Cada experimento tem a sua pasta `AAAA-MM-DD_<nome>/`, com um `README.md` (objetivo, dados, resultados e conclusões), os resultados brutos e os scripts para refazer.

Esta pasta vai para o Git, para o grupo acompanhar os resultados. As imagens usadas ficam em `datasets/`, que continua fora do Git; cada experimento tem um script para baixar ou montar as imagens de novo.

| Data | Pasta | Tipo | Resumo |
|---|---|---|---|
| 07-08/10/2026 | [2026-10-08_cf-89-imagens](2026-10-08_cf-89-imagens/README.md) | Exploratório | 89 imagens do Community Forensics (49 reais JPEG, 40 de IA PNG). CNNDetection e UnivFD quase não detectam geradores modernos; o NPR acertou 90%, mas convertendo as IAs para JPEG cai para 18%: viés de formato. |
| 08/10/2026 | [2026-10-08_bfree-teste-funcional](2026-10-08_bfree-teste-funcional/README.md) | Teste funcional | B-Free adicionado como 4º detector; roda com os outros 3 no `run.py`. 4 imagens, todas certas no B-Free (sem valor estatístico). ~17-20 s por imagem em CPU só no B-Free. |
| 09/10/2026 | [2026-10-09_bfree-50-imagens](2026-10-09_bfree-50-imagens/README.md) | Exploratório | Só o B-Free em 50 imagens do Community Forensics (20 reais, 16 de difusão, 4 Firefly 3 em JPEG, 10 GAN). 88% de acerto (27/30 IA, 17/20 reais), AUC 0,935. Convertendo as IAs para JPEG 95 não caiu (23/26 → 24/26): sem o viés de formato do NPR. ~11 s/imagem em CPU. |
