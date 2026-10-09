# Teste funcional do B-Free (4º detector)

**Data:** 08/10/2026 · **Branch:** `B-Free` · **Máquina:** notebook do João (só CPU)

## Objetivo

Só provar que o B-Free roda junto com os outros três detectores no `run.py`. Não é avaliação de desempenho: 4 imagens não dizem nada sobre acurácia.

## Configuração

- Modelo: B-Free oficial (`BFREE_dino2reg4`, ViT-Base/14 DINOv2 com registers, via timm), pesos de https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip (MD5 `f3f53fa647848b16cf81c913f148a198` conferido).
- Inferência igual à oficial (`main_bfree_single.py`): imagem inteira, normalização ImageNet, o próprio modelo tira a média de 5 recortes de 504×504 (centro + 4 cantos; imagem menor é replicada). Score = sigmoid do logit.
- Imagens (copiadas de `datasets/input/`): `real_001.jpg` (500×500), `real_002.jpg` (300×300), `fake_FLUX-schnell_01.png` (512×512), `fake_LCM_lora_sdxl_05.png` (1024×1024).

## Resultados

| Imagem | UnivFD | CNNDetection | NPR | B-Free |
|---|---|---|---|---|
| fake_FLUX-schnell_01.png | 0,000 | 0,000 | 1,000 | 0,989 |
| fake_LCM_lora_sdxl_05.png | 0,319 | 0,000 | 0,698 | 0,998 |
| real_001.jpg | 0,001 | 0,000 | 0,000 | 0,005 |
| real_002.jpg | 0,036 | 0,000 | 0,000 | 0,006 |

Valores completos em `results.csv`.

**Tempo:** carregar o B-Free leva ~15 s. Cada imagem leva ~17 a 20 s só no B-Free em CPU, e o tempo quase não depende do tamanho, porque o modelo sempre processa 5 recortes de 504×504. É o detector mais pesado dos quatro. Para as 89 imagens, contar com ~30 min em CPU; com GPU deve cair bastante.

## Observações

- As 4 imagens foram classificadas certo pelo B-Free, mas as 2 de IA são PNG e as 2 reais são JPEG, então o viés de formato visto no NPR ([2026-10-08_cf-89-imagens](../2026-10-08_cf-89-imagens/README.md)) ainda precisa ser testado no B-Free (rodada com IAs convertidas para JPEG, a fazer nos PCs dos colegas).
