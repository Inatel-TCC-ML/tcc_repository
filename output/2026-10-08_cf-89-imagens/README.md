# Teste exploratório: 89 imagens do Community Forensics (07-08/10/2026)

Primeiro teste com gabarito, feito num notebook só com CPU para entender como os três detectores se comportam. **É um teste pequeno e exploratório. Os números não devem ir para o artigo sem um experimento oficial com mais imagens.**

## Ambiente

- Branch `NPR-DeepfakeDetection-image` (commit `611d7ce`), com a correção do import em `src/detection/__init__.py` (`.cnnDetection` → `.cnndetection`).
- Python 3.13, torch 2.14.1 (CPU). O `requirements.txt` não instala no 3.13; foi usado com `numpy` e `scikit-image` sem versão fixa.
- Detectores: UnivFD (CLIP ViT-L/14), CNNDetection (Blur+JPEG 0.5) e NPR (ResNet50), com os pesos oficiais.
- Tempo: cerca de 5 minutos para as 89 imagens.

## Dados

- Origem: [Community Forensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval) (OwensLab, Hugging Face), split `CompEval`. Baixado pela API do dataset viewer com `scripts/baixar_amostra.py`.
- **49 reais**: fotos do LAION, todas em **JPEG**, de tamanhos variados (80×80 até 1024×768). Uma 50ª imagem veio corrompida (1×1 pixel) e foi descartada.
- **40 de IA**, todas em **PNG**, 10 de cada gerador:
  - FLUX-schnell (512×512);
  - IdeogramV2 (1024×1024);
  - LCM_lora_sdxl, que é SDXL (1024×1024);
  - MidjourneyV6_1 (1024×1024).
- O gabarito está no nome do arquivo: `real_NNN` ou `fake_<gerador>_NN`. A lista completa está em `lista_imagens.txt`.

## Resultado 1: os três detectores (`avaliacao.txt`)

Limiar 0,5 (nota ≥ 0,5 = IA).

| Score | Acurácia | Reais (49) | FLUX | Ideogram | SDXL | Midjourney |
|---|---|---|---|---|---|---|
| UnivFD | 56% | 46 | 0 | 1 | 2 | 1 |
| CNNDetection | 55% | 49 | 0 | 0 | 0 | 0 |
| NPR | 90% | 48 | 9 | 10 | 4 | 9 |
| Média | 58% | 49 | 0 | 1 | 1 | 1 |
| Percentil 75 | 91% | 48 | 9 | 10 | 5 | 9 |
| Máximo | 88% | 45 | 9 | 10 | 5 | 9 |

- CNNDetection e UnivFD quase não detectam os geradores modernos: chamam quase tudo de real.
- A média é puxada para baixo pelos dois modelos fracos. Percentil 75 e máximo seguem o NPR.

## Resultado 2: viés de formato no NPR (`teste_jpeg_npr.txt`)

Suspeita: como todas as reais são JPEG e todas as de IA são PNG, o detector pode estar distinguindo o formato, e não a IA. Teste: as mesmas 40 imagens de IA convertidas para JPEG.

| Imagens de IA | NPR acertou | FLUX | Ideogram | SDXL | Midjourney |
|---|---|---|---|---|---|
| PNG (original) | 32/40 | 9 | 10 | 4 | 9 |
| JPEG qualidade 95 | 17/40 | 0 | 7 | 4 | 6 |
| JPEG qualidade 75 | 7/40 | 0 | 4 | 1 | 2 |

**Conclusão:** a maior parte do acerto do NPR vinha do formato do arquivo. Em condições iguais de formato, nenhum dos três detectores reconhece bem esses geradores.

## O que levar para o TCC

- **Metodologia:** reais e IA precisam estar no mesmo formato e com a mesma compressão; senão o experimento mede o formato, não a detecção. Referência sobre esse viés: Grommelt et al., "Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets" ([arXiv 2403.17608](https://arxiv.org/abs/2403.17608)).
- **Resultados:** registrar a sensibilidade do NPR à compressão JPEG como limitação.
- **CSV do Thomas (~2k imagens):** o NPR deu nota perto de 1 em quase tudo. Pode ser o mesmo viés; falta saber a origem e o formato daquelas imagens.

## Arquivos

- `results.csv`: notas de cada modelo e agregações, gerado pelo `run.py`.
- `avaliacao.txt`: acerto, F1 e AP por score, gerado por `scripts/avaliar.py`.
- `teste_jpeg_npr.txt`: saída de `scripts/teste_jpeg_npr.py`.
- `lista_imagens.txt`: as 89 imagens usadas.
- `scripts/`: os scripts para refazer o teste. Rodar da raiz do projeto, com o `.venv` ativo:
  - `python output/2026-10-08_cf-89-imagens/scripts/avaliar.py output/2026-10-08_cf-89-imagens/results.csv`
  - `python output/2026-10-08_cf-89-imagens/scripts/teste_jpeg_npr.py .`
