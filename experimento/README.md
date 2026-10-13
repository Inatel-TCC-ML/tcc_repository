# Experimento oficial

Desenho e scripts do experimento que vai para o artigo. Todo mundo usa a mesma amostra de imagens (`amostra.csv`, sorteada com semente fixa), então os resultados de máquinas diferentes podem ser juntados e comparados.

## Perguntas

1. **Generalização:** cada detector vai bem no gerador que viu no treino e cai nos geradores que não viu?
2. **Agregação:** combinar as notas dos 4 detectores (média, mediana, P75, P90, máximo) reduz essa queda?
3. **Formato:** quanto do resultado vem da diferença de formato entre reais e IA (PNG × JPEG), e não da detecção em si? (No teste exploratório, o NPR caiu de 90% para 18% de acerto só convertendo as imagens de IA para JPEG; ver `output/2026-10-08_cf-89-imagens/`.)

## Dados

24 grupos, cada um com **50 imagens de IA de um gerador e 50 reais pareadas** (da mesma origem). São 2.400 linhas na amostra e 2.050 imagens diferentes, porque as 50 reais do RAISE são as mesmas nos 8 grupos do RAISE.

| Grupo | Arquitetura | Reais de | Formato reais | Formato IA | Visto no treino por |
|---|---|---|---|---|---|
| ProGAN | GAN | LSUN | PNG | PNG | CNNDetection, UnivFD, NPR |
| SD2.1 | LatDiff | COCO | PNG | PNG | B-Free |
| DFGAN | GAN | COCO | JPEG | PNG | - |
| Dalle2 | Commercial | LAION | JPEG/PNG | PNG | - |
| Dalle3 | Commercial | LAION | JPEG/PNG | PNG | - |
| DeciDiffusionV2 | LatDiff | RAISE | TIFF | PNG | - |
| FLUX-dev | Commercial | LAION | JPEG | PNG | - |
| FLUX-schnell | Commercial | LAION | JPEG/PNG/WEBP | PNG | - |
| Firefly_Image2 | Commercial | LAION | JPEG/WEBP | JPEG | - |
| Firefly_Image3 | Commercial | LAION | JPEG/PNG | JPEG | - |
| GALIP | GAN | RAISE | TIFF | PNG | - |
| Hourglass_ffhq | PixDiff | FFHQ | PNG | PNG | - |
| Hourglass_imagenet | PixDiff | ImageNet | JPEG | PNG | - |
| IdeogramV1 | Commercial | LAION | JPEG/PNG/WEBP | PNG | - |
| IdeogramV2 | Commercial | LAION | JPEG/PNG/WEBP | PNG | - |
| Imagen3 | Commercial | LAION | JPEG/PNG/WEBP | PNG | - |
| LCM_lora_sdv15 | LatDiff | RAISE | TIFF | PNG | - |
| LCM_lora_sdxl | LatDiff | RAISE | TIFF | PNG | - |
| LCM_lora_ssd1b | LatDiff | RAISE | TIFF | PNG | - |
| MidjourneyV5_2 | Commercial | LAION | JPEG/PNG | PNG | - |
| MidjourneyV6_1 | Commercial | LAION | JPEG | PNG | - |
| kandinsky_2_2 | LatDiff | RAISE | TIFF | PNG | - |
| kvikontent_midjourney_v6 | LatDiff | RAISE | TIFF | PNG | - |
| stable_cascade | Other | RAISE | TIFF | PNG | - |

**De onde vem cada parte:**

- **ProGAN** (gerador visto por 3 detectores): conjunto de teste oficial do CNNDetection (Wang et al., CVPR 2020), `progan_testset.zip` em [sywang/CNNDetection](https://huggingface.co/datasets/sywang/CNNDetection). Tem 20 categorias; o sorteio alterna as categorias.
- **SD2.1** (gerador visto pelo B-Free): validação publicada pelos autores do B-Free (Guillaro et al., CVPR 2025), em [grip.unina.it](https://www.grip.unina.it/download/prog/B-Free/training_data/). São imagens do COCO e as mesmas cenas regeneradas pelo SD 2.1 (*self-conditioned*), ambas PNG 512×512 e fora do treino do B-Free.
- **22 grupos não vistos:** [Community Forensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval) (Park e Owens, CVPR 2025), split `CompEval`. É o mesmo conjunto em que o artigo do dataset publicou mAP/mAcc do CNNDetection (0,543 / 0,515) e do UnivFD (0,583 / 0,540), então dá para comparar com os nossos números.

**Regras do sorteio** (`gerar_amostra.py`, semente 42):

- só imagens RGB com o menor lado ≥ 128 px (o dataset tem algumas imagens corrompidas de 1×1);
- reais só em JPEG, PNG, WEBP ou TIFF (ficam de fora GIF, MPO e CMYK);
- só imagens que a API do Hugging Face consegue entregar (blocos do parquet abaixo de 280 MB; a API recusa acima de 300 MB). O tamanho do bloco é a soma de 25 imagens vizinhas, então isso não tem relação com o conteúdo de cada imagem;
- RAISE: só 286 das 8.000 fotos reais do RAISE estão em blocos acessíveis (são TIFF de ~22 MB). Por isso os 8 grupos do RAISE usam **o mesmo conjunto de 50 fotos reais do RAISE**. A comparação continua sendo "IA do gerador × fotos reais do RAISE";
- a coluna `ordem` permite usar menos imagens: `--por-grupo 25` pega as 25 primeiras de cada classe em cada grupo, sempre as mesmas.

## Condições

- **original:** cada imagem como veio da base. É o protocolo do artigo do Community Forensics. Nos grupos do RAISE, no FFHQ, no ProGAN e no SD2.1, reais e IA já estão sem compressão; nos outros, o formato é diferente entre as classes.
- **jpeg95:** todas as imagens, reais e IA, recomprimidas em JPEG qualidade 95. Tira a diferença de formato. As reais que já eram JPEG ficam com compressão dupla, o que deve ser citado como limitação.

## Métricas

Em cada grupo: **AP** (*average precision*, não depende de limiar) e **acurácia** com limiar 0,5 (geral, só nas reais e só nas IA). **mAP** e **mAcc** são as médias nos 22 grupos do Community Forensics, como no artigo do dataset. As agregações usam as classes de `src/aggregation/`.

O `avaliar.py` gera:

1. mAP e mAcc de cada detector e agregação, nas duas condições, ao lado dos valores publicados;
2. visto × não visto: AP no ProGAN (ou SD2.1, no B-Free) contra mAP nos 22 grupos;
3. mAP por arquitetura (GAN, LatDiff, PixDiff, Commercial, Other);
4. efeito do formato: mAP nos grupos sem perda × demais, nas duas condições;
5. AP e acurácia de cada grupo.

## Como rodar

Na branch com os 4 detectores, depois do `make setup` (que baixa os pesos):

```
python experimento/baixar_dados.py --por-grupo 50
python experimento/rodar.py --saida output/AAAA-MM-DD_experimento-oficial/resultados.csv
python experimento/avaliar.py output/AAAA-MM-DD_experimento-oficial/resultados.csv
```

- **Download:** cerca de 4 GB (as 50 fotos TIFF do RAISE somam ~1,1 GB). No teste, cada imagem do Community Forensics levou ~0,8 s com 4 downloads simultâneos (`--paralelo 4`), então as 2.050 devem levar de 30 a 60 min, dependendo da internet. A API do Hugging Face às vezes devolve erro 502 ou derruba a conexão; o script tenta de novo sozinho.
- **Detectores:** medido num notebook só com CPU (Intel i5-1135G7, 16 GB): por imagem, CNNDetection 0,4 s, UnivFD 1,2 s, NPR 0,1 s e **B-Free 10,3 s**. Com 2.050 imagens × 2 condições, são **~14 h em CPU** (quase tudo do B-Free) ou ~7 h com `--por-grupo 25`. Com GPU deve cair bastante (o código usa a GPU sozinho quando ela existe). Para testar antes, `--limite 4` roda só 4 imagens.
- Os três scripts podem ser interrompidos e rodados de novo: continuam de onde pararam.
- **Dividir entre máquinas:** cada um roda uma parte com `--detectores` (ex.: um PC com `--detectores bfree`, outro com `--detectores cnndetection univfd npr`) e no fim o `avaliar.py` recebe os dois CSVs: `python experimento/avaliar.py parte1.csv parte2.csv`.
- **Menos imagens:** use o mesmo `--por-grupo` (ex.: 25) nos três scripts.

Depois, registrar em `output/` como os outros experimentos (README com máquina, versões e tempo + linha no `output/README.md`).

## Limitações já conhecidas

- 50 imagens por classe em cada grupo e uma única execução (não há treino, então não há sementes de treino; a variação vem só da amostra).
- Resoluções diferentes entre reais e IA (ex.: RAISE com 16 megapixels × IA com 1 megapixel). Cada detector usa o seu pré-processamento oficial, sem ajuste nosso.
- Compressão dupla nas reais que já eram JPEG, na condição jpeg95.
- Alguns geradores "não vistos" do Community Forensics são versões ajustadas do Stable Diffusion (LCM_lora_sdv15, kvikontent_midjourney_v6 etc.), parentes próximos do SD 2.1 visto pelo B-Free. Na discussão, vale separar esses grupos.
- Os números do artigo do Community Forensics usam todas as imagens e o pré-processamento deles; os nossos são de uma amostra e não devem bater exatamente.

## Arquivos

- `amostra.csv`: as 2.400 linhas (id, grupo, classe, ordem, base, gerador, arquitetura, origem das reais, formato, quem viu no treino, onde baixar).
- `baixar_dados.py`, `rodar.py`, `avaliar.py`: os três passos acima.
- `gerar_amostra.py`, `extrair_metadados.py`: como a amostra foi feita (não precisa rodar).
- `zip_remoto.py`: tira poucas imagens de zips grandes sem baixar o zip inteiro.
