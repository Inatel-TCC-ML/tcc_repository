# Detecção de imagens geradas por IA

Ferramenta experimental para comparar detectores de imagens sintéticas e avaliar se um ensemble de modelos melhora a detecção. O projeto faz parte de um TCC sobre generalização entre geradores de IA e impacto do redimensionamento das imagens.

O fluxo é simples:

```text
CommunityForensics-Eval → detectores individuais → ensemble → métricas por resolução
```

## Sumário

- [Detectores usados](#detectores-usados)
- [Estrutura](#estrutura)
- [Como rodar](#como-rodar)
  - [Configuração inicial](#configuração-inicial)
  - [Executar a inferência](#executar-a-inferência)
  - [Gerar as métricas](#gerar-as-métricas)
  - [Executar os testes](#executar-os-testes)
  - [Explorar os gráficos](#explorar-os-gráficos)
- [Arquivos de resultado](#arquivos-de-resultado)
  - [`experiment_resolution.csv`](#experiment_resolutioncsv)
  - [`metrics_by_resolution.csv`](#metrics_by_resolutioncsv)
- [Como analisar os modelos](#como-analisar-os-modelos)
- [Dataset e referências](#dataset-e-referências)

## Detectores usados

| Detector | Papel no experimento |
|---|---|
| UnivFD | Detector baseado em CLIP para generalização entre geradores. |
| CNNDetection | Baseline treinado para identificar artefatos de GANs. |
| NPR | Detector baseado em padrões entre pixels vizinhos. |
| B-Free | Detector baseado em DINOv2, voltado a imagens de difusão. |
| SPAI | Detector espectral de imagens sintéticas. |

Cada detector retorna um **score entre 0 e 1**. Quanto maior o score, maior a evidência de que a imagem é sintética segundo aquele detector. O score não deve ser interpretado automaticamente como uma probabilidade calibrada.

## Estrutura

```text
tcc_repository/
├── external/                 # repositórios originais dos detectores
├── weights/                  # pesos pré-treinados
├── src/
│   ├── detection/            # wrappers dos cinco detectores
│   ├── aggregation/          # estratégias de ensemble
│   └── config.py             # caminhos do projeto
├── results/                  # CSVs gerados pelos experimentos
├── run.py                    # executa a inferência no dataset
├── analyze_results.py        # transforma scores em métricas
├── tests/                    # testes dos agregadores
├── Makefile                  # comandos de configuração e execução
└── requirements.txt
```

## Como rodar

### Configuração inicial

No PowerShell, dentro da pasta do projeto:

```powershell
make setup
```

Esse comando cria `.venv`, baixa os submódulos, instala as dependências e baixa os pesos automatizados do CNNDetection e B-Free.

Se o submódulo CNNDetection falhar por usar SSH, execute uma vez:

```powershell
git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive
```

O checkpoint do SPAI deve estar em `weights/spai.pth`. Baixe-o pelo [link oficial](https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view?usp=sharing) antes da primeira execução.

### Executar a inferência

```powershell
make run
```

As opções do experimento, como os geradores selecionados e as resoluções, ficam no início de `run.py`. O resultado é salvo em `results/experiment_resolution.csv`. A execução pode demorar bastante; ela carrega cinco modelos e realiza uma inferência para cada imagem e resolução.

### Gerar as métricas

Após existir um CSV de resultados em `results/experiment_resolution.csv`, execute:

```powershell
make metrics
```

O script gera `results/metrics_by_resolution.csv`.

### Executar os testes

```powershell
make test
```

Os testes verificam as estratégias de ensemble e suas validações. Os testes
dos repositórios em `external/` não são executados, pois eles possuem
dependências próprias.

### Explorar os gráficos

Abra `notebooks/analysis_results.ipynb` no VS Code, selecione o kernel `Python (.venv)` e use **Run All**. As figuras são exibidas no notebook e salvas em `results/figures/`.

## Arquivos de resultado

### `experiment_resolution.csv`

Contém uma linha por imagem e resolução. As colunas mais importantes são:

| Coluna | Significado |
|---|---|
| `Indice_Dataset` | Identificador da imagem no dataset. O mesmo identificador permite comparar a mesma imagem entre resoluções. |
| `Tipo_Label` | Rótulo verdadeiro: `Fake` ou `Real`. |
| `Modelo_Gerador` | Gerador que criou a imagem sintética. |
| `Fonte_Real` | Fonte da imagem real, como COCO, FFHQ ou LAION. |
| `Arquitetura` | Categoria do gerador, por exemplo GAN, LatDiff ou Commercial. |
| `Resolução` | Condição usada na inferência. |
| colunas dos detectores | Score individual de cada detector. |
| `Média`, `Mediana`, `Percentil 75`, `Percentil 90`, `Máximo`, `Mínimo` | Formas de combinar os cinco scores em um score de ensemble. |
| `Votação da Maioria` | Proporção de detectores que classificaram a imagem como sintética usando threshold 0,5. |

### `metrics_by_resolution.csv`

É o arquivo principal para tabelas, gráficos e discussão do artigo. Cada linha representa um detector individual ou um ensemble em uma resolução.

| Coluna | Significado |
|---|---|
| `Resolução` | Condição de tamanho da imagem avaliada. |
| `Tipo` | `Detector individual` ou `Ensemble`. |
| `Modelo` | Nome do detector ou estratégia de combinação dos scores. |
| `Threshold` | Limite de classificação. Score maior ou igual a 0,5 é classificado como fake. |
| `Amostras`, `Fakes`, `Reais` | Tamanho e composição do conjunto avaliado. |
| `TP` | Imagem fake classificada como fake. |
| `FP` | Imagem real classificada incorretamente como fake. |
| `FN` | Imagem fake classificada incorretamente como real. |
| `TN` | Imagem real classificada como real. |
| `Acurácia` | Proporção total de classificações corretas. |
| `Precisão` | Entre as imagens classificadas como fake, quantas realmente eram fake. |
| `Recall` | Entre as imagens fake, quantas foram detectadas. |
| `Especificidade` | Entre as imagens reais, quantas foram reconhecidas como reais. |
| `F1-score` | Equilíbrio entre precisão e recall. |
| `AUC-ROC` | Capacidade de separar reais e fakes sem depender apenas do threshold 0,5. |

## Como analisar os modelos

1. Compare o `F1-score` de cada detector dentro da mesma resolução.
2. Compare `Recall` e `Especificidade` juntos. Recall alto com especificidade baixa significa muitos falsos positivos em imagens reais.
3. Compare o mesmo detector entre `original`, `256x256`, `512x512` e `1024x1024` para observar o impacto da resolução.
4. Compare os ensembles com os detectores individuais. Um ensemble é útil apenas se melhorar os resultados sem criar muitos falsos positivos.
5. Use `AUC-ROC` para verificar a separação geral dos scores e `F1-score` para avaliar a decisão prática usando threshold 0,5.

## Dataset e referências

O experimento usa o conjunto de avaliação [CommunityForensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval), que contém imagens reais e sintéticas, metadados do gerador, arquitetura e fonte das imagens reais. O uso é acadêmico e não comercial, conforme a licença do dataset.

Referência principal:

```bibtex
@InProceedings{Park_2025_CVPR,
  author    = {Park, Jeongsoo and Owens, Andrew},
  title     = {Community Forensics: Using Thousands of Generators to Train Fake Image Detectors},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025},
  pages     = {8245--8257}
}
```
