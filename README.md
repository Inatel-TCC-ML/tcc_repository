# Estrutura do Projeto

Este projeto segue uma arquitetura modular baseada no padrão **Strategy**: cada etapa do pipeline (extração de keyframes, detecção, agregação de scores) é uma interface intercambiável, permitindo adicionar novos métodos sem alterar o restante do código.

## Visão geral do pipeline

```
Vídeo → Extração de Keyframes → Detecção (1..N modelos) → Agregação de Scores → Resultado Final
```

## Árvore de diretórios

```
tcc_repository/
├── external/
│   └── CNNDetection/                  # submodule (código de terceiros - Wang et al., CVPR 2020)
│
├── weights/                           # pesos dos modelos baixados (gitignored)
│   └── cnndetection_blur_jpg.pth
│
├── src/
│   ├── config.py                      # caminhos centralizados do projeto
│   │
│   ├── extraction/                    # métodos de extração de keyframes
│   │   ├── __init__.py
│   │   ├── base.py                    # interface abstrata KeyframeExtractor
│   │   ├── ssim_extractor.py          # extração via MSSIM
│   │   └── uniform_extractor.py       # extração a cada N frames (futuro)
│   │
│   ├── detection/                     # modelos de detecção (ensemble)
│   │   ├── __init__.py
│   │   ├── base.py                    # interface abstrata Detector
│   │   ├── cnndetection.py            # wrapper do CNNDetection
│   │   └── your_next_model.py         # futuros modelos de detecção
│   │
│   ├── aggregation/                   # métodos de agregação de score
│   │   ├── __init__.py
│   │   ├── base.py                    # interface abstrata Aggregator
│   │   ├── mean_aggregator.py         # média simples
│   │   ├── weighted_aggregator.py     # média ponderada (futuro)
│   │   └── voting_aggregator.py       # votação majoritária (futuro)
│   │
│   ├── pipeline.py                    # orquestra: extração → detecção → agregação
│   │
│   └── interface/                     # interface para rodar o pipeline
│       ├── __init__.py
│       ├── cli.py                     # interface de linha de comando
│       └── api.py                     # opcional: API simples (FastAPI/Flask)
│
├── datasets/                          # dados de vídeo (gitignored)
├── output/                            # keyframes extraídos e resultados (gitignored)
├── notebooks/                         # experimentos exploratórios, análises
├── tests/                             # testes unitários por módulo
│
├── setup.sh                           # automatiza clone do submodule + setup inicial
├── requirements.txt
├── .gitignore
└── README.md
```

## O que cada pasta faz

### `external/`

Código de terceiros incluído via **git submodule**. Mantém separação clara entre código original (citado/referenciado) e código próprio — importante para a integridade acadêmica do TCC. Nunca editar diretamente; se precisar de mudanças, criar wrapper em `src/detection/`.

### `weights/`

Pesos pré-treinados dos modelos de detecção. Não versionado no Git (arquivos grandes); cada membro do grupo baixa manualmente seguindo instruções do `setup.sh`/README.

### `src/config.py`

Único lugar do projeto com caminhos absolutos, calculados a partir da raiz do repositório (via `os.path.dirname(os.path.abspath(__file__))`). Evita caminhos hardcoded específicos de cada máquina/usuário.

### `src/extraction/`

Métodos de extração de keyframes de vídeo. Todo método novo implementa a interface `KeyframeExtractor` (`base.py`), com um método `extract(video_path) -> list[str]`.

**Atual:** extração via similaridade estrutural (MSSIM), comparando cada frame ao último keyframe salvo.

### `src/detection/`

Modelos de detecção de imagens/vídeos gerados por IA — o coração do ensemble. Todo modelo novo implementa a interface `Detector` (`base.py`), com um método `predict(image_path) -> float` retornando o score de probabilidade.

**Atual:** CNNDetection (Wang et al., CVPR 2020), ResNet50 treinado em imagens geradas por GAN. Escolhido propositalmente por suas limitações documentadas de generalização contra modelos de difusão — serve de baseline para o eixo de generalização da pesquisa.

### `src/aggregation/`

Métodos de agregação dos scores retornados pelos múltiplos detectores. Todo método novo implementa a interface `Aggregator` (`base.py`), com um método `aggregate(scores: dict) -> float`.

Este é o módulo central do segundo eixo de pesquisa: testar se a agregação de scores consegue compensar a baixa generalização individual dos detectores.

### `src/pipeline.py`

Orquestrador central. Recebe um extrator, uma lista de detectores e um agregador, e executa o fluxo completo: extração → detecção (em cada keyframe, por cada modelo) → agregação → resultado final. É o único ponto do código que "conhece" o fluxo inteiro — módulos individuais não precisam saber uns dos outros.

### `src/interface/`

Ponto de entrada para rodar o pipeline sem precisar mexer no código-fonte. Recebe um caminho de vídeo e devolve o resultado (scores por frame, score agregado final).

```bash
python -m src.interface.cli "datasets/ai/video1.mp4"
```

### Analisar uma imagem diretamente

1. Coloque **uma única** imagem em `datasets/input/`.
2. Com o ambiente virtual ativo, execute:

```bash
python run_detector.py
```

O script carrega o UnivFD e mostra o score de probabilidade de a imagem ser
sintética. Arquivos em `datasets/` continuam fora do Git.

### Analisar um vídeo pelos keyframes

1. Coloque **um único** vídeo em `datasets/videos/input/`.
2. Com o ambiente virtual ativo, execute:

```bash
python run_video_detector.py
```

O pipeline extrai keyframes por MSSIM, grava-os em
`output/keyframes/<nome-do-video>/`, aplica o CNNDetection em cada imagem e
retorna a média dos scores como score final do vídeo.

Além do resultado principal, o executor mostra média, mediana, percentis 75 e
90, e máximo dos scores dos keyframes. Compare esses valores em um conjunto de
vídeos reais e sintéticos antes de escolher a agregação definitiva.

O executor usa o UnivFD (CLIP ViT-L/14) como detector padrão. Na primeira
execução, o CLIP pode baixar automaticamente o backbone pré-treinado; mantenha
conexão com a internet. O CNNDetection continua disponível como baseline para
os experimentos comparativos.

### `datasets/`

Vídeos de entrada (reais e gerados por IA), organizados por gerador de origem — importante para rastrear qual amostra veio de qual modelo generativo no teste de generalização.

### `output/`

Keyframes extraídos e resultados de execução do pipeline. Organizado por vídeo processado (ex: `output/{nome_do_video}/frame_000000.jpg`).

### `notebooks/`

Experimentos exploratórios, testes de threshold, análises de resultados — código descartável/investigativo, separado do código de produção em `src/`.

### `tests/`

Testes unitários, um por módulo (`test_extraction.py`, `test_detection.py`, `test_aggregation.py`, etc.), garantindo que cada interface implementada respeita o contrato esperado.

## Por que essa estrutura escala bem

- **Adicionar um novo modelo de detecção:** criar `detection/novo_modelo.py` implementando `Detector`, adicionar na lista passada ao `pipeline` — zero mudança no resto do código.
- **Testar um novo método de agregação:** criar novo `Aggregator`, trocar uma linha na chamada do pipeline.
- **Comparar extratores de keyframe diferentes:** mesma lógica — trocar apenas o `extractor` passado ao pipeline.
- A arquitetura reflete diretamente o desenho experimental do TCC (extração × detecção × agregação como eixos independentes e combináveis), o que facilita a escrita da metodologia no artigo.
