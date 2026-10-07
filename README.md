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
│
├── datasets/                          # dados de vídeo (gitignored)
│
├── requirements.txt
├── .gitignore
├── run.py                             # Destinado a inicar a detecção (deve ser inicializado no terminal)
└── README.md
```

## O que cada pasta faz

### `external/`

Código de terceiros incluído via **git submodule**. Mantém separação clara entre código original (citado/referenciado) e código próprio — importante para a integridade acadêmica do TCC. Nunca editar diretamente; se precisar de mudanças, criar wrapper em `src/detection/`.

### `weights/`

Pesos pré-treinados dos modelos de detecção. Não versionado no Git (arquivos grandes); cada membro do grupo baixa manualmente seguindo instruções do `setup.sh`/README.

### `src/config.py`

Único lugar do projeto com caminhos absolutos, calculados a partir da raiz do repositório (via `os.path.dirname(os.path.abspath(__file__))`). Evita caminhos hardcoded específicos de cada máquina/usuário.

### `src/detection/`

Modelos de detecção de imagens/vídeos gerados por IA — o coração do ensemble. Todo modelo novo implementa a interface `Detector` (`base.py`), com um método `predict(image_path) -> float` retornando o score de probabilidade.

**Atual:** CNNDetection (Wang et al., CVPR 2020), ResNet50 treinado em imagens geradas por GAN. Escolhido propositalmente por suas limitações documentadas de generalização contra modelos de difusão — serve de baseline para o eixo de generalização da pesquisa.

### `src/aggregation/`

Métodos de agregação dos scores retornados pelos múltiplos detectores. Todo método novo implementa a interface `Aggregator` (`base.py`), com um método `aggregate(scores: dict) -> float`.

Este é o módulo central do segundo eixo de pesquisa: testar se a agregação de scores consegue compensar a baixa generalização individual dos detectores.

### Analisar uma imagem diretamente

1. Coloque imagens em `datasets/input/`.
2. Com o ambiente virtual ativo, execute:

```bash
python run.py
```

O script carrega todos os modelos e mostra o score de probabilidade de a imagem ser
sintética. Arquivos em `datasets/` continuam fora do Git.

Além do resultado principal, o executor mostra média, mediana, percentis 75 e
90, e máximo dos scores dos modelos. Compare esses valores em um conjunto de
imagens reais e sintéticos antes de escolher a agregação definitiva.

O executor usa todos os modelos como detector padrão. Na primeira
execução, o CLIP pode baixar automaticamente o backbone pré-treinado; mantenha
conexão com a internet.

### `datasets/`

Imagens de entrada (reais e gerados por IA), organizados por gerador de origem — importante para rastrear qual amostra veio de qual modelo generativo no teste de generalização.

## Por que essa estrutura escala bem

- **Adicionar um novo modelo de detecção:** criar `detection/novo_modelo.py` implementando `Detector`, adicionar na lista passada ao `pipeline` — zero mudança no resto do código.
- **Testar um novo método de agregação:** criar novo `Aggregator`, trocar uma linha na chamada do pipeline.
- A arquitetura reflete diretamente o desenho experimental do TCC (detecção × agregação como eixos independentes e combináveis).
