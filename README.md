# Estrutura do Projeto

Este projeto segue uma arquitetura modular baseada no padrão **Strategy**: cada etapa do pipeline (detecção, agregação de scores) é uma interface intercambiável, permitindo adicionar novos métodos sem alterar o restante do código.

## Visão geral do pipeline

```
Imagem → Detecção (1..N modelos) → Agregação de Scores → Resultado Final
```

O `run.py` analisa as imagens de `datasets/input/` com todos os detectores e agrega os scores.

## Como configurar e rodar

Requisitos: Windows 10 ou 11, Python, Git e `make` (o `curl`, o `tar` e o `certutil` já vêm no Windows). Os comandos podem ser rodados no PowerShell ou no Prompt de Comando.

### Primeira vez

```
git clone https://github.com/Inatel-TCC-ML/tcc_repository.git
cd tcc_repository
make setup
```

O `make setup` faz tudo em sequência:

1. `make venv`: cria o ambiente virtual em `.venv/`;
2. `make submodules`: baixa o código dos detectores em `external/`;
3. `make install`: instala as dependências do `requirements.txt`;
4. `make weights`: baixa os pesos dos modelos em `weights/` (~600 MB no total; veja a tabela abaixo).

Depois é só colocar as imagens em `datasets/input/` e rodar:

```
make run
```

### Depois de um `git pull`

O `git pull` não traz os submódulos novos nem os pesos, que ficam fora do Git. Sempre que alguém adicionar um detector, rode de novo:

```
git pull
make setup
make run
```

O `make setup` pode ser repetido sem problema: o ambiente virtual que já existe é mantido e os pesos que já foram baixados não são baixados de novo.

### Pesos dos modelos

| Detector | Arquivo | De onde vem |
|---|---|---|
| CNNDetection | `weights/blur_jpg_prob0.5.pth` (~280 MB) | baixado pelo `make weights` |
| UnivFD | `external/UniversalFakeDetect/pretrained_weights/fc_weights.pth` | vem no submódulo |
| NPR | `external/NPR-DeepfakeDetection/NPR.pth` | vem no submódulo |
| B-Free | `weights/BFREE_dino2reg4/` (`config.yaml` + `model_epoch_best.pth`, ~320 MB) | baixado pelo `make weights` |

O `make weights` confere o MD5 de cada download e para com erro se o arquivo vier corrompido. Na primeira execução do `run.py`, o UnivFD também baixa o backbone do CLIP para `~/.cache/clip`, então é preciso estar com internet.

### Problemas comuns

- **`make submodules` falha no CNNDetection:** o submódulo usa um endereço SSH. Quem não tem chave SSH no GitHub pode baixar por HTTPS com:
  `git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive`
- **O B-Free é lento:** ele sempre processa 5 recortes de 504×504 por imagem. Em CPU são uns 17 a 20 s por imagem; para muitas imagens, use uma máquina com GPU (o código usa a GPU sozinho quando ela existe).
- **`results.csv` é sobrescrito a cada execução:** copie o arquivo para outro lugar antes de rodar de novo.

## Árvore de diretórios

```
tcc_repository/
├── external/                          # submódulos (código de terceiros, não editar)
│   ├── CNNDetection/                  # Wang et al., CVPR 2020
│   ├── UniversalFakeDetect/           # Ojha et al., CVPR 2023
│   ├── NPR-DeepfakeDetection/         # Tan et al., CVPR 2024
│   └── B-Free/                        # Guillaro et al., CVPR 2025
│
├── weights/                           # pesos baixados pelo make weights (gitignored)
│   ├── blur_jpg_prob0.5.pth
│   └── BFREE_dino2reg4/
│
├── src/
│   ├── config.py                      # caminhos centralizados do projeto
│   │
│   ├── detection/                     # modelos de detecção (ensemble)
│   │   ├── __init__.py
│   │   ├── base.py                    # interface abstrata Detector
│   │   ├── cnndetection.py            # wrapper do CNNDetection
│   │   ├── universalFakeDetect.py     # wrapper do UnivFD
│   │   ├── nprDeepfakeDetect.py       # wrapper do NPR
│   │   └── bFree.py                   # wrapper do B-Free
│   │
│   └── aggregation/                   # métodos de agregação de score
│       ├── __init__.py
│       ├── base.py                    # interface abstrata Aggregator
│       └── score_aggregators.py       # média, mediana, percentil e máximo
│
├── datasets/                          # imagens de entrada (gitignored)
│   ├── input/
│   └── experimento/                   # imagens baixadas pelo experimento oficial
│
├── experimento/                       # desenho e scripts do experimento oficial
├── output/                            # registro de cada experimento já rodado
│
├── Makefile                           # comandos de configuração e execução
├── requirements.txt
├── .gitignore
├── run.py                             # inicia a detecção (rodar no terminal)
└── README.md
```

## O que cada pasta faz

### `external/`

Código de terceiros incluído via **git submodule**. Mantém separação clara entre código original (citado/referenciado) e código próprio — importante para a integridade acadêmica do TCC. Nunca editar diretamente; se precisar de mudanças, criar wrapper em `src/detection/`.

### `weights/`

Pesos pré-treinados dos modelos de detecção. Não versionado no Git (arquivos grandes); o `make weights` baixa tudo (veja [Pesos dos modelos](#pesos-dos-modelos)).

### `src/config.py`

Único lugar do projeto com caminhos absolutos, calculados a partir da raiz do repositório (via `os.path.dirname(os.path.abspath(__file__))`). Evita caminhos hardcoded específicos de cada máquina/usuário.

### `src/detection/`

Modelos de detecção de imagens geradas por IA — o coração do ensemble. Todo modelo novo implementa a interface `Detector` (`base.py`), com um método `predict(image_path) -> float` retornando o score de probabilidade.

**Atuais:**

- **CNNDetection** (Wang et al., CVPR 2020): ResNet50 treinado em imagens geradas por GAN (ProGAN). Escolhido propositalmente por suas limitações documentadas de generalização contra modelos de difusão — serve de baseline para o eixo de generalização da pesquisa.
- **UnivFD** (Ojha et al., CVPR 2023): camada linear sobre as características do CLIP ViT-L/14, treinada em ProGAN.
- **NPR** (Tan et al., CVPR 2024): ResNet50 que analisa os artefatos de upsampling entre pixels vizinhos, treinado em ProGAN.
- **B-Free** (Guillaro et al., CVPR 2025): ViT DINOv2 treinado em imagens de difusão (Stable Diffusion 2.1), com treino pensado para evitar vieses de conteúdo, formato e resolução. É o único dos quatro que não foi treinado em ProGAN.

Os submódulos que têm uma pasta `networks/` (CNNDetection, NPR e B-Free) são carregados pelo caminho do arquivo via `importlib`, para um não sobrescrever o outro.

### `src/aggregation/`

Métodos de agregação dos scores retornados pelos múltiplos detectores. Os usados hoje pelo `run.py` ficam em `score_aggregators.py` e recebem a lista de scores: média, mediana, percentil (75 e 90) e máximo.

Este é o módulo central do segundo eixo de pesquisa: testar se a agregação de scores consegue compensar a baixa generalização individual dos detectores.

### Analisar uma imagem diretamente

1. Coloque imagens em `datasets/input/`.
2. Execute `make run` (ou, com o ambiente virtual ativo, `python run.py`).

O script carrega todos os modelos e grava em `results.csv` o score de cada modelo para cada imagem (probabilidade de a imagem ser sintética). Arquivos em `datasets/` continuam fora do Git.

Além dos scores de cada modelo, o `results.csv` traz média, mediana, percentis 75 e 90, e máximo dos scores. Compare esses valores em um conjunto de imagens reais e sintéticas antes de escolher a agregação definitiva.

### `datasets/`

Imagens de entrada (reais e geradas por IA), organizadas por gerador de origem — importante para rastrear qual amostra veio de qual modelo generativo no teste de generalização.

### `experimento/`

Experimento oficial do artigo: amostra fixa de 2.400 imagens (24 geradores, 50 reais e 50 de IA em cada), scripts para baixar as imagens, rodar os detectores e gerar as tabelas. Tudo explicado em [experimento/README.md](experimento/README.md).

### `output/`

Um registro por experimento já rodado (`AAAA-MM-DD_<nome>/`, com README, resultados e scripts), listados em [output/README.md](output/README.md).

## Por que essa estrutura escala bem

- **Adicionar um novo modelo de detecção:** criar `src/detection/novo_modelo.py` implementando `Detector`, colocar os caminhos em `config.py`, exportar em `__init__.py` e adicionar na lista `detectors` do `run.py`. Se o modelo precisar de pesos fora do Git, incluir o download no alvo `weights` do `Makefile`.
- **Testar um novo método de agregação:** criar um novo agregador em `src/aggregation/` e chamá-lo no `run.py`.
- A arquitetura reflete diretamente o desenho experimental do TCC (detecção × agregação como eixos independentes e combináveis).
