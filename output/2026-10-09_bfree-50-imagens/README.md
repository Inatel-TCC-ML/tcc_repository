# B-Free em 50 imagens de difusão, comerciais e GAN (09/10/2026)

Teste pequeno só com o B-Free, para servir de referência quando o grupo rodar o experimento grande: quem refizer com as mesmas imagens deve obter os mesmos números. **É exploratório (50 imagens); os números não devem ir para o artigo como resultado final.**

## Ambiente

- Branch `B-Free`, detector `BFreeDetector` (`src/detection/bFree.py`), pesos oficiais `BFREE_dino2reg4` (MD5 do zip `f3f53fa647848b16cf81c913f148a198`).
- Inferência igual à do código oficial (`main_bfree_single.py`): imagem inteira, normalização ImageNet, média dos logits de 5 recortes de 504×504 (centro e quatro cantos; imagem menor que 504 é repetida lado a lado para completar). Score = sigmoid do logit.
- Notebook com Intel Core i5-1135G7, 16 GB de RAM, sem GPU; Windows 11, Python 3.13, torch 2.14.1+cpu, timm 1.0.30.
- Limiar 0,5 (score ≥ 0,5 = IA).

## Dados

Todas as imagens vêm do [Community Forensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval), split `CompEval`, o mesmo do teste das 89 imagens ([2026-10-08_cf-89-imagens](../2026-10-08_cf-89-imagens/README.md)). A lista completa, com a origem de cada imagem, está em `gabarito.csv`.

| Grupo | Qtd | Formato | Tamanho | Origem |
|---|---|---|---|---|
| Reais (fotos do LAION) | 20 | JPEG | 300 a 1024 px | as 20 primeiras reais da amostra de 89 |
| Difusão: FLUX-schnell, IdeogramV2, SDXL (LCM_lora_sdxl), Midjourney 6.1 | 4 de cada = 16 | PNG | 512 a 1024 px | `_01` a `_04` de cada gerador da amostra de 89 |
| Comercial: Adobe Firefly 3 | 4 | **JPEG** (já vem assim no dataset) | 2048 px | baixadas (índices 41000 a 41003) |
| GAN: DFGAN | 5 | PNG | 256 px | baixadas (índices 100 a 104) |
| GAN: GALIP | 5 | PNG | 224 px | baixadas (índices 15000 a 15004) |

DFGAN e GALIP são GANs de texto para imagem. O B-Free foi treinado só com imagens de difusão (Stable Diffusion 2.1), então **nenhum dos 7 geradores fez parte do treino dele**.

Para checar o viés de formato visto no NPR, as 26 imagens de IA em PNG também foram convertidas para JPEG qualidade 95 e rodadas de novo. No total foram 76 passadas do modelo.

## Resultados (`avaliacao.txt`)

### 1. Formato original (50 imagens)

| Grupo | Acertos | Score médio | Mín | Máx |
|---|---|---|---|---|
| Reais (LAION, JPEG) | 17/20 | 0,191 | 0,001 | 0,965 |
| FLUX-schnell (PNG) | 3/4 | 0,751 | 0,090 | 0,992 |
| IdeogramV2 (PNG) | 3/4 | 0,724 | 0,039 | 0,998 |
| SDXL / LCM_lora_sdxl (PNG) | 4/4 | 0,994 | 0,986 | 0,998 |
| Midjourney 6.1 (PNG) | 4/4 | 0,886 | 0,737 | 0,957 |
| Firefly 3 (JPEG) | 4/4 | 0,953 | 0,908 | 1,000 |
| DFGAN (PNG) | 5/5 | 0,974 | 0,946 | 0,997 |
| GALIP (PNG) | 4/5 | 0,787 | 0,406 | 0,997 |

**Acurácia 44/50 = 88%, AP 0,957, AUC 0,935.** Acertou 27 de 30 imagens de IA e 17 de 20 reais.

### 2. Tudo em JPEG (reais + IA convertidas para JPEG 95 + Firefly 3)

**Acurácia 45/50 = 90%, AP 0,957, AUC 0,927.** Os acertos por gerador ficaram iguais aos do formato original, a não ser pelo GALIP, que foi de 4/5 para 5/5 (tabela completa em `avaliacao.txt`).

### 3. As mesmas imagens de IA em PNG e em JPEG 95

| Gerador | Acertos PNG | Acertos JPEG 95 | Score médio PNG | Score médio JPEG 95 |
|---|---|---|---|---|
| FLUX-schnell | 3/4 | 3/4 | 0,751 | 0,684 |
| IdeogramV2 | 3/4 | 3/4 | 0,724 | 0,740 |
| SDXL | 4/4 | 4/4 | 0,994 | 0,995 |
| Midjourney 6.1 | 4/4 | 4/4 | 0,886 | 0,852 |
| DFGAN | 5/5 | 5/5 | 0,974 | 0,980 |
| GALIP | 4/5 | 5/5 | 0,787 | 0,817 |

### Erros

- **Reais marcadas como IA (3):** `real_012` (0,965, castelo de papelão em fundo branco), `real_010` (0,761, óculos de realidade virtual que parece render 3D de produto) e `real_007` (0,519, capacho com moldura branca de catálogo). As três são fotos de produto bem editadas, não fotos comuns.
- **IA marcadas como real:** `fake_FLUX-schnell_03` (0,090 em PNG, 0,030 em JPEG), `fake_IdeogramV2_01` (0,039 em PNG, 0,030 em JPEG) e `fake_GALIP_01` (0,406 em PNG, 0,560 em JPEG, ou seja, perto do limiar nos dois casos).

### Tempo

Média de 10,9 s por imagem em CPU (mín. 7,5 s, máx. 19,4 s); total de 13,7 min para as 76 passadas. Carregar o modelo leva uns 15 s. O tempo quase não depende do tamanho da imagem, porque o modelo sempre processa 5 recortes de 504×504.

## Conclusões

- **O B-Free detecta bem geradores que não viu no treino:** 27 de 30 imagens de IA, incluindo geradores comerciais recentes (Midjourney 6.1, Firefly 3) e GANs. No teste das 89 imagens, CNNDetection e UnivFD quase não detectaram esses mesmos geradores de difusão.
- **Não mostrou o viés de formato do NPR.** Convertendo as imagens de IA de PNG para JPEG 95, os acertos não caíram (23/26 em PNG, 24/26 em JPEG), e o score médio de cada gerador mudou no máximo 0,07. Ele também acertou as 4 do Firefly 3, que já vêm em JPEG. No NPR, a mesma conversão derrubou os acertos de 32/40 para 17/40.
- **O ponto fraco foram os falsos positivos:** 3 de 20 reais marcadas como IA, todas fotos de produto editadas. Vale olhar isso no experimento grande, porque o LAION tem muitas imagens de catálogo.
- **Limitações:** só 50 imagens (4 ou 5 por gerador), uma única execução e o limiar fixo em 0,5. Só foi testado JPEG qualidade 95, que é uma compressão leve; o NPR já caía nessa qualidade, mas uma compressão mais forte (ex.: qualidade 75) ainda não foi testada no B-Free.

## Como refazer

Da raiz do projeto, na branch `B-Free`, com o `.venv` ativo e os pesos baixados (`make weights`):

1. Ter a amostra de 89 imagens em `datasets/input/` (`../2026-10-08_cf-89-imagens/scripts/baixar_amostra.py`).
2. `python output/2026-10-09_bfree-50-imagens/scripts/montar_conjunto.py`: copia as 36 imagens da amostra, baixa as 14 de Firefly 3, DFGAN e GALIP e gera as versões JPEG 95 em `datasets/bfree_50/`.
3. `python output/2026-10-09_bfree-50-imagens/scripts/rodar_bfree.py resultados_bfree.csv`: roda só o B-Free nas 76 imagens.
4. `python output/2026-10-09_bfree-50-imagens/scripts/avaliar_bfree.py resultados_bfree.csv`: gera as tabelas.

Os scores devem bater com `resultados_bfree.csv` até a 3ª casa decimal; diferenças mínimas podem aparecer entre CPU e GPU.

## Arquivos

- `resultados_bfree.csv`: score e tempo de cada uma das 76 passadas (colunas `variante` = original ou jpeg95, `classe`, `gerador`, `arquitetura`, `formato_testado`). Também foi copiado para o `results.csv` da raiz do projeto.
- `avaliacao.txt`: tabelas geradas por `avaliar_bfree.py`.
- `gabarito.csv`: as 50 imagens, com classe, gerador, formato e índice no dataset.
- `log_execucao.txt`: saída da execução.
- `scripts/`: `montar_conjunto.py`, `rodar_bfree.py` e `avaliar_bfree.py`.
