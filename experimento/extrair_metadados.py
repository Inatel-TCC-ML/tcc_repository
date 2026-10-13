"""Gera o CSV de metadados do Community Forensics-Eval (51.836 imagens).

Lê só as colunas de texto dos 413 parquets do Hugging Face (HTTP Range), sem
baixar as imagens (os parquets somam 206 GB; a leitura leva ~8 min). O índice
`idx` é a posição da linha no split CompEval, a mesma usada pela API
datasets-server, por onde o baixar_dados.py busca cada imagem.

A coluna `tamanho_bloco` é o tamanho do bloco (row group) do parquet onde a
imagem está. A API recusa blocos acima de 300 MB, então o gerar_amostra.py só
sorteia imagens de blocos menores.

Não precisa rodar de novo: só serve para refazer o gerar_amostra.py.
Requer duckdb (pip install duckdb). Uso:
    python experimento/extrair_metadados.py <saida.csv>
"""

import csv
import sys

import duckdb

saida = sys.argv[1]
urls = [
    f"https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval/resolve/main/data/CompEval-{i:05d}-of-00413.parquet"
    for i in range(413)
]
lista = ", ".join(f"'{u}'" for u in urls)
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET threads TO 8;")

linhas = con.execute(f"""
    SELECT row_number() OVER (ORDER BY filename, file_row_number) - 1 AS idx, filename, file_row_number,
           image_name, format, resolution, mode, model_name, real_source,
           subset, split, label, architecture
    FROM read_parquet([{lista}], filename = true, file_row_number = true)
    ORDER BY filename, file_row_number
""").fetchall()
colunas = [d[0] for d in con.description]

# Tamanho de cada bloco: (arquivo) -> [(primeira linha, última linha + 1, bytes)]
blocos = {}
for arquivo, _, num_linhas, tamanho in con.execute(f"""
    SELECT file_name, row_group_id, any_value(row_group_num_rows), sum(total_compressed_size)
    FROM parquet_metadata([{lista}])
    GROUP BY file_name, row_group_id ORDER BY file_name, row_group_id
""").fetchall():
    lista_blocos = blocos.setdefault(arquivo, [])
    inicio = lista_blocos[-1][1] if lista_blocos else 0
    lista_blocos.append((inicio, inicio + num_linhas, tamanho))

with open(saida, "w", newline="", encoding="utf-8") as arquivo:
    escritor = csv.writer(arquivo)
    escritor.writerow(colunas + ["tamanho_bloco"])
    for linha in linhas:
        registro = dict(zip(colunas, linha))
        n = registro["file_row_number"]
        tamanho = next(t for a, b, t in blocos[registro["filename"]] if a <= n < b)
        escritor.writerow(list(linha) + [tamanho])
print(f"{len(linhas)} linhas -> {saida}")
