"""Leitura de arquivos dentro de um zip remoto sem baixar o zip inteiro.

Usa requisições HTTP com cabeçalho Range: o zipfile lê o índice no fim do zip
e depois só os trechos das imagens pedidas. Serve para tirar poucas imagens de
zips de 0,8 a 21 GB (teste do ProGAN e dados do B-Free).
"""

import io
import time
import urllib.request
import zipfile


class ArquivoRemoto(io.RawIOBase):
    def __init__(self, url):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resposta:
            self.tamanho = int(resposta.headers["Content-Length"])
            self.url = resposta.url
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.tamanho}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.tamanho - self.pos
        n = min(n, self.tamanho - self.pos)
        if n <= 0:
            return b""
        fim = self.pos + n - 1
        for tentativa in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{fim}"})
                with urllib.request.urlopen(req, timeout=120) as resposta:
                    dados = resposta.read()
                break
            except OSError:  # inclui URLError, timeout e conexão derrubada
                if tentativa == 4:
                    raise
                time.sleep(5 * (tentativa + 1))
        self.pos += len(dados)
        return dados

    def readinto(self, b):
        dados = self.read(len(b))
        b[: len(dados)] = dados
        return len(dados)


def abrir_zip(url):
    return zipfile.ZipFile(io.BufferedReader(ArquivoRemoto(url), buffer_size=1 << 20))
