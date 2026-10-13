SHELL := cmd.exe

PYTHON := .venv/Scripts/python.exe
PIP := .venv/Scripts/pip.exe

CNNDETECTION_URL := https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1
CNNDETECTION_MD5 := 0c0bd6f572eaec0e8ea8ed8b033969dd
BFREE_URL := https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip
BFREE_MD5 := f3f53fa647848b16cf81c913f148a198

.PHONY: help

help:
	@echo.
	@echo "Comandos disponiveis:"
	@echo.
	@echo "  make setup       Configura o projeto completo"
	@echo "  make venv        Cria o ambiente virtual"
	@echo "  make install     Instala as dependencias"
	@echo "  make weights     Baixa os pesos dos modelos"
	@echo "  make submodules  Inicializa/atualiza os submodules"
	@echo "  make update      Atualiza os submodules"
	@echo "  make run         Executa a aplicacao"
	@echo "  make test        Executa os testes"
	@echo "  make clean       Remove caches do Python"
	@echo "  make check       Verifica o ambiente"
	@echo.


.PHONY: setup
setup: venv submodules install weights
	@echo.
	@echo "========================================"
	@echo " Projeto configurado com sucesso!"
	@echo "========================================"
	@echo.

.PHONY: venv
venv:
	@if not exist ".venv" ( \
		echo Criando ambiente virtual... && \
		python -m venv .venv \
	) else ( \
		echo Ambiente virtual ja existe. \
	)

.PHONY: install
install:
	@echo Instalando dependencias...
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: submodules
submodules:
	@echo Inicializando submodules...
	git submodule update --init --recursive

.PHONY: update
update:
	@echo Atualizando submodules...
	git submodule update --remote --merge

.PHONY: weights
weights:
	@if not exist "weights" mkdir weights
	@if not exist "weights\blur_jpg_prob0.5.pth" ( \
		echo Baixando pesos do CNNDetection, cerca de 280 MB... && \
		curl -L --fail -o weights\blur_jpg_prob0.5.tmp "$(CNNDETECTION_URL)" && \
		certutil -hashfile weights\blur_jpg_prob0.5.tmp MD5 | findstr /i "$(CNNDETECTION_MD5)" && \
		move /y weights\blur_jpg_prob0.5.tmp weights\blur_jpg_prob0.5.pth > nul \
	) else ( \
		echo Pesos do CNNDetection ja existem. \
	)
	@if not exist "weights\BFREE_dino2reg4\model_epoch_best.pth" ( \
		echo Baixando pesos do B-Free, cerca de 320 MB... && \
		curl -L --fail -o weights\BFREE_dino2reg4.zip "$(BFREE_URL)" && \
		certutil -hashfile weights\BFREE_dino2reg4.zip MD5 | findstr /i "$(BFREE_MD5)" && \
		tar -xf weights\BFREE_dino2reg4.zip -C weights && \
		del weights\BFREE_dino2reg4.zip \
	) else ( \
		echo Pesos do B-Free ja existem. \
	)

.PHONY: run
run:
	$(PYTHON) run.py

.PHONY: check
check:
	@echo.
	@echo Verificando ambiente...
	@echo.
	@$(PYTHON) --version
	@echo.
	@$(PIP) --version
	@echo.
	@echo Submodules:
	@git submodule status
	@echo.

.PHONY: clean
clean:
	@echo Removendo caches do Python...
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f"
	@echo Limpeza concluida.
