PYTHON := .venv/Scripts/python.exe
PIP := .venv/Scripts/pip.exe

.PHONY: help

help:
	@echo.
	@echo "Comandos disponiveis:"
	@echo.
	@echo "  make setup       Configura o projeto completo"
	@echo "  make venv        Cria o ambiente virtual"
	@echo "  make install     Instala as dependencias"
	@echo "  make submodules  Inicializa/atualiza os submodules"
	@echo "  make update      Atualiza os submodules"
	@echo "  make run         Executa a aplicacao"
	@echo "  make test        Executa os testes"
	@echo "  make clean       Remove caches do Python"
	@echo "  make check       Verifica o ambiente"
	@echo.


.PHONY: setup
setup: venv submodules install
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
