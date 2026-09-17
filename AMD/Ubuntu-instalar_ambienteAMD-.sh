#!/bin/bash

echo "🚀 Iniciando a configuração com o gerenciador UV (Python 3.11 isolado)..."

# 1. Carrega o ambiente do uv caso não esteja no PATH
export PATH="$HOME/.local/bin:$PATH"

# 2. Remove venv anterior se houver lixo corrompido
if [ -d ".venv_linux" ]; then
    echo "🗑️ Removendo ambiente virtual antigo..."
    rm -rf .venv_linux
fi

# 3. Cria o ambiente virtual isolado usando Python 3.11 via uv
echo "📦 Criando ambiente virtual com Python 3.11..."
uv venv .venv_linux --python 3.11

# 4. Ativa o ambiente virtual
echo "🔄 Ativando o ambiente virtual..."
source .venv_linux/bin/activate

# 5. Instala PRIMEIRO o PyTorch otimizado para AMD ROCm (versão 6.0 estável)
echo "🔥 Instalando PyTorch para AMD ROCm..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# 6. Instala as dependências base do projeto
echo "📚 Instalando dependências base..."
uv pip install -r requirements.txt --no-deps

echo "✅ Ambiente configurado com sucesso e isolado com UV!"
echo "👉 Dica: Para ativar o ambiente depois, use: source .venv_linux/bin/activate"