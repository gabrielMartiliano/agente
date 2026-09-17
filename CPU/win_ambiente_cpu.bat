echo [1/3] Limpando variaveis de certificado corrompidas do sistema...
set SSL_CERT_FILE=
set REQUESTS_CA_BUNDLE=

echo [2/3] Criando ambiente virtual limpo (.venv)...
python -m venv .venv

echo [3/3] Ativando e instalando dependencias via CPU...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --trusted-host download.pytorch.org --trusted-host pypi.org --trusted-host files.pythonhosted.org
pip install -r requirements.txt --no-deps --trusted-host pypi.org --trusted-host files.pythonhosted.org

echo ========================================================
echo Ambiente configurado com sucesso e pronto para uso!
echo ========================================================
pause