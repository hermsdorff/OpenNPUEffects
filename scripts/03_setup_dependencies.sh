#!/usr/bin/env bash
set -e

# Cores para terminal
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   [ETAPA 3/6] Instalação das Ferramentas e Venv     ${NC}"
echo -e "${CYAN}======================================================${NC}"

# 1. Instalar pacotes de sistema necessários
echo -e "${YELLOW}--> Instalando dependências de sistema via apt...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-venv \
    python3-pip \
    build-essential \
    libopenblas-dev \
    libportaudio2 \
    ffmpeg \
    v4l-utils \
    git \
    pipewire \
    pipewire-pulse \
    wireplumber

# 2. Criar diretório base global do projeto em /opt/npu-effects
OPT_DIR="/opt/npu-effects"
echo -e "${YELLOW}--> Criando estrutura compartilhada em $OPT_DIR...${NC}"
sudo mkdir -p "$OPT_DIR"/{bin,models/{audio,video},backgrounds,config,cameractrls}

# 3. Criar ou atualizar ambiente virtual Python compartilhado
VENV_DIR="$OPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}--> Criando ambiente virtual Python compartilhado em: $VENV_DIR...${NC}"
    sudo python3 -m venv "$VENV_DIR"
else
    echo -e "${CYAN}Ambiente virtual Python compartilhado já existente em: $VENV_DIR${NC}"
fi

# 4. Atualizar pip e instalar bibliotecas Python
echo -e "${YELLOW}--> Instalando pacotes Python essenciais no ambiente compartilhado (OpenVINO, OpenCV, etc.)...${NC}"
sudo "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel -q

sudo "$VENV_DIR/bin/pip" install -q \
    openvino \
    opencv-python-headless \
    numpy \
    scipy \
    sounddevice \
    pyvirtualcam

# Garantir permissão de leitura e execução para todos os usuários do sistema
sudo chmod -R a+rX "$OPT_DIR"

# 5. Testar detecção de hardware pelo OpenVINO
echo -e "${YELLOW}--> Testando suporte a NPU no OpenVINO Runtime...${NC}"
"$VENV_DIR/bin/python3" - << 'PYTEST'
import sys
import openvino as ov

core = ov.Core()
devices = core.available_devices
print(f"Dispositivos detectados pelo OpenVINO: {devices}")

if "NPU" in devices:
    print("\033[0;32m✓ SUCESSO: Intel NPU (AI Boost) ativa e disponível no OpenVINO!\033[0m")
else:
    print("\033[1;33mAviso: Dispositivo 'NPU' não reportado na lista imediata. Verifique se o módulo intel_vpu e o driver Level Zero foram carregados.\033[0m")
PYTEST

echo -e "${GREEN}✓ Etapa 3 concluída com sucesso!${NC}\n"
