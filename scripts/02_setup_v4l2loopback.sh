#!/usr/bin/env bash
set -e

# Cores para terminal
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   [ETAPA 2/6] Configuração da Câmera Virtual V4L2   ${NC}"
echo -e "${CYAN}======================================================${NC}"

# 1. Verificar privilégios sudo
if ! sudo -v; then
    echo -e "${RED}Erro: Privilégios sudo são necessários para configurar módulos de kernel.${NC}"
    exit 1
fi

# 2. Instalar v4l2loopback via DKMS se ainda não estiver instalado
echo -e "${YELLOW}--> Verificando pacotes v4l2loopback-dkms e v4l2loopback-utils...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq v4l2loopback-dkms v4l2loopback-utils v4l-utils

# 3. Configurar parâmetros do v4l2loopback (/etc/modprobe.d/v4l2loopback.conf)
echo -e "${YELLOW}--> Configurando /etc/modprobe.d/v4l2loopback.conf...${NC}"
sudo tee /etc/modprobe.d/v4l2loopback.conf > /dev/null << 'CONF'
options v4l2loopback devices=3 video_nr=70,71,72 card_label="Iriun Webcam","OBS Virtual Cam","Intel NPU Enhanced Webcam" exclusive_caps=1,1,1 max_buffers=6
CONF

# 4. Configurar inicialização automática no boot (/etc/modules-load.d/v4l2loopback.conf)
echo -e "${YELLOW}--> Garantindo carregamento automático do módulo no boot...${NC}"
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf > /dev/null

# 5. Carregar ou recarregar o módulo
echo -e "${YELLOW}--> Aplicando configurações no módulo de kernel v4l2loopback...${NC}"
if lsmod | grep -q v4l2loopback; then
    echo -e "${CYAN}Módulo v4l2loopback já carregado. Verificando /dev/video72...${NC}"
    if [ ! -e "/dev/video72" ]; then
        echo -e "${YELLOW}Recarregando v4l2loopback para registrar /dev/video72...${NC}"
        sudo modprobe -r v4l2loopback || echo -e "${YELLOW}(Módulo em uso por outra aplicação, mantendo atual)${NC}"
        sudo modprobe v4l2loopback || true
    fi
else
    sudo modprobe v4l2loopback
fi

# 6. Validar existência do dispositivo /dev/video72
if [ -e "/dev/video72" ]; then
    echo -e "${GREEN}✓ Sucesso! Câmera virtual criada em /dev/video72:${NC}"
    v4l2-ctl --device=/dev/video72 --info | head -n 8 || true
else
    echo -e "${RED}Aviso: /dev/video72 não encontrado. Pode ser necessário reiniciar o sistema se o módulo estava travado.${NC}"
fi

echo -e "${GREEN}✓ Etapa 2 concluída com sucesso!${NC}\n"
