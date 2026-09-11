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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
SOURCE_DIR="$BASE_DIR/src/v4l2loopback/source"
KERNEL_VER="$(uname -r)"

# 1. Verificar privilégios sudo
if ! sudo -v; then
    echo -e "${RED}Erro: Privilégios sudo são necessários para configurar módulos de kernel.${NC}"
    exit 1
fi

# 2. Instalar utilitários essenciais
echo -e "${YELLOW}--> Verificando pacotes utilitários do V4L2 e DKMS...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq v4l2loopback-utils v4l-utils dkms || true

# 3. Verificar/Instalar v4l2loopback no DKMS compatível com o kernel atual
echo -e "${YELLOW}--> Verificando módulo de kernel v4l2loopback para $KERNEL_VER...${NC}"

needs_patch=false

# Tentar instalar pacote apt padrão se não instalado
if ! dpkg -s v4l2loopback-dkms >/dev/null 2>&1; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq v4l2loopback-dkms || true
fi

# Verificar se o módulo está compilado e instalado no kernel atual
if ! dkms status 2>/dev/null | grep -E "v4l2loopback.*installed.*${KERNEL_VER}" >/dev/null 2>&1; then
    needs_patch=true
fi

if [ "$needs_patch" = true ]; then
    echo -e "${YELLOW}Detectado kernel recente ($KERNEL_VER) ou falha na compilação do pacote padrão do apt.${NC}"
    echo -e "${CYAN}--> Aplicando versão moderna e compatível do v4l2loopback no DKMS...${NC}"

    # Limpar qualquer build com erro do dkms para a versão 0.12.7
    sudo dkms remove -m v4l2loopback -v 0.12.7 --all 2>/dev/null || true

    # Garantir que o diretório de fontes no sistema exista
    TARGET_SRC="/usr/src/v4l2loopback-0.12.7"
    sudo mkdir -p "$TARGET_SRC"

    # Copiar código compatível bundled no projeto
    if [ -d "$SOURCE_DIR" ] && [ -f "$SOURCE_DIR/v4l2loopback.c" ]; then
        echo -e "${GREEN}Utilizando fontes do v4l2loopback otimizadas do projeto...${NC}"
        sudo cp -a "$SOURCE_DIR"/* "$TARGET_SRC/"
    else
        echo -e "${YELLOW}Baixando versão atualizada do upstream...${NC}"
        TMP_CLONE=$(mktemp -d)
        git clone --depth 1 https://github.com/umlaeute/v4l2loopback.git "$TMP_CLONE"
        sudo cp -a "$TMP_CLONE"/* "$TARGET_SRC/"
        rm -rf "$TMP_CLONE"
    fi

    # Ajustar dkms.conf para reportar a versão esperada pelo pacote .deb do apt
    sudo sed -i 's/PACKAGE_VERSION=".*"/PACKAGE_VERSION="0.12.7"/' "$TARGET_SRC/dkms.conf"

    # Registrar, compilar e instalar módulo no DKMS para o kernel atual
    sudo dkms add -m v4l2loopback -v 0.12.7 2>/dev/null || true
    echo -e "${YELLOW}Compilando v4l2loopback para o kernel $KERNEL_VER...${NC}"
    sudo dkms build -m v4l2loopback -v 0.12.7 -k "$KERNEL_VER"
    sudo dkms install -m v4l2loopback -v 0.12.7 -k "$KERNEL_VER" --force

    # Sanear status do dpkg
    sudo dpkg --configure -a 2>/dev/null || true
fi

# 4. Configurar parâmetros do v4l2loopback (/etc/modprobe.d/v4l2loopback.conf)
echo -e "${YELLOW}--> Configurando /etc/modprobe.d/v4l2loopback.conf...${NC}"
sudo tee /etc/modprobe.d/v4l2loopback.conf > /dev/null << 'CONF'
options v4l2loopback devices=3 video_nr=70,71,72 card_label="Iriun Webcam","OBS Virtual Cam","Intel NPU Enhanced Webcam" exclusive_caps=1,1,1 max_buffers=6
CONF

# 5. Configurar inicialização automática no boot (/etc/modules-load.d/v4l2loopback.conf)
echo -e "${YELLOW}--> Garantindo carregamento automático do módulo no boot...${NC}"
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf > /dev/null

# 6. Carregar ou recarregar o módulo
echo -e "${YELLOW}--> Aplicando configurações no módulo de kernel v4l2loopback...${NC}"
if lsmod | grep -q v4l2loopback; then
    echo -e "${CYAN}Módulo v4l2loopback já carregado. Verificando /dev/video72...${NC}"
    if [ ! -e "/dev/video72" ]; then
        echo -e "${YELLOW}Recarregando v4l2loopback para registrar /dev/video72...${NC}"
        sudo modprobe -r v4l2loopback 2>/dev/null || echo -e "${YELLOW}(Módulo em uso por outra aplicação, mantendo atual)${NC}"
        sudo modprobe v4l2loopback || true
    fi
else
    sudo modprobe v4l2loopback
fi

# 7. Validar existência do dispositivo /dev/video72
if [ -e "/dev/video72" ]; then
    echo -e "${GREEN}✓ Sucesso! Câmera virtual criada em /dev/video72:${NC}"
    v4l2-ctl --device=/dev/video72 --info 2>/dev/null | head -n 8 || true
else
    echo -e "${RED}Aviso: /dev/video72 não encontrado imediatamente. Se o módulo foi atualizado agora, pode ser necessário reiniciar o sistema.${NC}"
fi

echo -e "${GREEN}✓ Etapa 2 concluída com sucesso!${NC}\n"
