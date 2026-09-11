#!/usr/bin/env bash
set -e

# Cores para terminal
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   [ETAPA 1/6] Instalação dos Drivers da Intel NPU   ${NC}"
echo -e "${CYAN}======================================================${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DRIVERS_DIR="$BASE_DIR/drivers"

# 1. Verificar privilégios sudo
if ! sudo -v; then
    echo -e "${RED}Erro: Privilégios sudo são necessários para instalar drivers no sistema.${NC}"
    exit 1
fi

# 2. Verificar suporte do kernel (driver intel_vpu)
echo -e "${YELLOW}--> Verificando driver do kernel intel_vpu...${NC}"
if ! lsmod | grep -q intel_vpu; then
    echo -e "${YELLOW}Carregando módulo de kernel intel_vpu...${NC}"
    sudo modprobe intel_vpu || true
fi

# 3. Instalar pacotes de driver (.deb)
echo -e "${YELLOW}--> Instalando Level Zero Loader e Drivers UMD da Intel NPU...${NC}"

if ls "$DRIVERS_DIR"/*.deb 1> /dev/null 2>&1; then
    echo -e "${GREEN}Utilizando pacotes de driver locais em: $DRIVERS_DIR${NC}"
    # Prioridade para libze1 primeiro
    if [ -f "$DRIVERS_DIR/libze1_1.28.2.deb" ]; then
        sudo dpkg -i "$DRIVERS_DIR/libze1_1.28.2.deb" || sudo apt-get install -f -y
    fi
    sudo dpkg -i "$DRIVERS_DIR"/intel-*.deb || sudo apt-get install -f -y
else
    echo -e "${YELLOW}Drivers locais não encontrados. Baixando versão oficial mais recente do GitHub da Intel...${NC}"
    TMP_DIR=$(mktemp -d)
    cd "$TMP_DIR"
    
    # URL de release oficial do Intel NPU Driver
    RELEASE_URL="https://api.github.com/repos/intel/linux-npu-driver/releases/latest"
    ASSET_URL=$(curl -s "$RELEASE_URL" | grep -o 'https://github.com/intel/linux-npu-driver/releases/download/[^"]*ubuntu2404\.tar\.gz' | head -n 1)
    
    if [ -n "$ASSET_URL" ]; then
        echo -e "${CYAN}Baixando: $ASSET_URL${NC}"
        curl -L -o npu-driver.tar.gz "$ASSET_URL"
        tar -xzf npu-driver.tar.gz
        sudo dpkg -i ./*.deb || sudo apt-get install -f -y
    else
        echo -e "${RED}Não foi possível obter a URL automática de download. Verifique sua conexão.${NC}"
        exit 1
    fi
    cd - > /dev/null
    rm -rf "$TMP_DIR"
fi

# 4. Configurar permissões de grupo para o usuário atual
echo -e "${YELLOW}--> Configurando grupos de usuário (render, video)...${NC}"
sudo usermod -aG render,video "$USER"
if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "$USER" ]; then
    sudo usermod -aG render,video "$SUDO_USER"
fi

# 5. Garantir regra udev para /dev/accel/accel*
UDEV_RULE="/etc/udev/rules.d/10-intel-vpu.rules"
if [ ! -f "$UDEV_RULE" ]; then
    echo -e "${YELLOW}--> Criando regra udev para acesso sem root à NPU...${NC}"
    echo 'SUBSYSTEM=="accel", KERNEL=="accel*", GROUP="render", MODE="0660"' | sudo tee "$UDEV_RULE" > /dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
fi

# 6. Verificação do dispositivo NPU
if [ -e "/dev/accel/accel0" ]; then
    echo -e "${GREEN}✓ Sucesso! Dispositivo NPU detectado em /dev/accel/accel0:${NC}"
    ls -l /dev/accel/accel0
else
    echo -e "${YELLOW}Aviso: /dev/accel/accel0 ainda não apareceu. Pode ser necessário reiniciar o sistema após instalar o driver do kernel.${NC}"
fi

echo -e "${GREEN}✓ Etapa 1 concluída com sucesso!${NC}\n"
