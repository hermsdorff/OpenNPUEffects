#!/usr/bin/env bash
set -e

# Cores para terminal
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   [ETAPA 5/6] Instalação dos Daemons e Serviços     ${NC}"
echo -e "${CYAN}======================================================${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$BASE_DIR/src"

OPT_DIR="/opt/npu-effects"
BIN_DEST="$OPT_DIR/bin"
CONFIG_DEFAULT="$OPT_DIR/config"
SYSTEMD_GLOBAL="/etc/systemd/user"
USER_CONFIG="$HOME/.config/npu-effects"

echo -e "${YELLOW}--> Preparando diretórios do sistema e do usuário...${NC}"
sudo mkdir -p "$BIN_DEST" "$CONFIG_DEFAULT" "$SYSTEMD_GLOBAL"
mkdir -p "$USER_CONFIG/backgrounds" "$HOME/.local/bin"

# 1. Instalar binários dos Daemons e CLI em /opt/npu-effects/bin
echo -e "${YELLOW}--> Instalando scripts de execução globais em $BIN_DEST...${NC}"
sudo cp -a "$SRC_DIR/bin/"* "$BIN_DEST/"
sudo chmod +x "$BIN_DEST/"*

# 2. Criar symlink do utilitário npu-ctl em /usr/local/bin (acessível por todos os usuários)
echo -e "${YELLOW}--> Configurando comando 'npu-ctl' no PATH global (/usr/local/bin/npu-ctl)...${NC}"
sudo ln -sf "$BIN_DEST/npu-ctl" "/usr/local/bin/npu-ctl"
ln -sf "$BIN_DEST/npu-ctl" "$HOME/.local/bin/npu-ctl" 2>/dev/null || true

# 3. Instalar arquivo de configuração padrão em /opt e na home do usuário
echo -e "${YELLOW}--> Instalando templates de configuração padrão...${NC}"
sudo cp -a "$SRC_DIR/config/config.json" "$CONFIG_DEFAULT/config.json"
if [ ! -f "$USER_CONFIG/config.json" ]; then
    cp -a "$SRC_DIR/config/config.json" "$USER_CONFIG/config.json"
fi

# 4. Instalar units do systemd user em /etc/systemd/user (disponíveis para TODOS os usuários)
echo -e "${YELLOW}--> Instalando units de serviço em $SYSTEMD_GLOBAL...${NC}"
sudo cp -a "$SRC_DIR/systemd/"*.service "$SYSTEMD_GLOBAL/"
sudo chmod 644 "$SYSTEMD_GLOBAL/"npu-*.service

# Garantir permissões globais
sudo chmod -R a+rX "$OPT_DIR"

# 5. Recarregar e ativar serviços para o usuário atual
echo -e "${YELLOW}--> Recarregando daemon do systemd e iniciando serviços do usuário...${NC}"
systemctl --user daemon-reload
systemctl --user enable --now npu-webcam.service npu-audio.service

# 6. Validar status
sleep 1.5
echo -e "${YELLOW}--> Verificando status dos serviços:${NC}"
CAM_STATUS=$(systemctl --user is-active npu-webcam.service || echo "inactive")
AUD_STATUS=$(systemctl --user is-active npu-audio.service || echo "inactive")

if [ "$CAM_STATUS" = "active" ]; then
    echo -e "  npu-webcam.service: ${GREEN}ATIVO (Running)${NC}"
else
    echo -e "  npu-webcam.service: ${RED}$CAM_STATUS${NC}"
fi

if [ "$AUD_STATUS" = "active" ]; then
    echo -e "  npu-audio.service:  ${GREEN}ATIVO (Running)${NC}"
else
    echo -e "  npu-audio.service:  ${RED}$AUD_STATUS${NC}"
fi

echo -e "${GREEN}✓ Etapa 5 concluída com sucesso!${NC}\n"
