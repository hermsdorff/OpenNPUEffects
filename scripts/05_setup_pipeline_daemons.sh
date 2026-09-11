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

TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME=$(eval echo "~$TARGET_USER")
TARGET_UID=$(id -u "$TARGET_USER" 2>/dev/null || echo "1000")

OPT_DIR="/opt/npu-effects"
BIN_DEST="$OPT_DIR/bin"
CONFIG_DEFAULT="$OPT_DIR/config"
SYSTEMD_GLOBAL="/etc/systemd/user"
USER_CONFIG="$TARGET_HOME/.config/npu-effects"

echo -e "${YELLOW}--> Preparando diretórios do sistema e do usuário ($TARGET_USER)...${NC}"
sudo mkdir -p "$BIN_DEST" "$CONFIG_DEFAULT" "$SYSTEMD_GLOBAL"
sudo mkdir -p "$USER_CONFIG/backgrounds"
sudo chown -R "$TARGET_USER:$TARGET_USER" "$USER_CONFIG"
[ -d "$TARGET_HOME/.local/bin" ] || sudo -u "$TARGET_USER" mkdir -p "$TARGET_HOME/.local/bin" 2>/dev/null || true

# 1. Instalar binários dos Daemons e CLI em /opt/npu-effects/bin
echo -e "${YELLOW}--> Instalando scripts de execução globais em $BIN_DEST...${NC}"
sudo cp -a "$SRC_DIR/bin/"* "$BIN_DEST/"
sudo chmod +x "$BIN_DEST/"*

# 2. Criar symlink do utilitário npu-ctl em /usr/local/bin (acessível por todos os usuários)
echo -e "${YELLOW}--> Configurando comando 'npu-ctl' no PATH global (/usr/local/bin/npu-ctl)...${NC}"
sudo ln -sf "$BIN_DEST/npu-ctl" "/usr/local/bin/npu-ctl"
sudo -u "$TARGET_USER" ln -sf "$BIN_DEST/npu-ctl" "$TARGET_HOME/.local/bin/npu-ctl" 2>/dev/null || true

# 3. Instalar arquivo de configuração padrão em /opt e na home do usuário
echo -e "${YELLOW}--> Instalando templates de configuração padrão...${NC}"
sudo cp -a "$SRC_DIR/config/config.json" "$CONFIG_DEFAULT/config.json"
if [ ! -f "$USER_CONFIG/config.json" ]; then
    sudo cp -a "$SRC_DIR/config/config.json" "$USER_CONFIG/config.json"
    sudo chown "$TARGET_USER:$TARGET_USER" "$USER_CONFIG/config.json"
fi

# 4. Instalar units do systemd user em /etc/systemd/user (disponíveis para TODOS os usuários)
echo -e "${YELLOW}--> Instalando units de serviço em $SYSTEMD_GLOBAL...${NC}"
sudo cp -a "$SRC_DIR/systemd/"*.service "$SYSTEMD_GLOBAL/"
sudo chmod 644 "$SYSTEMD_GLOBAL/"npu-*.service

# Garantir permissões globais
sudo chmod -R a+rX "$OPT_DIR"

# 5. Recarregar e ativar serviços para o usuário atual
echo -e "${YELLOW}--> Recarregando daemon do systemd e iniciando serviços do usuário ($TARGET_USER)...${NC}"

# Tentar recarregar e habilitar systemd user de forma segura
if [ "$(id -u)" -eq 0 ] && [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
    sudo -u "$TARGET_USER" XDG_RUNTIME_DIR="/run/user/$TARGET_UID" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$TARGET_UID/bus" systemctl --user daemon-reload 2>/dev/null || true
    sudo -u "$TARGET_USER" XDG_RUNTIME_DIR="/run/user/$TARGET_UID" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$TARGET_UID/bus" systemctl --user enable --now npu-webcam.service npu-audio.service 2>/dev/null || true
else
    systemctl --user daemon-reload 2>/dev/null || true
    systemctl --user enable --now npu-webcam.service npu-audio.service 2>/dev/null || true
fi

# 6. Validar status
sleep 1.5
echo -e "${YELLOW}--> Verificando status dos serviços:${NC}"
if [ "$(id -u)" -eq 0 ] && [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
    CAM_STATUS=$(sudo -u "$TARGET_USER" XDG_RUNTIME_DIR="/run/user/$TARGET_UID" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$TARGET_UID/bus" systemctl --user is-active npu-webcam.service 2>/dev/null || echo "inactive")
    AUD_STATUS=$(sudo -u "$TARGET_USER" XDG_RUNTIME_DIR="/run/user/$TARGET_UID" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$TARGET_UID/bus" systemctl --user is-active npu-audio.service 2>/dev/null || echo "inactive")
else
    CAM_STATUS=$(systemctl --user is-active npu-webcam.service 2>/dev/null || echo "inactive")
    AUD_STATUS=$(systemctl --user is-active npu-audio.service 2>/dev/null || echo "inactive")
fi

if [ "$CAM_STATUS" = "active" ]; then
    echo -e "  npu-webcam.service: ${GREEN}ATIVO (Running)${NC}"
else
    echo -e "  npu-webcam.service: ${YELLOW}$CAM_STATUS (Pronto para iniciar)${NC}"
fi

if [ "$AUD_STATUS" = "active" ]; then
    echo -e "  npu-audio.service:  ${GREEN}ATIVO (Running)${NC}"
else
    echo -e "  npu-audio.service:  ${YELLOW}$AUD_STATUS (Pronto para iniciar)${NC}"
fi

echo -e "${GREEN}✓ Etapa 5 concluída com sucesso!${NC}\n"
