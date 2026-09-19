#!/usr/bin/env bash
set -e

# Cores para terminal
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${CYAN}${BOLD}======================================================${NC}"
echo -e "${CYAN}${BOLD}   Atualizando Open NPU Effects para Todos os Usuários ${NC}"
echo -e "${CYAN}${BOLD}======================================================${NC}"

# 1. Copiar binários e templates para /opt/npu-effects
echo -e "${YELLOW}--> Atualizando binários em /opt/npu-effects/bin...${NC}"
sudo cp -a "$SCRIPT_DIR/src/bin/"* /opt/npu-effects/bin/
sudo chmod +x /opt/npu-effects/bin/*

echo -e "${YELLOW}--> Atualizando template de configuração padrão...${NC}"
sudo cp -a "$SCRIPT_DIR/src/config/config.json" /opt/npu-effects/config/config.json

echo -e "${YELLOW}--> Atualizando integração do Cameractrls...${NC}"
sudo cp -a "$SCRIPT_DIR/src/cameractrls/cameractrls.py" /opt/npu-effects/cameractrls/cameractrls.py
sudo cp -a "$SCRIPT_DIR/src/cameractrls/cameractrls.py" /opt/npu-effects/cameractrls/cameractrls.py.npu_backup
sudo cp -a "$SCRIPT_DIR/src/cameractrls/cameractrlsgtk.py" /opt/npu-effects/cameractrls/cameractrlsgtk.py
sudo cp -a "$SCRIPT_DIR/src/cameractrls/cameractrlsgtk.py" /opt/npu-effects/cameractrls/cameractrlsgtk.py.npu_backup
sudo cp -a "$SCRIPT_DIR/src/cameractrls/cameractrls" /usr/local/bin/cameractrls
sudo chmod +x /usr/local/bin/cameractrls
sudo ln -sf /opt/npu-effects/bin/npu-ctl /usr/local/bin/npu-ctl

# 2. Configurar e sincronizar para todos os usuários
echo -e "${YELLOW}--> Sincronizando configurações para todos os usuários...${NC}"
"$SCRIPT_DIR/scripts/07_setup_users.sh" --all

# 3. Reiniciar serviços no usuário atual se estiver rodando
echo -e "${YELLOW}--> Reiniciando serviços para o usuário atual...${NC}"
if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
    TARGET_UID=$(id -u "$SUDO_USER" 2>/dev/null || echo "1000")
    sudo -u "$SUDO_USER" XDG_RUNTIME_DIR="/run/user/$TARGET_UID" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$TARGET_UID/bus" systemctl --user restart npu-webcam.service npu-audio.service 2>/dev/null || true
else
    systemctl --user restart npu-webcam.service npu-audio.service 2>/dev/null || true
fi

echo -e "${GREEN}${BOLD}✓ Atualização concluída com sucesso para todo o sistema!${NC}\n"
