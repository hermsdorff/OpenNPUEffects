#!/usr/bin/env bash
set -e

# Cores para terminal
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

clear || true
echo -e "${CYAN}${BOLD}"
echo "================================================================"
echo "    🚀 INSTALADOR COMPLETO: OPEN NPU EFFECTS (METEOR LAKE)   "
echo "        Webcam 1080p + Áudio de Estúdio + Cameractrls GUI       "
echo "================================================================"
echo -e "${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Solicitar credenciais sudo logo no início
echo -e "${YELLOW}Solicitando permissões administrativas (sudo) para configuração de sistema...${NC}"
sudo -v

# Manter sudo vivo durante o processo
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &
SUDO_PID=$!
trap 'kill $SUDO_PID 2>/dev/null || true' EXIT

# Executar scripts modulares sequencialmente
"$SCRIPT_DIR/scripts/01_install_npu_drivers.sh"
"$SCRIPT_DIR/scripts/02_setup_v4l2loopback.sh"
"$SCRIPT_DIR/scripts/03_setup_dependencies.sh"
"$SCRIPT_DIR/scripts/04_download_models.sh"
"$SCRIPT_DIR/scripts/05_setup_pipeline_daemons.sh"
"$SCRIPT_DIR/scripts/06_setup_cameractrls.sh"
"$SCRIPT_DIR/scripts/07_setup_users.sh" "${SUDO_USER:-$USER}"

echo -e "${GREEN}${BOLD}"
echo "================================================================"
echo "    🎉 INSTALAÇÃO E CONFIGURAÇÃO CONCLUÍDAS COM SUCESSO!       "
echo "================================================================"
echo -e "${NC}"

# Exibir status do painel NPU
TARGET_USER="${SUDO_USER:-$USER}"
TARGET_UID=$(id -u "$TARGET_USER" 2>/dev/null || echo "1000")

if [ "$(id -u)" -eq 0 ] && [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
    sudo -u "$TARGET_USER" XDG_RUNTIME_DIR="/run/user/$TARGET_UID" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$TARGET_UID/bus" /usr/local/bin/npu-ctl status 2>/dev/null || true
elif [ -x "/usr/local/bin/npu-ctl" ]; then
    /usr/local/bin/npu-ctl status || true
fi

echo -e "\n${CYAN}Dica Multi-Usuário:${NC}"
echo -e "  A NPU está instalada centralmente em ${BOLD}/opt/npu-effects/${NC} compartilhada para todos."
echo -e "  Para habilitar para outros usuários do PC, execute:"
echo -e "    ${BOLD}sudo ./scripts/07_setup_users.sh <usuario>${NC}"
echo -e "    ${BOLD}sudo ./scripts/07_setup_users.sh --all${NC} (para todos os usuários)"
echo -e "  - ${BOLD}STEP_BY_STEP.md${NC}: Technical step-by-step guide and architecture details."
echo -e "  - ${BOLD}NPU_STUDIO_GUIDE.md${NC}: Complete user manual and audio/video effect catalogue."
echo -e "\nPara abrir a interface gráfica:"
echo -e "  ${BOLD}cameractrls${NC}"
echo ""
