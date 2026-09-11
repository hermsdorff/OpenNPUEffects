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
echo "    🚀 INSTALADOR COMPLETO: INTEL NPU AI STUDIO (METEOR LAKE)   "
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
if [ -x "/usr/local/bin/npu-ctl" ]; then
    /usr/local/bin/npu-ctl status || true
elif [ -x "$HOME/.local/bin/npu-ctl" ]; then
    "$HOME/.local/bin/npu-ctl" status || true
fi

echo -e "\n${CYAN}Dica Multi-Usuário:${NC}"
echo -e "  A NPU está instalada centralmente em ${BOLD}/opt/npu-effects/${NC} compartilhada para todos."
echo -e "  Para habilitar para outros usuários do PC, execute:"
echo -e "    ${BOLD}sudo ./scripts/07_setup_users.sh <usuario>${NC}"
echo -e "    ${BOLD}sudo ./scripts/07_setup_users.sh --all${NC} (para todos os usuários)"
echo -e "\nConsulte os guias detalhados nesta pasta:"
echo -e "  - ${BOLD}PASSO_A_PASSO.md${NC}: Explicação técnica detalhada da arquitetura centralizada."
echo -e "  - ${BOLD}GUIA_NPU_STUDIO.md${NC}: Catálogo completo de efeitos de áudio/vídeo e como usar."
echo -e "\nPara abrir a interface gráfica:"
echo -e "  ${BOLD}cameractrls${NC}"
echo ""
