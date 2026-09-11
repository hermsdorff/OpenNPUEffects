#!/usr/bin/env bash
# ==============================================================================
# Open NPU Effects - Script de Desinstalação Completa
# https://github.com/hermsdorff/OpenNPUEffects
# ==============================================================================

set -e

# Cores para terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

AUTO_YES=false
PURGE_ALL=false

# Processar argumentos de linha de comando
for arg in "$@"; do
    case "$arg" in
        -y|--yes)
            AUTO_YES=true
            ;;
        -p|--purge)
            PURGE_ALL=true
            AUTO_YES=true
            ;;
        -h|--help)
            echo "Uso: ./uninstall.sh [OPÇÕES]"
            echo ""
            echo "Opções:"
            echo "  -y, --yes     Executa a desinstalação sem confirmação interativa"
            echo "  -p, --purge   Remove completamente tudo, incluindo configs de usuário (~/.config/npu-effects) e configs de kernel"
            echo "  -h, --help    Exibe esta mensagem de ajuda"
            exit 0
            ;;
    esac
done

echo -e "${RED}${BOLD}"
echo "================================================================"
echo "       🗑️  DESINSTALADOR: OPEN NPU EFFECTS FOR LINUX           "
echo "================================================================"
echo -e "${NC}"

# Detectar usuário real chamador
REAL_USER="${SUDO_USER:-$USER}"
USER_HOME=$(eval echo "~$REAL_USER")

if [ "$AUTO_YES" = false ]; then
    echo -e "${YELLOW}Este script irá remover:${NC}"
    echo -e "  - Serviços systemd em segundo plano (npu-webcam e npu-audio)"
    echo -e "  - Diretório de instalação global (/opt/npu-effects/)"
    echo -e "  - Comandos e atalhos (/usr/local/bin/npu-ctl, /usr/local/bin/cameractrls)"
    echo -e "  - Atalho do menu de aplicativos (hu.irl.cameractrls.desktop)"
    echo ""
    read -p "Deseja continuar com a desinstalação? [s/N]: " -r CONFIRM
    if [[ ! "$CONFIRM" =~ ^[sSyY]$ ]]; then
        echo -e "${CYAN}Desinstalação cancelada pelo usuário.${NC}"
        exit 0
    fi
    echo ""
    read -p "Deseja também remover as configurações pessoais e fundos virtuais em ~/.config/npu-effects? [s/N]: " -r CONFIRM_PURGE
    if [[ "$CONFIRM_PURGE" =~ ^[sSyY]$ ]]; then
        PURGE_ALL=true
    fi
fi

echo -e "\n${YELLOW}--> Solicitando permissões administrativas (sudo)...${NC}"
sudo -v

# Manter sudo vivo durante o processo
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &
SUDO_PID=$!
trap 'kill $SUDO_PID 2>/dev/null || true' EXIT

# ------------------------------------------------------------------------------
# 1. Parar e desabilitar serviços systemd do usuário
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}--> [1/5] Parando e desabilitando serviços em segundo plano...${NC}"

stop_user_services() {
    local target_user="$1"
    if [ -z "$target_user" ]; then return; fi
    local uid=$(id -u "$target_user" 2>/dev/null || true)
    if [ -z "$uid" ]; then return; fi

    echo -e "    Parando serviços para o usuário: ${BOLD}$target_user${NC}..."
    if [ "$target_user" = "$USER" ] && [ -n "$XDG_RUNTIME_DIR" ]; then
        systemctl --user stop npu-webcam.service npu-audio.service 2>/dev/null || true
        systemctl --user disable npu-webcam.service npu-audio.service 2>/dev/null || true
    else
        # Executar dentro do contexto da sessão do usuário alvo
        sudo -u "$target_user" XDG_RUNTIME_DIR="/run/user/$uid" systemctl --user stop npu-webcam.service npu-audio.service 2>/dev/null || true
        sudo -u "$target_user" XDG_RUNTIME_DIR="/run/user/$uid" systemctl --user disable npu-webcam.service npu-audio.service 2>/dev/null || true
    fi
}

stop_user_services "$REAL_USER"

# Se purge ativo, parar para todos os usuários com home
if [ "$PURGE_ALL" = true ]; then
    while IFS=: read -r username _ uid _ _ homedir _; do
        if [ "$uid" -ge 1000 ] && [ "$uid" -lt 60000 ] && [[ "$homedir" == /home/* ]] && [ "$username" != "$REAL_USER" ]; then
            stop_user_services "$username"
        fi
    done < /etc/passwd
fi

# ------------------------------------------------------------------------------
# 2. Remover arquivos de serviço systemd globais
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}--> [2/5] Removendo arquivos de serviço systemd...${NC}"
if [ -f "/etc/systemd/user/npu-webcam.service" ]; then
    sudo rm -f "/etc/systemd/user/npu-webcam.service"
    echo -e "    Removido /etc/systemd/user/npu-webcam.service"
fi
if [ -f "/etc/systemd/user/npu-audio.service" ]; then
    sudo rm -f "/etc/systemd/user/npu-audio.service"
    echo -e "    Removido /etc/systemd/user/npu-audio.service"
fi

# Recarregar systemd para aplicar a remoção
if [ -n "$XDG_RUNTIME_DIR" ]; then
    systemctl --user daemon-reload 2>/dev/null || true
fi
sudo -u "$REAL_USER" XDG_RUNTIME_DIR="/run/user/$(id -u "$REAL_USER")" systemctl --user daemon-reload 2>/dev/null || true

# ------------------------------------------------------------------------------
# 3. Remover comandos do PATH e atalhos (.desktop)
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}--> [3/5] Removendo utilitários e atalhos do sistema...${NC}"
sudo rm -f "/usr/local/bin/npu-ctl"
sudo rm -f "/usr/local/bin/cameractrls"
rm -f "$USER_HOME/.local/bin/npu-ctl" 2>/dev/null || true
rm -f "$USER_HOME/.local/bin/cameractrls" 2>/dev/null || true

if [ -f "/usr/share/applications/hu.irl.cameractrls.desktop" ]; then
    sudo rm -f "/usr/share/applications/hu.irl.cameractrls.desktop"
    echo -e "    Removido /usr/share/applications/hu.irl.cameractrls.desktop"
    if command -v update-desktop-database > /dev/null 2>&1; then
        sudo update-desktop-database "/usr/share/applications" || true
    fi
fi

# ------------------------------------------------------------------------------
# 4. Remover diretório global de instalação (/opt/npu-effects)
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}--> [4/5] Removendo diretório central /opt/npu-effects...${NC}"
if [ -d "/opt/npu-effects" ]; then
    sudo rm -rf "/opt/npu-effects"
    echo -e "    ${GREEN}✓ /opt/npu-effects removido com sucesso.${NC}"
else
    echo -e "    Diretório /opt/npu-effects já não existe."
fi

# ------------------------------------------------------------------------------
# 5. Limpeza de dados de usuário e configurações de sistema
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}--> [5/5] Verificando configurações e parâmetros do sistema...${NC}"

if [ "$PURGE_ALL" = true ]; then
    echo -e "    ${RED}Modo Purge ativado: removendo arquivos de configuração...${NC}"
    
    # Remover ~/.config/npu-effects de todos os usuários
    while IFS=: read -r username _ uid _ _ homedir _; do
        if [ "$uid" -ge 1000 ] && [ "$uid" -lt 60000 ] && [[ "$homedir" == /home/* ]]; then
            if [ -d "$homedir/.config/npu-effects" ]; then
                echo -e "    Removendo $homedir/.config/npu-effects"
                sudo rm -rf "$homedir/.config/npu-effects"
            fi
        fi
    done < /etc/passwd

    # Remover regras do v4l2loopback criadas pelo projeto
    if [ -f "/etc/modprobe.d/v4l2loopback.conf" ]; then
        echo -e "    Removendo /etc/modprobe.d/v4l2loopback.conf"
        sudo rm -f "/etc/modprobe.d/v4l2loopback.conf"
    fi
    if [ -f "/etc/modules-load.d/v4l2loopback.conf" ]; then
        echo -e "    Removendo /etc/modules-load.d/v4l2loopback.conf"
        sudo rm -f "/etc/modules-load.d/v4l2loopback.conf"
    fi

    # Remover regras udev da NPU criadas pelo projeto
    if [ -f "/etc/udev/rules.d/10-intel-vpu.rules" ]; then
        echo -e "    Removendo /etc/udev/rules.d/10-intel-vpu.rules"
        sudo rm -f "/etc/udev/rules.d/10-intel-vpu.rules"
        sudo udevadm control --reload-rules 2>/dev/null || true
        sudo udevadm trigger 2>/dev/null || true
    fi

    # Tentar descarregar o módulo v4l2loopback
    if lsmod | grep -q v4l2loopback; then
        echo -e "    Tentando descarregar o módulo v4l2loopback..."
        sudo modprobe -r v4l2loopback 2>/dev/null || echo -e "    ${YELLOW}(Módulo v4l2loopback em uso por outra aplicação, mantido na memória)${NC}"
    fi
else
    echo -e "    ${CYAN}Configs mantidas em ~/.config/npu-effects (use --purge para removê-las).${NC}"
    echo -e "    ${CYAN}Parâmetros do kernel v4l2loopback mantidos.${NC}"
fi

echo -e "\n${GREEN}${BOLD}"
echo "================================================================"
echo "    ✨ OPEN NPU EFFECTS DESINSTALADO COM SUCESSO!              "
echo "================================================================"
echo -e "${NC}"
