#!/usr/bin/env bash
set -e

# Cores para terminal
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   [ETAPA 7] Configuração de Usuários para Intel NPU  ${NC}"
echo -e "${CYAN}======================================================${NC}"

setup_one_user() {
    local target_user="$1"
    if [ -z "$target_user" ]; then
        return
    fi
    if ! id "$target_user" >/dev/null 2>&1; then
        echo -e "${RED}Aviso: Usuário '$target_user' não existe no sistema.${NC}"
        return
    fi

    echo -e "${YELLOW}--> Configurando usuário: ${BOLD}$target_user${NC}..."

    # 1. Permissões de hardware (render, video)
    sudo usermod -aG render,video "$target_user"
    echo -e "    Grupos 'render' e 'video': ${GREEN}OK${NC}"

    # 2. Diretório de configurações do usuário
    local user_home=$(eval echo "~$target_user")
    local user_cfg="$user_home/.config/npu-effects"
    local user_bg="$user_cfg/backgrounds"

    sudo mkdir -p "$user_bg"
    if [ ! -f "$user_cfg/config.json" ]; then
        if [ -f "/opt/npu-effects/config/config.json" ]; then
            sudo cp "/opt/npu-effects/config/config.json" "$user_cfg/config.json"
        elif [ -f "$(dirname "$(dirname "${BASH_SOURCE[0]}")")/src/config/config.json" ]; then
            sudo cp "$(dirname "$(dirname "${BASH_SOURCE[0]}")")/src/config/config.json" "$user_cfg/config.json"
        fi
    fi
    sudo chown -R "$target_user:$target_user" "$user_cfg"
    echo -e "    Diretório $user_cfg: ${GREEN}OK${NC}"

    # 3. Se for o usuário executando, recarregar e habilitar systemd
    if [ "$target_user" = "$USER" ] || [ "$target_user" = "$SUDO_USER" ]; then
        systemctl --user daemon-reload 2>/dev/null || true
        systemctl --user enable --now npu-webcam.service npu-audio.service 2>/dev/null || true
        echo -e "    Serviços systemd da sessão atual: ${GREEN}Ativados${NC}"
    else
        echo -e "    ${CYAN}Dica: Na sessão do usuário '$target_user', os serviços podem ser iniciados com:${NC}"
        echo -e "    ${BOLD}npu-ctl start${NC}"
    fi

    echo -e "${GREEN}✓ Usuário '$target_user' pronto para usar a Intel NPU!${NC}\n"
}

# Processar argumentos
if [ "$1" = "--all" ] || [ "$1" = "-a" ]; then
    echo -e "${CYAN}Configurando todos os usuários do sistema com pasta em /home...${NC}"
    while IFS=: read -r username _ uid _ _ homedir _; do
        if [ "$uid" -ge 1000 ] && [ "$uid" -lt 60000 ] && [[ "$homedir" == /home/* ]]; then
            setup_one_user "$username"
        fi
    done < /etc/passwd
elif [ -n "$1" ]; then
    setup_one_user "$1"
else
    # Configurar usuário chamador (SUDO_USER ou USER)
    CURRENT="${SUDO_USER:-$USER}"
    setup_one_user "$CURRENT"
    echo -e "${CYAN}Dica:${NC} Para configurar outros usuários, execute:"
    echo -e "  ${BOLD}sudo ./scripts/07_setup_users.sh <nome_do_usuario>${NC}"
    echo -e "  ${BOLD}sudo ./scripts/07_setup_users.sh --all${NC} (para todos os usuários do PC)"
fi

echo -e "${GREEN}✓ Etapa de configuração de usuários concluída!${NC}\n"
