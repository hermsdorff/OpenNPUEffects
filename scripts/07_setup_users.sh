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

    # 4. Configurar câmera e microfone virtuais como padrão do usuário (Navegadores e Áudio)
    echo -e "    Configurando dispositivos virtuais como padrão..."
    python3 - "$user_home" << 'PYEOF' 2>/dev/null || true
import sys
import json
from pathlib import Path

user_home = Path(sys.argv[1])

# 1. Atualizar navegadores Chromium/Chrome/Brave/Edge
browser_roots = [
    user_home / '.config/google-chrome',
    user_home / '.config/chromium',
    user_home / '.config/BraveSoftware/Brave-Browser',
    user_home / '.config/microsoft-edge',
    user_home / '.config/microsoft-edge-dev',
]
cam_name = 'Intel NPU Enhanced Webcam'
for b_dir in browser_roots:
    if not b_dir.exists():
        continue
    for pref_file in b_dir.glob('**/Preferences'):
        try:
            with open(pref_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            media = data.setdefault('media', {})
            # Video: NPU virtual webcam top priority
            v_in = media.setdefault('video_input', {})
            v_rank = v_in.setdefault('user_preference_ranking', [])
            if cam_name in v_rank:
                v_rank.remove(cam_name)
            v_rank.insert(0, cam_name)
            # Audio: default top priority (inherits npu_clearvoice)
            a_in = media.setdefault('audio_input', {})
            a_rank = a_in.setdefault('user_preference_ranking', [])
            if 'default' in a_rank:
                a_rank.remove('default')
            a_rank.insert(0, 'default')
            with open(pref_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

# 2. Atualizar WirePlumber default-nodes state se existir
wp_nodes = user_home / '.local/state/wireplumber/default-nodes'
if wp_nodes.exists():
    try:
        content = wp_nodes.read_text(encoding='utf-8')
        lines = []
        has_src = False
        for line in content.splitlines():
            if line.startswith('default.configured.audio.source='):
                lines.append('default.configured.audio.source=npu_clearvoice')
                has_src = True
            else:
                lines.append(line)
        if not has_src:
            lines.append('default.configured.audio.source=npu_clearvoice')
        wp_nodes.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    except Exception:
        pass
PYEOF

    local target_uid=$(id -u "$target_user" 2>/dev/null || echo "1000")
    if [ -d "/run/user/$target_uid" ]; then
        sudo -u "$target_user" XDG_RUNTIME_DIR="/run/user/$target_uid" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$target_uid/bus" pactl set-default-source npu_clearvoice 2>/dev/null || true
    fi
    echo -e "    Câmera e microfone padrão: ${GREEN}Configurados${NC}"

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
