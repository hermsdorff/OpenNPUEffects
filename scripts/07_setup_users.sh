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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

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
    local user_home=$(getent passwd "$target_user" | cut -d: -f6)
    if [ -z "$user_home" ] || [ ! -d "$user_home" ]; then
        user_home=$(eval echo "~$target_user")
    fi
    if [ ! -d "$user_home" ]; then
        echo -e "${RED}    Diretório home '$user_home' não encontrado.${NC}"
        return
    fi

    local user_cfg="$user_home/.config/npu-effects"
    local user_bg="$user_cfg/backgrounds"
    local user_absence="$user_cfg/absence"
    sudo mkdir -p "$user_bg" "$user_absence"

    local def_cfg="/opt/npu-effects/config/config.json"
    [ ! -f "$def_cfg" ] && def_cfg="$BASE_DIR/src/config/config.json"

    # Sincronizar/fundir config.json preservando personalizações do usuário
    if [ ! -f "$user_cfg/config.json" ]; then
        if [ -f "$def_cfg" ]; then
            sudo cp -a "$def_cfg" "$user_cfg/config.json"
        fi
    else
        if [ -f "$def_cfg" ]; then
            sudo python3 -c "
import json
def merge(d, u):
    for k, v in d.items():
        if k not in u:
            u[k] = v
        elif isinstance(v, dict) and isinstance(u[k], dict):
            merge(v, u[k])
    return u

try:
    with open('$def_cfg', 'r', encoding='utf-8') as f:
        default_data = json.load(f)
    with open('$user_cfg/config.json', 'r', encoding='utf-8') as f:
        user_data = json.load(f)
    merged = merge(default_data, user_data)
    with open('$user_cfg/config.json', 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)
except Exception:
    pass
" 2>/dev/null || true
        fi
    fi

    # Assets de ausência/privacidade e fundos
    local assets_src="/opt/npu-effects/assets"
    [ ! -d "$assets_src" ] && assets_src="$BASE_DIR/assets"
    if [ -d "$assets_src/absence" ]; then
        sudo cp -an "$assets_src/absence/"* "$user_absence/" 2>/dev/null || true
    fi
    if [ -d "$assets_src/backgrounds" ]; then
        sudo cp -an "$assets_src/backgrounds/"* "$user_bg/" 2>/dev/null || true
    fi

    sudo chown -R "$target_user:$target_user" "$user_cfg"
    echo -e "    Configurações em $user_cfg: ${GREEN}OK${NC}"

    # 3. Atalhos locais (.local/bin e .local/share/applications)
    sudo mkdir -p "$user_home/.local/bin" "$user_home/.local/share/applications" "$user_home/.local/share/icons/hicolor/scalable/apps"
    sudo ln -sf "/usr/local/bin/npu-ctl" "$user_home/.local/bin/npu-ctl"
    sudo ln -sf "/usr/local/bin/cameractrls" "$user_home/.local/bin/cameractrls"

    if [ -f "/usr/share/applications/hu.irl.cameractrls.desktop" ]; then
        sudo cp -a "/usr/share/applications/hu.irl.cameractrls.desktop" "$user_home/.local/share/applications/hu.irl.cameractrls.desktop"
    fi
    if [ -f "/usr/share/icons/hicolor/scalable/apps/hu.irl.cameractrls.svg" ]; then
        sudo cp -a "/usr/share/icons/hicolor/scalable/apps/hu.irl.cameractrls.svg" "$user_home/.local/share/icons/hicolor/scalable/apps/hu.irl.cameractrls.svg"
    fi

    if [ -f "$user_home/.config/autostart/hu.irl.cameractrls.desktop" ]; then
        sudo sed -i 's|Exec=flatpak run.*|Exec=/usr/local/bin/cameractrls|g' "$user_home/.config/autostart/hu.irl.cameractrls.desktop" 2>/dev/null || true
    fi

    sudo chown -R "$target_user:$target_user" "$user_home/.local/bin" "$user_home/.local/share/applications" "$user_home/.local/share/icons" 2>/dev/null || true

    # 4. Configurar inicialização dos serviços systemd para o usuário
    sudo mkdir -p "$user_home/.config/systemd/user/default.target.wants"
    if [ -f "/etc/systemd/user/npu-webcam.service" ]; then
        sudo ln -sf "/etc/systemd/user/npu-webcam.service" "$user_home/.config/systemd/user/default.target.wants/npu-webcam.service"
    fi
    if [ -f "/etc/systemd/user/npu-audio.service" ]; then
        sudo ln -sf "/etc/systemd/user/npu-audio.service" "$user_home/.config/systemd/user/default.target.wants/npu-audio.service"
    fi
    sudo chown -R "$target_user:$target_user" "$user_home/.config/systemd"

    local target_uid=$(id -u "$target_user" 2>/dev/null || echo "1000")
    if [ -d "/run/user/$target_uid" ]; then
        sudo -u "$target_user" XDG_RUNTIME_DIR="/run/user/$target_uid" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$target_uid/bus" systemctl --user daemon-reload 2>/dev/null || true
        sudo -u "$target_user" XDG_RUNTIME_DIR="/run/user/$target_uid" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$target_uid/bus" systemctl --user enable npu-webcam.service npu-audio.service 2>/dev/null || true
        sudo -u "$target_user" XDG_RUNTIME_DIR="/run/user/$target_uid" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$target_uid/bus" systemctl --user restart npu-webcam.service npu-audio.service 2>/dev/null || true
        echo -e "    Serviços systemd da sessão ($target_user): ${GREEN}Ativados e Atualizados${NC}"
    else
        echo -e "    Serviços systemd ($target_user): ${GREEN}Habilitados (iniciarão no login)${NC}"
    fi

    # 5. Configurar câmera e microfone virtuais como padrão do usuário (Navegadores e Áudio)
    echo -e "    Configurando dispositivos virtuais como padrão..."
    sudo python3 - "$user_home" << 'PYEOF' 2>/dev/null || true
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

    if [ -d "/run/user/$target_uid" ]; then
        sudo -u "$target_user" XDG_RUNTIME_DIR="/run/user/$target_uid" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$target_uid/bus" pactl set-default-source npu_clearvoice 2>/dev/null || true
    fi

    if command -v update-desktop-database > /dev/null 2>&1; then
        sudo -u "$target_user" update-desktop-database "$user_home/.local/share/applications" 2>/dev/null || true
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
    # Sincronizar/atualizar outros usuários que já possuem ~/.config/npu-effects
    while IFS=: read -r username _ uid _ _ homedir _; do
        if [ "$uid" -ge 1000 ] && [ "$uid" -lt 60000 ] && [[ "$homedir" == /home/* ]] && [ "$username" != "$1" ]; then
            if [ -d "$homedir/.config/npu-effects" ]; then
                echo -e "${CYAN}Atualizando configurações do outro usuário existente: $username...${NC}"
                setup_one_user "$username"
            fi
        fi
    done < /etc/passwd
else
    # Configurar usuário chamador (SUDO_USER ou USER)
    CURRENT="${SUDO_USER:-$USER}"
    setup_one_user "$CURRENT"
    # Sincronizar/atualizar outros usuários que já possuem ~/.config/npu-effects
    while IFS=: read -r username _ uid _ _ homedir _; do
        if [ "$uid" -ge 1000 ] && [ "$uid" -lt 60000 ] && [[ "$homedir" == /home/* ]] && [ "$username" != "$CURRENT" ]; then
            if [ -d "$homedir/.config/npu-effects" ]; then
                echo -e "${CYAN}Atualizando configurações do outro usuário existente: $username...${NC}"
                setup_one_user "$username"
            fi
        fi
    done < /etc/passwd
    echo -e "${CYAN}Dica Multi-Usuário:${NC} Para configurar outros usuários, execute:"
    echo -e "  ${BOLD}sudo npu-ctl setup-user <nome_do_usuario>${NC}"
    echo -e "  ${BOLD}sudo npu-ctl setup-all-users${NC} (para todos os usuários do PC)"
fi

echo -e "${GREEN}✓ Etapa de configuração de usuários concluída!${NC}\n"
