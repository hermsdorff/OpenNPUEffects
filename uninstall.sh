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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    echo -e "  - Comando de controle (/usr/local/bin/npu-ctl)"
    echo -e "  - Configurações e controles customizados do Cameractrls (o Cameractrls é mantido)"
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
# 1. Parar e desabilitar serviços systemd para todos os usuários
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
        sudo -u "$target_user" XDG_RUNTIME_DIR="/run/user/$uid" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$uid/bus" systemctl --user stop npu-webcam.service npu-audio.service 2>/dev/null || true
        sudo -u "$target_user" XDG_RUNTIME_DIR="/run/user/$uid" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$uid/bus" systemctl --user disable npu-webcam.service npu-audio.service 2>/dev/null || true
    fi
}

while IFS=: read -r username _ uid _ _ homedir _; do
    if [ "$uid" -ge 1000 ] && [ "$uid" -lt 60000 ] && [[ "$homedir" == /home/* ]]; then
        stop_user_services "$username"
        sudo rm -f "$homedir/.config/systemd/user/default.target.wants/npu-webcam.service" 2>/dev/null || true
        sudo rm -f "$homedir/.config/systemd/user/default.target.wants/npu-audio.service" 2>/dev/null || true
        if [ -d "/run/user/$uid" ]; then
            sudo -u "$username" XDG_RUNTIME_DIR="/run/user/$uid" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$uid/bus" systemctl --user daemon-reload 2>/dev/null || true
        fi
    fi
done < /etc/passwd

# ------------------------------------------------------------------------------
# 2. Remover arquivos de serviço systemd globais
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}--> [2/5] Removendo arquivos de serviço systemd globais...${NC}"
if [ -f "/etc/systemd/user/npu-webcam.service" ]; then
    sudo rm -f "/etc/systemd/user/npu-webcam.service"
    echo -e "    Removido /etc/systemd/user/npu-webcam.service"
fi
if [ -f "/etc/systemd/user/npu-audio.service" ]; then
    sudo rm -f "/etc/systemd/user/npu-audio.service"
    echo -e "    Removido /etc/systemd/user/npu-audio.service"
fi

# ------------------------------------------------------------------------------
# 3. Remover utilitários e restaurar Cameractrls limpo
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}--> [3/5] Removendo customizações do Cameractrls e utilitários NPU...${NC}"

# 3a. Remover npu-ctl
sudo rm -f "/usr/local/bin/npu-ctl"
while IFS=: read -r username _ uid _ _ homedir _; do
    if [ "$uid" -ge 1000 ] && [ "$uid" -lt 60000 ] && [[ "$homedir" == /home/* ]]; then
        sudo rm -f "$homedir/.local/bin/npu-ctl" 2>/dev/null || true
    fi
done < /etc/passwd
echo -e "    Removido utilitário npu-ctl"

# 3b. Remover arquivos de desktop customizados com branding NPU
sudo rm -f "/usr/share/applications/hu.irl.cameractrls.desktop"
sudo rm -f "/usr/local/share/applications/hu.irl.cameractrls.desktop"

while IFS=: read -r username _ uid _ _ homedir _; do
    if [ "$uid" -ge 1000 ] && [ "$uid" -lt 60000 ] && [[ "$homedir" == /home/* ]]; then
        # Remover desktop file customizado da pasta do usuário
        if [ -f "$homedir/.local/share/applications/hu.irl.cameractrls.desktop" ]; then
            sudo rm -f "$homedir/.local/share/applications/hu.irl.cameractrls.desktop"
        fi

        # Restaurar autostart do Flatpak se tiver sido alterado para /usr/local/bin/cameractrls
        if [ -f "$homedir/.config/autostart/hu.irl.cameractrls.desktop" ]; then
            if command -v flatpak >/dev/null 2>&1 && flatpak list 2>/dev/null | grep -q "hu.irl.cameractrls"; then
                sudo sed -i 's|Exec=/usr/local/bin/cameractrls.*|Exec=flatpak run hu.irl.cameractrls|g' "$homedir/.config/autostart/hu.irl.cameractrls.desktop" 2>/dev/null || true
            fi
        fi

        # Remover presets e configurações customizadas da câmera virtual NPU criados no Cameractrls
        for cdir in "$homedir/.config/hu.irl.cameractrls" "$homedir/.var/app/hu.irl.cameractrls/config/hu.irl.cameractrls"; do
            if [ -d "$cdir" ]; then
                sudo find "$cdir" -type f \( -name "*video72*.ini" -o -name "*Intel*NPU*.ini" \) -delete 2>/dev/null || true
            fi
        done

        # Limpar symlink antigo se quebrado
        if [ -L "$homedir/.local/bin/cameractrls" ]; then
            if [ ! -e "$homedir/.local/bin/cameractrls" ]; then
                sudo rm -f "$homedir/.local/bin/cameractrls"
            fi
        fi
    fi
done < /etc/passwd

# 3c. Garantir que o Cameractrls original seja preservado e permaneça funcional
if command -v flatpak >/dev/null 2>&1 && flatpak list 2>/dev/null | grep -q "hu.irl.cameractrls"; then
    echo -e "    ${GREEN}✓ Instalação Flatpak do Cameractrls detectada.${NC}"
    echo -e "    Configurando executável /usr/local/bin/cameractrls limpo para o Flatpak..."
    sudo tee "/usr/local/bin/cameractrls" > /dev/null << 'EOF_CAM'
#!/bin/sh
exec flatpak run hu.irl.cameractrls "$@"
EOF_CAM
    sudo chmod 755 "/usr/local/bin/cameractrls"
elif [ -x "/usr/bin/cameractrls" ] || [ -x "/usr/bin/cameractrlsgtk.py" ]; then
    echo -e "    ${GREEN}✓ Instalação de sistema (apt) do Cameractrls detectada.${NC}"
    sudo rm -f "/usr/local/bin/cameractrls"
else
    # Standalone Cameractrls em /opt/npu-effects/cameractrls:
    # Mover para /opt/cameractrls e restaurar código limpo para que Cameractrls NÃO seja removido
    if [ -d "/opt/npu-effects/cameractrls" ]; then
        echo -e "    ${YELLOW}Cameractrls autônomo detectado em /opt/npu-effects/cameractrls.${NC}"
        echo -e "    Preservando Cameractrls em /opt/cameractrls sem as customizações de NPU..."
        sudo mkdir -p "/opt/cameractrls"
        sudo cp -r "/opt/npu-effects/cameractrls"/* "/opt/cameractrls/" 2>/dev/null || true

        # Restaurar arquivos originais upstream limpos
        if [ -f "/opt/cameractrls/cameractrls.py.upstream_clean" ]; then
            sudo cp -f "/opt/cameractrls/cameractrls.py.upstream_clean" "/opt/cameractrls/cameractrls.py"
        elif [ -f "$SCRIPT_DIR/src/cameractrls/upstream/cameractrls.py" ]; then
            sudo cp -f "$SCRIPT_DIR/src/cameractrls/upstream/cameractrls.py" "/opt/cameractrls/cameractrls.py"
        fi

        if [ -f "/opt/cameractrls/cameractrlsgtk.py.upstream_clean" ]; then
            sudo cp -f "/opt/cameractrls/cameractrlsgtk.py.upstream_clean" "/opt/cameractrls/cameractrlsgtk.py"
        elif [ -f "$SCRIPT_DIR/src/cameractrls/upstream/cameractrlsgtk.py" ]; then
            sudo cp -f "$SCRIPT_DIR/src/cameractrls/upstream/cameractrlsgtk.py" "/opt/cameractrls/cameractrlsgtk.py"
        fi

        sudo rm -f "/opt/cameractrls"/*.npu_backup

        # Instalar desktop file original limpo
        if [ -f "/opt/cameractrls/pkg/hu.irl.cameractrls.desktop" ]; then
            sudo cp -f "/opt/cameractrls/pkg/hu.irl.cameractrls.desktop" "/usr/share/applications/hu.irl.cameractrls.desktop"
            sudo sed -i 's|Exec=cameractrlsgtk.py|Exec=/usr/local/bin/cameractrls|g' "/usr/share/applications/hu.irl.cameractrls.desktop"
        elif [ -f "$SCRIPT_DIR/src/cameractrls/upstream/hu.irl.cameractrls.desktop" ]; then
            sudo cp -f "$SCRIPT_DIR/src/cameractrls/upstream/hu.irl.cameractrls.desktop" "/usr/share/applications/hu.irl.cameractrls.desktop"
            sudo sed -i 's|Exec=cameractrlsgtk.py|Exec=/usr/local/bin/cameractrls|g' "/usr/share/applications/hu.irl.cameractrls.desktop"
        fi

        # Criar executável limpo apontando para /opt/cameractrls
        sudo tee "/usr/local/bin/cameractrls" > /dev/null << 'EOF_STANDALONE'
#!/bin/sh
exec /usr/bin/python3 /opt/cameractrls/cameractrlsgtk.py "$@"
EOF_STANDALONE
        sudo chmod 755 "/usr/local/bin/cameractrls"
    fi
fi

# 3d. Atualizar bancos de dados de aplicativos (.desktop)
if command -v update-desktop-database > /dev/null 2>&1; then
    sudo update-desktop-database "/usr/share/applications" 2>/dev/null || true
    sudo update-desktop-database "/usr/local/share/applications" 2>/dev/null || true
fi
while IFS=: read -r username _ uid _ _ homedir _; do
    if [ "$uid" -ge 1000 ] && [ "$uid" -lt 60000 ] && [[ "$homedir" == /home/* ]]; then
        if [ -d "$homedir/.local/share/applications" ] && command -v update-desktop-database > /dev/null 2>&1; then
            sudo -u "$username" update-desktop-database "$homedir/.local/share/applications" 2>/dev/null || true
        fi
    fi
done < /etc/passwd

echo -e "    ${GREEN}✓ Customizações do Cameractrls removidas com sucesso (o aplicativo continua instalado e funcional).${NC}"

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
# 5. Limpeza de dados de áudio, configurações de usuário e sistema
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}--> [5/5] Verificando configurações e parâmetros do sistema...${NC}"

# Limpar fonte de áudio npu_clearvoice do WirePlumber
while IFS=: read -r username _ uid _ _ homedir _; do
    if [ "$uid" -ge 1000 ] && [ "$uid" -lt 60000 ] && [[ "$homedir" == /home/* ]]; then
        wp_nodes="$homedir/.local/state/wireplumber/default-nodes"
        if [ -f "$wp_nodes" ]; then
            sudo sed -i '/default\.configured\.audio\.source=npu_clearvoice/d' "$wp_nodes" 2>/dev/null || true
        fi
    fi
done < /etc/passwd

if [ "$PURGE_ALL" = true ]; then
    echo -e "    ${RED}Modo Purge ativado: removendo arquivos de configuração de todos os usuários...${NC}"
    
    # Remover ~/.config/npu-effects de todos os usuários
    while IFS=: read -r username _ uid _ _ homedir _; do
        if [ "$uid" -ge 1000 ] && [ "$uid" -lt 60000 ] && [[ "$homedir" == /home/* ]]; then
            if [ -d "$homedir/.config/npu-effects" ]; then
                echo -e "    Removendo $homedir/.config/npu-effects"
                sudo rm -rf "$homedir/.config/npu-effects"
            fi
        fi
    done < /etc/passwd

    # Remover regras do v4l2loopback criadas pelo projeto (preservando outras câmeras se houver)
    if [ -f "/etc/modprobe.d/v4l2loopback.conf" ]; then
        echo -e "    Removendo Intel NPU Enhanced Webcam de /etc/modprobe.d/v4l2loopback.conf..."
        sudo python3 -c '
import re
from pathlib import Path

conf_path = Path("/etc/modprobe.d/v4l2loopback.conf")
target_label = "Intel NPU Enhanced Webcam"
target_nr = 72

if conf_path.exists():
    lines = conf_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    has_other_cameras = False
    for line in lines:
        if line.strip().startswith("options v4l2loopback"):
            opt_line = line.strip()
            m_card = re.search(r"card_label=(.*?)(?=\s+[a-z_]+|\s*$)", opt_line)
            existing_labels = []
            if m_card:
                existing_labels = [l.strip(" \"'\t") for l in m_card.group(1).split(",") if l.strip(" \"'\t")]
            
            if target_label in existing_labels:
                idx = existing_labels.index(target_label)
                existing_labels.pop(idx)
                
                m_nr = re.search(r"video_nr=([0-9,]+)", opt_line)
                existing_nrs = [int(n) for n in m_nr.group(1).split(",") if n] if m_nr else []
                if idx < len(existing_nrs):
                    existing_nrs.pop(idx)
                elif target_nr in existing_nrs:
                    existing_nrs.remove(target_nr)
                    
                m_caps = re.search(r"exclusive_caps=([0-9,]+)", opt_line)
                existing_caps = [c for c in m_caps.group(1).split(",") if c] if m_caps else []
                if idx < len(existing_caps):
                    existing_caps.pop(idx)
                    
                m_buf = re.search(r"max_buffers=(\d+)", opt_line)
                max_buf = max(2, int(m_buf.group(1))) if m_buf else 2

                if existing_labels:
                    has_other_cameras = True
                    dev_count = len(existing_labels)
                    new_lines.append(
                        f"options v4l2loopback devices={dev_count} "
                        f"video_nr={\",\".join(map(str, existing_nrs))} "
                        f"card_label=\"{\",\".join(existing_labels)}\" "
                        f"exclusive_caps={\",\".join(existing_caps)} "
                        f"max_buffers={max_buf}"
                    )
            else:
                new_lines.append(line)
                has_other_cameras = True
        else:
            new_lines.append(line)
            
    if has_other_cameras:
        conf_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print("    ✓ Outras câmeras virtuais foram preservadas em /etc/modprobe.d/v4l2loopback.conf.")
    else:
        conf_path.unlink(missing_ok=True)
        print("    ✓ Removido /etc/modprobe.d/v4l2loopback.conf (nenhuma outra câmera restante).")
'
    fi
    if [ ! -f "/etc/modprobe.d/v4l2loopback.conf" ] && [ -f "/etc/modules-load.d/v4l2loopback.conf" ]; then
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
