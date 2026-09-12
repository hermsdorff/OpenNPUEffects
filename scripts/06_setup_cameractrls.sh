#!/usr/bin/env bash
set -e

# Cores para terminal
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   [ETAPA 6/6] Integração com a Interface Cameractrls${NC}"
echo -e "${CYAN}======================================================${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
SRC_CAMERACTRLS="$BASE_DIR/src/cameractrls"

TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME=$(eval echo "~$TARGET_USER")

OPT_DIR="/opt/npu-effects"
CAMERACTRLS_DIR="$OPT_DIR/cameractrls"
GLOBAL_BIN="/usr/local/bin"
GLOBAL_DESKTOP="/usr/share/applications"

sudo mkdir -p "$CAMERACTRLS_DIR" "$GLOBAL_BIN" "$GLOBAL_DESKTOP"
[ -d "$TARGET_HOME" ] && sudo -u "$TARGET_USER" mkdir -p "$TARGET_HOME/.local/bin" "$TARGET_HOME/.local/share/applications" 2>/dev/null || true

# 1. Instalar dependências GTK do Cameractrls via apt
echo -e "${YELLOW}--> Verificando dependências GTK/PyGObject...${NC}"
sudo apt-get install -y -qq \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-3.0 \
    gir1.2-gtk-4.0 \
    libglib2.0-dev

# 2. Obter base do Cameractrls (reaproveitar local ou clonar)
if [ ! -f "$CAMERACTRLS_DIR/cameraview.py" ]; then
    if [ -f "$TARGET_HOME/.local/share/cameractrls/cameraview.py" ]; then
        echo -e "${GREEN}Copiando base do Cameractrls de ~/.local/share para $CAMERACTRLS_DIR...${NC}"
        sudo cp -r "$TARGET_HOME/.local/share/cameractrls"/* "$CAMERACTRLS_DIR/"
    else
        echo -e "${YELLOW}--> Baixando base do Cameractrls do repositório oficial...${NC}"
        TMP_CLONE=$(mktemp -d)
        git clone --depth 1 https://github.com/soyersoyer/cameractrls.git "$TMP_CLONE"
        sudo cp -r "$TMP_CLONE"/* "$CAMERACTRLS_DIR/"
        rm -rf "$TMP_CLONE"
    fi
fi

# 3. Aplicar arquivos integrados com suporte à NPU Intel em /opt/npu-effects/cameractrls
echo -e "${YELLOW}--> Injetando controles de IA, barra de rolagem fixa e redimensionamento...${NC}"
sudo cp -a "$SRC_CAMERACTRLS/cameractrls.py" "$CAMERACTRLS_DIR/cameractrls.py"
sudo cp -a "$SRC_CAMERACTRLS/cameractrlsgtk.py" "$CAMERACTRLS_DIR/cameractrlsgtk.py"

# Criar cópias de backup persistentes (.npu_backup) para restauração rápida
sudo cp -a "$SRC_CAMERACTRLS/cameractrls.py" "$CAMERACTRLS_DIR/cameractrls.py.npu_backup"
sudo cp -a "$SRC_CAMERACTRLS/cameractrlsgtk.py" "$CAMERACTRLS_DIR/cameractrlsgtk.py.npu_backup"

# Garantir permissão de leitura/execução global
sudo chmod -R a+rX "$CAMERACTRLS_DIR"

# 4. Instalar script de inicialização universal no PATH (/usr/local/bin/cameractrls)
echo -e "${YELLOW}--> Configurando executável global no PATH ($GLOBAL_BIN/cameractrls)...${NC}"
sudo cp -a "$SRC_CAMERACTRLS/cameractrls" "$GLOBAL_BIN/cameractrls"
sudo chmod +x "$GLOBAL_BIN/cameractrls"
[ -d "$TARGET_HOME" ] && sudo -u "$TARGET_USER" ln -sf "$GLOBAL_BIN/cameractrls" "$TARGET_HOME/.local/bin/cameractrls" 2>/dev/null || true

# 5. Instalar ícone oficial e atalho de aplicativo no menu (.desktop) para todos os usuários
echo -e "${YELLOW}--> Configurando atalhos de aplicativo e ícones no sistema...${NC}"
ICON_SRC="$SRC_CAMERACTRLS/hu.irl.cameractrls.svg"
[ ! -f "$ICON_SRC" ] && ICON_SRC="$CAMERACTRLS_DIR/pkg/hu.irl.cameractrls.svg"

if [ -f "$ICON_SRC" ]; then
    sudo mkdir -p /usr/share/icons/hicolor/scalable/apps
    sudo cp -a "$ICON_SRC" /usr/share/icons/hicolor/scalable/apps/hu.irl.cameractrls.svg 2>/dev/null || true
    if [ -d "$TARGET_HOME" ]; then
        sudo -u "$TARGET_USER" mkdir -p "$TARGET_HOME/.local/share/icons/hicolor/scalable/apps" 2>/dev/null || true
        sudo -u "$TARGET_USER" cp -a "$ICON_SRC" "$TARGET_HOME/.local/share/icons/hicolor/scalable/apps/hu.irl.cameractrls.svg" 2>/dev/null || true
    fi
fi

# Instalar .desktop global e no perfil do usuário (para ter prioridade absoluta sobre Flatpaks)
sudo cp -a "$SRC_CAMERACTRLS/hu.irl.cameractrls.desktop" "$GLOBAL_DESKTOP/hu.irl.cameractrls.desktop"
sudo mkdir -p /usr/local/share/applications
sudo cp -a "$SRC_CAMERACTRLS/hu.irl.cameractrls.desktop" /usr/local/share/applications/hu.irl.cameractrls.desktop 2>/dev/null || true

if [ -d "$TARGET_HOME" ]; then
    sudo -u "$TARGET_USER" mkdir -p "$TARGET_HOME/.local/share/applications" 2>/dev/null || true
    sudo -u "$TARGET_USER" cp -a "$SRC_CAMERACTRLS/hu.irl.cameractrls.desktop" "$TARGET_HOME/.local/share/applications/hu.irl.cameractrls.desktop" 2>/dev/null || true
    
    # Se houver autostart do Flatpak antigo apontando para o sandbox, atualizar para o binário nativo
    if [ -f "$TARGET_HOME/.config/autostart/hu.irl.cameractrls.desktop" ]; then
        if grep -q "flatpak" "$TARGET_HOME/.config/autostart/hu.irl.cameractrls.desktop"; then
            echo -e "${YELLOW}--> Atualizando autostart de Flatpak para o Cameractrls nativo da NPU...${NC}"
            sudo -u "$TARGET_USER" sed -i 's|Exec=flatpak run.*|Exec=/usr/local/bin/cameractrls|g' "$TARGET_HOME/.config/autostart/hu.irl.cameractrls.desktop" 2>/dev/null || true
        fi
    fi
fi

if command -v update-desktop-database > /dev/null 2>&1; then
    sudo update-desktop-database "$GLOBAL_DESKTOP" 2>/dev/null || true
    sudo update-desktop-database /usr/local/share/applications 2>/dev/null || true
    [ -d "$TARGET_HOME" ] && sudo -u "$TARGET_USER" update-desktop-database "$TARGET_HOME/.local/share/applications" 2>/dev/null || true
fi

# Detectar se o Cameractrls Flatpak está instalado e avisar sobre conflito potencial
if command -v flatpak >/dev/null 2>&1 && flatpak list 2>/dev/null | grep -q "hu.irl.cameractrls"; then
    echo -e "${YELLOW}⚠️  Atenção: Foi detectada uma instalação anterior do Cameractrls via Flatpak.${NC}"
    echo -e "${YELLOW}   O atalho do usuário em ~/.local/share/applications foi configurado com prioridade máxima.${NC}"
    echo -e "${YELLOW}   Para evitar qualquer conflito, recomenda-se desinstalar o pacote Flatpak:${NC}"
    echo -e "${BOLD}   flatpak uninstall hu.irl.cameractrls${NC}"
fi

# 6. Testar sintaxe dos arquivos modificados
python3 -m py_compile "$CAMERACTRLS_DIR/cameractrls.py" "$CAMERACTRLS_DIR/cameractrlsgtk.py"
echo -e "${GREEN}✓ Sintaxe do Cameractrls validada com sucesso!${NC}"

echo -e "${GREEN}✓ Etapa 6 concluída com sucesso!${NC}\n"
