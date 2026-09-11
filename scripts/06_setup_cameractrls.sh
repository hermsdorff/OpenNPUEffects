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

# 5. Instalar atalho de aplicativo no menu global (.desktop) para todos os usuários
echo -e "${YELLOW}--> Configurando atalho no menu de aplicativos global ($GLOBAL_DESKTOP)...${NC}"
sudo cp -a "$SRC_CAMERACTRLS/hu.irl.cameractrls.desktop" "$GLOBAL_DESKTOP/hu.irl.cameractrls.desktop"
if command -v update-desktop-database > /dev/null 2>&1; then
    sudo update-desktop-database "$GLOBAL_DESKTOP" || true
fi

# 6. Testar sintaxe dos arquivos modificados
python3 -m py_compile "$CAMERACTRLS_DIR/cameractrls.py" "$CAMERACTRLS_DIR/cameractrlsgtk.py"
echo -e "${GREEN}✓ Sintaxe do Cameractrls validada com sucesso!${NC}"

echo -e "${GREEN}✓ Etapa 6 concluída com sucesso!${NC}\n"
