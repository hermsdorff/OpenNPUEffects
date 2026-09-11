#!/usr/bin/env bash
#
# remove_ipu6_config.sh - Remove configurações antigas da IPU6 (Galaxy Book webcam)
# Desativa os 48 nós dummy (/dev/video0..47) e impede que apareçam no Cameractrls
#
set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}"
echo "================================================================"
echo "    🧹 REMOÇÃO DE CONFIGURAÇÕES RESIDUAIS DA INTEL IPU6        "
echo "================================================================"
echo -e "${NC}"

if [ "$(id -u)" -ne 0 ]; then
    echo -e "${YELLOW}Solicitando privilégios administrativos (sudo)...${NC}"
    exec sudo bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# 1. Parar e remover serviço systemd de rotação Samsung se existir
echo -e "${YELLOW}--> Verificando serviços residuais da IPU6...${NC}"
if systemctl is-active --quiet ipu-bridge-check-upstream.service 2>/dev/null; then
    systemctl stop ipu-bridge-check-upstream.service 2>/dev/null || true
fi
if [ -f "/etc/systemd/system/ipu-bridge-check-upstream.service" ]; then
    systemctl disable ipu-bridge-check-upstream.service 2>/dev/null || true
    rm -f "/etc/systemd/system/ipu-bridge-check-upstream.service"
    rm -f "/usr/local/sbin/ipu-bridge-check-upstream.sh"
    systemctl daemon-reload
    echo -e "    Serviço 'ipu-bridge-check-upstream': ${GREEN}Removido${NC}"
fi

# 2. Remover módulos DKMS do sensor e bridge da câmera do notebook
echo -e "${YELLOW}--> Verificando módulos DKMS (ipu-bridge-fix e ov02c10)...${NC}"
if command -v dkms >/dev/null 2>&1; then
    dkms remove -m ipu-bridge-fix --all 2>/dev/null || true
    dkms remove -m ov02c10 --all 2>/dev/null || true
fi
rm -rf /usr/src/ipu-bridge-fix* /usr/src/ov02c10* 2>/dev/null || true
echo -e "    Módulos DKMS: ${GREEN}Limpos${NC}"

# 3. Limpar linha '-e intel_ipu6' de /etc/modules
echo -e "${YELLOW}--> Limpando /etc/modules...${NC}"
if [ -f "/etc/modules" ]; then
    sed -i '/intel_ipu6/d' /etc/modules
    sed -i '/-e/d' /etc/modules
    awk '!seen[$0]++' /etc/modules > /etc/modules.tmp && mv /etc/modules.tmp /etc/modules
    echo -e "    /etc/modules: ${GREEN}Limpo${NC}"
fi

# 4. Remover arquivos de configuração específicos da webcam interna
echo -e "${YELLOW}--> Removendo configurações residuais em /etc...${NC}"
rm -f /etc/modprobe.d/ivsc-camera.conf
rm -f /etc/udev/hwdb.d/60-galaxybook-webcam.hwdb
if command -v udevadm >/dev/null 2>&1; then
    udevadm hwdb --update 2>/dev/null || true
fi

# 5. Criar Blacklist para impedir o carregamento do driver IPU6 (que cria 48 nós dummy)
echo -e "${YELLOW}--> Configurando blacklist dos módulos IPU6 em /etc/modprobe.d/blacklist-ipu6.conf...${NC}"
cat << 'BLACKLIST_EOF' > /etc/modprobe.d/blacklist-ipu6.conf
# Desativa módulos da IPU6 da webcam interna que geram dezenas de nós /dev/video dummy
blacklist intel_ipu6
blacklist intel_ipu6_isys
blacklist ov02c10
blacklist ipu_bridge
BLACKLIST_EOF
echo -e "    Blacklist: ${GREEN}Ativada${NC}"

# 6. Descarregar os módulos da memória agora se estiverem carregados
echo -e "${YELLOW}--> Descarregando módulos do kernel ativo...${NC}"
modprobe -r ov02c10 2>/dev/null || true
modprobe -r intel_ipu6_isys 2>/dev/null || true
modprobe -r intel_ipu6 2>/dev/null || true
modprobe -r ipu_bridge 2>/dev/null || true

# 7. Sincronizar Cameractrls com o filtro IPU6 atualizado
if [ -d "/opt/npu-effects/cameractrls" ]; then
    echo -e "${YELLOW}--> Atualizando Cameractrls em /opt/npu-effects/cameractrls...${NC}"
    cp -a "$BASE_DIR/src/cameractrls/cameractrls.py" "/opt/npu-effects/cameractrls/cameractrls.py"
    cp -a "$BASE_DIR/src/cameractrls/cameractrls.py" "/opt/npu-effects/cameractrls/cameractrls.py.npu_backup"
    python3 -m py_compile "/opt/npu-effects/cameractrls/cameractrls.py" 2>/dev/null || true
    echo -e "    Filtro no Cameractrls: ${GREEN}Atualizado${NC}"
fi

echo -e "\n${GREEN}${BOLD}✓ Limpeza dos dispositivos IPU6 concluída com sucesso!${NC}"
echo -e "${CYAN}Agora o Cameractrls exibirá apenas sua câmera real (EMEET PIXY) e a Câmera Virtual NPU.${NC}\n"
