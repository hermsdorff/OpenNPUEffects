#!/usr/bin/env bash
set -e

# Cores para terminal
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   [ETAPA 2/6] Configuração da Câmera Virtual V4L2   ${NC}"
echo -e "${CYAN}======================================================${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
SOURCE_DIR="$BASE_DIR/src/v4l2loopback/source"
KERNEL_VER="$(uname -r)"

# 1. Verificar privilégios sudo
if ! sudo -v; then
    echo -e "${RED}Erro: Privilégios sudo são necessários para configurar módulos de kernel.${NC}"
    exit 1
fi

# 2. Instalar utilitários essenciais
echo -e "${YELLOW}--> Verificando pacotes utilitários do V4L2...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq v4l2loopback-utils v4l-utils || true

# 3. Verificar suporte do kernel ao módulo v4l2loopback
echo -e "${YELLOW}--> Verificando suporte do kernel ao módulo v4l2loopback ($KERNEL_VER)...${NC}"

# Se o kernel já fornece o módulo v4l2loopback oficialmente pré-compilado (ex: linux-modules-7.0.*)
if modinfo v4l2loopback >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Módulo v4l2loopback oficial já fornecido nativamente pelo pacote de módulos do kernel!${NC}"
    
    # Se o pacote v4l2loopback-dkms ficou em estado quebrado (half-configured) por tentativas anteriores, removê-lo
    if dpkg -s v4l2loopback-dkms 2>/dev/null | grep -q "Status: install ok"; then
        echo -e "${CYAN}Limpando pacote v4l2loopback-dkms redundante para usar o módulo oficial do kernel...${NC}"
        sudo dkms remove -m v4l2loopback -v 0.12.7 --all 2>/dev/null || true
        sudo dpkg --purge --force-all v4l2loopback-dkms 2>/dev/null || true
    fi
else
    # Caso o kernel em uso não inclua o módulo nativo, compilar via DKMS
    echo -e "${YELLOW}Módulo nativo não encontrado no kernel. Configurando via DKMS...${NC}"
    sudo apt-get install -y -qq dkms || true

    # Garantir que os fontes compatíveis estejam em /usr/src/v4l2loopback-0.12.7
    TARGET_SRC="/usr/src/v4l2loopback-0.12.7"
    sudo mkdir -p "$TARGET_SRC"

    if [ -d "$SOURCE_DIR" ] && [ -f "$SOURCE_DIR/v4l2loopback.c" ]; then
        echo -e "${GREEN}Utilizando fontes do v4l2loopback otimizadas do projeto...${NC}"
        sudo cp -a "$SOURCE_DIR"/* "$TARGET_SRC/"
    else
        echo -e "${YELLOW}Baixando versão atualizada do upstream...${NC}"
        TMP_CLONE=$(mktemp -d)
        git clone --depth 1 https://github.com/umlaeute/v4l2loopback.git "$TMP_CLONE"
        sudo cp -a "$TMP_CLONE"/* "$TARGET_SRC/"
        rm -rf "$TMP_CLONE"
    fi

    sudo sed -i 's/PACKAGE_VERSION=".*"/PACKAGE_VERSION="0.12.7"/' "$TARGET_SRC/dkms.conf"
    sudo dkms remove -m v4l2loopback -v 0.12.7 --all 2>/dev/null || true
    sudo dkms add -m v4l2loopback -v 0.12.7 2>/dev/null || true
    echo -e "${YELLOW}Compilando v4l2loopback para o kernel $KERNEL_VER...${NC}"
    sudo dkms build -m v4l2loopback -v 0.12.7 -k "$KERNEL_VER"
    sudo dkms install -m v4l2loopback -v 0.12.7 -k "$KERNEL_VER" --force
    sudo dpkg --configure -a 2>/dev/null || true
fi

# 4. Configurar parâmetros do v4l2loopback com detecção dinâmica (/etc/modprobe.d/v4l2loopback.conf)
echo -e "${YELLOW}--> Configurando /etc/modprobe.d/v4l2loopback.conf com alocação automática de porta livre...${NC}"
sudo python3 - "$BASE_DIR" << 'PYEOF'
import sys
import re
import json
from pathlib import Path

base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/opt/npu-effects")
modprobe_dir = Path("/etc/modprobe.d")
conf_path = modprobe_dir / "v4l2loopback.conf"
target_label = "Intel NPU Enhanced Webcam"

# 1. Identificar dispositivos fisicos em /sys/class/video4linux
physical_nums = set()
sysfs_v4l = Path("/sys/class/video4linux")
if sysfs_v4l.exists():
    for p in sysfs_v4l.glob("video*"):
        try:
            num = int(p.name.replace("video", ""))
            name_file = p / "name"
            card_name = name_file.read_text(encoding="utf-8", errors="ignore").strip() if name_file.exists() else ""
            is_virtual = any(k in card_name.lower() for k in ["loopback", "virtual", "npu", "iriun", "obs"])
            if not is_virtual and (p / "device").exists():
                physical_nums.add(num)
        except Exception:
            pass

# 2. Ler configuracoes existentes em /etc/modprobe.d/
all_labels = []
all_nrs = []
caps = []
lines = []

if conf_path.exists():
    try:
        lines = conf_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = []

opt_idx = -1
for idx, line in enumerate(lines):
    s = line.strip()
    if s.startswith("options v4l2loopback") and not s.startswith("#"):
        opt_idx = idx
        m_card = re.search(r"card_label=(.*?)(?=\s+[a-z_]+|\s*$)", s)
        if m_card:
            all_labels = [l.strip(" \"'\t") for l in m_card.group(1).split(",") if l.strip(" \"'\t")]
        m_nr = re.search(r"video_nr=([0-9,]+)", s)
        if m_nr:
            all_nrs = [int(n) for n in m_nr.group(1).split(",") if n]
        m_caps = re.search(r"exclusive_caps=([0-9,]+)", s)
        if m_caps:
            caps = [c for c in m_caps.group(1).split(",") if c]
        break

# Ler outros arquivos modprobe (ex: iriunwebcam-options.conf) se labels ainda nao estiverem em all_labels
for other_conf in modprobe_dir.glob("*.conf"):
    if other_conf.name == "v4l2loopback.conf":
        continue
    try:
        for l in other_conf.read_text(encoding="utf-8").splitlines():
            s = l.strip()
            if s.startswith("options v4l2loopback") and not s.startswith("#"):
                m_card = re.search(r"card_label=(.*?)(?=\s+[a-z_]+|\s*$)", s)
                if m_card:
                    lbls = [x.strip(" \"'\t") for x in m_card.group(1).split(",") if x.strip(" \"'\t")]
                    m_devs = re.search(r"devices=(\d+)", s)
                    if m_devs:
                        lbls = lbls[:int(m_devs.group(1))]
                    for lbl in lbls:
                        if lbl and lbl not in all_labels:
                            all_labels.append(lbl)
    except Exception:
        pass

if target_label not in all_labels:
    all_labels.append(target_label)

# 3. Alocacao dinamica inteligente
assigned = {}
used_numbers = set(physical_nums)

# Reservar dispositivos ja alocados que sejam validos (sem colisao fisica)
for i, lbl in enumerate(all_labels):
    if i < len(all_nrs):
        nr = all_nrs[i]
        # Iriun precisa obrigatoriamente de numero <= 9 (single-digit)
        if "iriun" in lbl.lower() and nr > 9:
            continue
        # Se colidir com hardware fisico real, descartar para re-alocar
        if nr in physical_nums:
            continue
        # Se for range legado (70, 71, 72) e faixa 7..9 estiver livre, migrar para faixa segura
        if nr in [70, 71, 72] and not (set(range(7, 10)) & physical_nums):
            continue
        assigned[lbl] = nr
        used_numbers.add(nr)

# Alocar para Iriun se necessario (garantir <= 9)
for lbl in all_labels:
    if "iriun" in lbl.lower() and lbl not in assigned:
        cand_pool = [7, 8, 6, 9, 5, 4, 3, 2]
        chosen = None
        for c in cand_pool:
            if c not in used_numbers:
                chosen = c
                break
        if chosen is None:
            for c in range(10):
                if c not in used_numbers:
                    chosen = c
                    break
        if chosen is not None:
            assigned[lbl] = chosen
            used_numbers.add(chosen)

# Alocar para as demais labels (incluindo Intel NPU Enhanced Webcam)
for lbl in all_labels:
    if lbl not in assigned:
        if "npu" in lbl.lower():
            if 9 not in used_numbers:
                chosen = 9
            else:
                cand = max(7, max(physical_nums, default=0) + 1)
                while cand in used_numbers:
                    cand += 1
                chosen = cand
        else:
            cand = max(7, max(physical_nums, default=0) + 1)
            while cand in used_numbers:
                cand += 1
            chosen = cand
        assigned[lbl] = chosen
        used_numbers.add(chosen)

final_nrs = [assigned[lbl] for lbl in all_labels]
while len(caps) < len(all_labels):
    caps.append("1")

dev_count = len(all_labels)
conf_line = (
    f'options v4l2loopback devices={dev_count} '
    f'video_nr={",".join(map(str, final_nrs))} '
    f'card_label="{",".join(all_labels)}" '
    f'exclusive_caps={",".join(caps)} '
    f'max_buffers=2'
)

if opt_idx == -1:
    lines.append(conf_line)
else:
    lines[opt_idx] = conf_line

conf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"    ✓ /etc/modprobe.d/v4l2loopback.conf configurado: {list(zip(all_labels, final_nrs))}")

# 4. Propagar output_device para configs do projeto
npu_nr = assigned.get(target_label, 9)
cfg_files = [
    base_dir / "src/config/config.json",
    Path("/opt/npu-effects/config/config.json")
]

for cfg_p in cfg_files:
    if cfg_p.exists():
        try:
            data = json.loads(cfg_p.read_text(encoding="utf-8"))
            if "video" in data and isinstance(data["video"], dict):
                data["video"]["output_device"] = f"/dev/video{npu_nr}"
            data["output_device"] = f"/dev/video{npu_nr}"
            cfg_p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"    ✓ Sincronizado {cfg_p}: output_device = /dev/video{npu_nr}")
        except Exception as e:
            print(f"    Aviso: Falha ao atualizar {cfg_p}: {e}")

Path("/tmp/npu_vcam_nr").write_text(str(npu_nr), encoding="utf-8")
PYEOF

NPU_NR=$(cat /tmp/npu_vcam_nr 2>/dev/null || echo "9")
sudo rm -f /tmp/npu_vcam_nr

# 5. Configurar inicialização automática no boot (/etc/modules-load.d/v4l2loopback.conf)
echo -e "${YELLOW}--> Garantindo carregamento automático do módulo no boot...${NC}"
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf > /dev/null

# 6. Carregar ou recarregar o módulo
echo -e "${YELLOW}--> Aplicando configurações no módulo de kernel v4l2loopback...${NC}"
if lsmod | grep -q v4l2loopback; then
    echo -e "${CYAN}Módulo v4l2loopback já carregado. Verificando /dev/video$NPU_NR...${NC}"
    if [ ! -e "/dev/video$NPU_NR" ]; then
        echo -e "${YELLOW}Recarregando v4l2loopback para registrar /dev/video$NPU_NR...${NC}"
        sudo modprobe -r v4l2loopback 2>/dev/null || echo -e "${YELLOW}(Módulo em uso por outra aplicação, mantendo atual)${NC}"
        sudo modprobe v4l2loopback || true
    fi
else
    sudo modprobe v4l2loopback
fi

# 7. Validar existência do dispositivo /dev/video$NPU_NR
if [ -e "/dev/video$NPU_NR" ]; then
    echo -e "${GREEN}✓ Sucesso! Câmera virtual criada em /dev/video$NPU_NR:${NC}"
    v4l2-ctl --device="/dev/video$NPU_NR" --info 2>/dev/null | head -n 8 || true
else
    echo -e "${RED}Aviso: /dev/video$NPU_NR não encontrado imediatamente. Se o módulo foi atualizado agora, pode ser necessário reiniciar o sistema.${NC}"
fi

echo -e "${GREEN}✓ Etapa 2 concluída com sucesso!${NC}\n"
