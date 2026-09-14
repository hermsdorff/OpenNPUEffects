#!/usr/bin/env bash
set -e

# Cores para terminal
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   [ETAPA 4/6] Download e Preparação dos Modelos IA  ${NC}"
echo -e "${CYAN}======================================================${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_MODELS="$BASE_DIR/models"

OPT_DIR="/opt/npu-effects"
TARGET_AUDIO="$OPT_DIR/models/audio"
TARGET_VIDEO="$OPT_DIR/models/video"
if [ -x "$OPT_DIR/venv/bin/python3" ]; then
    VENV_PYTHON="$OPT_DIR/venv/bin/python3"
else
    VENV_PYTHON="$HOME/.local/share/npu-effects/venv/bin/python3"
fi

sudo mkdir -p "$TARGET_AUDIO" "$TARGET_VIDEO"

# 1. Modelos de Áudio (Intel PoCoNet Noise Suppression)
echo -e "${YELLOW}--> Verificando modelo de Áudio (PoCoNet FP16)...${NC}"
if [ -f "$LOCAL_MODELS/audio/noise-suppression-poconetlike-0001.xml" ] && [ -f "$LOCAL_MODELS/audio/noise-suppression-poconetlike-0001.bin" ]; then
    echo -e "${GREEN}Copiando modelo de áudio local para $TARGET_AUDIO...${NC}"
    sudo cp -u "$LOCAL_MODELS/audio/noise-suppression-poconetlike-0001".* "$TARGET_AUDIO/"
else
    echo -e "${CYAN}Baixando modelo PoCoNet do repositório oficial da Intel (Open Model Zoo)...${NC}"
    OMZ_URL="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/noise-suppression-poconetlike-0001/FP16"
    sudo curl -L --retry 3 -o "$TARGET_AUDIO/noise-suppression-poconetlike-0001.xml" "$OMZ_URL/noise-suppression-poconetlike-0001.xml"
    sudo curl -L --retry 3 -o "$TARGET_AUDIO/noise-suppression-poconetlike-0001.bin" "$OMZ_URL/noise-suppression-poconetlike-0001.bin"
fi

# 2. Modelos de Vídeo (YuNet Face Detection & Selfie Segmentation)
echo -e "${YELLOW}--> Verificando modelo de Rastreamento Facial (YuNet)...${NC}"
if [ -f "$LOCAL_MODELS/video/face_detection_yunet_2023mar.onnx" ]; then
    echo -e "${GREEN}Copiando YuNet local para $TARGET_VIDEO...${NC}"
    sudo cp -u "$LOCAL_MODELS/video/face_detection_yunet_2023mar.onnx" "$TARGET_VIDEO/"
else
    echo -e "${CYAN}Baixando YuNet do repositório oficial OpenCV Zoo...${NC}"
    YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    sudo curl -L --retry 3 -o "$TARGET_VIDEO/face_detection_yunet_2023mar.onnx" "$YUNET_URL"
fi

echo -e "${YELLOW}--> Verificando modelo de Segmentação de Pessoas (Selfie Multiclass & Static NPU IR)...${NC}"
if [ -f "$LOCAL_MODELS/video/selfie_multiclass.xml" ] && [ -f "$LOCAL_MODELS/video/selfie_multiclass.bin" ]; then
    echo -e "${GREEN}Copiando modelo de segmentação Multiclass local para $TARGET_VIDEO...${NC}"
    sudo cp -u "$LOCAL_MODELS/video/selfie_multiclass".* "$TARGET_VIDEO/"
fi

if [ -f "$LOCAL_MODELS/video/selfie_segmentation_static.xml" ] && [ -f "$LOCAL_MODELS/video/selfie_segmentation_static.bin" ]; then
    echo -e "${GREEN}Copiando modelo de segmentação estático local para $TARGET_VIDEO...${NC}"
    sudo cp -u "$LOCAL_MODELS/video/selfie_segmentation_static".* "$TARGET_VIDEO/"
else
    echo -e "${CYAN}Baixando modelo de segmentação e convertendo para formato estático otimizado para NPU...${NC}"
    ONNX_URL="https://github.com/PINTO0309/PINTO_model_zoo/raw/main/082_MediaPipe_Selfie_Segmentation/selfie_landscape_fp16.onnx"
    sudo curl -L --retry 3 -o "$TARGET_VIDEO/selfie_landscape_fp16.onnx" "$ONNX_URL"
    
    sudo "$VENV_PYTHON" - << 'PYCONV'
import openvino as ov
from pathlib import Path

video_dir = Path("/opt/npu-effects/models/video") if Path("/opt/npu-effects/models/video").exists() else Path.home() / ".local/share/npu-effects/models/video"
onnx_path = video_dir / "selfie_landscape_fp16.onnx"
out_path = video_dir / "selfie_segmentation_static.xml"

core = ov.Core()
model = core.read_model(str(onnx_path))
# Ajuste de forma estática 144x256 ideal para os SHAVE cores da NPU
model.reshape({"pixel_values": [1, 3, 144, 256]})
ov.save_model(model, str(out_path))
print("Modelo de segmentação estático gerado com sucesso!")
PYCONV
fi

echo -e "${YELLOW}--> Verificando modelo de Segmentação de Instâncias (Cadeira/Objetos YOLACT)...${NC}"
if [ -f "$LOCAL_MODELS/video/chair_instance_segmenter.xml" ] && [ -f "$LOCAL_MODELS/video/chair_instance_segmenter.bin" ]; then
    echo -e "${GREEN}Copiando modelo de segmentação de cadeira (YOLACT) para $TARGET_VIDEO...${NC}"
    sudo cp -u "$LOCAL_MODELS/video/chair_instance_segmenter".* "$TARGET_VIDEO/"
fi

echo -e "${YELLOW}--> Verificando modelo de Segmentação de Rosto e Óculos (BiSeNet Face Parsing)...${NC}"
if [ -f "$LOCAL_MODELS/video/face_parsing_bisenet.xml" ] && [ -f "$LOCAL_MODELS/video/face_parsing_bisenet.bin" ]; then
    echo -e "${GREEN}Copiando modelo de face parsing (BiSeNet) para $TARGET_VIDEO...${NC}"
    sudo cp -u "$LOCAL_MODELS/video/face_parsing_bisenet".* "$TARGET_VIDEO/"
else
    echo -e "${CYAN}Baixando modelo BiSeNet Face Parsing e convertendo para OpenVINO IR FP16...${NC}"
    BISENET_URL="https://huggingface.co/litert-community/BiSeNet-Face-Parsing-LiteRT/resolve/main/faceparsing.tflite"
    sudo curl -L --retry 3 -o "$TARGET_VIDEO/faceparsing.tflite" "$BISENET_URL"

    sudo "$VENV_PYTHON" - << 'PYBISENET'
import openvino as ov
from pathlib import Path

video_dir = Path("/opt/npu-effects/models/video") if Path("/opt/npu-effects/models/video").exists() else Path.home() / ".local/share/npu-effects/models/video"
tflite_path = video_dir / "faceparsing.tflite"
out_path = video_dir / "face_parsing_bisenet.xml"

core = ov.Core()
model = core.read_model(str(tflite_path))
ov.save_model(model, str(out_path), compress_to_fp16=True)
print("Modelo BiSeNet Face Parsing convertido para IR FP16 com sucesso!")
PYBISENET
    sudo rm -f "$TARGET_VIDEO/faceparsing.tflite"
fi

# 3. Testar compilação dos modelos na NPU
echo -e "${YELLOW}--> Testando compilação dos modelos baixados na NPU...${NC}"
"$VENV_PYTHON" - << 'PYTEST'
import openvino as ov
from pathlib import Path

core = ov.Core()
device = "NPU" if "NPU" in core.available_devices else "CPU"
base = Path("/opt/npu-effects/models") if Path("/opt/npu-effects/models").exists() else Path.home() / ".local/share/npu-effects/models"

# Audio
audio_m = base / "audio/noise-suppression-poconetlike-0001.xml"
if audio_m.exists():
    m = core.read_model(str(audio_m))
    compiled = core.compile_model(m, device)
    print(f"\033[0;32m✓ Modelo de Áudio PoCoNet compilado com sucesso no dispositivo '{device}'!\033[0m")

# Video
video_m = base / "video/selfie_segmentation_static.xml"
if video_m.exists():
    m = core.read_model(str(video_m))
    compiled = core.compile_model(m, device)
    print(f"\033[0;32m✓ Modelo de Vídeo Selfie compilado com sucesso no dispositivo '{device}'!\033[0m")

chair_m = base / "video/chair_instance_segmenter.xml"
if chair_m.exists():
    m = core.read_model(str(chair_m))
    compiled = core.compile_model(m, device)
    print(f"\033[0;32m✓ Modelo de Segmentação de Cadeira YOLACT compilado com sucesso no dispositivo '{device}'!\033[0m")

glasses_m = base / "video/face_parsing_bisenet.xml"
if glasses_m.exists():
    m = core.read_model(str(glasses_m))
    compiled = core.compile_model(m, device)
    print(f"\033[0;32m✓ Modelo Face Parsing (BiSeNet Óculos) compilado com sucesso no dispositivo '{device}'!\033[0m")
PYTEST

# Garantir permissão de leitura para todos os usuários
sudo chmod -R a+rX "$OPT_DIR/models"

echo -e "${GREEN}✓ Etapa 4 concluída com sucesso!${NC}\n"
