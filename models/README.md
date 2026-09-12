# 🧠 Modelos de Redes Neurais NPU (Neural Models)

Este diretório armazena os modelos de aprendizado profundo (Deep Learning) pré-otimizados em formato OpenVINO IR (Intermediate Representation: `.xml` + `.bin`) e ONNX (`.onnx`) para execução acelerada em tempo real na NPU Intel AI Boost (Meteor Lake, Lunar Lake, Arrow Lake).

Para detalhes completos de direitos autorais e licenças individuais de cada modelo, consulte [`LICENSE.md`](LICENSE.md).

---

## 📋 Tabela de Modelos

| Arquivo(s) | Arquitetura | Resolução / Shape | Precisão | Origem | Licença |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `audio/noise-suppression-poconetlike-0001.*` | PoCoNet (Intel Speech Enhancement) | Frame de áudio 16 kHz com estados recorrentes | FP16 | [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) | Apache 2.0 |
| `video/face_detection_yunet_2023mar.onnx` | YuNet (Face Detection & Landmarks) | Dinâmico / Adaptativo | FP32 | [OpenCV Zoo](https://github.com/opencv/opencv_zoo) | Apache 2.0 |
| `video/selfie_segmentation_static.*` | MediaPipe Selfie Landscape | `[1, 3, 144, 256]` (NPU Shave Core optimized) | FP16 | [Google MediaPipe](https://github.com/google/mediapipe) / PINTO Zoo | Apache 2.0 |
| `video/selfie_multiclass.*` | MediaPipe Selfie Multiclass | `[1, 256, 256, 3]` (6 classes) | FP16 | [Google MediaPipe](https://github.com/google/mediapipe) / PINTO Zoo | Apache 2.0 |
| `video/chair_instance_segmenter.*` | YOLACT ResNet-50 FPN Instance Segmenter | `[1, 3, 550, 550]` | FP16 | [Daniel Bolya et al. (UC Davis)](https://github.com/dbolya/yolact) / Intel OMZ | MIT |
| `video/chair_segmenter.*` | MobileNetV3 LRASPP Segmenter | `[1, 3, 256, 256]` (21 classes VOC) | FP16 | [TorchVision](https://github.com/pytorch/vision) | BSD-3-Clause |

---

## ⬇️ Download e Conversão Automatizada

Os modelos locais estão prontos para compilação direta na NPU. O script de instalação automatizado [`scripts/04_download_models.sh`](../scripts/04_download_models.sh) é responsável por:
1. Copiar os modelos deste diretório para a pasta central do sistema `/opt/npu-effects/models/`.
2. Baixar atualizações dos repositórios oficiais caso algum arquivo esteja ausente.
3. Testar a compilação do modelo na NPU (`/dev/accel/accel0`) via OpenVINO Runtime.
