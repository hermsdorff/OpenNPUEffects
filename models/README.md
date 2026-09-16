# 🧠 NPU Neural Network Models

This directory stores deep learning models pre-optimized in OpenVINO IR format (Intermediate Representation: `.xml` + `.bin`) and ONNX (`.onnx`) for real-time hardware-accelerated execution on Intel AI Boost NPUs (Meteor Lake, Lunar Lake, Arrow Lake).

For full copyright details and individual model licenses, see [`LICENSE.md`](LICENSE.md).

---

## 📋 Model Specifications

| File(s) | Architecture | Resolution / Shape | Precision | Source | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `audio/noise-suppression-poconetlike-0001.*` | PoCoNet (Intel Speech Enhancement) | 16 kHz audio frame with recurrent states | FP16 | [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) | Apache 2.0 |
| `video/face_detection_yunet_2023mar.onnx` | YuNet (Face Detection & Landmarks) | Dynamic / Adaptive | FP32 | [OpenCV Zoo](https://github.com/opencv/opencv_zoo) | Apache 2.0 |
| `video/selfie_segmentation_static.*` | MediaPipe Selfie Landscape | `[1, 3, 144, 256]` (NPU Shave Core optimized) | FP16 | [Google MediaPipe](https://github.com/google/mediapipe) / PINTO Zoo | Apache 2.0 |
| `video/selfie_multiclass.*` | MediaPipe Selfie Multiclass | `[1, 256, 256, 3]` (6 classes) | FP16 | [Google MediaPipe](https://github.com/google/mediapipe) / PINTO Zoo | Apache 2.0 |
| `video/chair_instance_segmenter.*` | YOLACT ResNet-50 FPN Instance Segmenter | `[1, 3, 550, 550]` | FP16 | [Daniel Bolya et al. (UC Davis)](https://github.com/dbolya/yolact) / Intel OMZ | MIT |
| `video/face_parsing_bisenet.*` | BiSeNet ResNet-18 Face Parser | `[1, 3, 512, 512]` (19 classes CelebAMask-HQ) | FP16 | [LiteRT Community](https://huggingface.co/litert-community/BiSeNet-Face-Parsing-LiteRT) / [zllrunning](https://github.com/zllrunning/face-parsing.PyTorch) | MIT |
| `video/modnet_portrait_matting.*` | MODNet Portrait Matting (optional assist, off by default) | `[1, 3, 512, 512]` | FP16 | [ZHKKKe](https://github.com/ZHKKKe/MODNet) / Intel OMZ | Apache 2.0 |

---

## ⬇️ Automated Download & Conversion

Local models are ready for direct compilation on the NPU. The automated installation script [`scripts/04_download_models.sh`](../scripts/04_download_models.sh):
1. Copies models from this directory to the central system directory `/opt/npu-effects/models/`.
2. Downloads updates from official repositories if any required file is missing.
3. Tests compilation on the NPU (`/dev/accel/accel0`) using OpenVINO Runtime.
