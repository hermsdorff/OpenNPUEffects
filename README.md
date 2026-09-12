<div align="center">

# 🚀 Open NPU Effects for Linux

**Hardware-Accelerated AI Webcam & Broadcast Audio Suite for Linux on Intel Core Ultra (Meteor Lake, Lunar Lake, Arrow Lake) NPUs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20(Kernel%206.8%2B)-blue.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-Intel%20Core%20Ultra%20(AI%20Boost)-0071C5.svg)]()
[![OpenVINO](https://img.shields.io/badge/Backend-OpenVINO%20NPU-purple.svg)]()
[![Video](https://img.shields.io/badge/Webcam-1080p%20%40%2030FPS-brightgreen.svg)]()
[![Audio](https://img.shields.io/badge/Audio-PipeWire%20Low--Latency-red.svg)]()

*The open-source alternative to Windows Studio Effects on Linux.*  
Run background blur, exact chair contour retention, auto-framing, eye contact correction, 3D parallax, and deep neural noise suppression natively on the **Intel NPU** with near-zero CPU and GPU overhead!

[Documentation](README.md) • [User Guide & Effects Catalogue](NPU_STUDIO_GUIDE.md) • [Technical Step-by-Step](STEP_BY_STEP.md)

</div>

---

## 🌟 Why This Project?

Modern AI PCs powered by **Intel Core Ultra** laptops (Samsung Galaxy Book4 Ultra, Dell XPS, ThinkPad, Asus Zenbook, HP Spectre) feature an integrated Neural Processing Unit (NPU / Intel AI Boost). Under Windows, *Windows Studio Effects* uses this chip for camera and microphone enhancements.

Under **Linux**, the NPU has traditionally remained 100% idle. Existing Linux background removal tools run on the CPU (burning 60–90% CPU, generating heat, and spinning fans) or consume battery on the GPU.

**Open NPU Effects** brings full hardware offloading to the Linux desktop:
- **1080p @ 30 FPS video pipeline** running on the NPU via OpenVINO (~18 ms inference latency).
- **Sub-watt power efficiency** — zero CPU fan spikes during Google Meet, Zoom, Teams, Discord, or OBS calls.
- **Broadcast vocal chain** powered by Intel PoCoNet deep noise suppression directly inside PipeWire.
- **Native GUI control** via integrated [Cameractrls](https://github.com/soyersoyer/cameractrls) GTK interface and intuitive `npu-ctl` CLI.

---

## ✨ Key Features

### 📹 Video Effects (1080p @ 30 FPS on NPU)
- **Neural Person Silhouette**: High-fidelity multiclass segmentation with hand and skin recovery.
- **Precise Chair & Furniture Retention (YOLACT)**: Instance segmentation detects the exact curves, headrest wings, leather stitching, and backrest of gaming/office chairs — **zero convex-hull blobs**.
- **Edge-Snapping Guided Filter**: 1080p camera luminance edge-refinement eliminates white boundary halos and preserves loose hair and finger gestures.
- **Cinematic Depth Blur & Virtual Backgrounds**: Configurable blur strength or custom photo/video backgrounds.
- **Auto-Framing / Face Tracking**: Smooth kinematic subject centering powered by YuNet face detection.
- **Eye Contact Teleprompter**: Natural gaze redirection toward the camera lens.
- **3D Parallax Perspective**: Shifts the virtual background dynamically as you move your head.
- **Studio & Rim Lighting**: Adds a soft warm or cool contour highlight around hair and shoulders.
- **Hand Gesture Recognition**: Detects hands/thumbs-up/open palm to trigger auto-reactions or instant mic mute.
- **Handheld Object Retention**: Keeps coffee mugs, pens, notebooks, and smartphones in the foreground.
- **Smart Auto-Standby & PTZ Hardware Parking**: Automatically releases the camera and physically tilts PTZ lenses downward (e.g., EMEET Pixy) to protect privacy when not in a call.
- **Privacy Screen with Smooth Fade**: Elegant fade-in/fade-out transition when leaving camera view.

### 🎙️ Audio Effects (PipeWire Broadcast Vocal DSP)
- **Deep Noise Suppression (Intel PoCoNet)**: Neural suppression of keyboard clicks, dog barks, street traffic, and fan hum.
- **High-Pass Low-Cut Filter (80 Hz)**: Removes desk rumble and sub-bass vibrations.
- **Vocal Presence EQ**: Studio clarity boost in the 2.5 kHz – 4 kHz vocal frequency band.
- **Fast De-Esser**: Tames harsh sibilance ("s", "ch", "sh") at 6.8 kHz.
- **Adaptive Voice Activity Gate (VAD)**: Absolute silence when you stop speaking (threshold at -45 dB).
- **Smooth Broadcast Compressor / AGC**: Consistent vocal loudness normalized to -18 dB target.
- **Room Dereverberation**: Cancels echo in tiled, minimalist, or untreated rooms.

---

## 💻 Hardware & System Requirements

- **Processor**: Intel Core Ultra (Series 1 *Meteor Lake* or Series 2 *Lunar Lake* / *Arrow Lake*) with integrated Intel AI Boost NPU (`/dev/accel/accel0`).
- **Operating System**: Linux with Kernel 6.8+ (Ubuntu 24.04 LTS, KDE neon, Debian 12+, Fedora 40+, Arch Linux).
- **Graphics / Acceleration**: Intel Level Zero NPU Driver (`intel-driver-compiler-npu`, `intel-level-zero-npu`).
- **Camera Backend**: `v4l2loopback-dkms` (virtual video device creation).
- **Audio Backend**: `pipewire`, `pipewire-pulse`, `wireplumber`.

---

## ⚡ Quick Start (One-Command Setup)

Clone the repository and run the automated installer:

```bash
git clone https://github.com/hermsdorff/OpenNPUEffects.git
cd OpenNPUEffects
chmod +x install_all.sh
./install_all.sh
```

The installer automatically:
1. Verifies and installs official Intel NPU Level Zero drivers and compiler UMDs.
2. Configures `v4l2loopback` (`/dev/video72`) with persistent kernel module settings.
3. Sets up a centralized Python virtual environment in `/opt/npu-effects/venv` with OpenVINO runtime.
4. Prepares pre-optimized OpenVINO IR neural models (Selfie Multiclass, YOLACT, PoCoNet, YuNet).
5. Installs and starts systemd user daemons (`npu-webcam.service` and `npu-audio.service`).
6. Integrates control sliders directly into `cameractrls` and links `npu-ctl` to `/usr/local/bin`.

---

## 🎮 How to Use

### 1. In Any Video Conferencing App (Meet, Zoom, Teams, Discord, OBS, Slack)
- **Camera**: Select **`Intel NPU Enhanced Webcam`** (`/dev/video72`).
- **Microphone**: Select **`npu_clearvoice`** (or default PipeWire virtual sink).

### 2. Graphical Control (GUI)
Launch the camera control center:
```bash
cameractrls
```
Navigate to the **NPU Effects** section to toggle background blur, virtual backgrounds, chair retention, auto-framing, eye contact, and lighting with live preview.

### 3. Command-Line Utility (`npu-ctl`)
Manage effects from terminal or hotkeys:

```bash
# General Status
npu-ctl status

# Camera Controls
npu-ctl blur on | off | 20          # Toggle / adjust background blur
npu-ctl chair on | off | 60         # Toggle exact chair retention (YOLACT NPU)
npu-ctl framing on | off            # Auto-framing face tracking
npu-ctl eye-contact on | off        # Teleprompter gaze correction
npu-ctl bg "/path/to/image.jpg"     # Set virtual background
npu-ctl rim-light on warm 50        # Studio rim light
npu-ctl privacy on | off            # Instant privacy screen

# Audio Controls
npu-ctl denoise on | off            # Intel PoCoNet NPU noise removal
npu-ctl studio on | off             # Toggle full vocal broadcast DSP chain
npu-ctl dereverb on 40              # Room echo cancellation
```

### 4. Uninstallation
To cleanly stop background daemons and remove Open NPU Effects:
```bash
./uninstall.sh

# Or for a full purge (including ~/.config/npu-effects and kernel module settings):
./uninstall.sh --purge
```

---

## 📁 Project Architecture

```
OpenNPUEffects/
├── install_all.sh                    # Master 1-click automated installer
├── uninstall.sh                      # Clean automated uninstaller
├── README.md                         # Project documentation
├── NPU_STUDIO_GUIDE.md               # Full user guide and effect catalogue
├── STEP_BY_STEP.md                   # Technical architecture walkthrough
├── LICENSE                           # MIT License
├── scripts/
│   ├── 01_install_npu_drivers.sh     # Intel NPU Level Zero UMD drivers
│   ├── 02_setup_v4l2loopback.sh      # V4L2 virtual camera creation
│   ├── 03_setup_dependencies.sh      # Python venv and system dependencies
│   ├── 04_download_models.sh         # Model downloader and NPU compiler test
│   ├── 05_setup_pipeline_daemons.sh  # Daemons, npu-ctl CLI, and systemd units
│   ├── 06_setup_cameractrls.sh       # Cameractrls GTK UI integration
│   └── 07_setup_users.sh             # Multi-user permission setup
├── src/
│   ├── bin/
│   │   ├── npu_webcam_daemon.py      # Video daemon (OpenVINO NPU + Guided Filter)
│   │   ├── npu_audio_daemon.py       # Audio daemon (PoCoNet NPU + PipeWire DSP)
│   │   └── npu-ctl                   # Fast CLI management tool
│   ├── config/
│   │   └── config.json               # Default configuration schema
│   ├── systemd/
│   │   ├── npu-webcam.service        # Systemd user unit for video
│   │   └── npu-audio.service         # Systemd user unit for audio
│   └── cameractrls/                  # Enhanced Cameractrls GUI & backend
├── models/                           # Pre-compiled static OpenVINO IR models
│   ├── LICENSE.md                    # Individual licenses for all neural models
│   ├── README.md                     # Model specifications and architectures
│   ├── audio/                        # Intel PoCoNet FP16
│   └── video/                        # YuNet, Selfie Multiclass, YOLACT
└── assets/                           # Default privacy and background assets
```

---

## 👥 Multi-User Support

The installation resides globally in `/opt/npu-effects/` so all users on the same machine share the precompiled models and drivers. To enable NPU effects for another user account:

```bash
sudo npu-ctl setup-user username
# Or for all users at once:
sudo npu-ctl setup-all-users
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License & Third-Party AI Models

This project source code is distributed under the **MIT License**. See [`LICENSE`](LICENSE) for complete terms.

### Neural Network Model Licenses
The neural network models bundled or downloaded by this project are subject to their respective open-source licenses:
- **Intel PoCoNet Noise Suppression**: [Apache License 2.0](models/LICENSE.md#1-intel-poconet-noise-suppression) (Intel Corporation)
- **YuNet Face Detector**: [Apache License 2.0](models/LICENSE.md#2-yunet-face-detector-opencv-zoo) (Shiqi Yu / OpenCV Zoo)
- **MediaPipe Selfie Segmentation**: [Apache License 2.0](models/LICENSE.md#3-mediapipe-selfie-segmentation-google-llc) (Google LLC)
- **MediaPipe Selfie Multiclass**: [Apache License 2.0](models/LICENSE.md#4-mediapipe-selfie-multiclass-google-llc) (Google LLC)
- **MobileNetV3 LRASPP Segmenter**: [BSD 3-Clause License](models/LICENSE.md#5-mobilenetv3-lraspp-segmenter-pytorch--torchvision) (TorchVision / PyTorch Contributors)
- **YOLACT Instance Segmenter**: [MIT License](models/LICENSE.md#6-yolact-resnet-50-fpn-instance-segmenter-uc-davis--intel-omz) (Daniel Bolya et al. / UC Davis)

For full license texts and copyright notices, see [`models/LICENSE.md`](models/LICENSE.md).

---

<div align="center">
Developed with ❤️ for the Linux & Intel Core Ultra community.
</div>
