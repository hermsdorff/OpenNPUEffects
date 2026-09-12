# 🛠️ Complete Step-by-Step Technical Guide: Intel NPU Setup & Architecture
### Samsung Galaxy Book4 Ultra — Intel Core Ultra 9 185H — KDE neon / Ubuntu 24.04

This document provides an exhaustive technical breakdown of how the entire Intel NPU (Intel AI Boost) infrastructure was configured for **Full HD 1080p Webcam Processing** and **Broadcast Studio Audio**, enabling you to reproduce, audit, or reinstall each subsystem at any time.

---

## 📂 1. Directory & File Structure

The project directory is structured in a modular, self-contained architecture:

```
OpenNPUEffects/
├── install_all.sh                    # Master 1-click installation script
├── uninstall.sh                      # Master uninstallation and cleanup script
├── STEP_BY_STEP.md                   # This technical step-by-step guide
├── NPU_STUDIO_GUIDE.md               # User manual with effect catalogue and commands
├── scripts/
│   ├── 01_install_npu_drivers.sh     # Level Zero drivers and NPU UMD packages
│   ├── 02_setup_v4l2loopback.sh      # Virtual camera /dev/video72 with 6 buffers
│   ├── 03_setup_dependencies.sh      # System dependencies and Python venv
│   ├── 04_download_models.sh         # Download and optimization of neural models (Audio & Video)
│   ├── 05_setup_pipeline_daemons.sh  # Streaming daemons, npu-ctl utility, and systemd services
│   ├── 06_setup_cameractrls.sh       # Integration into Cameractrls GUI
│   └── 07_setup_users.sh             # User permissions and multi-user profile setup
├── src/
│   ├── bin/                          # Source code for daemons and npu-ctl utility
│   │   ├── npu_webcam_daemon.py      # NPU AI webcam daemon with PTZ control
│   │   ├── npu_audio_daemon.py       # NPU AI audio daemon with vocal DSP
│   │   └── npu-ctl                   # Command-line interface tool (CLI)
│   ├── config/
│   │   └── config.json               # Default configuration parameters and effects
│   ├── systemd/
│   │   ├── npu-webcam.service        # Systemd user unit for webcam service
│   │   └── npu-audio.service         # Systemd user unit for audio service
│   ├── v4l2loopback/
│   │   └── v4l2loopback.conf         # Kernel module configuration
│   └── cameractrls/                  # Integrated files for the Cameractrls GUI
│       ├── cameractrls.py            # Backend with IntelNPUCtrls class
│       ├── cameractrlsgtk.py         # GUI with fixed scrollbar and comfortable window geometry
│       ├── cameractrls               # Executable launcher in PATH
│       └── hu.irl.cameractrls.desktop # Application menu desktop shortcut
├── models/                           # Pre-compiled static OpenVINO IR models
│   ├── LICENSE.md                    # Individual licenses for all neural models
│   ├── README.md                     # Model specifications and architectures
│   ├── audio/                        # Intel PoCoNet FP16 for deep noise suppression
│   └── video/                        # YuNet (face) + Selfie Segmentation + YOLACT (Chair/Objects)
└── drivers/                          # Official Intel NPU driver .deb packages
    ├── libze1_1.28.2.deb
    ├── intel-fw-npu_*.deb
    ├── intel-driver-compiler-npu_*.deb
    └── intel-level-zero-npu_*.deb
```

---

## ⚡ 2. How to Run the Installation

### Option A: Automated Complete Installation (Recommended)
To run all installation stages in sequence with automated health checks:
```bash
git clone https://github.com/hermsdorff/OpenNPUEffects.git
cd OpenNPUEffects
./install_all.sh
```

### Option B: Step-by-Step Manual Installation
You can execute each script individually in numeric sequence:
```bash
cd scripts
./01_install_npu_drivers.sh
./02_setup_v4l2loopback.sh
./03_setup_dependencies.sh
./04_download_models.sh
./05_setup_pipeline_daemons.sh
./06_setup_cameractrls.sh
./07_setup_users.sh [username|--all]
```

---

## 🔍 3. Technical Breakdown of Each Stage

---

### STAGE 1: Intel NPU Hardware & Kernel Drivers (`01_install_npu_drivers.sh`)

#### What it does:
1. **Kernel Driver Verification**: Ensures the `intel_vpu` kernel module (Linux kernel driver communicating with the Meteor Lake NPU coprocessor) is loaded.
2. **Installation of Official Intel Driver Packages**:
   - `libze1` (v1.28.2): OneAPI Level Zero library loader.
   - `intel-fw-npu`: Official binary firmware loaded onto the NPU hardware chip (`vpu_37xx_v1.bin`).
   - `intel-driver-compiler-npu`: Microcode graph compiler converting OpenVINO intermediate representations into instructions for the neural SHAVE processing cores.
   - `intel-level-zero-npu`: User-space driver implementation (`libze_intel_npu.so.1`).
3. **User & Group Permissions**:
   - The NPU device node is exposed by the kernel at `/dev/accel/accel0` under the `render` group.
   - The script adds the user to the `render` and `video` groups (`sudo usermod -aG render,video $USER`).
4. **UDEV Rules**: Creates `/etc/udev/rules.d/10-intel-vpu.rules` ensuring `/dev/accel/accel*` is automatically granted read/write permissions for the `render` group without requiring root privileges.

#### How to audit/verify:
```bash
# Check presence of the NPU device node:
ls -l /dev/accel/accel0

# Verify active kernel driver messages:
dmesg | grep -i vpu
```

---

### STAGE 2: V4L2 Virtual Camera Device (`02_setup_v4l2loopback.sh`)

#### What it does:
1. Installs dynamic kernel module support (`v4l2loopback-dkms`) and video utilities (`v4l-utils`).
2. Creates persistent kernel module options at `/etc/modprobe.d/v4l2loopback.conf`:
   ```ini
   options v4l2loopback devices=3 video_nr=70,71,72 card_label="Iriun Webcam","OBS Virtual Cam","Intel NPU Enhanced Webcam" exclusive_caps=1,1,1 max_buffers=6
   ```
   - **`video_nr=70,71,72`**: Allocates high, deterministic minor numbers to prevent collisions with physical USB webcams on `/dev/video0` or `/dev/video1`.
   - **`card_label`**: Sets the human-readable display name *"Intel NPU Enhanced Webcam"*, making device selection seamless in Google Meet, Zoom, Teams, OBS, and Slack.
   - **`max_buffers=6`**: **Critical requirement!** Cameractrls (`cameraview.py:509`) strictly requires 6 streaming queue buffers. Without this setting, Cameractrls fails with an *insufficient buffer memory* error.
   - **`exclusive_caps=1`**: Prevents web browsers from attempting to open the virtual camera as an output device instead of a capture device.
3. Adds `v4l2loopback` to `/etc/modules-load.d/v4l2loopback.conf` to guarantee loading on every system boot.

#### How to audit/verify:
```bash
v4l2-ctl --device=/dev/video72 --all
```

---

### STAGE 3: System Dependencies & Python Runtime (`03_setup_dependencies.sh`)

#### What it does:
1. Installs system packages and math libraries: `python3-venv`, `python3-pip`, `build-essential`, `libopenblas-dev`, `ffmpeg`, `libportaudio2`.
2. Creates a centralized, system-wide Python virtual environment at:
   👉 `/opt/npu-effects/venv/` (with execute permissions for all local users).
3. Installs compatible Python package versions:
   - **`openvino`**: Intel's official runtime with native backend support for Meteor Lake NPU hardware.
   - **`opencv-python-headless`**: High-performance computer vision with AVX2/AVX-512 acceleration.
   - **`sounddevice`**: Real-time low-latency audio capture and playback via ALSA / PulseAudio / PipeWire.
   - **`pyvirtualcam`**: High-throughput zero-copy frame injection into `/dev/video72` in YUV420p format without external conversion overhead.
   - **`numpy` & `scipy`**: Vectorized tensor operations for vocal DSP filters and spatial interpolation.

#### How to audit/verify:
```bash
/opt/npu-effects/venv/bin/python3 -c "import openvino as ov; print(ov.Core().available_devices)"
# Expected output: ['CPU', 'GPU', 'NPU']
```

---

### STAGE 4: AI Model Deployment & Optimization (`04_download_models.sh`)

#### What it does:
Deploys models centrally into `/opt/npu-effects/models/` (shared in read-only mode for all system users):

1. **Audio Model — Intel PoCoNet (`noise-suppression-poconetlike-0001`)**:
   - Deep recurrent neural network for vocal isolation and acoustic noise cancellation developed by Intel.
   - Processes 128ms audio chunks in ~30ms on the NPU, guaranteeing continuous real-time processing with low latency.
   - Stored in native OpenVINO IR format (`.xml` topology + `.bin` FP16 weights).
2. **Video Model — Facial Detection & Tracking (`face_detection_yunet_2023mar.onnx`)**:
   - Ultra-lightweight facial detector from OpenCV Zoo with rotation support and 5-point facial landmarks (eyes, nose, mouth corners).
   - Runs in less than 1.2ms per frame, powering kinematic smooth auto-framing.
3. **Video Model — Static Person Segmentation (`selfie_segmentation_static`)**:
   - Full-body segmentation model with static `[1, 3, 144, 256]` input shape, customized for optimal NPU SHAVE core pipelining.
   - Executes background segmentation in **~2.4 ms per frame (~400 FPS)** on the NPU.
4. **Video Model — Neural Chair Retention (`chair_instance_segmenter`)**:
   - Instance segmentation model based on **YOLACT ResNet-50 FPN** (MIT License).
   - Distinguishes gaming/office chairs and couches from human subjects to retain chair backs without unnatural polygon bounding halos.

---

### STAGE 5: Background Daemons & Systemd Services (`05_setup_pipeline_daemons.sh`)

#### What it does:
1. **Daemon Installation**:
   - Copies [`npu_webcam_daemon.py`](src/bin/npu_webcam_daemon.py) and [`npu_audio_daemon.py`](src/bin/npu_audio_daemon.py) into `/opt/npu-effects/bin/`.
   - Reads individual user configuration and runtime parameters from `~/.config/npu-effects/config.json`.
2. **Installation of `npu-ctl` CLI Utility**:
   - Creates a symbolic link in `/usr/local/bin/npu-ctl` (globally accessible in every user's `$PATH`).
3. **Global User Systemd Services**:
   - Installs unit files `npu-webcam.service` and `npu-audio.service` into `/etc/systemd/user/`.
   - Being located in `/etc/systemd/user/`, any user on the machine can manage their personal daemon instances without duplicating service unit definitions.
   - Enables and starts the services for the installing user (`systemctl --user enable --now ...`).

#### How to audit/verify:
```bash
# Check service execution status:
systemctl --user status npu-webcam.service npu-audio.service

# View interactive CLI control center:
npu-ctl status
```

---

### STAGE 6: Cameractrls GUI Integration (`06_setup_cameractrls.sh`)

#### What it does:
1. Installs GTK binding dependencies (`python3-gi`, `gir1.2-gtk-3.0`, `gir1.2-gtk-4.0`).
2. Installs the Cameractrls application base in `/opt/npu-effects/cameractrls/`.
3. Injects a dedicated **"Intel NPU"** control tab containing sliders for AI Blur, Chair Retention, Auto-Framing, Teleprompter Eye Contact, Studio Lighting, Hand Gestures, and Audio Denoising linked directly to `~/.config/npu-effects/config.json`.
4. Enhances the GTK layout with persistent vertical scrollbars and comfortable window dimensions.
5. Configures the global launcher `/usr/local/bin/cameractrls` and desktop menu entry `/usr/share/applications/hu.irl.cameractrls.desktop`.

---

### STAGE 7: Multi-User Profiles & Access (`07_setup_users.sh`)

#### What it does:
1. Adds target user accounts to the hardware groups `render` (NPU `/dev/accel/accel0` access) and `video` (physical & virtual camera access).
2. Provisions user configuration directories at `~/.config/npu-effects/` and background images at `~/.config/npu-effects/backgrounds/`.
3. Enables configuring a specific user or all system users with a single command:
   ```bash
   # Configure a specific user:
   sudo ./scripts/07_setup_users.sh username
   # Or using npu-ctl:
   npu-ctl setup-user username

   # Configure ALL user accounts at once:
   sudo ./scripts/07_setup_users.sh --all
   # Or using npu-ctl:
   npu-ctl setup-all-users
   ```

---

## 🎯 4. Testing & Validation

After running the installer scripts, execute the following validation steps:

### 1. Test 1080p Video on the NPU:
```bash
# Open Cameractrls graphical interface:
cameractrls
```
- In the camera dropdown, select: **`Intel NPU Enhanced Webcam (/dev/video72)`**.
- Click the Camera preview icon in the top right corner.
- Confirm Full HD 1080p video stream with real-time background blur and face tracking.

### 2. Test Audio & Listen to Noise Suppression:
```bash
# Loop your live microphone back through headphones with NPU filtering active:
npu-ctl mic-listen on

# When finished testing:
npu-ctl mic-listen off

# Or record a 5-second test clip:
npu-ctl audio-test 5
```

### 3. Verify Hardware Camera & Microphone Detection:
```bash
# List detected physical cameras:
npu-ctl cam-source

# List detected physical microphones:
npu-ctl mic-source
```

---

## 🔧 5. Troubleshooting & FAQ

### Camera reports "Device or resource busy" on startup:
- Another application may have an exclusive lock on the physical camera node (`/dev/video0`). Close apps like Zoom, Teams, or browser tabs, then restart the daemon:
  ```bash
  npu-ctl restart
  ```

### Cameractrls NPU controls disappeared after a repository update:
- If Cameractrls is re-cloned or upgraded, restore the custom NPU tab by running:
  ```bash
  npu-ctl restore-cameractrls
  ```

### Galaxy Book4 Ultra internal webcam returns after kernel update:
- When the Intel IPU6 driver initializes `/dev/video0`, switch the active input device by running:
  ```bash
  npu-ctl cam-source notebook
  ```

---

## 🗑️ 6. How to Uninstall (`uninstall.sh`)

To cleanly remove Open NPU Effects and restore original system settings:

```bash
# Standard uninstallation (removes services, /opt/npu-effects, and shortcuts while keeping user configs):
./uninstall.sh

# Complete purge (also removes ~/.config/npu-effects and kernel module options):
./uninstall.sh --purge
```

---

*For complete effect descriptions, hotkeys, and command-line usage, refer to the user manual:*
👉 [**`NPU_STUDIO_GUIDE.md`**](NPU_STUDIO_GUIDE.md)
