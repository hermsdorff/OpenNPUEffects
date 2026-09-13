# 🚀 Complete Guide: Intel NPU AI Studio (Webcam & Audio)
### Samsung Galaxy Book4 Ultra — Intel Core Ultra 9 185H (Intel AI Boost) — KDE neon 24.04

This document covers everything about the deployed architecture, daily workflows, the complete effects catalogue, and how to manage the system via the graphical interface (**Cameractrls**) and the command-line utility (**`npu-ctl`**).

---

## 📋 Table of Contents
1. [Overview & Architecture](#1-overview--architecture)
2. [Daily Usage & Workflow](#2-daily-usage--workflow)
3. [Video Effects Catalogue (AI & Computer Vision)](#3-video-effects-catalogue)
4. [Audio Effects Catalogue (NPU & Vocal Studio DSP)](#4-audio-effects-catalogue)
5. [Graphical Control (Cameractrls GUI)](#5-graphical-control-cameractrls-gui)
6. [Command-Line Interface (`npu-ctl`)](#6-command-line-interface-npu-ctl)
7. [Hardware Device Selection (Cameras & Microphones)](#7-hardware-device-selection)
8. [Configuration File (`config.json`)](#8-configuration-file)
9. [System Updates & Persistence](#9-system-updates--persistence)
10. [Multi-User Support](#10-multi-user-support)
11. [Licenses & AI Model Terms](#11-licenses--ai-model-terms)

---

## 1. Overview & Architecture

We have established a complete hardware-accelerated video and audio processing suite running on the **integrated Neural Processing Unit (NPU) of the Intel Core Ultra 9 processor**, functioning natively and transparently on Linux:

```
[Physical Devices]
 ├─ Physical Webcam (EMEET PIXY / Laptop IPU6) ──► npu-webcam.service (OpenVINO + PTZ) ──► /dev/video72 (v4l2loopback 1080p)
 └─ Physical Microphone (USB / PIXY / Laptop)    ──► npu-audio.service  (PoCoNet + DSP)   ──► PipeWire (npu_clearvoice)
                                                           ▲
                                                   Intel AI Boost NPU
                                                   (/dev/accel/accel0)
```

### Core Configured Components:
1. **Intel NPU Drivers**:
   - Official binary firmware `intel-fw-npu` (`vpu_37xx_v1.bin`) loaded by the kernel.
   - Level Zero user-mode driver (`intel-level-zero-npu` and `intel-driver-compiler-npu`).
   - Device node `/dev/accel/accel0` integrated directly with OpenVINO Runtime.
2. **V4L2 Virtual Camera (`v4l2loopback`)**:
   - Kernel module configured in `/etc/modprobe.d/v4l2loopback.conf` exposing `/dev/video72` with the friendly label `"Intel NPU Enhanced Webcam"`.
   - Native support for 6 streaming queue buffers (`max_buffers=6`), ensuring total compatibility with Cameractrls and stutter-free 1080p @ 30 FPS video delivery.
3. **Background Daemons (`systemd --user`)**:
   - `npu-webcam.service`: Video daemon with automatic startup, on-demand standby, and motorized PTZ gimbal control.
   - `npu-audio.service`: Audio daemon with instant hot-reload upon configuration changes.
4. **Command-Line Management (`npu-ctl`)**:
   - Installed globally to `/usr/local/bin/npu-ctl` for adjusting any effect parameter in real-time.
5. **Cameractrls Graphical Integration**:
   - Dedicated "Intel NPU" tab in Cameractrls featuring intuitive sliders, toggles, comfortable window geometry, and permanent vertical scrollbars.

---

## 2. Daily Usage & Workflow

### In Video Conferencing Applications (Google Meet, Teams, Zoom, Discord, Browsers):
1. **Video**:
   - In any application, select as your camera:
     👉 **`Intel NPU Enhanced Webcam`** (or `/dev/video72`).
   - It delivers Full HD 1080p video with all active effects applied.
2. **Microphone**:
   - Select as your input device:
     👉 **`Intel NPU ClearVoice Microphone`** (`npu_clearvoice`).
   - It is automatically set as the default audio source in PipeWire.

### Smart On-Demand Behavior (Standby & PTZ Hardware Parking):
* When **no application** is actively consuming the virtual camera feed, it enters low-power standby after 3 seconds: the camera LED turns off, the physical sensor is released, and the **EMEET PIXY** physically tilts downward (tilt `-324000`), guaranteeing complete physical privacy.
* The moment you enter a meeting or open the Cameractrls preview, the camera wakes up in under **0.3 seconds**, resets tilt to the forward-facing position, and streams immediately.

---

## 3. Video Effects Catalogue

All video effects run accelerated on the NPU and CPU with AVX-512/AVX2 SIMD vector extensions, maintaining latency well below 33 ms (solid 30 FPS at 1080p).

| Effect | Description | How to Configure |
| :--- | :--- | :--- |
| **Background Blur (AI Blur)** | Segments the subject using neural models on the NPU and applies silky bokeh blur to the background. | `npu-ctl blur [on/off/0-100]` or slider in Cameractrls |
| **Optical Portrait Mode (Depth-Aware)** | Emulates physical DSLR lenses with gradual depth blur (increasing bokeh on distant surfaces). | `npu-ctl blur-mode [standard/portrait]` |
| **Edge Transition (Feathering)** | Hermite smoothstep boundary blending eliminating harsh cutouts or halo artifacts around hair. | `npu-ctl feather [10-80]` or slider in Cameractrls |
| **Chair & Furniture Retention (NPU AI)** | Retains gaming/office chair headrests, wings, and backrests in the foreground using YOLACT instance segmentation. | `npu-ctl chair [on/off/10-100]` or in Cameractrls |
| **Virtual Background (Image Replacement)** | Replaces the physical background with any custom photo stored in `~/.config/npu-effects/backgrounds/`. | `npu-ctl background /path/to/photo.jpg` |
| **3D Parallax Perspective** | Subtly shifts background perspective based on head movement, providing realistic three-dimensional depth. | `npu-ctl parallax [on/off/10-100]` |
| **Hand Gesture Recognition (AI)** | Real-time recognition of gestures (**👍 Thumbs Up**, **✌️ Peace**, **🖐️ Open Palm**) for visual animated reactions or mic mute shortcut. | `npu-ctl gesture [on/off/reaction/mute/all]` or in Cameractrls |
| **Handheld Object Retention** | Prevents held smartphones, mugs, pens, notebooks, and documents from blurring into the background. | `npu-ctl objects [on/off/10-100]` or in Cameractrls |
| **Absence Privacy Fade** | Smoothly dissolves (*crossfade*) video when entering/leaving the privacy screen, eliminating abrupt cuts. | `npu-ctl privacy-fade [on/off]` or in Cameractrls |
| **Artistic Creative Filters** | Windows Studio Effects-inspired styling: **Illustrated (HQ)**, **Animated (Cartoon)**, and **Watercolor**. | `npu-ctl artistic [off/illustrated/animated/watercolor]` |
| **Cinematic Color Grading (LUTs)** | Professional cinematic palettes: **Teal & Orange**, **Warm Vintage**, **Noir B&W**, and **Vibrant Pop**. | `npu-ctl color [none/teal_orange/vintage/noir/vibrant]` |
| **Virtual Rim Light** | Three-point studio contour highlight along hair edges and shoulders (Warm Amber, Clean White, or Cool Blue). | `npu-ctl rim [on/off/10-100] [warm/white/cool]` |
| **Monitor Glare Suppression** | Neutralizes bluish light cast onto faces and glasses by bright monitors in dark environments. | `npu-ctl glare [on/off/10-100]` |
| **Auto-Framing (Single & Group)** | Smooth cinematic face centering for one person, or dynamic zoom widening to keep everyone in frame (*Group Framing*). | `npu-ctl framing [on/off]` / `npu-ctl framing-mode [single/group]` |
| **Eye Contact (Natural & Teleprompter)** | Realigns irises toward the lens. **Teleprompter** mode applies extra vertical correction for reading notes on screen. | `npu-ctl eyecontact [off/natural/teleprompter]` |
| **Skin Smoothing & Denoise** | Bilateral filtering that balances skin tone and removes sensor noise without softening eyes or hair. | `npu-ctl smooth [on/off/10-100]` |
| **Super-Resolution & Sharpening** | Adaptive edge unsharp-masking enhancing fine textures in eyes, clothing, and hair. | `npu-ctl sharpen [on/off/10-100]` |
| **Studio Light (Virtual Ring Light)** | Emulates a diffuse softbox or ring light illuminating the face from the front. | `npu-ctl studio [on/off/10-100]` |
| **Adaptive Low-Light Compensation** | Smart tone-mapping curve that brightens dark scenes without washing out deep blacks. | `npu-ctl lowlight [on/off/10-100]` |
| **Absence Privacy Mode** | Replaces video with an elegant pause card when you step away from the camera for more than 3 seconds. | `npu-ctl privacy [on/off]` |
| **Automatic On-Demand Standby** | Powers down the physical camera sensor and indicator LED when no applications are reading `/dev/video72`. | `npu-ctl standby [on/off]` |
| **Motorized PTZ Parking** | Automatically tilts the EMEET PIXY downward (`tilt = -324000`) during standby and reopens upon wake-up. | `npu-ctl ptz-park [on/off]` |

---

## 4. Audio Effects Catalogue

The audio pipeline combines the **Intel PoCoNet** neural model running on the NPU with a broadcast-grade vocal digital signal processing (DSP) chain:

```
Physical Microphone
  └──► Low-Cut Filter (80Hz)
        └──► NPU PoCoNet Neural Denoise
              └──► De-Reverb Filter
                    └──► Smart Noise Gate (VAD)
                          └──► Vocal Presence EQ
                                └──► Vocal Compressor / AGC (-18dB)
                                      └──► Dynamic De-Esser (5.2-7.5kHz)
                                            └──► Virtual PipeWire Node (npu_clearvoice)
```

| Effect | Description | How to Configure |
| :--- | :--- | :--- |
| **AI Noise Suppression (PoCoNet)** | Recurrent neural network running on the NPU that eliminates dog barking, vacuum cleaners, wind, traffic, and fan noise while preserving clean voice. | `npu-ctl mic-noise [on/off]` |
| **Vocal Studio Suite (Studio Mic)** | Master switch that activates the entire broadcast acoustic post-processing chain. | `npu-ctl studio-mic [on/off]` |
| **High-Pass / Low-Cut Filter (80Hz)** | Eliminates inaudible sub-bass vibrations: desk thumps, mechanical keyboard typing rumble, and foot tapping. | `npu-ctl mic-lowcut [on/off]` |
| **Smart Noise Gate (Smart Gate)** | Enforces **absolute digital silence** during breathing pauses and silence using Voice Activity Detection (VAD) with soft attack/decay envelopes. | `npu-ctl mic-gate [-30 to -60 dB]` |
| **Vocal Presence Equalizer (Vocal EQ)** | Dips muddy frequencies at 250 Hz (eliminating the boxy sound) and boosts 3,500 Hz (introducing air, presence, and crisp articulation). | `npu-ctl mic-eq [on/off]` |
| **Vocal Compressor & AGC** | Automatically balances vocal volume to broadcast standard (-18 dBFS). Quiet speech is raised, and loud peaks are gently compressed. | `npu-ctl mic-comp [-12 to -28 dB]` |
| **Dynamic De-Esser** | Suppresses harsh, ear-piercing sibilance ("S", "SH", and "CH" sounds) in the 5.2 kHz to 7.5 kHz spectrum. | `npu-ctl mic-deesser [on/off]` |
| **Acoustic De-Reverb** | Subtracts room reflections and reverberation typical of untreated rooms with hard surfaces or bare walls. | `npu-ctl mic-dereverb [10-100]` |

### Voice Testing & Monitoring Tools:
* **Real-time headphone monitoring**:
  ```bash
  npu-ctl mic-listen on   # Enables live headphone loopback to audition effects
  npu-ctl mic-listen off  # Disables loopback
  ```
* **Quick 5-second test recording**:
  ```bash
  npu-ctl audio-test 5    # Records 5 seconds through the NPU and plays it back
  ```

---

## 5. Graphical Control (Cameractrls GUI)

The camera control utility features custom modifications tailored to Open NPU Effects:

1. Launch **Cameractrls** from the application menu or terminal (`cameractrls`).
2. Select **`Intel NPU Enhanced Webcam (/dev/video72)`**.
3. **UI Enhancements**:
   - **Comfortable Window Sizing**: Opens at 520x680px with a locked minimum height of 580px to prevent cramped views.
   - **Permanent Scrollbars**: The vertical scrollbar remains visible at all times, making navigation through all effects smooth.
4. **"Intel NPU" Control Tab**:
   - Sliders and switches for background blur, chair retention, framing speed, studio lighting, standby power saving, PTZ gimbal tilt, and audio effects.
5. **Live Video Preview**:
   - Click the Camera preview icon in the upper-right corner to see the 1080p stream and observe changes in real time.

---

## 6. Command-Line Interface (`npu-ctl`)

The `npu-ctl` command-line utility provides instant control from any terminal:

```bash
# === STATUS & DIAGNOSTICS ===
npu-ctl status                  # Comprehensive status of NPU, devices, and effects
npu-ctl make-default            # Set virtual camera & mic as system & browser defaults
npu-ctl logs [cam|audio]        # Follow live service logs

# === SERVICE MANAGEMENT ===
npu-ctl restart                 # Restart both video and audio services
npu-ctl start | npu-ctl stop    # Start or stop services

# === VIDEO & COMPUTER VISION ===
npu-ctl blur on | off           # Toggle background blur
npu-ctl blur 45                 # Adjust blur intensity (0 to 100%)
npu-ctl blur-mode standard|portrait # Uniform blur vs Depth-aware optical portrait
npu-ctl feather 40              # Edge feathering transition width (10 to 80px)
npu-ctl background /path/to/img # Set virtual background image (or 'none' to clear)
npu-ctl parallax on | off | 40  # 3D parallax perspective effect
npu-ctl gesture on | off | reaction | mute | all # Hand gesture recognition (👍, ✌️, 🖐️)
npu-ctl objects on | off | 60   # Handheld object retention
npu-ctl chair on | off | 50     # Precise gaming/office chair retention
npu-ctl privacy-fade on | off   # Smooth crossfade transition on absence
npu-ctl artistic off|illustrated|animated|watercolor [10-100] # Creative filters
npu-ctl color none|teal_orange|vintage|noir|vibrant [10-100]  # Cinematic LUTs
npu-ctl rim on | off | 50 warm|white|cool # Studio rim/hair contour lighting
npu-ctl glare on | off | 50     # Monitor blue glare reduction
npu-ctl framing on | off        # Auto-framing face tracking
npu-ctl framing-mode single|group # Single subject vs Group framing
npu-ctl framing smooth 3        # Framing pan speed (1=slow cinematic, 10=fast)
npu-ctl eyecontact off|natural|teleprompter # Eye contact correction
npu-ctl smooth 35               # Skin smoothing & noise reduction (10 to 100)
npu-ctl studio on | 50 | off    # Virtual ring light (studio fill light)
npu-ctl lowlight on | 40 | off  # Low-light compensation
npu-ctl sharpen on | 35 | off   # Edge sharpening and super-resolution
npu-ctl privacy on | off        # Absence privacy mode

# === POWER, STANDBY & PTZ GIMBAL ===
npu-ctl standby on | off        # Automatic low-power standby
npu-ctl ptz-park on | off       # Motorized physical camera parking in standby
npu-ctl ptz-delay 1.0           # Delay (s) before driving gimbal motors
npu-ctl ptz-tilt -324000 0      # Configure standby and active tilt positions
npu-ctl ptz-test -324000        # Test PTZ motor movement immediately

# === VOCAL STUDIO AUDIO ===
npu-ctl mic-noise on | off      # Intel PoCoNet neural noise suppression
npu-ctl studio-mic on | off     # Broadcast vocal studio processing suite
npu-ctl mic-gate -45            # Absolute digital silence gate (-30 to -60 dB)
npu-ctl mic-comp -18            # Vocal target compressor level (-12 to -28 dB)
npu-ctl mic-eq on | off         # Vocal presence equalizer
npu-ctl mic-deesser on | off    # Sibilance de-esser (S/SH)
npu-ctl mic-dereverb 40         # Room echo & reflection reduction (10-100%)
npu-ctl mic-lowcut on | off     # 80 Hz high-pass sub-bass filter
npu-ctl mic-listen on | off     # Live headphone voice loopback
npu-ctl audio-test 5            # Record and playback a 5-second test clip
```

---

## 7. Hardware Device Selection

You can select which physical webcam and microphone feed the AI processing pipelines:

### Webcams:
```bash
# List all detected physical cameras and identify the active source:
npu-ctl cam-source

# Select input device:
npu-ctl cam-source pixy        # Use external EMEET PIXY (/dev/video1)
npu-ctl cam-source notebook    # Use built-in laptop webcam (/dev/video0)
npu-ctl cam-source auto        # Automatic detection
```
> When the built-in webcam on the Galaxy Book4 Ultra is enabled after Linux IPU6 driver updates, switch directly by executing `npu-ctl cam-source notebook`.

### Microphones:
```bash
# List all detected physical microphones in PipeWire:
npu-ctl mic-source

# Select input device:
npu-ctl mic-source pixy        # EMEET PIXY built-in microphone
npu-ctl mic-source usb         # Dedicated external USB microphone
npu-ctl mic-source notebook    # Built-in laptop microphone array
npu-ctl mic-source auto        # Automatic (prioritizes USB, then internal)
```

---

## 8. Configuration File

All user preferences are maintained in a structured JSON file:
👉 `~/.config/npu-effects/config.json`

The audio daemon hot-reloads this file immediately upon modification (**without dropping ongoing calls**). The video daemon applies parameter updates on each inference cycle.

Example structure of `config.json`:
```json
{
  "device": "NPU",
  "video_device": "auto",
  "input_device": "auto",
  "effects": {
    "blur": { "enabled": true, "strength": 40 },
    "virtual_background": { "enabled": false, "image_path": "" },
    "auto_framing": { "enabled": true, "smoothness": 3.0, "deadzone": 0.10 },
    "skin_smoothing": { "enabled": true, "strength": 35 },
    "sharpen": { "enabled": false, "strength": 35 },
    "studio_light": { "enabled": false, "intensity": 40 },
    "low_light": { "enabled": false, "gain": 40 },
    "eye_contact": { "enabled": false },
    "feather": 40,
    "standby_on_idle": true,
    "idle_timeout_sec": 3.0,
    "ptz_park_enabled": true,
    "ptz_standby_tilt": -324000,
    "ptz_wakeup_tilt": 0
  },
  "audio_effects": {
    "noise_suppression": { "enabled": true },
    "studio_voice": { "enabled": true },
    "low_cut": { "enabled": true, "cutoff_hz": 80 },
    "smart_gate": { "enabled": true, "threshold_db": -45.0 },
    "vocal_eq": { "enabled": true },
    "compressor": { "enabled": true, "target_db": -18.0 },
    "deesser": { "enabled": true },
    "dereverb": { "enabled": true, "strength": 0.4 }
  }
}
```

---

## 9. System Updates & Persistence

* **KDE neon / Ubuntu OS Updates (`pkcon`, `apt upgrade`):**
  - **100% preserved.** Model binaries and libraries reside in `/opt/npu-effects/` while user preferences reside in `~/.config/npu-effects/`. Operating system package upgrades will not overwrite these directories.
* **Kernel Upgrades (`v4l2loopback`):**
  - Managed via DKMS and `/etc/modprobe.d/v4l2loopback.conf`. The virtual camera kernel module is automatically recompiled for each new installed kernel version.
* **Reinstalling or Updating Cameractrls via GitHub:**
  - If you re-clone Cameractrls or pull upstream updates, re-apply the NPU integration instantly:
    ```bash
    npu-ctl restore-cameractrls
    ```
  - Backups of modified files are saved with the `.npu_backup` extension.

---

## 10. Multi-User Support

The architecture enables **all user accounts on the machine** to share a single centralized installation:
- **Centralized (Shared):** Python virtual environment, AI models, and binaries in `/opt/npu-effects/`.
- **Per-User (Isolated):** Individual configuration files (`~/.config/npu-effects/config.json`) and backgrounds (`~/.config/npu-effects/backgrounds/`).

### Provisioning access for additional users:
```bash
# 1. Enable for a specific username:
sudo npu-ctl setup-user username
# Or:
sudo ./scripts/07_setup_users.sh username

# 2. Enable for ALL local users at once:
sudo npu-ctl setup-all-users
# Or:
sudo ./scripts/07_setup_users.sh --all
```

When logging into the newly provisioned user account:
- The **Cameractrls** desktop shortcut is immediately available.
- The `npu-ctl` CLI command is accessible in terminal.
- Running `npu-ctl start` initializes video and audio effects for that user's session.

---

## 11. Licenses & AI Model Terms

The source code of **Open NPU Effects** is distributed under the **MIT License** (see [`LICENSE`](LICENSE)).

### Individual AI Model Licenses
The artificial intelligence models bundled or prepared by the installer operate under their respective permissive open-source licenses:
- **Intel PoCoNet (Noise Suppression)**: [Apache 2.0](models/LICENSE.md#1-intel-poconet-noise-suppression) (Intel Open Model Zoo)
- **YuNet (Face Detection & Landmarks)**: [Apache 2.0](models/LICENSE.md#2-yunet-face-detector-opencv-zoo) (Shiqi Yu / OpenCV Zoo)
- **MediaPipe Selfie Segmentation**: [Apache 2.0](models/LICENSE.md#3-mediapipe-selfie-segmentation-google-llc) (Google LLC)
- **MediaPipe Selfie Multiclass**: [Apache 2.0](models/LICENSE.md#4-mediapipe-selfie-multiclass-google-llc) (Google LLC)
- **MobileNetV3 LRASPP Segmenter**: [BSD 3-Clause](models/LICENSE.md#5-mobilenetv3-lraspp-segmenter-pytorch--torchvision) (TorchVision / PyTorch)
- **YOLACT Instance Segmenter (Chair Retention)**: [MIT](models/LICENSE.md#6-yolact-resnet-50-fpn-instance-segmenter-uc-davis--intel-omz) (Daniel Bolya et al. / UC Davis)

For complete license texts and attribution notices, see [`models/LICENSE.md`](models/LICENSE.md).

---

*Designed and calibrated for the Linux & Intel Core Ultra community.*
