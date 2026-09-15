#!/usr/bin/env python3
"""
Intel NPU Webcam Daemon
Offloads AI Selfie Segmentation & Auto-Framing to Intel Meteor Lake NPU.
Features Edge-Preserving Noise Reduction (Skin Smoothing / Denoising) in 1080p.
Outputs to /dev/video72 (v4l2loopback) for transparent use in any application.
"""
import os
import sys
import time
import json
import signal
import subprocess
import logging
from pathlib import Path
import numpy as np
import cv2
import openvino as ov
import pyvirtualcam

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [NPU-CAM] %(levelname)s: %(message)s'
)

CONFIG_PATH = Path.home() / ".config/npu-effects/config.json"
USER_MODEL_DIR = Path.home() / ".local/share/npu-effects/models/video"
OPT_MODEL_DIR = Path("/opt/npu-effects/models/video")
REPO_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models/video"

MODEL_DIR = OPT_MODEL_DIR if OPT_MODEL_DIR.exists() else (USER_MODEL_DIR if USER_MODEL_DIR.exists() else Path("/opt/npu-effects/models/video"))
SEG_MULTICLASS_PATH = MODEL_DIR / "selfie_multiclass.xml"
SEG_MODEL_PATH = MODEL_DIR / "selfie_segmentation_static.xml"
YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"

# MODNet Portrait Matting (Apache 2.0) - optional alternative to Selfie Multiclass with
# higher-fidelity edges (hair/fingers), selectable via cfg["video"]["segmentation_model"] == "modnet"
SEG_MODNET_PATH = None
for p in [
    USER_MODEL_DIR / "modnet_portrait_matting.xml",
    REPO_MODEL_DIR / "modnet_portrait_matting.xml",
    OPT_MODEL_DIR / "modnet_portrait_matting.xml",
    MODEL_DIR / "modnet_portrait_matting.xml",
]:
    if p.exists():
        SEG_MODNET_PATH = p
        break
if SEG_MODNET_PATH is None:
    SEG_MODNET_PATH = MODEL_DIR / "modnet_portrait_matting.xml"

SEG_CHAIR_PATH = None
for p in [
    USER_MODEL_DIR / "chair_instance_segmenter.xml",
    REPO_MODEL_DIR / "chair_instance_segmenter.xml",
    OPT_MODEL_DIR / "chair_instance_segmenter.xml",
    MODEL_DIR / "chair_instance_segmenter.xml",
    MODEL_DIR / "chair_segmenter.xml"
]:
    if p.exists():
        SEG_CHAIR_PATH = p
        break
if SEG_CHAIR_PATH is None:
    SEG_CHAIR_PATH = MODEL_DIR / "chair_instance_segmenter.xml"

SEG_GLASSES_PATH = None
for p in [
    USER_MODEL_DIR / "face_parsing_bisenet.xml",
    REPO_MODEL_DIR / "face_parsing_bisenet.xml",
    OPT_MODEL_DIR / "face_parsing_bisenet.xml",
    MODEL_DIR / "face_parsing_bisenet.xml",
]:
    if p.exists():
        SEG_GLASSES_PATH = p
        break
if SEG_GLASSES_PATH is None:
    SEG_GLASSES_PATH = MODEL_DIR / "face_parsing_bisenet.xml"

USER_ASSETS_DIR = Path.home() / ".local/share/npu-effects/assets"
OPT_ASSETS_DIR = Path("/opt/npu-effects/assets")
REPO_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"

DEFAULT_PRIVACY_IMAGE = None
for p in [
    Path.home() / ".config/npu-effects/absence/Ausente.png",
    Path.home() / ".local/share/npu-effects/absence/Ausente.png",
    REPO_ASSETS_DIR / "absence" / "Ausente.png",
    Path("/home/kleber/picture/ausente.png"),
    Path("/home/kleber/Pictures/ausente.png"),
    OPT_ASSETS_DIR / "absence" / "Ausente.png",
    OPT_ASSETS_DIR / "default_privacy.png",
    USER_ASSETS_DIR / "default_privacy.png",
    REPO_ASSETS_DIR / "default_privacy.png",
]:
    if p.exists():
        DEFAULT_PRIVACY_IMAGE = p
        break

running = True

def handle_signal(sig, frame):
    global running
    logging.info("Shutting down webcam daemon...")
    running = False

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def load_config():
    default_config = {
        "video": {
            "enabled": True,
            "input_device": "auto",
            "output_device": "/dev/video72",
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "blur_enabled": True,
            "blur_strength": 35,
            "blur_mode": "standard",
            "mask_feather": 40,
            "segmentation_model": "multiclass",
            "auto_standby": True,
            "standby_timeout": 3.0,
            "auto_framing": True,
            "framing_mode": "single",
            "framing_smoothness": 0.04,
            "framing_deadzone": 0.10,
            "smooth_enabled": True,
            "smooth_strength": 50,
            "background_image": "",
            "parallax_enabled": False,
            "parallax_strength": 40,
            "rim_light_enabled": False,
            "rim_light_color": "warm",
            "rim_light_intensity": 50,
            "studio_light_enabled": False,
            "studio_light_gain": 40,
            "low_light_enabled": False,
            "low_light_gain": 45,
            "screen_glare_enabled": False,
            "screen_glare_strength": 50,
            "sharpen_enabled": False,
            "sharpen_strength": 35,
            "eye_contact_mode": "natural",
            "eye_contact_enabled": False,
            "artistic_filter": "off",
            "artistic_strength": 70,
            "color_filter": "none",
            "color_strength": 70,
            "object_retention_enabled": True,
            "object_retention_strength": 60,
            "gesture_detection_enabled": True,
            "gesture_action": "all",
            "privacy_fade_enabled": True,
            "privacy_fade_speed": 0.07,
            "privacy_enabled": False,
            "privacy_timeout": 3.0,
            "privacy_mute_mic": True,
            "privacy_image": "",
            "preserve_glasses": True,
            "glasses_protection": True,
            "guided_filter_guide_size": [640, 360],
            "guided_filter_radius": 5,
            "guided_filter_eps": 0.001,
            "guided_filter_color_guide": True
        }
    }
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
                return cfg.get("video", default_config["video"])
    except Exception as e:
        logging.warning(f"Error reading config: {e}")
    return default_config["video"]

def resolve_cam_index(device_str="auto"):
    if isinstance(device_str, int):
        return device_str
    if device_str and device_str != "auto":
        dev_path = Path(str(device_str))
        if dev_path.exists():
            try:
                target = dev_path.resolve()
                if os.access(target, os.R_OK):
                    return int(target.name.replace("video", ""))
            except Exception:
                pass
    # Auto detect: check /dev/v4l/by-id for primary physical video capture stream
    by_id = Path("/dev/v4l/by-id")
    if by_id.exists():
        for link in sorted(by_id.iterdir()):
            if "video-index0" in link.name:
                try:
                    target = link.resolve()
                    if os.access(target, os.R_OK):
                        return int(target.name.replace("video", ""))
                except Exception:
                    pass
        for link in sorted(by_id.iterdir()):
            try:
                target = link.resolve()
                if target.name.startswith("video") and target.name not in ["video70", "video71", "video72", "video50"]:
                    if os.access(target, os.R_OK):
                        return int(target.name.replace("video", ""))
            except Exception:
                pass
    # Fallback to any readable physical video capture device (skipping virtual loopbacks)
    for p in sorted(Path("/dev").glob("video*")):
        try:
            num = int(p.name.replace("video", ""))
            if num not in [70, 71, 72, 50] and os.access(p, os.R_OK):
                return num
        except Exception:
            pass
    return 48

class AutoFramer:
    def __init__(self, in_w, in_h, out_w, out_h, smoothness=0.04, deadzone=0.10, framing_mode="single"):
        self.in_w = in_w
        self.in_h = in_h
        self.out_w = out_w
        self.out_h = out_h
        self.smoothness = smoothness
        self.deadzone = deadzone
        self.aspect_ratio = out_w / out_h
        self.framing_mode = framing_mode
        
        self.curr_box = np.array([0.0, 0.0, float(in_w), float(in_h)], dtype=np.float32)
        self.target_box = np.copy(self.curr_box)
        self.last_face_time = time.time()
        self.face_lost_timeout = 4.5
        self.last_face_info = {"has_face": False}
        self.smooth_face = None  # Filtered [fx, fy, fw, fh, rx, ry, lx, ly]

        self.detector = None
        if YUNET_MODEL_PATH.exists():
            try:
                self.det_w = 320
                self.det_h = int(320 / self.aspect_ratio)
                self.detector = cv2.FaceDetectorYN.create(
                    model=str(YUNET_MODEL_PATH),
                    config="",
                    input_size=(self.det_w, self.det_h),
                    score_threshold=0.52,
                    nms_threshold=0.3,
                    top_k=5
                )
                logging.info("YuNet face detector loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize YuNet detector: {e}")

    def update(self, frame, enabled=True, framing_mode="single"):
        self.framing_mode = framing_mode
        now = time.time()
        faces = None
        if self.detector is not None:
            small = cv2.resize(frame, (self.det_w, self.det_h))
            self.detector.setInputSize((self.det_w, self.det_h))
            retval, faces = self.detector.detect(small)

        # Sempre atualiza o rastreamento facial e o instante do ultimo rosto visto quando ha deteccao,
        # mesmo com o Auto-Framing desativado (enabled=False). Dessa forma, o detector de ausencia
        # e recursos dependentes da face (Preservacao de Oculos, Studio Light, Eye Contact) continuam
        # funcionando com total precisao.
        if faces is not None and len(faces) > 0:
            self.last_face_time = now
            scale_x = self.in_w / self.det_w
            scale_y = self.in_h / self.det_h

            if self.smooth_face is not None:
                prev_cx = self.smooth_face[0] + self.smooth_face[2] / 2.0
                prev_cy = self.smooth_face[1] + self.smooth_face[3] / 2.0
                def face_score(f):
                    area = (f[2] * scale_x) * (f[3] * scale_y)
                    fcx = (f[0] + f[2] / 2.0) * scale_x
                    fcy = (f[1] + f[3] * 0.5) * scale_y
                    dist = np.hypot(fcx - prev_cx, fcy - prev_cy)
                    return area / (1.0 + (dist / 120.0))
                best_face = max(faces, key=face_score)
            else:
                best_face = max(faces, key=lambda f: f[2] * f[3])

            fx = float(best_face[0] * scale_x)
            fy = float(best_face[1] * scale_y)
            fw = float(best_face[2] * scale_x)
            fh = float(best_face[3] * scale_y)
            rx = float(best_face[4] * scale_x)
            ry = float(best_face[5] * scale_y)
            lx = float(best_face[6] * scale_x)
            ly = float(best_face[7] * scale_y)

            raw_face = np.array([fx, fy, fw, fh, rx, ry, lx, ly], dtype=np.float32)

            # Exponential Moving Average filter on face position & dimensions (eliminates frame jitter)
            if self.smooth_face is None or len(self.smooth_face) != 8:
                self.smooth_face = raw_face
            else:
                self.smooth_face = self.smooth_face * 0.82 + raw_face * 0.18

            s_fx, s_fy, s_fw, s_fh, s_rx, s_ry, s_lx, s_ly = self.smooth_face

            self.last_face_info = {
                "has_face": True,
                "box_phys": (s_fx, s_fy, s_fw, s_fh),
                "r_eye_phys": (s_rx, s_ry),
                "l_eye_phys": (s_lx, s_ly),
                "time": now
            }
        else:
            if (now - self.last_face_time) > 1.0:
                self.last_face_info = {"has_face": False}

        full_frame_box = np.array([0.0, 0.0, float(self.in_w), float(self.in_h)], dtype=np.float32)

        if not enabled:
            self.target_box = full_frame_box
            if np.all(np.abs(self.curr_box - self.target_box) < 0.5):
                self.curr_box = np.copy(self.target_box)
            else:
                self.curr_box = self.curr_box * (1.0 - self.smoothness) + self.target_box * self.smoothness
            return self.crop_and_resize(frame, self.curr_box)

        if faces is not None and len(faces) > 0:
            scale_x = self.in_w / self.det_w
            scale_y = self.in_h / self.det_h

            if self.framing_mode == "group" and len(faces) > 1:
                # Group mode: calculate bounding box enclosing all detected people
                all_boxes = []
                for f in faces:
                    score = float(f[-1]) if len(f) > 8 else 1.0
                    if score >= 0.38:
                        fx = float(f[0] * scale_x)
                        fy = float(f[1] * scale_y)
                        fw = float(f[2] * scale_x)
                        fh = float(f[3] * scale_y)
                        all_boxes.append((fx, fy, fx + fw, fy + fh, fw, fh))
                if not all_boxes:
                    for f in faces:
                        fx = float(f[0] * scale_x)
                        fy = float(f[1] * scale_y)
                        fw = float(f[2] * scale_x)
                        fh = float(f[3] * scale_y)
                        all_boxes.append((fx, fy, fx + fw, fy + fh, fw, fh))

                min_x = min(b[0] for b in all_boxes)
                min_y = min(b[1] for b in all_boxes)
                max_x = max(b[2] for b in all_boxes)
                max_y = max(b[3] for b in all_boxes)
                group_w = max(10.0, max_x - min_x)
                group_h = max(10.0, max_y - min_y)

                g_fx, g_fy, g_fw, g_fh = min_x, min_y, group_w, group_h

                target_h = max(g_fh * 2.2, self.in_h * 0.58)
                target_h = min(target_h, float(self.in_h))
                target_w = target_h * self.aspect_ratio
                if target_w < (g_fw * 1.35):
                    target_w = min(float(self.in_w), g_fw * 1.35)
                    target_h = target_w / self.aspect_ratio
                    if target_h > self.in_h:
                        target_h = float(self.in_h)
                        target_w = target_h * self.aspect_ratio

                center_x = g_fx + g_fw / 2.0
                center_y = g_fy + g_fh * 0.48
                x1 = center_x - target_w / 2.0
                y1 = center_y - target_h * 0.42
                x2 = x1 + target_w
                y2 = y1 + target_h
            else:
                s_fx, s_fy, s_fw, s_fh, s_rx, s_ry, s_lx, s_ly = self.smooth_face

                # Comfortable framing: ~3.2x face height, min crop 55% of sensor height
                target_h = max(s_fh * 3.2, self.in_h * 0.55)
                target_h = min(target_h, float(self.in_h))
                target_w = target_h * self.aspect_ratio
                if target_w > self.in_w:
                    target_w = float(self.in_w)
                    target_h = target_w / self.aspect_ratio

                center_x = s_fx + s_fw / 2.0
                center_y = s_fy + s_fh * 0.52

                x1 = center_x - target_w / 2.0
                y1 = center_y - target_h * 0.40
                x2 = x1 + target_w
                y2 = y1 + target_h

            # Sensor edge bounds clamping
            if x1 < 0:
                x2 += -x1
                x1 = 0.0
            if x2 > self.in_w:
                x1 -= (x2 - self.in_w)
                x2 = float(self.in_w)
            x1 = max(0.0, x1)

            if y1 < 0:
                y2 += -y1
                y1 = 0.0
            if y2 > self.in_h:
                y1 -= (y2 - self.in_h)
                y2 = float(self.in_h)
            y1 = max(0.0, y1)

            cand_box = np.array([x1, y1, x2, y2], dtype=np.float32)

            # Deadzone Hysteresis Filter:
            # Only shift the target framing if user movement or zoom change exceeds deadzone thresholds
            t_w = self.target_box[2] - self.target_box[0]
            t_h = self.target_box[3] - self.target_box[1]
            t_cx = (self.target_box[0] + self.target_box[2]) / 2.0
            t_cy = (self.target_box[1] + self.target_box[3]) / 2.0

            c_w = cand_box[2] - cand_box[0]
            c_h = cand_box[3] - cand_box[1]
            c_cx = (cand_box[0] + cand_box[2]) / 2.0
            c_cy = (cand_box[1] + cand_box[3]) / 2.0

            # If target is currently full frame (e.g. startup or reset), lock immediately
            is_full = (t_w >= self.in_w * 0.98 and t_h >= self.in_h * 0.98)
            if is_full:
                self.target_box = cand_box
            else:
                pos_diff = np.hypot(c_cx - t_cx, c_cy - t_cy) / max(1.0, t_h)
                zoom_diff = abs(c_h - t_h) / max(1.0, t_h)

                pos_threshold = self.deadzone
                zoom_threshold = self.deadzone * 1.3  # Extra deadband tolerance against zoom hunting

                if pos_diff > pos_threshold or zoom_diff > zoom_threshold:
                    self.target_box = cand_box

        else:
            # Face not detected in current frame
            if now - self.last_face_time > self.face_lost_timeout:
                self.last_face_info = {"has_face": False}
                self.smooth_face = None
                self.target_box = full_frame_box
            # While within timeout, we hold the current target_box! No fluttering or zoom hunting.

        # Smooth asymptotic convergence to target
        if np.all(np.abs(self.curr_box - self.target_box) < 0.5):
            self.curr_box = np.copy(self.target_box)
        else:
            self.curr_box = self.curr_box * (1.0 - self.smoothness) + self.target_box * self.smoothness

        return self.crop_and_resize(frame, self.curr_box)

    def get_framed_face_info(self):
        if not self.last_face_info or not self.last_face_info.get("has_face"):
            return {"has_face": False}
        if (time.time() - self.last_face_info.get("time", 0)) > 1.0:
            return {"has_face": False}
        x1, y1, x2, y2 = self.curr_box
        cw = max(1.0, x2 - x1)
        ch = max(1.0, y2 - y1)
        sx = self.out_w / cw
        sy = self.out_h / ch
        fx, fy, fw, fh = self.last_face_info["box_phys"]
        rx, ry = self.last_face_info["r_eye_phys"]
        lx, ly = self.last_face_info["l_eye_phys"]
        return {
            "has_face": True,
            "box": (int((fx - x1) * sx), int((fy - y1) * sy), int(fw * sx), int(fh * sy)),
            "r_eye": (int((rx - x1) * sx), int((ry - y1) * sy)),
            "l_eye": (int((lx - x1) * sx), int((ly - y1) * sy)),
        }

    def crop_and_resize(self, frame, box):
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(self.in_w, int(box[2]))
        y2 = min(self.in_h, int(box[3]))
        
        if x2 <= x1 or y2 <= y1:
            return cv2.resize(frame, (self.out_w, self.out_h))
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)

def count_active_consumers(dev_target="/dev/video72", self_pid=None):
    if self_pid is None:
        self_pid = os.getpid()
    target_pattern = os.path.basename(dev_target)
    consumer_count = 0

    try:
        for pid_dir in os.listdir('/proc'):
            if not pid_dir.isdigit():
                continue
            pid = int(pid_dir)
            if pid == self_pid:
                continue

            fd_dir = f'/proc/{pid_dir}/fd'
            try:
                for fd in os.listdir(fd_dir):
                    try:
                        link = os.readlink(f'{fd_dir}/{fd}')
                        if target_pattern in link:
                            cmd = ''
                            try:
                                with open(f'/proc/{pid_dir}/cmdline', 'rb') as f:
                                    cmd = f.read().decode('utf-8', errors='ignore').replace('\0', ' ').strip()
                            except Exception:
                                pass

                            if 'npu_webcam_daemon' in cmd:
                                continue
                            if ('cameractrlsgtk' in cmd or 'cameractrls' in cmd) and 'cameraview' not in cmd:
                                continue

                            consumer_count += 1
                            break
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass

    return consumer_count

saved_tilt = None
saved_pan = None

def get_v4l2_ctrl(dev, ctrl_name):
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "-d", str(dev), f"--get-ctrl={ctrl_name}"],
            text=True, stderr=subprocess.DEVNULL, timeout=1.0
        )
        if ":" in out:
            return int(out.split(":")[1].strip())
    except Exception:
        pass
    return None

def execute_standby_hooks(cfg, phys_dev):
    global saved_tilt, saved_pan
    # 1. Custom standby shell command if configured
    custom_cmd = cfg.get("standby_command", "").strip()
    if custom_cmd:
        try:
            formatted_cmd = custom_cmd.replace("{device}", str(phys_dev)).replace("{virtual_device}", str(cfg.get("output_device", "/dev/video72")))
            subprocess.Popen(formatted_cmd, shell=True)
            logging.info(f"Comando customizado de standby executado: {formatted_cmd}")
        except Exception as e:
            logging.warning(f"Falha ao executar standby_command: {e}")

    # 2. PTZ park (tilt downwards and optional pan to close/hide camera)
    if cfg.get("ptz_park_on_standby", True):
        current_tilt = get_v4l2_ctrl(phys_dev, "tilt_absolute")
        standby_tilt = int(cfg.get("standby_tilt_value", -324000))
        if current_tilt is not None and current_tilt != standby_tilt:
            saved_tilt = current_tilt

        pan_val = cfg.get("standby_pan_value")
        if pan_val is not None:
            current_pan = get_v4l2_ctrl(phys_dev, "pan_absolute")
            if current_pan is not None and current_pan != int(pan_val):
                saved_pan = current_pan
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", str(phys_dev), f"--set-ctrl=pan_absolute={int(pan_val)}"],
                    capture_output=True, timeout=3.0
                )
                logging.info(f"PTZ Park: Pan ajustado para {pan_val} em {phys_dev}.")
            except Exception as e:
                logging.warning(f"Falha ao ajustar pan de standby: {e}")

        try:
            res = subprocess.run(
                ["v4l2-ctl", "-d", str(phys_dev), f"--set-ctrl=tilt_absolute={standby_tilt}"],
                capture_output=True, timeout=3.0
            )
            if res.returncode == 0:
                logging.info(f"PTZ Park: Tilt ajustado para {standby_tilt} em {phys_dev} (Câmera fechada/recolhida).")
            else:
                logging.warning(f"PTZ Park retorno {res.returncode}: {res.stderr.decode(errors='ignore')}")
        except Exception as e:
            logging.warning(f"Falha ao executar PTZ park: {e}")

def execute_wakeup_hooks(cfg, phys_dev):
    global saved_tilt, saved_pan
    # 1. PTZ unpark (restore to open upright position, e.g. 0)
    if cfg.get("ptz_park_on_standby", True):
        target_tilt = int(cfg.get("wakeup_tilt_value", 0))
        standby_tilt = int(cfg.get("standby_tilt_value", -324000))
        if saved_tilt is not None and saved_tilt != standby_tilt:
            target_tilt = saved_tilt
        try:
            res = subprocess.run(
                ["v4l2-ctl", "-d", str(phys_dev), f"--set-ctrl=tilt_absolute={target_tilt}"],
                capture_output=True, timeout=2.0
            )
            if res.returncode == 0:
                logging.info(f"PTZ Wakeup: Tilt restaurado para {target_tilt} em {phys_dev} (Câmera aberta/ativa).")
            else:
                logging.warning(f"PTZ Wakeup retorno {res.returncode}: {res.stderr.decode(errors='ignore')}")
        except Exception as e:
            logging.warning(f"Falha ao executar PTZ wakeup: {e}")

        pan_val = cfg.get("wakeup_pan_value")
        if pan_val is not None:
            target_pan = int(pan_val)
            if saved_pan is not None:
                target_pan = saved_pan
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", str(phys_dev), f"--set-ctrl=pan_absolute={target_pan}"],
                    capture_output=True, timeout=2.0
                )
                logging.info(f"PTZ Wakeup: Pan restaurado para {target_pan} em {phys_dev}.")
            except Exception as e:
                logging.warning(f"Falha ao restaurar pan de wakeup: {e}")

    # 2. Custom wakeup shell command if configured
    custom_cmd = cfg.get("wakeup_command", "").strip()
    if custom_cmd:
        try:
            formatted_cmd = custom_cmd.replace("{device}", str(phys_dev)).replace("{virtual_device}", str(cfg.get("output_device", "/dev/video72")))
            subprocess.Popen(formatted_cmd, shell=True)
            logging.info(f"Comando customizado de wakeup executado: {formatted_cmd}")
        except Exception as e:
            logging.warning(f"Falha ao executar wakeup_command: {e}")

def draw_privacy_screen(w, h, custom_path=None):
    if custom_path:
        cp = os.path.expanduser(custom_path)
        if os.path.exists(cp):
            try:
                img = cv2.imread(cp)
                if img is not None:
                    return cv2.resize(img, (w, h))
            except Exception as e:
                logging.warning(f"Erro ao carregar imagem de privacidade {cp}: {e}")

    # Checar na pasta de presets de ausencia (~/.config/npu-effects/absence)
    absence_dirs = [
        Path.home() / ".config/npu-effects/absence",
        Path.home() / ".local/share/npu-effects/absence",
        REPO_ASSETS_DIR / "absence",
        OPT_ASSETS_DIR / "absence",
    ]
    for d in absence_dirs:
        for candidate in ["Ausente.png", "ausente.png", "default.png", "privacy.jpg", "privacy.png", "ausencia.jpg", "ausencia.png", "away.jpg", "away.png"]:
            p = d / candidate
            if p.exists():
                try:
                    img = cv2.imread(str(p))
                    if img is not None:
                        return cv2.resize(img, (w, h))
                except Exception:
                    pass

    # Checar imagem padrao na pasta de backgrounds caso o usuario tenha adicionado
    bg_dir = Path.home() / ".config/npu-effects/backgrounds"
    for candidate in ["privacy.jpg", "privacy.png", "ausencia.jpg", "ausencia.png", "away.jpg", "away.png"]:
        p = bg_dir / candidate
        if p.exists():
            try:
                img = cv2.imread(str(p))
                if img is not None:
                    return cv2.resize(img, (w, h))
            except Exception:
                pass

    # Imagem padrao empacotada com o projeto (assets/absence/Ausente.png ou assets/default_privacy.png)
    if DEFAULT_PRIVACY_IMAGE is not None:
        try:
            img = cv2.imread(str(DEFAULT_PRIVACY_IMAGE))
            if img is not None:
                return cv2.resize(img, (w, h))
        except Exception as e:
            logging.warning(f"Erro ao carregar imagem padrao de privacidade {DEFAULT_PRIVACY_IMAGE}: {e}")

    screen = np.full((h, w, 3), 26, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.circle(screen, (cx, cy - 35), 65, (45, 45, 55), -1)
    cv2.circle(screen, (cx, cy - 35), 67, (90, 90, 110), 2)
    cv2.circle(screen, (cx, cy - 50), 24, (180, 185, 200), -1)
    cv2.ellipse(screen, (cx, cy - 8), (40, 24), 0, 0, 180, (180, 185, 200), -1)
    cv2.putText(screen, "MODO PRIVACIDADE", (cx - 175, cy + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (245, 245, 250), 2, cv2.LINE_AA)
    cv2.putText(screen, "Camera pausada por ausencia de usuario", (cx - 225, cy + 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 165, 180), 1, cv2.LINE_AA)
    return screen

def set_audio_mute(mute=True):
    try:
        subprocess.run(["pactl", "set-source-mute", "npu_clearvoice", "1" if mute else "0"],
                       capture_output=True, timeout=0.3)
    except Exception:
        pass

def check_audio_muted():
    try:
        out = subprocess.check_output(["pactl", "get-source-mute", "npu_clearvoice"],
                                      text=True, stderr=subprocess.DEVNULL, timeout=0.3)
        return "sim" in out.lower() or "yes" in out.lower() or "1" in out.lower()
    except Exception:
        return False

def apply_neural_chair_retention(p_person, framed, chair_infer_req=None, chair_inp_name=None, strength=50, cached_chair_mask=None, run_inference=True, inp_w=550, inp_h=550):
    """
    Retencao Neural Exata de Cadeira & Encosto com Segmentacao de Instancias (YOLACT - MIT License) na NPU:
    Detecta os contornos e bordas anatomicas reais da cadeira (encosto, apoio de cabeca, abas e bracos),
    sem aproximacoes poligonais ou convexHull artificiais.
    1. Pre-processamento letterbox preservando aspect ratio.
    2. Inferencia na NPU do modelo de segmentacao de instancias YOLACT (MIT License).
    3. NMS por classe e combinacao linear dos coeficientes com os mapas prototipo (Protonet).
    4. Ancoragem espacial: retem apenas a cadeira conectada/apoiada ao corpo do usuario.
    5. Suavizacao temporal e fusao limpa na mascara p_person.
    """
    try:
        if chair_infer_req is None or chair_inp_name is None:
            return p_person, cached_chair_mask

        # Reutiliza mascara estavel no frame intermediario para manter 30+ FPS solidos
        if not run_inference and cached_chair_mask is not None:
            return np.maximum(p_person, cached_chair_mask), cached_chair_mask

        h_orig, w_orig = framed.shape[:2]
        scale = min(inp_w / float(w_orig), inp_h / float(h_orig))
        scaled_w = int(round(w_orig * scale))
        scaled_h = int(round(h_orig * scale))
        pad_top = (inp_h - scaled_h) // 2
        pad_bottom = inp_h - scaled_h - pad_top
        pad_left = (inp_w - scaled_w) // 2
        pad_right = inp_w - scaled_w - pad_left

        resized = cv2.resize(framed, (scaled_w, scaled_h))
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[np.newaxis, ...]

        res = chair_infer_req.infer({chair_inp_name: blob})
        out_dict = {}
        for k, v in res.items():
            name = k.get_any_name() if hasattr(k, "get_any_name") else str(k)
            out_dict[name] = v

        is_yolact = ("conf" in out_dict and "proto" in out_dict and "mask" in out_dict and "boxes" in out_dict)
        thr_chair = max(0.12, 0.36 - (float(strength) / 100.0) * 0.24)

        if is_yolact:
            conf = np.squeeze(out_dict["conf"], axis=0)          # (19248, 81)
            mask_coeffs = np.squeeze(out_dict["mask"], axis=0)   # (19248, 32)
            proto = np.squeeze(out_dict["proto"], axis=0)        # (138, 138, 32)
            boxes = np.squeeze(out_dict["boxes"], axis=0)        # (19248, 4)

            # Em YOLACT COCO (81 classes com fundo em 0): 57 = cadeira, 58 = sofa/poltrona
            chair_scores = conf[:, 57]
            couch_scores = conf[:, 58]
            c_scores = np.maximum(chair_scores, couch_scores)

            c_mask = c_scores > thr_chair
            if not np.any(c_mask):
                return p_person, None

            sel_indices = np.where(c_mask)[0]
            sel_boxes = boxes[sel_indices]
            sel_confs = c_scores[sel_indices]

            x1 = sel_boxes[:, 0] * inp_w
            y1 = sel_boxes[:, 1] * inp_h
            x2 = sel_boxes[:, 2] * inp_w
            y2 = sel_boxes[:, 3] * inp_h
            bw = np.maximum(0.0, x2 - x1)
            bh = np.maximum(0.0, y2 - y1)
            nms_boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()

            nms_res = cv2.dnn.NMSBoxes(nms_boxes, sel_confs.tolist(), score_threshold=thr_chair, nms_threshold=0.45)
            if len(nms_res) == 0:
                return p_person, None

            nms_keep = np.array(nms_res).flatten()
            final_sel_idx = sel_indices[nms_keep]
            final_boxes = sel_boxes[nms_keep]

            if proto.ndim == 3 and proto.shape[-1] == 32:
                mh, mw, c = proto.shape
                proto_flat = proto.reshape(-1, c)
                coeffs = mask_coeffs[final_sel_idx]
                raw_masks = (coeffs @ proto_flat.T).reshape(-1, mh, mw)
            else:
                c, mh, mw = proto.shape
                coeffs = mask_coeffs[final_sel_idx]
                raw_masks = (coeffs @ proto.reshape(c, -1)).reshape(-1, mh, mw)

            sig_masks = 1.0 / (1.0 + np.exp(-raw_masks))

            proto_chair = np.zeros((mh, mw), dtype=np.float32)
            for i in range(len(final_boxes)):
                bx1 = max(0, int(final_boxes[i, 0] * mw))
                by1 = max(0, int(final_boxes[i, 1] * mh))
                bx2 = min(mw, int(np.ceil(final_boxes[i, 2] * mw)))
                by2 = min(mh, int(np.ceil(final_boxes[i, 3] * mh)))

                m = np.zeros((mh, mw), dtype=np.float32)
                m[by1:by2, bx1:bx2] = sig_masks[i, by1:by2, bx1:bx2]
                proto_chair = np.maximum(proto_chair, m)

            rx = mw / float(inp_w)
            ry = mh / float(inp_h)
        else:
            out_boxes = None
            out_protos = None
            for v in res.values():
                if v.ndim == 3:
                    out_boxes = v
                elif v.ndim == 4:
                    out_protos = v

            if out_boxes is None or out_protos is None:
                return p_person, cached_chair_mask

            preds = np.squeeze(out_boxes, axis=0).T  # (5040, 116)
            scores_chair = preds[:, 4 + 56]
            scores_couch = preds[:, 4 + 57]
            c_scores = np.maximum(scores_chair, scores_couch)

            c_mask = c_scores > thr_chair
            if not np.any(c_mask):
                return p_person, None

            c_preds = preds[c_mask]
            cx = c_preds[:, 0]
            cy = c_preds[:, 1]
            w = c_preds[:, 2]
            h = c_preds[:, 3]
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            boxes_list = np.stack([x1, y1, w, h], axis=1).tolist()
            confs_list = c_scores[c_mask].tolist()

            nms_idx = cv2.dnn.NMSBoxes(boxes_list, confs_list, score_threshold=thr_chair, nms_threshold=0.45)
            if len(nms_idx) == 0:
                return p_person, None

            nms_idx = np.array(nms_idx).flatten()
            sel_preds = c_preds[nms_idx]
            sel_boxes = np.array(boxes_list)[nms_idx]

            protos = np.squeeze(out_protos, axis=0)  # (32, 96, 160)
            c, mh, mw = protos.shape
            mask_coeffs = sel_preds[:, 84:]

            raw_masks = (mask_coeffs @ protos.reshape(c, -1)).reshape(-1, mh, mw)
            sig_masks = 1.0 / (1.0 + np.exp(-raw_masks))

            rx = mw / float(inp_w)
            ry = mh / float(inp_h)

            proto_chair = np.zeros((mh, mw), dtype=np.float32)
            for i in range(len(nms_idx)):
                bx1 = max(0, int(sel_boxes[i, 0] * rx))
                by1 = max(0, int(sel_boxes[i, 1] * ry))
                bx2 = min(mw, int(np.ceil((sel_boxes[i, 0] + sel_boxes[i, 2]) * rx)))
                by2 = min(mh, int(np.ceil((sel_boxes[i, 1] + sel_boxes[i, 3]) * ry)))

                m = np.zeros((mh, mw), dtype=np.float32)
                m[by1:by2, bx1:bx2] = sig_masks[i, by1:by2, bx1:bx2]
                proto_chair = np.maximum(proto_chair, m)

        p_top = int(round(pad_top * ry))
        p_bottom = mh - int(round(pad_bottom * ry))
        p_left = int(round(pad_left * rx))
        p_right = mw - int(round(pad_right * rx))

        chair_unpad = proto_chair[p_top:p_bottom, p_left:p_right]
        chair_mask = cv2.resize(chair_unpad, (p_person.shape[1], p_person.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Ancoragem espacial: reter apenas componentes da cadeira que tocam ou estao atras do usuario
        person_bin = (p_person > 0.18).astype(np.uint8)
        person_anchor = cv2.dilate(person_bin, np.ones((11, 11), np.uint8))
        chair_bin = (chair_mask > 0.32).astype(np.uint8)

        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(chair_bin)
        valid_chair = np.zeros_like(chair_mask)
        for lbl in range(1, num_labels):
            comp = (labels == lbl)
            if np.any(comp & (person_anchor > 0)):
                valid_chair = np.maximum(valid_chair, np.where(comp, chair_mask, 0.0))

        # Suavizacao temporal com mascara cacheada
        if cached_chair_mask is not None:
            updated_cache = cached_chair_mask * 0.70 + valid_chair * 0.30
        else:
            updated_cache = valid_chair

        return np.maximum(p_person, updated_cache), updated_cache
    except Exception as e:
        return p_person, cached_chair_mask

def apply_neural_glasses_retention(
    framed,
    face_info,
    glasses_infer_req=None,
    glasses_inp_name=None,
    glasses_out_name=None,
    cached_glasses_mask=None,
    run_inference=True,
    inp_w=512,
    inp_h=512
):
    """
    Retencao Neural de Armacao e Hastes de Oculos (BiSeNet Face Parsing - MIT License) na NPU:
    Detecta e preserva com precisao sub-pixel a armacao dos oculos e as hastes laterais ate as orelhas,
    impedindo que bordas finas sejam cortadas ou borradas pelo algoritmo de segmentacao de pessoa.
    1. Recorte focado (crop) a partir da bounding box do rosto rastreada pelo YuNet com expansao lateral (~35%) e superior (~30%).
    2. Inferencia na NPU do modelo BiSeNet Face Parsing (CelebAMask-HQ, 19 classes, classe 6 = eyeglass).
    3. Extracao da mascara binaria dos oculos, redimensionamento para coordenadas originais e dilatacao suave + feathering.
    4. Suavizacao temporal e cache entre quadros para manter 30+ FPS solidos.
    """
    try:
        if glasses_infer_req is None or glasses_inp_name is None:
            return cached_glasses_mask

        if not face_info or not face_info.get("has_face"):
            return None

        # Reutiliza mascara estavel no frame intermediario para economizar ciclos na NPU
        if not run_inference and cached_glasses_mask is not None:
            return cached_glasses_mask

        h_orig, w_orig = framed.shape[:2]
        fx, fy, fw, fh = face_info["box"]

        if fw < 16 or fh < 16:
            return cached_glasses_mask

        # Margens expandidas para cobrir as hastes dos oculos ate a regiao das orelhas (~38% nas laterais, ~30% superior, ~15% inferior)
        margin_x = int(fw * 0.38)
        margin_top = int(fh * 0.30)
        margin_bottom = int(fh * 0.15)

        x1 = max(0, fx - margin_x)
        y1 = max(0, fy - margin_top)
        x2 = min(w_orig, fx + fw + margin_x)
        y2 = min(h_orig, fy + fh + margin_bottom)

        cw = x2 - x1
        ch = y2 - y1
        if cw < 16 or ch < 16:
            return cached_glasses_mask

        crop = framed[y1:y2, x1:x2]
        resized = cv2.resize(crop, (inp_w, inp_h))

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Normalizacao ImageNet padrao do BiSeNet Face Parsing
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        blob = ((rgb - mean) / std).transpose(2, 0, 1)[np.newaxis, ...]

        glasses_infer_req.infer({glasses_inp_name: blob})
        out_tensor = glasses_infer_req.get_tensor(glasses_out_name).data[0]
        classes = np.argmax(out_tensor, axis=0)

        # Classe 6: eyeglass no CelebAMask-HQ
        glasses_bin = (classes == 6).astype(np.uint8)

        # Se nenhum pixel de oculos foi detectado
        if np.count_nonzero(glasses_bin) < 25:
            # Se havia mascara anterior, decai suavemente antes de zerar
            if cached_glasses_mask is not None:
                decayed = cached_glasses_mask * 0.4
                return decayed if np.max(decayed) > 0.08 else None
            return None

        # Redimensiona mascara de volta para as dimensoes do crop no frame
        glasses_crop_unpad = cv2.resize(glasses_bin, (cw, ch), interpolation=cv2.INTER_NEAREST)

        # Dilate suave (3x3) para garantir continuidade das hastes finas e cobertura total da armacao
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(glasses_crop_unpad, k_dilate, iterations=1)

        # Feathering com GaussianBlur para bordas anti-aliased e fusao natural com o primeiro plano
        feathered = cv2.GaussianBlur(dilated.astype(np.float32), (5, 5), sigmaX=1.0)
        core_mask = glasses_crop_unpad.astype(np.float32)
        feathered = np.maximum(core_mask, feathered)
        feathered = np.clip(feathered, 0.0, 1.0)

        # Posiciona na mascara full frame
        glasses_full = np.zeros((h_orig, w_orig), dtype=np.float32)
        glasses_full[y1:y2, x1:x2] = feathered

        return glasses_full
    except Exception as e:
        return cached_glasses_mask

GESTURE_EMOJI_CACHE = {}

def get_gesture_emoji(gesture_name):
    global GESTURE_EMOJI_CACHE
    if gesture_name in GESTURE_EMOJI_CACHE:
        return GESTURE_EMOJI_CACHE[gesture_name]

    candidate_dirs = [
        "/home/kleber/Development/NPU/assets/emojis",
        os.path.expanduser("~/.local/share/npu-effects/assets/emojis"),
        "/opt/npu-effects/assets/emojis",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "emojis"),
    ]

    for d in candidate_dirs:
        p = os.path.join(d, f"{gesture_name}.png")
        if os.path.isfile(p):
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is not None and len(im.shape) == 3 and im.shape[2] == 4:
                GESTURE_EMOJI_CACHE[gesture_name] = im
                return im
    return None

def detect_hand_gesture(hand_skin, out_w=1920, out_h=1080):
    hand_bin = (hand_skin > 0.22).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hand_bin = cv2.morphologyEx(hand_bin, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(hand_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    valid_candidates = []
    for idx, cnt in enumerate(contours):
        # Ignore inner holes as candidate hands
        if hierarchy is not None and hierarchy[0][idx][3] != -1:
            continue
        area = cv2.contourArea(cnt)
        if area < 120 or area > 7500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 12 or h < 14:
            continue

        # Ignore arms resting on desk at bottom edge of screen (bottom 45px when resting)
        if (y + h >= 253) and (y > 195):
            continue

        aspect = float(h) / max(1.0, float(w))
        if aspect < 0.45 or aspect > 3.20:
            continue

        valid_candidates.append((area, cnt, (x, y, w, h), idx, aspect))

    if not valid_candidates:
        return None, None

    valid_candidates.sort(key=lambda c: c[0], reverse=True)
    area, cnt, (x, y, w, h), cnt_idx, aspect = valid_candidates[0]

    cx = int((x + w / 2.0) * out_w / 256.0)
    cy = int((y + h / 2.0) * out_h / 256.0)

    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return None, None

    hull_pts = cv2.convexHull(cnt, returnPoints=True)
    hull_area = cv2.contourArea(hull_pts)
    solidity = float(area) / max(1.0, hull_area)

    # Check for inner hole (e.g. OK hand loop)
    has_inner_hole = False
    if hierarchy is not None:
        for ch_idx in range(len(contours)):
            if hierarchy[0][ch_idx][3] == cnt_idx:
                hole_area = cv2.contourArea(contours[ch_idx])
                if hole_area >= 20:
                    has_inner_hole = True
                    break

    valid_defects = []
    try:
        defects = cv2.convexityDefects(cnt, hull_idx)
        if defects is not None:
            for i in range(defects.shape[0]):
                row = defects[i, 0] if len(defects.shape) == 3 else defects[i]
                s, e, f, d = int(row[0]), int(row[1]), int(row[2]), float(row[3])
                d_px = d / 256.0
                if d_px < 3.2:
                    continue

                start = cnt[s][0]
                end = cnt[e][0]
                far = cnt[f][0]

                a = np.hypot(end[0] - start[0], end[1] - start[1])
                b = np.hypot(far[0] - start[0], far[1] - start[1])
                c = np.hypot(end[0] - far[0], end[1] - far[1])

                if b < 5.5 or c < 5.5:
                    continue

                cos_val = (b**2 + c**2 - a**2) / (2.0 * b * c + 1e-6)
                angle = np.arccos(np.clip(cos_val, -1.0, 1.0)) * 180.0 / np.pi

                if 12.0 <= angle <= 95.0:
                    valid_defects.append((d_px, angle, far, start, end))
    except Exception:
        pass

    fc = len(valid_defects)

    h_top = max(3, int(h * 0.25))
    h_bot = max(3, int(h * 0.25))
    top_slice = hand_bin[y : y + h_top, x : x + w]
    mid_slice = hand_bin[y + int(h * 0.35) : y + int(h * 0.75), x : x + w]
    bot_slice = hand_bin[y + h - h_bot : y + h, x : x + w]

    top_w = np.max(np.sum(top_slice > 0, axis=1)) if top_slice.size > 0 and np.any(top_slice > 0) else 0
    mid_w = np.max(np.sum(mid_slice > 0, axis=1)) if mid_slice.size > 0 and np.any(mid_slice > 0) else 1
    bot_w = np.max(np.sum(bot_slice > 0, axis=1)) if bot_slice.size > 0 and np.any(bot_slice > 0) else 0

    top_ratio = top_w / max(1.0, float(mid_w))
    bot_ratio = bot_w / max(1.0, float(mid_w))

    gesture = None

    # 1. OK Hand: loop formed by thumb and index
    if has_inner_hole and aspect >= 0.90:
        gesture = "ok_hand"

    # 2. Open Palm / Wave: 3 or more deep finger valleys
    elif fc >= 3 and 0.40 <= solidity <= 0.85 and 0.65 <= aspect <= 1.70:
        gesture = "open_palm"

    # 3. Two-finger or Horn gestures (Peace vs Rock)
    elif fc in [1, 2] and aspect >= 1.00 and 0.42 <= solidity <= 0.88:
        deepest = max(valid_defects, key=lambda d: d[0])
        tip_dist = np.hypot(deepest[3][0] - deepest[4][0], deepest[3][1] - deepest[4][1])
        tip_dist_ratio = tip_dist / max(1.0, float(w))

        # Rock / Horns: index and pinky extended wide apart
        if tip_dist_ratio >= 0.42 and deepest[0] >= 4.5:
            gesture = "rock"
        # Peace: index and middle extended together in upper region
        elif deepest[0] >= 3.8 and deepest[1] <= 75.0 and deepest[2][1] < (y + h * 0.75):
            gesture = "peace"

    # 4. Zero defects: Thumbs Up, Thumbs Down, Pointing Up, Call Me (Shaka), Fist
    elif fc == 0:
        # Call Me / Shaka: wide horizontal profile (w > h)
        if aspect <= 0.95 and 0.40 <= solidity <= 0.88:
            gesture = "call_me"
        # Pointing Up: tall and slender single index finger
        elif aspect >= 1.45 and top_ratio < 0.55 and solidity <= 0.85:
            gesture = "pointing_up"
        # Thumbs Up: vertical, top is narrow thumb, middle is wide fist
        elif aspect >= 1.06 and solidity >= 0.64 and top_ratio < 0.72 and mid_w >= 12:
            gesture = "thumbs_up"
        # Thumbs Down: vertical, top is wide fist, bottom is narrow thumb
        elif aspect >= 1.06 and solidity >= 0.64 and bot_ratio < 0.72 and top_ratio >= 0.65 and mid_w >= 12:
            gesture = "thumbs_down"
        # Fist Bump / Clenched Fist: compact ball, high solidity, no protrusions
        elif 0.75 <= aspect <= 1.30 and solidity >= 0.78 and top_ratio >= 0.65 and bot_ratio >= 0.65:
            gesture = "fist"

    if gesture:
        return gesture, (cx, cy)

    return None, None

def draw_gesture_reaction_fast(frame, gesture_type, hand_pos, progress):
    emoji_img = get_gesture_emoji(gesture_type)
    fh, fw = frame.shape[:2]
    hx, hy = hand_pos

    # Rising trajectory with subtle natural sway as it floats up
    cy = int(hy - 45 - progress * 165)
    cx = int(hx + np.sin(progress * np.pi * 2.5) * 12)

    # Pop-in bounce scale, then subtle expansion as it floats up
    if progress < 0.15:
        scale = 0.45 + (progress / 0.15) * 0.70  # 0.45 -> 1.15
    elif progress < 0.28:
        scale = 1.15 - ((progress - 0.15) / 0.13) * 0.15  # 1.15 -> 1.0
    else:
        scale = 1.0 + (progress - 0.28) * 0.12  # 1.0 -> 1.08

    # Alpha fade-in and smooth fade-out
    if progress < 0.12:
        alpha = progress / 0.12
    elif progress > 0.68:
        alpha = (1.0 - progress) / 0.32
    else:
        alpha = 1.0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.01:
        return frame

    if emoji_img is not None:
        base_sz = 110
        target_sz = max(24, int(base_sz * scale))
        if target_sz % 2 != 0:
            target_sz += 1

        scaled_emoji = cv2.resize(emoji_img, (target_sz, target_sz), interpolation=cv2.INTER_LINEAR)

        # Circular frosted bubble backdrop (modern slate with translucent border)
        bubble_radius = int(target_sz * 0.58)
        x1 = cx - bubble_radius
        y1 = cy - bubble_radius
        x2 = cx + bubble_radius
        y2 = cy + bubble_radius

        if x2 <= 0 or y2 <= 0 or x1 >= fw or y1 >= fh:
            return frame

        fx1 = max(0, x1)
        fy1 = max(0, y1)
        fx2 = min(fw, x2)
        fy2 = min(fh, y2)
        box_w = fx2 - fx1
        box_h = fy2 - fy1
        if box_w <= 0 or box_h <= 0:
            return frame

        roi = frame[fy1:fy2, fx1:fx2]

        bw = 2 * bubble_radius
        bubble_mask = np.zeros((bw, bw), dtype=np.float32)
        cv2.circle(bubble_mask, (bubble_radius, bubble_radius), bubble_radius - 2, 0.65 * alpha, -1, cv2.LINE_AA)
        cv2.circle(bubble_mask, (bubble_radius, bubble_radius), bubble_radius - 2, 0.90 * alpha, 2, cv2.LINE_AA)

        bx1 = fx1 - x1
        by1 = fy1 - y1
        b_crop = bubble_mask[by1:by1 + box_h, bx1:bx1 + box_w]
        b_3c = np.repeat(b_crop[:, :, np.newaxis], 3, axis=2)
        bubble_color = np.array([24, 24, 34], dtype=np.uint8)
        roi[:] = (roi * (1.0 - b_3c) + bubble_color * b_3c).astype(np.uint8)

        # Overlay high-res emoji with alpha channel
        ex1 = cx - target_sz // 2
        ey1 = cy - target_sz // 2
        ex2 = ex1 + target_sz
        ey2 = ey1 + target_sz

        efx1 = max(0, ex1)
        efy1 = max(0, ey1)
        efx2 = min(fw, ex2)
        efy2 = min(fh, ey2)

        if efx2 > efx1 and efy2 > efy1:
            e_crop = scaled_emoji[efy1 - ey1 : efy1 - ey1 + (efy2 - efy1), efx1 - ex1 : efx1 - ex1 + (efx2 - efx1)]
            e_alpha = (e_crop[:, :, 3].astype(np.float32) / 255.0) * alpha
            e_alpha_3c = np.repeat(e_alpha[:, :, np.newaxis], 3, axis=2)
            e_bgr = e_crop[:, :, :3]
            e_roi = frame[efy1:efy2, efx1:efx2]
            e_roi[:] = (e_roi * (1.0 - e_alpha_3c) + e_bgr * e_alpha_3c).astype(np.uint8)
    else:
        # Fallback if image asset is unexpectedly missing
        tag = gesture_type.upper()
        cv2.putText(frame, tag, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

def apply_low_light_booster(img, gain_percent=45):
    gamma = 1.0 + (gain_percent / 100.0) * 0.55
    inv_gamma = 1.0 / gamma
    lift = int((gain_percent / 100.0) * 12.0)
    lut = np.array([min(255, int(((i / 255.0) ** inv_gamma) * 255.0 + lift)) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)

def apply_studio_light(img, face_info, intensity=40):
    if not face_info or not face_info.get("has_face"):
        return img
    h, w = img.shape[:2]
    fx, fy, fw, fh = face_info["box"]
    cx = fx + fw // 2
    cy = fy + int(fh * 0.45)
    rx = max(20, int(fw * 0.85))
    ry = max(20, int(fh * 1.1))

    sw, sh = 240, 135
    cx_s = int(cx * sw / w)
    cy_s = int(cy * sh / h)
    rx_s = max(5, int(rx * sw / w))
    ry_s = max(5, int(ry * sh / h))

    Y, X = np.ogrid[:sh, :sw]
    dist_sq = ((X - cx_s) / rx_s)**2 + ((Y - cy_s) / ry_s)**2
    mask_s = np.clip(1.0 - dist_sq, 0.0, 1.0)
    max_val = (intensity / 100.0) * 42.0
    mask_s = (mask_s * mask_s * max_val).astype(np.uint8)
    light_map = cv2.resize(mask_s, (w, h), interpolation=cv2.INTER_LINEAR)
    light_3c = cv2.merge([light_map, light_map, light_map])
    return cv2.add(img, light_3c)

def apply_screen_glare_suppressor(img, strength=50):
    if strength <= 0:
        return img
    alpha = (strength / 100.0) * 0.65
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    target_warm = (r * 0.80 + g * 0.20)
    excess = np.maximum(0.0, b - target_warm)
    corrected_b = np.clip(b - excess * alpha, 0.0, 255.0).astype(np.uint8)
    return cv2.merge([corrected_b, img[:, :, 1], img[:, :, 2]])

def apply_eye_contact(img, face_info, mode="natural"):
    if not face_info or not face_info.get("has_face") or mode in ["off", False]:
        return img
    h, w = img.shape[:2]
    fx, fy, fw, fh = face_info["box"]
    r_eye = face_info["r_eye"]
    l_eye = face_info["l_eye"]

    result = img.copy()
    ew = max(12, int(fw * 0.13))
    eh = max(8, int(fh * 0.09))
    shift_y = 4.6 if mode == "teleprompter" else 2.5
    r_mult = 0.82 if mode == "teleprompter" else 0.65

    for ex, ey in [r_eye, l_eye]:
        x1 = max(0, ex - ew)
        y1 = max(0, ey - eh)
        x2 = min(w, ex + ew)
        y2 = min(h, ey + eh)
        crop = result[y1:y2, x1:x2]
        if crop.size > 0:
            ch, cw = crop.shape[:2]
            map_y, map_x = np.indices((ch, cw), dtype=np.float32)
            cy_c, cx_c = ch / 2.0, cw / 2.0
            r_c = max(1.0, cw / 2.4)
            dist_sq = ((map_x - cx_c)/r_c)**2 + ((map_y - cy_c)/(r_c * r_mult))**2
            falloff = np.clip(1.0 - dist_sq, 0.0, 1.0)
            map_y_shifted = map_y + falloff * shift_y
            shifted = cv2.remap(crop, map_x, map_y_shifted, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            result[y1:y2, x1:x2] = shifted
    return result

def apply_rim_light(fg_img, mask_full, color_mode='warm', intensity=50):
    if intensity <= 0:
        return fg_img
    h, w = fg_img.shape[:2]
    small_m = cv2.resize((mask_full * 255).astype(np.uint8), (640, 360))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eroded = cv2.erode(small_m, k, iterations=1)
    rim_small = cv2.subtract(small_m, eroded)
    rim_small[int(360 * 0.85):, :] = 0
    rim_blurred = cv2.GaussianBlur(rim_small, (5, 5), 0)
    rim_full = cv2.resize(rim_blurred, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    if color_mode == 'warm':
        c_b, c_g, c_r = 120, 205, 255
    elif color_mode == 'cool':
        c_b, c_g, c_r = 255, 220, 130
    else:  # white
        c_b, c_g, c_r = 245, 245, 250

    alpha = (intensity / 100.0) * 0.85
    rim_b = (rim_full * c_b * alpha).astype(np.uint8)
    rim_g = (rim_full * c_g * alpha).astype(np.uint8)
    rim_r = (rim_full * c_r * alpha).astype(np.uint8)
    rim_3c = cv2.merge([rim_b, rim_g, rim_r])
    return cv2.add(fg_img, rim_3c)

def apply_color_grading(img, mode='none', strength=70):
    if mode == 'none' or strength <= 0:
        return img
    alpha = strength / 100.0
    if mode == 'noir':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lut = np.array([np.clip(int(255.0 / (1.0 + np.exp(-0.03 * (i - 128)))), 0, 255) for i in range(256)], dtype=np.uint8)
        c_gray = cv2.LUT(gray, lut)
        graded = cv2.cvtColor(c_gray, cv2.COLOR_GRAY2BGR)
    elif mode == 'teal_orange':
        lut_b = np.array([min(255, int(i * 1.12 + 10)) if i < 128 else max(0, int(i * 0.90)) for i in range(256)], dtype=np.uint8)
        lut_g = np.array([min(255, int(i * 1.04 + 4)) if i < 128 else min(255, int(i * 1.02)) for i in range(256)], dtype=np.uint8)
        lut_r = np.array([max(0, int(i * 0.92)) if i < 128 else min(255, int(i * 1.15 + 12)) for i in range(256)], dtype=np.uint8)
        b = cv2.LUT(img[:, :, 0], lut_b)
        g = cv2.LUT(img[:, :, 1], lut_g)
        r = cv2.LUT(img[:, :, 2], lut_r)
        graded = cv2.merge([b, g, r])
    elif mode == 'warm_vintage':
        lut_b = np.array([min(255, int(i * 0.88 + 15)) for i in range(256)], dtype=np.uint8)
        lut_g = np.array([min(255, int(i * 0.98 + 12)) for i in range(256)], dtype=np.uint8)
        lut_r = np.array([min(255, int(i * 1.08 + 18)) for i in range(256)], dtype=np.uint8)
        b = cv2.LUT(img[:, :, 0], lut_b)
        g = cv2.LUT(img[:, :, 1], lut_g)
        r = cv2.LUT(img[:, :, 2], lut_r)
        graded = cv2.merge([b, g, r])
    elif mode == 'vibrant':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int32) * 1.30, 0, 255).astype(np.uint8)
        graded = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        return img
    if alpha >= 1.0:
        return graded
    return cv2.addWeighted(graded, alpha, img, 1.0 - alpha, 0)

def apply_artistic_filter(img, mode='off', strength=70):
    if mode == 'off' or strength <= 0:
        return img
    alpha = strength / 100.0
    h, w = img.shape[:2]
    small = cv2.resize(img, (480, 270))
    if mode == 'illustrated':
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
        color = cv2.bilateralFilter(small, d=5, sigmaColor=60, sigmaSpace=60)
        color = (color // 42) * 42 + 21
        edges_3c = cv2.merge([edges, edges, edges])
        res_small = cv2.bitwise_and(color, edges_3c)
    elif mode == 'animated':
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
        color = cv2.bilateralFilter(small, d=7, sigmaColor=100, sigmaSpace=100)
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int32) * 1.35, 0, 255).astype(np.uint8)
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        edges_3c = cv2.merge([edges, edges, edges])
        res_small = cv2.bitwise_and(color, edges_3c)
    elif mode == 'watercolor':
        smooth = cv2.bilateralFilter(small, d=7, sigmaColor=90, sigmaSpace=90)
        smooth = cv2.bilateralFilter(smooth, d=5, sigmaColor=90, sigmaSpace=90)
        res_small = smooth
    else:
        return img
    res_full = cv2.resize(res_small, (w, h), interpolation=cv2.INTER_LINEAR)
    if alpha >= 1.0:
        return res_full
    return cv2.addWeighted(res_full, alpha, img, 1.0 - alpha, 0)

def apply_smart_sharpen(img, strength=35):
    amount = (strength / 100.0) * 0.45
    kernel = np.array([[0, -amount, 0],
                       [-amount, 1.0 + 4.0 * amount, -amount],
                       [0, -amount, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)

def fast_guided_filter(guide_small, mask_small, r=5, eps=1e-3, out_shape=(1920, 1080), color_guide=False):
    """
    Fast Guided Filter (snaps coarse mask to physical edges of hair and fingers).
    guide_small: grayscale (2D) or color (3D BGR) guide image, uint8 or float32
    mask_small: segmentation mask at model resolution (e.g. 256x256), float32 [0, 1]
    r: filter radius (ksize = 2*r + 1)
    eps: regularization parameter
    out_shape: (width, height) to upscale the final refined edge mask (1920, 1080)
    color_guide: boolean, whether to use multi-channel color edge guidance
    """
    gw, gh = guide_small.shape[1], guide_small.shape[0]
    p = cv2.resize(mask_small, (gw, gh), interpolation=cv2.INTER_LINEAR)
    r = max(1, int(r))
    eps = max(1e-6, float(eps))
    ksize = (2 * r + 1, 2 * r + 1)

    if color_guide and guide_small.ndim == 3:
        # Multi-channel color guided filter (joint linear regression across B, G, R channels)
        # Model: q = a0*I0 + a1*I1 + a2*I2 + b
        I = guide_small.astype(np.float32) * (1.0 / 255.0) if guide_small.dtype == np.uint8 else guide_small
        mean_I = cv2.boxFilter(I, -1, ksize)
        mean_p = cv2.boxFilter(p, -1, ksize)

        I0, I1, I2 = I[:, :, 0], I[:, :, 1], I[:, :, 2]
        mI0, mI1, mI2 = mean_I[:, :, 0], mean_I[:, :, 1], mean_I[:, :, 2]

        v0 = cv2.boxFilter(I0 * I0, -1, ksize) - mI0 * mI0
        v1 = cv2.boxFilter(I1 * I1, -1, ksize) - mI1 * mI1
        v2 = cv2.boxFilter(I2 * I2, -1, ksize) - mI2 * mI2
        inv_var = 1.0 / (v0 + v1 + v2 + eps)

        cov0 = cv2.boxFilter(I0 * p, -1, ksize) - mI0 * mean_p
        cov1 = cv2.boxFilter(I1 * p, -1, ksize) - mI1 * mean_p
        cov2 = cv2.boxFilter(I2 * p, -1, ksize) - mI2 * mean_p

        a0 = cov0 * inv_var
        a1 = cov1 * inv_var
        a2 = cov2 * inv_var
        b = mean_p - (a0 * mI0 + a1 * mI1 + a2 * mI2)

        ma0 = cv2.boxFilter(a0, -1, ksize)
        ma1 = cv2.boxFilter(a1, -1, ksize)
        ma2 = cv2.boxFilter(a2, -1, ksize)
        mb = cv2.boxFilter(b, -1, ksize)

        q = ma0 * I0 + ma1 * I1 + ma2 * I2 + mb
    else:
        # Grayscale guided filter (classic single-channel)
        if guide_small.ndim == 3:
            guide_gray = cv2.cvtColor(guide_small, cv2.COLOR_BGR2GRAY)
        else:
            guide_gray = guide_small
        I = guide_gray.astype(np.float32) / 255.0 if guide_gray.dtype == np.uint8 else guide_gray
        mean_I = cv2.boxFilter(I, -1, ksize)
        mean_p = cv2.boxFilter(p, -1, ksize)
        cov_Ip = cv2.boxFilter(I * p, -1, ksize) - mean_I * mean_p
        var_I = cv2.boxFilter(I * I, -1, ksize) - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, -1, ksize)
        mean_b = cv2.boxFilter(b, -1, ksize)
        q = mean_a * I + mean_b

    return cv2.resize(np.clip(q, 0.0, 1.0), out_shape, interpolation=cv2.INTER_LINEAR)

def main():
    global running
    logging.info("Starting Intel NPU Webcam Daemon (1080p + Denoising)...")

    cfg = load_config()
    in_device = cfg.get("input_device", "/dev/video0")
    out_device = cfg.get("output_device", "/dev/video72")
    out_w = int(cfg.get("width", 1920))
    out_h = int(cfg.get("height", 1080))
    target_fps = int(cfg.get("fps", 30))

    # Initialize OpenVINO for NPU
    logging.info("Initializing OpenVINO NPU Core...")
    core = ov.Core()
    devices = core.available_devices
    npu_device = "NPU" if "NPU" in devices else "CPU"
    logging.info(f"Targeting inference device: {npu_device} (Available: {devices})")

    wants_modnet = (cfg.get("video", {}).get("segmentation_model", "multiclass") == "modnet")
    if wants_modnet and SEG_MODNET_PATH.exists():
        active_seg_path = SEG_MODNET_PATH
    elif SEG_MULTICLASS_PATH.exists():
        active_seg_path = SEG_MULTICLASS_PATH
    else:
        active_seg_path = SEG_MODEL_PATH
    if wants_modnet and not SEG_MODNET_PATH.exists():
        logging.warning(f"segmentation_model=modnet requested but {SEG_MODNET_PATH} not found. Falling back to {active_seg_path}.")
    if not active_seg_path.exists():
        logging.error(f"Model not found at {active_seg_path}")
        sys.exit(1)

    logging.info(f"Loading segmentation model {active_seg_path}...")
    model = core.read_model(str(active_seg_path))
    compiled_model = core.compile_model(model, npu_device)
    infer_request = compiled_model.create_infer_request()
    seg_inp_name = model.inputs[0].get_any_name()
    seg_out_name = model.outputs[0].get_any_name()
    seg_inp_shape = list(model.inputs[0].get_shape())
    is_modnet = (active_seg_path == SEG_MODNET_PATH)
    is_multiclass = (not is_modnet) and ((active_seg_path == SEG_MULTICLASS_PATH) or (len(seg_inp_shape) == 4 and seg_inp_shape[-1] == 3))
    seg_type_label = "MODNet Portrait Matting" if is_modnet else ("Multiclass (6-class + Hand/Skin)" if is_multiclass else "Legacy Landscape")
    logging.info(f"NPU segmentation model compiled successfully! Type: {seg_type_label}, In: {seg_inp_shape}")

    # Load Neural Chair Segmentation Model (YOLACT Instance Segmenter - MIT License)
    chair_infer_req = None
    chair_inp_name = None
    chair_inp_w = 550
    chair_inp_h = 550
    if SEG_CHAIR_PATH and SEG_CHAIR_PATH.exists():
        try:
            logging.info(f"Loading neural chair model {SEG_CHAIR_PATH}...")
            chair_model = core.read_model(str(SEG_CHAIR_PATH))
            chair_compiled = core.compile_model(chair_model, npu_device)
            chair_infer_req = chair_compiled.create_infer_request()
            chair_inp_name = chair_model.inputs[0].get_any_name()
            c_shape = chair_model.inputs[0].shape
            if len(c_shape) >= 4:
                chair_inp_h = int(c_shape[2])
                chair_inp_w = int(c_shape[3])
            logging.info(f"Neural chair model (YOLACT) compiled successfully on {npu_device} ({chair_inp_w}x{chair_inp_h})!")
        except Exception as e:
            logging.warning(f"Could not compile chair model on {npu_device}: {e}. Retrying on CPU...")
            try:
                chair_compiled = core.compile_model(chair_model, "CPU")
                chair_infer_req = chair_compiled.create_infer_request()
                chair_inp_name = chair_model.inputs[0].get_any_name()
                c_shape = chair_model.inputs[0].shape
                if len(c_shape) >= 4:
                    chair_inp_h = int(c_shape[2])
                    chair_inp_w = int(c_shape[3])
                logging.info(f"Neural chair model compiled successfully on CPU ({chair_inp_w}x{chair_inp_h})!")
            except Exception as e2:
                logging.error(f"Failed to load neural chair model: {e2}")

    glasses_infer_req = None
    glasses_inp_name = None
    glasses_out_name = None
    glasses_inp_w = 512
    glasses_inp_h = 512
    if SEG_GLASSES_PATH.exists():
        logging.info(f"Loading BiSeNet Face Parsing model from: {SEG_GLASSES_PATH}")
        try:
            glasses_model = core.read_model(str(SEG_GLASSES_PATH))
            try:
                glasses_compiled = core.compile_model(glasses_model, npu_device)
                glasses_infer_req = glasses_compiled.create_infer_request()
                glasses_inp_name = glasses_model.inputs[0].get_any_name()
                glasses_out_name = glasses_model.outputs[0].get_any_name()
                g_shape = glasses_model.inputs[0].shape
                if len(g_shape) >= 4:
                    glasses_inp_h = int(g_shape[2])
                    glasses_inp_w = int(g_shape[3])
                logging.info(f"BiSeNet Face Parsing model (CelebAMask-HQ) compiled successfully on {npu_device} ({glasses_inp_w}x{glasses_inp_h})!")
            except Exception as e:
                logging.warning(f"Could not compile BiSeNet Face Parsing model on {npu_device}: {e}. Retrying on CPU...")
                glasses_compiled = core.compile_model(glasses_model, "CPU")
                glasses_infer_req = glasses_compiled.create_infer_request()
                glasses_inp_name = glasses_model.inputs[0].get_any_name()
                glasses_out_name = glasses_model.outputs[0].get_any_name()
                g_shape = glasses_model.inputs[0].shape
                if len(g_shape) >= 4:
                    glasses_inp_h = int(g_shape[2])
                    glasses_inp_w = int(g_shape[3])
                logging.info(f"BiSeNet Face Parsing model compiled successfully on CPU ({glasses_inp_w}x{glasses_inp_h})!")
        except Exception as e2:
            logging.error(f"Failed to load BiSeNet Face Parsing model: {e2}")

    last_bg_path = ""
    cached_bg = None
    oversized_bg = None
    smooth_head_pos = None
    last_config_check = time.time()
    last_config_mtime = 0

    logging.info(f"Opening virtual camera: {out_device} ({out_w}x{out_h} @ {target_fps}fps)...")
    with pyvirtualcam.Camera(width=out_w, height=out_h, fps=target_fps, device=out_device, fmt=pyvirtualcam.PixelFormat.BGR) as vcam:
        logging.info(f"Virtual camera started on {vcam.device} at {out_w}x{out_h}!")

        placeholder_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        cv2.putText(placeholder_frame, "Intel NPU 1080p Webcam - Conectando...", (150, out_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 2)
        vcam.send(placeholder_frame)

        cap = None
        framer = None
        privacy_muted = False
        privacy_screen = None
        last_privacy_path = None
        privacy_fade_alpha = 0.0
        candidate_gesture = None
        candidate_gesture_time = 0.0
        gesture_match_count = 0
        no_hand_count = 0
        gesture_animation = None
        last_gesture_mute_time = 0.0
        gesture_latched = False
        gesture_cooldown_until = 0.0
        prev_mask = None
        chair_frame_counter = 0
        cached_chair_mask = None
        glasses_frame_counter = 0
        cached_glasses_mask = None
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Pre-warm gesture emoji cache into memory
        for g_name in ["thumbs_up", "thumbs_down", "peace", "rock", "call_me", "pointing_up", "fist", "ok_hand", "open_palm"]:
            get_gesture_emoji(g_name)

        last_consumer_check = 0.0
        last_active_time = 0.0
        has_active_consumers = False
        in_standby = False

        while running:
            now = time.time()

            # Periodic config reload check
            if now - last_config_check > 1.0:
                last_config_check = now
                if CONFIG_PATH.exists():
                    try:
                        mtime = CONFIG_PATH.stat().st_mtime
                        if mtime != last_config_mtime:
                            last_config_mtime = mtime
                            cfg = load_config()
                            if framer:
                                framer.smoothness = float(cfg.get("framing_smoothness", 0.04))
                                framer.deadzone = float(cfg.get("framing_deadzone", 0.10))
                    except Exception:
                        pass

            # On-Demand Auto-Standby: Check if any application is reading from /dev/video72
            auto_standby = cfg.get("auto_standby", True)
            standby_timeout = float(cfg.get("standby_timeout", 3.0))

            if auto_standby:
                if now - last_consumer_check > 0.5:
                    last_consumer_check = now
                    consumer_cnt = count_active_consumers(out_device)
                    if consumer_cnt > 0:
                        last_active_time = now
                        has_active_consumers = True
                    else:
                        if (now - last_active_time) > standby_timeout:
                            has_active_consumers = False

                if not has_active_consumers:
                    if not in_standby:
                        in_standby = True
                        if privacy_muted:
                            set_audio_mute(False)
                            privacy_muted = False

                        # 1. Desativar a camera fisica primeiro (libera dispositivo V4L2 e apaga LED)
                        if cap is not None:
                            logging.info("Modo Standby On-Demand: nenhum aplicativo usando a camera. Desativando sensor fisico (LED OFF)...")
                            cap.release()
                            cap = None
                            framer = None
                            prev_mask = None
                        else:
                            logging.info("Modo Standby On-Demand ativo: sensor fisico ja desligado (LED OFF). Aguardando aplicativo em /dev/video72...")

                        # 2. Aguardar 1 segundo com a camera desativada antes do parking
                        park_delay = float(cfg.get("standby_park_delay", 1.0))
                        if park_delay > 0:
                            time.sleep(park_delay)

                        # 3. Executar o parking da camera (PTZ tilt/pan para recolher)
                        phys_dev = f"/dev/video{resolve_cam_index(cfg.get('input_device', 'auto'))}"
                        execute_standby_hooks(cfg, phys_dev)
                    vcam.send(placeholder_frame)
                    time.sleep(0.3)
                    continue

                if in_standby:
                    in_standby = False
                    logging.info("Modo Standby On-Demand: aplicativo detectado em /dev/video72! Ligando sensor fisico...")
                    phys_dev = f"/dev/video{resolve_cam_index(cfg.get('input_device', 'auto'))}"
                    execute_wakeup_hooks(cfg, phys_dev)

            # Check camera connection
            if cap is None or not cap.isOpened():
                in_dev_cfg = cfg.get("input_device", "auto")
                idx = resolve_cam_index(in_dev_cfg)
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, out_w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, out_h)
                    cap.set(cv2.CAP_PROP_FPS, target_fps)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        framer = AutoFramer(
                            actual_w, actual_h, out_w, out_h,
                            smoothness=float(cfg.get("framing_smoothness", 0.04)),
                            deadzone=float(cfg.get("framing_deadzone", 0.10))
                        )
                        logging.info(f"Physical camera /dev/video{idx} connected at {actual_w}x{actual_h} ({target_fps} FPS)!")
                    else:
                        cap.release()
                        cap = None
                else:
                    cap.release()
                    cap = None
                
                if cap is None:
                    vcam.send(placeholder_frame)
                    vcam.sleep_until_next_frame()
                    time.sleep(0.5)
                    continue

            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("Failed to read frame from physical camera. Will reconnect...")
                cap.release()
                cap = None
                continue

            # 1. Framing and Face Landmark Tracking (Single vs Group Mode)
            auto_framing_enabled = cfg.get("auto_framing", True)
            framing_mode = cfg.get("framing_mode", "single")
            framed = framer.update(frame, enabled=auto_framing_enabled, framing_mode=framing_mode) if framer else cv2.resize(frame, (out_w, out_h))

            # 2. Smart Auto-Privacy on Absence with Smooth Fade-in / Fade-out Transition
            privacy_enabled = cfg.get("privacy_enabled", False)
            is_absent = False
            if privacy_enabled and framer:
                ptimeout = float(cfg.get("privacy_timeout", 3.0))
                is_absent = (now - framer.last_face_time) > ptimeout

            fade_enabled = cfg.get("privacy_fade_enabled", True)
            fade_step = float(cfg.get("privacy_fade_speed", 0.07)) if fade_enabled else 1.0

            if is_absent:
                privacy_fade_alpha = min(1.0, privacy_fade_alpha + fade_step)
            else:
                privacy_fade_alpha = max(0.0, privacy_fade_alpha - fade_step)

            if privacy_fade_alpha > 0.0:
                custom_priv = cfg.get("privacy_image", "")
                if privacy_screen is None or custom_priv != last_privacy_path:
                    privacy_screen = draw_privacy_screen(out_w, out_h, custom_priv)
                    last_privacy_path = custom_priv

            if privacy_fade_alpha >= 1.0:
                if not privacy_muted and cfg.get("privacy_mute_mic", True):
                    set_audio_mute(True)
                    privacy_muted = True
                vcam.send(privacy_screen)
                vcam.sleep_until_next_frame()
                continue
            elif privacy_fade_alpha < 0.2 and privacy_muted:
                set_audio_mute(False)
                privacy_muted = False

            face_info = framer.get_framed_face_info() if framer else {"has_face": False}

            # 3. Adaptive Low-Light Booster
            if cfg.get("low_light_enabled", False):
                ll_gain = int(cfg.get("low_light_gain", 45))
                framed = apply_low_light_booster(framed, gain_percent=ll_gain)

            # 3.1 Screen Glare Suppressor (Monitor Reflection Neutralizer)
            if cfg.get("screen_glare_enabled", False):
                glare_str = int(cfg.get("screen_glare_strength", 50))
                framed = apply_screen_glare_suppressor(framed, strength=glare_str)

            # 4. Virtual Studio Light (Face Relighting)
            if cfg.get("studio_light_enabled", False) and face_info.get("has_face"):
                sl_gain = int(cfg.get("studio_light_gain", 40))
                framed = apply_studio_light(framed, face_info, intensity=sl_gain)

            # 5. Eye Contact (Natural vs Teleprompter Mode)
            eye_mode = cfg.get("eye_contact_mode", "natural" if cfg.get("eye_contact_enabled", False) else "off")
            if eye_mode != "off" and face_info.get("has_face"):
                framed = apply_eye_contact(framed, face_info, mode=eye_mode)

            # 6. Edge-Preserving Denoising & Skin Smoothing
            smooth_enabled = cfg.get("smooth_enabled", True)
            if smooth_enabled:
                sc = int(cfg.get("smooth_strength", 50))
                sc = max(10, min(100, sc))
                processed_fg = cv2.bilateralFilter(framed, d=5, sigmaColor=sc, sigmaSpace=sc)
            else:
                processed_fg = framed

            # 7. Super-Resolution & Detail Sharpening
            if cfg.get("sharpen_enabled", False):
                sh_strength = int(cfg.get("sharpen_strength", 35))
                processed_fg = apply_smart_sharpen(processed_fg, strength=sh_strength)

            # 3. Background Blur / Replacement
            blur_enabled = cfg.get("blur_enabled", True)
            bg_path = os.path.expanduser(cfg.get("background_image", ""))

            if blur_enabled or bg_path:
                if is_multiclass:
                    # High quality 256x256 RGB input for NPU multiclass model
                    small = cv2.resize(framed, (256, 256))
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    blob = np.expand_dims(rgb, axis=0)

                    # NPU Inference (~13.5ms)
                    infer_request.infer({seg_inp_name: blob})
                    raw = infer_request.get_tensor(seg_out_name).data[0]  # (256, 256, 6)

                    # Softmax probabilities
                    exp_raw = np.exp(raw - np.max(raw, axis=-1, keepdims=True))
                    probs = exp_raw / np.sum(exp_raw, axis=-1, keepdims=True)

                    # 1.0 - background gives full human silhouette (classes 1..5)
                    p_person = 1.0 - probs[:, :, 0]

                    # Direct foreground vs background logit differential:
                    # In multiclass, the sum of human classes (hair, skin, clothes, etc.) can be diluted
                    # when individual classes are split. fg_max_logits compares the strongest human class
                    # against the background logit directly!
                    fg_max_logits = np.max(raw[:, :, 1:], axis=-1)
                    bg_logits = raw[:, :, 0]
                    logit_diff = fg_max_logits - bg_logits
                    p_direct_fg = 1.0 / (1.0 + np.exp(-1.8 * logit_diff))
                    p_person = np.maximum(p_person, p_direct_fg)

                    # Boost human foreground components: body-skin and clothes (arms, sleeves, torso)
                    body_skin = probs[:, :, 2]
                    clothes = probs[:, :, 4]
                    human_body = np.maximum(body_skin * 1.45, clothes * 1.40)
                    p_person = np.maximum(p_person, np.clip(human_body, 0.0, 1.0))

                    # Anatomical Torso Core Boost:
                    # Directly below the chin/neck, the central column beneath the face
                    # is the user's torso. Reinforce any signal here so shirts never become transparent.
                    if face_info.get("has_face"):
                        fx_b, fy_b, fw_b, fh_b = face_info["box"]
                        fx_sb = int(fx_b * 256 / out_w)
                        fy_sb = int(fy_b * 256 / out_h)
                        fw_sb = int(fw_b * 256 / out_w)
                        fh_sb = int(fh_b * 256 / out_h)
                        cx_sb = max(0, min(255, fx_sb + fw_sb // 2))

                        torso_y_start = min(250, fy_sb + int(fh_sb * 1.10))
                        if torso_y_start < 255:
                            y_idx_arr = np.arange(torso_y_start, 256)
                            prog = (y_idx_arr - torso_y_start) / max(1.0, 255 - torso_y_start)
                            # Expands from shoulders down to waist
                            hws = (fw_sb * (0.85 + prog * 0.65)).astype(int)
                            torso_mask = np.zeros((256, 256), dtype=bool)
                            for y_idx, hw in zip(y_idx_arr, hws):
                                x1_t = max(0, cx_sb - hw)
                                x2_t = min(256, cx_sb + hw)
                                torso_mask[y_idx, x1_t:x2_t] = True

                            torso_signal = p_person[torso_mask]
                            if len(torso_signal) > 0 and np.mean(torso_signal) > 0.15:
                                p_person[torso_mask] = np.maximum(p_person[torso_mask], 0.92)

                    # Morphological vertical closing to seal cloth folds, buttons, creases and arm-torso gaps
                    torso_close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 21))
                    p_closed = cv2.morphologyEx(p_person, cv2.MORPH_CLOSE, torso_close_k)
                    p_person = np.maximum(p_person, p_closed)

                    # Solidify interior holes inside the silhouette (flood-fill background from corners)
                    p_bin = (p_person > 0.25).astype(np.uint8)
                    if face_info.get("has_face"):
                        b_x1 = max(0, cx_sb - int(fw_sb * 1.4))
                        b_x2 = min(256, cx_sb + int(fw_sb * 1.4))
                        p_bin[253:256, b_x1:b_x2] = 1

                    flood_padded = cv2.copyMakeBorder(p_bin, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                    f_h, f_w = flood_padded.shape
                    flood_mask = np.zeros((f_h + 2, f_w + 2), np.uint8)
                    cv2.floodFill(flood_padded, flood_mask, (0, 0), 255)
                    internal_holes = ((flood_padded[1:-1, 1:-1] == 0) & (p_bin == 0)).astype(np.float32)
                    p_person = np.maximum(p_person, internal_holes * 0.98)

                    # 1. Hand isolation (excluding face and strictly anatomical throat)
                    hand_skin = body_skin.copy()
                    if face_info.get("has_face"):
                        fx, fy, fw, fh = face_info["box"]
                        fx_s = int(fx * 256 / out_w)
                        fy_s = int(fy * 256 / out_h)
                        fw_s = int(fw * 256 / out_w)
                        fh_s = int(fh * 256 / out_h)
                        # Head / Face exclusion box
                        y1_ex = max(0, fy_s - int(fh_s * 0.20))
                        y2_ex = min(256, fy_s + int(fh_s * 1.15))
                        x1_ex = max(0, fx_s - int(fw_s * 0.20))
                        x2_ex = min(256, fx_s + int(fw_s * 1.20))
                        hand_skin[y1_ex:y2_ex, x1_ex:x2_ex] = 0.0

                        # Anatomical throat / neck exclusion (strictly between chin and collarbone)
                        y1_neck = max(0, fy_s + int(fh_s * 0.92))
                        y2_neck = min(256, fy_s + int(fh_s * 1.45))
                        x1_neck = max(0, fx_s + int(fw_s * 0.18))
                        x2_neck = min(256, fx_s + int(fw_s * 0.82))
                        hand_skin[y1_neck:y2_neck, x1_neck:x2_neck] = 0.0

                    has_hands = bool(np.max(hand_skin) > 0.20)

                    # 2. Handheld Object Retention (Preservação de Objetos Segurados na Mão)
                    if cfg.get("object_retention_enabled", True) and has_hands:
                        obj_str = int(cfg.get("object_retention_strength", 60))
                        k_sz = max(7, int(15 * (obj_str / 50.0)))
                        if k_sz % 2 == 0:
                            k_sz += 1
                        k_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
                        hand_bin = (hand_skin > 0.22).astype(np.uint8)
                        interaction_zone = cv2.dilate(hand_bin, k_el, iterations=2)
                        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz * 2 + 1, k_sz * 2 + 1))
                        closed_mask = cv2.morphologyEx(p_person, cv2.MORPH_CLOSE, close_k)
                        retained_candidate = np.where(interaction_zone > 0, closed_mask, p_person)
                        p_person = np.maximum(p_person, retained_candidate)

                    # 2.1 Neural Chair & Headrest Retention (Modelo de Segmentacao de Instancias YOLACT na NPU)
                    if cfg.get("chair_retention_enabled", True):
                        chair_str = int(cfg.get("chair_retention_strength", 50))
                        should_infer_chair = (chair_frame_counter % 2 == 0) or (cached_chair_mask is None)
                        p_person, cached_chair_mask = apply_neural_chair_retention(
                            p_person, framed,
                            chair_infer_req=chair_infer_req,
                            chair_inp_name=chair_inp_name,
                            strength=chair_str,
                            cached_chair_mask=cached_chair_mask,
                            run_inference=should_infer_chair,
                            inp_w=chair_inp_w,
                            inp_h=chair_inp_h
                        )
                        chair_frame_counter += 1
                    else:
                        cached_chair_mask = None

                    # 3. Real-time Hand Gesture Recognition & Triggers
                    if cfg.get("gesture_detection_enabled", True):
                        g_act = cfg.get("gesture_action", "all")
                        g_type, g_pos = (None, None)
                        if has_hands:
                            g_type, g_pos = detect_hand_gesture(hand_skin, out_w, out_h)

                        if g_type is not None:
                            no_hand_count = 0
                            if g_type == candidate_gesture:
                                gesture_match_count += 1
                            else:
                                candidate_gesture = g_type
                                gesture_match_count = 1
                                candidate_gesture_time = now

                            # Only consider gesture after strictly 0.5 seconds of uninterrupted detection of the exact same gesture
                            time_held = now - candidate_gesture_time
                            if time_held >= 0.50:
                                if not gesture_latched and (now >= gesture_cooldown_until):
                                    if g_act in ["reaction", "all"]:
                                        gesture_animation = {
                                            "type": g_type,
                                            "pos": g_pos,
                                            "start_time": now,
                                            "duration": 1.8
                                        }
                                    if g_type == "open_palm" and g_act in ["mute_toggle", "all"]:
                                        if (now - last_gesture_mute_time) > 1.8:
                                            is_currently_muted = check_audio_muted()
                                            set_audio_mute(not is_currently_muted)
                                            last_gesture_mute_time = now

                                    gesture_latched = True
                                    gesture_cooldown_until = now + 0.9
                        else:
                            # Hand absent or gesture broken: reset candidate and streak immediately
                            candidate_gesture = None
                            candidate_gesture_time = 0.0
                            gesture_match_count = 0
                            no_hand_count += 1
                            if no_hand_count >= 2:
                                gesture_latched = False
                    else:
                        candidate_gesture = None
                        candidate_gesture_time = 0.0
                        gesture_match_count = 0
                        no_hand_count = 0
                        gesture_latched = False
                else:
                    # Legacy 144x256 model fallback
                    small = cv2.resize(framed, (256, 144))
                    blob = np.expand_dims(np.transpose(small.astype(np.float32) / 255.0, (2, 0, 1)), axis=0)
                    infer_request.infer({seg_inp_name: blob})
                    p_person = infer_request.get_tensor(seg_out_name).data[0, 0]

                # 2.2 Neural Eyeglasses & Frame Retention (Modelo BiSeNet Face Parsing na NPU - MIT License)
                preserve_glasses = cfg.get("preserve_glasses", cfg.get("glasses_protection", True))
                if preserve_glasses and face_info.get("has_face") and glasses_infer_req is not None:
                    should_infer_glasses = (glasses_frame_counter % 2 == 1) or (cached_glasses_mask is None)
                    cached_glasses_mask = apply_neural_glasses_retention(
                        framed, face_info,
                        glasses_infer_req=glasses_infer_req,
                        glasses_inp_name=glasses_inp_name,
                        glasses_out_name=glasses_out_name,
                        cached_glasses_mask=cached_glasses_mask,
                        run_inference=should_infer_glasses,
                        inp_w=glasses_inp_w,
                        inp_h=glasses_inp_h
                    )
                    glasses_frame_counter += 1
                    if cached_glasses_mask is not None:
                        g_low = cv2.resize(cached_glasses_mask, (p_person.shape[1], p_person.shape[0]), interpolation=cv2.INTER_LINEAR)
                        p_person = np.maximum(p_person, g_low)
                else:
                    cached_glasses_mask = None

                # Motion-Adaptive Temporal Filtering:
                # Kills pixel jitter on static areas, but responds promptly to arm/hand motion
                if prev_mask is None:
                    mask_256 = p_person
                else:
                    motion_delta = np.abs(p_person - prev_mask)
                    adaptive_alpha = np.clip(0.35 + motion_delta * 1.2, 0.35, 0.85)
                    mask_256 = prev_mask * (1.0 - adaptive_alpha) + p_person * adaptive_alpha
                prev_mask = mask_256

                # Natural Smoothstep Feathering
                feather_px = float(cfg.get("mask_feather", 40))
                feather_px = max(10.0, min(80.0, feather_px))
                half_width = 0.20 * (feather_px / 40.0)
                center = 0.32
                low = max(0.06, center - half_width)
                high = min(0.92, center + half_width)

                u = np.clip((mask_256 - low) / (high - low), 0.0, 1.0)
                p_curved = u * u * (3.0 - 2.0 * u)

                # Fast Guided Filter:
                # Uses camera color/luminance to snap the mask to exact real-world 1080p hair & finger edges!
                gf_size = cfg.get("guided_filter_guide_size", [640, 360])
                if isinstance(gf_size, (list, tuple)) and len(gf_size) == 2:
                    gw, gh = int(gf_size[0]), int(gf_size[1])
                else:
                    gw, gh = 640, 360
                gw = max(640, gw)
                gh = max(360, gh)

                gf_r = int(cfg.get("guided_filter_radius", 5))
                gf_eps = float(cfg.get("guided_filter_eps", 0.001))
                gf_color = bool(cfg.get("guided_filter_color_guide", True))

                if gf_color:
                    guide_small = cv2.resize(framed, (gw, gh))
                else:
                    guide_small = cv2.resize(cv2.cvtColor(framed, cv2.COLOR_BGR2GRAY), (gw, gh))

                mask_full = fast_guided_filter(
                    guide_small, p_curved,
                    r=gf_r, eps=gf_eps,
                    out_shape=(out_w, out_h),
                    color_guide=gf_color
                )
                if cached_glasses_mask is not None:
                    mask_full = np.maximum(mask_full, cached_glasses_mask)

                # Solid Core Protection:
                # Ensure the body interior is 100% opaque without virtual background bleeding through clothes!
                mask_full = np.where(mask_full > 0.80, 1.0, mask_full)
                mask_3c = cv2.merge([mask_full, mask_full, mask_full])

                # Rim Light (Contour / Hair Light)
                if cfg.get("rim_light_enabled", False):
                    processed_fg = apply_rim_light(
                        processed_fg, mask_full,
                        color_mode=cfg.get("rim_light_color", "warm"),
                        intensity=int(cfg.get("rim_light_intensity", 50))
                    )

                if bg_path and os.path.exists(bg_path):
                    if bg_path != last_bg_path or cached_bg is None:
                        bg_img = cv2.imread(bg_path)
                        if bg_img is not None:
                            cached_bg = cv2.resize(bg_img, (out_w, out_h))
                            ow, oh = int(out_w * 1.08), int(out_h * 1.08)
                            oversized_bg = cv2.resize(bg_img, (ow, oh))
                            last_bg_path = bg_path

                    if cfg.get("parallax_enabled", False) and face_info.get("has_face") and oversized_bg is not None:
                        fx, fy, fw, fh = face_info["box"]
                        cx = fx + fw / 2.0
                        cy = fy + fh * 0.45
                        norm_x = (cx - out_w / 2.0) / (out_w / 2.0)
                        norm_y = (cy - out_h / 2.0) / (out_h / 2.0)
                        if smooth_head_pos is None:
                            smooth_head_pos = np.array([norm_x, norm_y], dtype=np.float32)
                        else:
                            smooth_head_pos = smooth_head_pos * 0.85 + np.array([norm_x, norm_y], dtype=np.float32) * 0.15

                        p_str = int(cfg.get("parallax_strength", 40))
                        max_shift = int((p_str / 100.0) * 45)
                        shift_x = int(-smooth_head_pos[0] * max_shift)
                        shift_y = int(-smooth_head_pos[1] * max_shift * 0.5)

                        ow, oh = oversized_bg.shape[1], oversized_bg.shape[0]
                        cx_bg = ow // 2 + shift_x
                        cy_bg = oh // 2 + shift_y
                        x1_bg = max(0, min(ow - out_w, cx_bg - out_w // 2))
                        y1_bg = max(0, min(oh - out_h, cy_bg - out_h // 2))
                        bg = oversized_bg[y1_bg:y1_bg + out_h, x1_bg:x1_bg + out_w]
                    else:
                        bg = cached_bg if cached_bg is not None else processed_fg
                else:
                    # Blur Mode: standard or portrait_depth
                    blur_mode = cfg.get("blur_mode", "standard")
                    blur_k = int(cfg.get("blur_strength", 35))
                    blur_k = max(5, min(99, blur_k))

                    if blur_mode == "portrait_depth":
                        # Depth-Aware Portrait Blur (Bokeh com Gradiente Óptico)
                        bk_mid = max(3, blur_k // 3)
                        if bk_mid % 2 == 0: bk_mid += 1
                        bk_deep = max(5, blur_k)
                        if bk_deep % 2 == 0: bk_deep += 1

                        small_w, small_h = out_w // 3, out_h // 3
                        small_bg = cv2.resize(framed, (small_w, small_h))
                        mask_s = cv2.resize((p_curved * 255).astype(np.uint8), (small_w, small_h))
                        inv_m = cv2.bitwise_not(mask_s)
                        dist = cv2.distanceTransform(inv_m, cv2.DIST_L2, 3)
                        dist_norm = np.clip(dist / 45.0, 0.0, 1.0)

                        y_coords = np.linspace(1.0, 0.2, small_h, dtype=np.float32).reshape(-1, 1)
                        vert_grad = np.repeat(y_coords, small_w, axis=1)

                        depth_s = np.clip(dist_norm * 0.70 + vert_grad * 0.30, 0.0, 1.0)
                        depth_3c = cv2.merge([depth_s, depth_s, depth_s])

                        b_mid = cv2.blur(small_bg, (bk_mid, bk_mid))
                        b_deep = cv2.blur(small_bg, (bk_deep, bk_deep))
                        bokeh_s = (b_mid * (1.0 - depth_3c) + b_deep * depth_3c).astype(np.uint8)
                        bg = cv2.resize(bokeh_s, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        bk = max(3, blur_k // 2)
                        if bk % 2 == 0:
                            bk += 1

                        small_bg = cv2.resize(framed, (out_w // 2, out_h // 2))
                        blur_small = cv2.blur(small_bg, (bk, bk))
                        bg = cv2.resize(blur_small, (out_w, out_h))

                # Smooth natural blend
                output_frame = (processed_fg * mask_3c + bg * (1.0 - mask_3c)).astype(np.uint8)

                # 8. Post-Processing: Cinematic Color Grading
                color_f = cfg.get("color_filter", "none")
                if color_f != "none":
                    output_frame = apply_color_grading(output_frame, mode=color_f, strength=int(cfg.get("color_strength", 70)))

                # 9. Post-Processing: Creative Artistic Filter
                art_f = cfg.get("artistic_filter", "off")
                if art_f != "off":
                    output_frame = apply_artistic_filter(output_frame, mode=art_f, strength=int(cfg.get("artistic_strength", 70)))
            else:
                prev_mask = None
                cached_glasses_mask = None
                cached_chair_mask = None
                output_frame = processed_fg

                # Post-Processing: Cinematic Color Grading & Artistic Filters even without blur
                color_f = cfg.get("color_filter", "none")
                if color_f != "none":
                    output_frame = apply_color_grading(output_frame, mode=color_f, strength=int(cfg.get("color_strength", 70)))

                art_f = cfg.get("artistic_filter", "off")
                if art_f != "off":
                    output_frame = apply_artistic_filter(output_frame, mode=art_f, strength=int(cfg.get("artistic_strength", 70)))

            # Render Gesture Reaction Animation if active
            if gesture_animation is not None:
                g_prog = (now - gesture_animation["start_time"]) / gesture_animation["duration"]
                if g_prog < 1.0:
                    output_frame = draw_gesture_reaction_fast(
                        output_frame,
                        gesture_animation["type"],
                        gesture_animation["pos"],
                        g_prog
                    )
                else:
                    gesture_animation = None

            # Apply smooth privacy fade transition if dissolving in or out
            if privacy_fade_alpha > 0.0 and privacy_screen is not None:
                output_frame = cv2.addWeighted(privacy_screen, privacy_fade_alpha, output_frame, 1.0 - privacy_fade_alpha, 0)

            # Send 1080p frame to virtual camera
            vcam.send(output_frame)
            vcam.sleep_until_next_frame()

        if cap is not None:
            logging.info("Encerrando daemon: desativando sensor fisico (LED OFF)...")
            cap.release()
            cap = None
        try:
            park_delay = float(cfg.get("standby_park_delay", 1.0))
            if park_delay > 0:
                time.sleep(park_delay)
            phys_dev = f"/dev/video{resolve_cam_index(cfg.get('input_device', 'auto'))}"
            execute_standby_hooks(cfg, phys_dev)
        except Exception:
            pass

    logging.info("Webcam daemon stopped.")

if __name__ == "__main__":
    main()
