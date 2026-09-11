#!/usr/bin/env python3
"""
Intel NPU Audio Daemon
Real-time AI Noise Suppression & Voice Isolation offloaded to Intel Meteor Lake NPU.
Outputs to PipeWire virtual microphone 'npu_clearvoice' for transparent system-wide usage.
"""
import os
import sys
import time
import json
import signal
import logging
import subprocess
from pathlib import Path
import numpy as np
import openvino as ov
import sounddevice as sd
from scipy import signal as dsp_signal

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [NPU-AUDIO] %(levelname)s: %(message)s'
)

CONFIG_PATH = Path.home() / ".config/npu-effects/config.json"
OPT_AUDIO_MODEL = Path("/opt/npu-effects/models/audio/noise-suppression-poconetlike-0001.xml")
MODEL_PATH = OPT_AUDIO_MODEL if OPT_AUDIO_MODEL.exists() else Path.home() / ".local/share/npu-effects/models/audio/noise-suppression-poconetlike-0001.xml"

running = True
original_default_source = None
loaded_modules = []

def handle_signal(sig, frame):
    global running
    logging.info("Shutting down audio daemon...")
    running = False

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        logging.error(f"Command failed '{cmd}': {e}")
        return ""

def load_config():
    default_config = {
        "audio": {
            "enabled": True,
            "noise_suppression": True,
            "input_device": "auto",
            "output_sink": "NPU_Mic_Sink",
            "output_source": "npu_clearvoice",
            "studio_mic_enabled": True,
            "lowcut_enabled": True,
            "studio_eq_enabled": True,
            "gate_enabled": True,
            "gate_threshold_db": -45.0,
            "deesser_enabled": True,
            "compressor_enabled": True,
            "comp_target_db": -18.0,
            "dereverb_enabled": True,
            "dereverb_strength": 40
        }
    }
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
                return cfg.get("audio", default_config["audio"])
    except Exception as e:
        logging.warning(f"Error reading config: {e}")
    return default_config["audio"]

def is_virtual_source(name):
    lower = name.lower()
    return lower.startswith("npu") or "clearvoice" in lower or "monitor" in lower or "null" in lower

def get_current_default_source():
    out = run_cmd("pactl get-default-source")
    return out if out and not is_virtual_source(out) else None

def get_physical_microphones():
    out = run_cmd("pactl list sources short")
    mics = []
    for line in out.splitlines():
        parts = line.split('\t')
        if len(parts) >= 2:
            name = parts[1]
            if not is_virtual_source(name):
                mics.append(name)
    return mics

def select_hardware_microphone(cfg):
    configured = cfg.get("input_device", "auto")
    avail = get_physical_microphones()
    if not avail:
        return None
    if configured != "auto":
        if configured in avail:
            return configured
        return None
    # In auto mode: prefer dedicated USB microphone adapter over webcam or internal mics
    usb_external = [m for m in avail if "usb" in m.lower() and "pixy" not in m.lower()]
    if usb_external:
        return usb_external[0]
    usb_mics = [m for m in avail if "usb" in m.lower()]
    if usb_mics:
        return usb_mics[0]
    return avail[0]

def get_current_default_sink():
    out = run_cmd("pactl get-default-sink")
    if out and "npu" not in out.lower() and "null" not in out.lower():
        return out
    sinks_out = run_cmd("pactl list sinks short")
    for line in sinks_out.splitlines():
        parts = line.split('\t')
        if len(parts) >= 2:
            s_name = parts[1]
            if "npu" not in s_name.lower() and "null" not in s_name.lower():
                return s_name
    return None

def setup_pipewire_virtual_devices(source_name="npu_clearvoice"):
    global loaded_modules
    # 1. Clean up legacy intermediate sinks (NPU_Mic_Sink) or filters if present
    try:
        modules_out = run_cmd("pactl list modules short")
        for line in modules_out.splitlines():
            parts = line.split('\t')
            if len(parts) >= 3:
                mod_id = parts[0]
                args = parts[2]
                if "NPU_Mic_Sink" in args or ("module-remap-source" in parts[1] and "npu_clearvoice" in args):
                    run_cmd(f"pactl unload-module {mod_id}")
                    logging.info(f"Cleaned up legacy audio module {mod_id}")
    except Exception as e:
        logging.warning(f"Error checking legacy modules: {e}")

    # 2. Register virtual microphone source directly in PipeWire
    sources = run_cmd("pactl list sources short")
    source_exists = source_name in sources

    if not source_exists:
        cmd = (
            f'pactl load-module module-null-sink media.class=Audio/Source/Virtual '
            f'sink_name={source_name} channel_map=front-left,front-right '
            f'\'sink_properties=device.description="Intel_NPU_ClearVoice_Microphone" device.class="sound"\''
        )
        mod_id = run_cmd(cmd)
        if mod_id:
            loaded_modules.append(mod_id)
            logging.info(f"Created virtual microphone source '{source_name}' (module {mod_id})")

    time.sleep(0.3)

def cleanup():
    global original_default_source, loaded_modules
    logging.info("Cleaning up audio virtual devices...")
    if original_default_source:
        logging.info(f"Restoring default source to '{original_default_source}'...")
        run_cmd(f"pactl set-default-source {original_default_source}")
    for mod_id in loaded_modules:
        run_cmd(f"pactl unload-module {mod_id}")
    logging.info("Cleanup completed.")

class AudioEffectsChain:
    def __init__(self, fs=16000):
        self.fs = fs
        # 1. High-Pass Filter (Low-cut 80Hz Butterworth 2nd order)
        self.sos_hp = dsp_signal.butter(2, 80, btype='highpass', fs=fs, output='sos')
        self.zi_hp = dsp_signal.sosfilt_zi(self.sos_hp)

        # 2. Studio Vocal EQ
        # Warmth at 250Hz (+1.5 dB, Q=1.0)
        self.sos_warmth = self._biquad_peaking(250, 1.5, 1.0, fs)
        self.zi_warmth = dsp_signal.sosfilt_zi(self.sos_warmth)
        # Presence & Articulation at 3500Hz (+2.5 dB, Q=1.2)
        self.sos_presence = self._biquad_peaking(3500, 2.5, 1.2, fs)
        self.zi_presence = dsp_signal.sosfilt_zi(self.sos_presence)

        # 3. De-Esser bandpass (5200-7500 Hz)
        self.sos_deess = dsp_signal.butter(2, [5200, 7500], btype='bandpass', fs=fs, output='sos')
        self.zi_deess = dsp_signal.sosfilt_zi(self.sos_deess)

        # 4. Smart Noise Gate (VAD)
        self.gate_gain = 1.0

        # 5. Compressor / AGC
        self.comp_gain = 1.0

        # 6. De-Reverb
        self.reverb_tail = 0.0

    def _biquad_peaking(self, f0, gain_db, q, fs):
        w0 = 2 * np.pi * f0 / fs
        alpha = np.sin(w0) / (2 * q)
        A = 10 ** (gain_db / 40.0)
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        b = np.array([b0/a0, b1/a0, b2/a0], dtype=np.float64)
        a = np.array([1.0, a1/a0, a2/a0], dtype=np.float64)
        return dsp_signal.tf2sos(b, a)

    def process(self, chunk, cfg):
        out = chunk.copy().astype(np.float32)

        # Master toggle for studio suite
        studio_on = cfg.get("studio_mic_enabled", True)

        # 1. High-Pass Filter (Low-Cut 80Hz) - elimina vibração da mesa, toques no teclado e vento
        if studio_on and cfg.get("lowcut_enabled", True):
            out, self.zi_hp = dsp_signal.sosfilt(self.sos_hp, out, zi=self.zi_hp)

        # 2. Desreverberação de Sala (De-Reverb) - reduz eco oco de paredes e piso frio
        if studio_on and cfg.get("dereverb_enabled", True):
            strength = float(cfg.get("dereverb_strength", 40)) / 100.0
            abs_x = np.abs(out)
            self.reverb_tail = max(float(np.mean(abs_x)), self.reverb_tail * 0.985)
            reverb_est = self.reverb_tail * (strength * 0.45)
            out = np.sign(out) * np.maximum(0.0, abs_x - reverb_est)

        # 3. Studio EQ (Presença Vocal & Calor Broadcast)
        if studio_on and cfg.get("studio_eq_enabled", True):
            out, self.zi_warmth = dsp_signal.sosfilt(self.sos_warmth, out, zi=self.zi_warmth)
            out, self.zi_presence = dsp_signal.sosfilt(self.sos_presence, out, zi=self.zi_presence)

        # 4. De-Esser (Atenuação Inteligente de Sibilâncias "S" e "CH")
        if studio_on and cfg.get("deesser_enabled", True):
            sibilance, self.zi_deess = dsp_signal.sosfilt(self.sos_deess, out, zi=self.zi_deess)
            sib_rms = np.sqrt(np.mean(sibilance**2) + 1e-9)
            tot_rms = np.sqrt(np.mean(out**2) + 1e-9)
            ratio = sib_rms / tot_rms
            if ratio > 0.38 and tot_rms > 0.01:
                excess = min(1.0, (ratio - 0.38) * 4.0)
                deess_gain = 1.0 - (excess * 0.5)  # atenua sibilâncias estridentes até 6dB
                out = out - sibilance * (1.0 - deess_gain)

        # 5. Smart Noise Gate (VAD - Silêncio absoluto durante pausas na fala)
        if studio_on and cfg.get("gate_enabled", True):
            thresh_db = float(cfg.get("gate_threshold_db", -45.0))
            thresh_lin = 10 ** (thresh_db / 20.0)
            rms = np.sqrt(np.mean(out**2) + 1e-9)
            if rms < thresh_lin:
                target_gain = 0.0
                attack_coeff = 0.15
            else:
                target_gain = 1.0
                attack_coeff = 0.70
            gains = np.linspace(
                self.gate_gain,
                self.gate_gain * (1.0 - attack_coeff) + target_gain * attack_coeff,
                len(out),
                dtype=np.float32
            )
            self.gate_gain = float(gains[-1])
            out = out * gains

        # 6. Compressor Vocal & Auto Gain Control (AGC / Normalizador de Ganho)
        if studio_on and cfg.get("compressor_enabled", True):
            target_rms = 10 ** (float(cfg.get("comp_target_db", -18.0)) / 20.0)
            cur_rms = np.sqrt(np.mean(out**2) + 1e-9)
            if cur_rms > 1e-4:
                desired_gain = target_rms / cur_rms
                desired_gain = np.clip(desired_gain, 0.4, 3.5)
                alpha_c = 0.90 if desired_gain < self.comp_gain else 0.97
                self.comp_gain = alpha_c * self.comp_gain + (1.0 - alpha_c) * desired_gain
                out = out * self.comp_gain

            # Soft Limiter / Anti-Clipping Brickwall
            out = np.tanh(out * 1.05) * 0.95

        return out.astype(np.float32)

def main():
    global running, original_default_source
    logging.info("Starting Intel NPU Audio Daemon...")

    cfg = load_config()
    sink_name = cfg.get("output_sink", "NPU_Mic_Sink")
    source_name = cfg.get("output_source", "npu_clearvoice")

    # Save original physical default source
    orig = get_current_default_source()
    if orig:
        original_default_source = orig
        logging.info(f"Detected original default microphone: {original_default_source}")

    # Determine initial input hardware mic
    hw_mic = select_hardware_microphone(cfg)
    if hw_mic:
        logging.info(f"Physical microphone selected: {hw_mic}")
    else:
        logging.warning("No physical microphone currently detected. Waiting in standby...")

    # Setup PipeWire virtual microphone node
    setup_pipewire_virtual_devices(source_name)

    # Initialize OpenVINO NPU Core
    logging.info("Initializing OpenVINO NPU Engine for Audio...")
    core = ov.Core()
    devices = core.available_devices
    npu_device = "NPU" if "NPU" in devices else "CPU"
    logging.info(f"Targeting audio inference device: {npu_device}")

    if not MODEL_PATH.exists():
        logging.error(f"Audio model not found at {MODEL_PATH}")
        sys.exit(1)

    model = core.read_model(str(MODEL_PATH))
    compiled_model = core.compile_model(model, npu_device)
    infer_request = compiled_model.create_infer_request()
    logging.info("Audio NPU model compiled successfully!")

    inp_shapes = {item.get_any_name(): item.shape for item in model.inputs}
    state_names = [n for n in inp_shapes.keys() if "state" in n]
    states = {n: np.zeros(inp_shapes[n], dtype=np.float32) for n in state_names}

    # Set virtual mic as default source for all system apps
    logging.info(f"Setting '{source_name}' as system default microphone...")
    run_cmd(f"pactl set-default-source {source_name}")

    chunk_size = 2048
    rate = 16000
    silence_chunk = np.zeros(chunk_size, dtype=np.float32).tobytes()
    frame_duration = chunk_size / rate  # ~0.128s

    effects_chain = AudioEffectsChain(rate)
    last_config_check = time.time()
    last_config_mtime = 0
    last_loopback_check = time.time()

    logging.info(f"Starting real-time audio filter stream via PipeWire (chunk={chunk_size}, rate={rate}Hz)...")
    play_node_name = f"npu_mic_feed_{os.getpid()}"
    out_cmd = ["pw-play", "--target", "0", "-P", f"node.name={play_node_name}", "--format", "f32", "--rate", str(rate), "--channels", "1", "-"]

    def spawn_out_proc():
        p = subprocess.Popen(out_cmd, stdin=subprocess.PIPE, bufsize=chunk_size * 4)
        time.sleep(0.15)
        run_cmd(f"pw-link {play_node_name}:output_MONO {source_name}:input_FL 2>/dev/null || true")
        run_cmd(f"pw-link {play_node_name}:output_MONO {source_name}:input_FR 2>/dev/null || true")
        return p

    def spawn_in_proc(target_mic):
        if not target_mic:
            return None
        cmd = [
            "pw-record",
            "--target", target_mic,
            "-P", "node.dont-reconnect=true",
            "--format", "f32",
            "--rate", str(rate),
            "--channels", "1",
            "-"
        ]
        logging.info(f"Connecting audio capture stream to physical microphone '{target_mic}'...")
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=chunk_size * 4)

    out_proc = spawn_out_proc()
    in_proc = spawn_in_proc(hw_mic)

    try:
        while running:
            now = time.time()

            # Dynamic config and hot-plug check
            if now - last_config_check > 1.0:
                last_config_check = now
                if CONFIG_PATH.exists():
                    try:
                        mtime = CONFIG_PATH.stat().st_mtime
                        if mtime != last_config_mtime:
                            last_config_mtime = mtime
                            cfg = load_config()
                    except Exception:
                        pass

                # Check if preferred microphone reconnected or changed
                best_mic = select_hardware_microphone(cfg)
                if best_mic and best_mic != hw_mic:
                    logging.info(f"Preferred physical microphone '{best_mic}' detected (was '{hw_mic}'). Switching...")
                    if in_proc:
                        try:
                            in_proc.terminate()
                            in_proc.wait(timeout=0.3)
                        except Exception:
                            pass
                    hw_mic = best_mic
                    in_proc = spawn_in_proc(hw_mic)
                    states = {n: np.zeros(inp_shapes[n], dtype=np.float32) for n in state_names}

            # Safeguard: prevent illegal loopback (pw-record connected to npu_clearvoice)
            if in_proc and (now - last_loopback_check > 3.0):
                last_loopback_check = now
                loop_check = run_cmd("pw-link -l | grep -A 1 '^pw-record:input' | grep npu_clearvoice || true")
                if loop_check:
                    logging.warning("Detected illegal loopback routing (pw-record connected to npu_clearvoice). Resetting capture...")
                    try:
                        in_proc.terminate()
                        in_proc.wait(timeout=0.3)
                    except Exception:
                        pass
                    in_proc = None

            # Handle case where physical microphone is temporarily unavailable (e.g. USB adapter restarted)
            if in_proc is None:
                new_mic = select_hardware_microphone(cfg)
                if new_mic:
                    hw_mic = new_mic
                    in_proc = spawn_in_proc(hw_mic)
                    states = {n: np.zeros(inp_shapes[n], dtype=np.float32) for n in state_names}
                else:
                    # Stream clean silence so Teams / apps keep receiving a smooth, alive stream
                    time.sleep(frame_duration)
                    try:
                        out_proc.stdin.write(silence_chunk)
                        out_proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        out_proc = spawn_out_proc()
                    continue

            # Read audio chunk from physical microphone
            raw = in_proc.stdout.read(chunk_size * 4)
            if not running:
                break

            # Handle physical microphone disconnect (read returns empty or short)
            if not raw or len(raw) < chunk_size * 4:
                logging.warning(f"Physical microphone '{hw_mic}' disconnected or stream ended. Waiting for reconnection...")
                try:
                    in_proc.terminate()
                    in_proc.wait(timeout=0.3)
                except Exception:
                    pass
                in_proc = None
                continue

            data = np.frombuffer(raw, dtype=np.float32)

            if cfg.get("noise_suppression", True):
                # Feed chunk to NPU
                chunk = data[None, :]  # shape (1, 2048)
                feed = {"input": chunk}
                feed.update(states)

                infer_request.infer(feed)
                cleaned = infer_request.get_tensor("output").data  # shape (1, 2048)

                # Update state tensors for seamless next iteration
                for n in state_names:
                    out_name = n.replace("inp", "out")
                    states[n] = infer_request.get_tensor(out_name).data

                audio_for_dsp = cleaned[0]
            else:
                audio_for_dsp = data

            # Apply studio DSP chain (Low-cut, De-reverb, Studio EQ, De-esser, Gate/VAD, Compressor/AGC)
            processed = effects_chain.process(audio_for_dsp, cfg)
            try:
                out_proc.stdin.write(processed.tobytes())
                out_proc.stdin.flush()
            except (BrokenPipeError, OSError):
                if not running:
                    break
                logging.warning("PipeWire playback process pipe closed, restarting...")
                out_proc = spawn_out_proc()

    except Exception as e:
        logging.error(f"Error in audio streaming loop: {e}")
    finally:
        if in_proc:
            try:
                in_proc.terminate()
            except Exception:
                pass
        try:
            out_proc.stdin.close()
            out_proc.terminate()
        except Exception:
            pass
        cleanup()
        logging.info("Audio daemon stopped.")

if __name__ == "__main__":
    main()
