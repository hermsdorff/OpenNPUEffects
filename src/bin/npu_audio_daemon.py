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
import struct
from pathlib import Path
import numpy as np
import openvino as ov
import sounddevice as sd
from scipy import signal as dsp_signal
import copy
import uuid
from npu_pipeline.telemetry import (
    BoundedTelemetry,
    NullTelemetry,
    ObservationOrigin,
    SCHEMA_VERSION,
    duration_ms,
    event_json,
    provenance_fields,
)
from npu_pipeline.telemetry_runtime import TelemetryRuntime, validate_m0_config
from npu_pipeline.audio_capture import (
    CaptureReadStatus,
    AudioCaptureState,
    CaptureBlockResult,
    WriteResult,
    AudioCaptureReader,
    AudioOutputWriter,
    CaptureRetryPolicy,
    AudioSwitchCoordinator,
)

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

def stop_process(p, timeout=0.4):
    if p is None:
        return
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(p, stream_name, None)
        if stream is not None and hasattr(stream, "fileno"):
            try:
                fd = stream.fileno()
                if isinstance(fd, int) and fd >= 0:
                    os.set_blocking(fd, False)
            except Exception:
                pass
    try:
        p.terminate()
        p.wait(timeout=timeout)
    except Exception:
        try:
            p.kill()
            p.wait(timeout=timeout)
        except Exception:
            pass
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(p, stream_name, None)
        if stream is not None and not getattr(stream, "closed", False):
            try:
                if hasattr(stream, "detach"):
                    raw = stream.detach()
                    if raw is not None:
                        raw.close()
                else:
                    stream.close()
            except Exception:
                pass

def run_cmd(cmd, timeout=0.5):
    try:
        env = dict(os.environ, LC_ALL="C")
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True, env=env, timeout=timeout)
        return res.stdout.strip()
    except Exception as e:
        logging.debug(f"Command failed '{cmd}': {e}")
        return ""

def load_config(last_valid=None, return_status=False):
    default_audio = {
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
        "dereverb_strength": 40,
        "auto_standby": True,
        "standby_timeout": 1.0,
    }
    fallback = dict(last_valid) if isinstance(last_valid, dict) else dict(default_audio)

    bool_fields = (
        "enabled",
        "noise_suppression",
        "studio_mic_enabled",
        "lowcut_enabled",
        "studio_eq_enabled",
        "gate_enabled",
        "deesser_enabled",
        "compressor_enabled",
        "dereverb_enabled",
        "auto_standby",
    )
    str_fields = ("input_device", "output_sink", "output_source")
    num_fields = ("gate_threshold_db", "comp_target_db", "dereverb_strength", "standby_timeout")

    try:
        if not CONFIG_PATH.exists():
            return (fallback, True) if return_status else fallback
        with open(CONFIG_PATH, "r") as f:
            raw_cfg = json.load(f)
        if not isinstance(raw_cfg, dict):
            logging.warning("Invalid config root (expected JSON object); preserving last valid config.")
            return (fallback, False) if return_status else fallback
        if "audio" in raw_cfg:
            audio_sec = raw_cfg["audio"]
            if not isinstance(audio_sec, dict):
                logging.warning("Invalid 'audio' config section (expected JSON object); preserving last valid config.")
                return (fallback, False) if return_status else fallback
        else:
            audio_sec = {}

        merged = dict(default_audio)
        for k, v in audio_sec.items():
            if k in bool_fields:
                if not isinstance(v, bool):
                    logging.warning(f"Invalid boolean config field '{k}'; preserving last valid config.")
                    return (fallback, False) if return_status else fallback
            elif k in str_fields:
                if not isinstance(v, str) or not v.strip():
                    logging.warning(f"Invalid string config field '{k}'; preserving last valid config.")
                    return (fallback, False) if return_status else fallback
            elif k in num_fields:
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    logging.warning(f"Invalid numeric config field '{k}'; preserving last valid config.")
                    return (fallback, False) if return_status else fallback
            merged[k] = v

        return (merged, True) if return_status else merged
    except Exception as e:
        logging.warning(f"Error reading config: {e}")
        return (fallback, False) if return_status else fallback

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

def select_hardware_microphone(cfg, avail=None):
    configured = cfg.get("input_device", "auto") if isinstance(cfg, dict) else "auto"
    if not isinstance(configured, str) or not configured.strip():
        configured = "auto"
    if avail is None:
        avail = get_physical_microphones()
    avail = [m for m in (avail or []) if isinstance(m, str) and not is_virtual_source(m)]
    if not avail:
        return None
    if configured != "auto":
        for m in avail:
            if configured == m:
                return m
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

def get_pactl_object_id(list_cmd, target_name):
    """Percorre a saida 'curta' do pactl (id \t nome \t ...) e retorna o id
    numerico do objeto (source, sink, etc.) cujo nome bate com target_name."""
    out = run_cmd(list_cmd)
    for line in out.splitlines():
        parts = line.split('\t')
        if len(parts) >= 2 and parts[1] == target_name:
            return parts[0]
    return None

def count_active_source_consumers(source_name):
    """Conta quantos source-outputs (consumidores reais, ex.: Zoom/Teams/Meet)
    estao conectados ao microfone virtual da NPU. A propria injecao de audio
    do daemon (out_proc via pw-play) entra como sink-input no null-sink, e nao
    como source-output, entao nao precisa ser excluida do resultado."""
    src_id = get_pactl_object_id("pactl list sources short", source_name)
    if src_id is None:
        return 0
    try:
        src_id_int = int(src_id)
    except ValueError:
        src_id_int = None

    # 1. Metodo primario estruturado via JSON (imune a locale)
    out_json = run_cmd("pactl --format=json list source-outputs")
    if out_json:
        try:
            data = json.loads(out_json)
            return sum(1 for item in data if item.get("source") in (src_id, src_id_int))
        except Exception:
            pass

    # 2. Fallback de texto com suporte multi-idioma (English, Portugues, Espanhol)
    out = run_cmd("pactl list source-outputs")
    consumer_count = 0
    for raw_line in out.splitlines():
        line = raw_line.strip().lower()
        if line.startswith("source:") or line.startswith("fonte:") or line.startswith("fuente:"):
            current_source_id = line.split(":", 1)[1].strip()
            if current_source_id == str(src_id):
                consumer_count += 1
    return consumer_count

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
        self.last_speech_time = 0.0

        # 5. Compressor / AGC
        self.comp_gain = 1.0

        # 6. De-Reverb
        self.reverb_tail = 0.0

    def reset_state(self):
        """Clears temporal filter states and adaptive gains without recalculating biquad coefficients."""
        self.zi_hp = dsp_signal.sosfilt_zi(self.sos_hp)
        self.zi_warmth = dsp_signal.sosfilt_zi(self.sos_warmth)
        self.zi_presence = dsp_signal.sosfilt_zi(self.sos_presence)
        self.zi_deess = dsp_signal.sosfilt_zi(self.sos_deess)
        self.gate_gain = 1.0
        self.last_speech_time = 0.0
        self.comp_gain = 1.0
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
        is_speech = False
        rms = float(np.sqrt(np.mean(out**2) + 1e-9))
        if studio_on and cfg.get("gate_enabled", True):
            thresh_db = float(cfg.get("gate_threshold_db", -45.0))
            thresh_lin = 10 ** (thresh_db / 20.0)
            if rms < thresh_lin:
                target_gain = 0.0
                attack_coeff = 0.15
            else:
                target_gain = 1.0
                attack_coeff = 0.70
                is_speech = True
            gains = np.linspace(
                self.gate_gain,
                self.gate_gain * (1.0 - attack_coeff) + target_gain * attack_coeff,
                len(out),
                dtype=np.float32
            )
            self.gate_gain = float(gains[-1])
            out = out * gains
        else:
            is_speech = (rms > 0.015)

        now = time.time()
        if is_speech:
            self.last_speech_time = now

        # Atualiza estado de fala em memória compartilhada (/dev/shm) para o auto-zoom do daemon de vídeo
        try:
            shm_path = f"/dev/shm/npu_speech_{os.getuid()}.bin"
            data = struct.pack("<?ddf", is_speech, self.last_speech_time, now, rms)
            with open(shm_path, "wb") as f_shm:
                f_shm.write(data)
        except Exception:
            pass

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
    last_config_check = time.monotonic()
    last_config_mtime = 0
    last_loopback_check = time.monotonic()

    audio_session_id = str(uuid.uuid4())
    audio_gen_id = f"audio-gen-{uuid.uuid4().hex[:8]}"
    audio_block_id = 0
    first_sample_index = 0
    config_version = 1
    last_applied_cfg = copy.deepcopy(cfg)

    # Dynamic Capture & Switch Infrastructure
    coordinator = AudioSwitchCoordinator(
        initial_desired_source=cfg.get("input_device", "auto"),
        initial_active_source=hw_mic,
    )
    coordinator.audio_gen_id = audio_gen_id
    capture_reader = AudioCaptureReader(
        block_size_bytes=chunk_size * 4,
        poll_timeout_sec=0.025,
        block_deadline_sec=0.750,
        initial_tolerance_sec=1.500,
    )
    output_writer = AudioOutputWriter(
        block_size_bytes=chunk_size * 4,
        poll_timeout_sec=0.025,
        write_deadline_sec=0.500,
    )
    retry_policy = CaptureRetryPolicy(
        backoff_steps=(0.250, 0.500, 1.000, 2.000),
        stable_blocks_for_reset=10,
    )
    sink_retry_policy = CaptureRetryPolicy(
        backoff_steps=(0.250, 0.500, 1.000, 2.000),
        stable_blocks_for_reset=10,
    )

    # Authoritative owner counters
    read_attempt_total = 0
    received_bytes_total = 0
    full_input_blocks_total = 0
    short_read_total = 0
    input_samples_observed_total = 0
    submitted_model_samples_total = 0
    normal_output_samples_accepted_total = 0
    silence_output_samples_accepted_total = 0
    write_attempt_total = 0
    write_or_flush_failure_total = 0
    capture_reconnect_total = 0
    playback_reconnect_total = 0
    model_state_reset_total = 0
    audio_input_switch_total = 0
    capture_stall_total = 0
    capture_partial_discard_bytes_total = 0
    audio_stale_block_rejected_total = 0

    m0_runtime = TelemetryRuntime("audio", session_id=audio_session_id, config=cfg.get("audio", cfg))
    m0_runtime.register_owner_counter("read_attempt_total", lambda: read_attempt_total)
    m0_runtime.register_owner_counter("received_bytes_total", lambda: received_bytes_total)
    m0_runtime.register_owner_counter("full_input_blocks_total", lambda: full_input_blocks_total)
    m0_runtime.register_owner_counter("short_read_total", lambda: short_read_total)
    m0_runtime.register_owner_counter("input_samples_observed_total", lambda: input_samples_observed_total)
    m0_runtime.register_owner_counter("submitted_model_samples_total", lambda: submitted_model_samples_total)
    m0_runtime.register_owner_counter("normal_output_samples_accepted_total", lambda: normal_output_samples_accepted_total)
    m0_runtime.register_owner_counter("silence_output_samples_accepted_total", lambda: silence_output_samples_accepted_total)
    m0_runtime.register_owner_counter("write_attempt_total", lambda: write_attempt_total)
    m0_runtime.register_owner_counter("write_or_flush_failure_total", lambda: write_or_flush_failure_total)
    m0_runtime.register_owner_counter("capture_reconnect_total", lambda: capture_reconnect_total)
    m0_runtime.register_owner_counter("playback_reconnect_total", lambda: playback_reconnect_total)
    m0_runtime.register_owner_counter("model_state_reset_total", lambda: model_state_reset_total)
    m0_runtime.register_owner_counter("audio_input_switch_total", lambda: audio_input_switch_total)
    m0_runtime.register_owner_counter("capture_stall_total", lambda: capture_stall_total)
    m0_runtime.register_owner_counter("capture_partial_discard_bytes_total", lambda: capture_partial_discard_bytes_total)
    m0_runtime.register_owner_counter("audio_stale_block_rejected_total", lambda: audio_stale_block_rejected_total)

    logging.info(f"Starting real-time audio filter stream via PipeWire (chunk={chunk_size}, rate={rate}Hz)...")
    play_node_name = f"npu_mic_feed_{os.getpid()}"
    rec_node_name = f"npu_mic_capture_{os.getpid()}"
    out_cmd = ["pw-play", "--raw", "--target", "0", "-P", f"node.name={play_node_name}", "--format", "f32", "--rate", str(rate), "--channels", "1", "-"]

    def spawn_out_proc():
        nonlocal playback_reconnect_total
        playback_reconnect_total += 1
        p = subprocess.Popen(out_cmd, stdin=subprocess.PIPE, bufsize=0)
        time.sleep(0.15)
        run_cmd(f"pw-link {play_node_name}:output_MONO {source_name}:input_FL 2>/dev/null || true")
        run_cmd(f"pw-link {play_node_name}:output_MONO {source_name}:input_FR 2>/dev/null || true")
        return p

    def spawn_in_proc(target_mic):
        nonlocal capture_reconnect_total, audio_gen_id, first_sample_index, capture_partial_discard_bytes_total
        if not target_mic:
            return None
        capture_reconnect_total += 1
        audio_gen_id = f"audio-gen-{uuid.uuid4().hex[:8]}"
        first_sample_index = 0
        coordinator.begin_connecting(target_mic, audio_gen_id)
        cmd = [
            "pw-record",
            "--raw",
            "--target", target_mic,
            "-P", f"node.name={rec_node_name}",
            "-P", "node.dont-reconnect=true",
            "--format", "f32",
            "--rate", str(rate),
            "--channels", "1",
            "-"
        ]
        logging.info(f"Connecting audio capture stream to physical microphone '{target_mic}' (gen={audio_gen_id}, epoch={coordinator.epoch})...")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        discarded = capture_reader.attach(p, generation_id=audio_gen_id, epoch=coordinator.epoch, is_first_block=True)
        if discarded > 0:
            capture_partial_discard_bytes_total += discarded
        return p

    frame_duration_ns = int(frame_duration * 1_000_000_000)
    m0_runtime._frame_duration_ns = frame_duration_ns
    m0_runtime._next_silence_ns = time.monotonic_ns()
    m0_runtime._last_silence_ns = 0
    m0_runtime._last_health_ns = time.monotonic_ns()
    m0_runtime._last_complete_block_ns = None
    m0_runtime._last_accepted_output_ns = None

    def _build_health_payload(now_ns):
        last_comp = m0_runtime._last_complete_block_ns
        last_out = m0_runtime._last_accepted_output_ns
        return {
            "capture_state": coordinator.state.value,
            "desired_source": coordinator.desired_source,
            "active_source": coordinator.active_source,
            "epoch": coordinator.epoch,
            "audio_gen_id": coordinator.audio_gen_id,
            "read_pending": capture_reader.is_pending,
            "partial_bytes": capture_reader.buffer_len,
            "pending_block_age_ms": round(capture_reader.pending_age_ms(), 2),
            "last_complete_block_age_ms": round((now_ns - last_comp) / 1_000_000, 2) if last_comp is not None else None,
            "last_accepted_output_age_ms": round((now_ns - last_out) / 1_000_000, 2) if last_out is not None else None,
            "total_discarded_bytes": capture_reader.total_discarded_bytes,
            "output_pending": output_writer.has_pending,
            "output_partial_bytes": output_writer.pending_offset,
            "output_pending_age_ms": round(output_writer.pending_age_ms(), 2),
            "output_total_discarded_bytes": output_writer.total_discarded_bytes,
            "consecutive_failures": retry_policy.consecutive_failures,
            "sink_consecutive_failures": sink_retry_policy.consecutive_failures,
        }

    m0_runtime._health_payload_fn = _build_health_payload

    def _fail_and_close_out_proc(reason="sink_error"):
        nonlocal out_proc
        output_writer.reset(reason=reason)
        if out_proc is not None:
            stop_process(out_proc)
            out_proc = None
        if not running:
            return
        if sink_retry_policy.can_retry():
            out_proc = spawn_out_proc()
            if out_proc is None:
                sink_retry_policy.record_failure()

    def _reconnect_out_proc_for_alignment():
        nonlocal out_proc
        output_writer.reset(reason="stream_realign")
        if out_proc is not None:
            stop_process(out_proc)
            out_proc = None
        if not running:
            return
        out_proc = spawn_out_proc()
        if out_proc is None:
            sink_retry_policy.record_failure()

    def _ensure_out_proc():
        nonlocal out_proc
        if out_proc is not None:
            return True
        if not running:
            return False
        if not sink_retry_policy.can_retry():
            return False
        out_proc = spawn_out_proc()
        if out_proc is None:
            sink_retry_policy.record_failure()
            return False
        return True

    def _handle_output_result(wres, now_ns):
        nonlocal normal_output_samples_accepted_total, silence_output_samples_accepted_total, write_or_flush_failure_total
        if wres.pending:
            return False
        if wres.success and wres.samples_written == chunk_size:
            meta = wres.metadata or {}
            kind = meta.get("kind", "normal")
            write_start_ns = meta.get("write_start_ns", now_ns)
            sink_retry_policy.record_success()
            m0_runtime._last_accepted_output_ns = now_ns
            if kind == "silence":
                silence_output_samples_accepted_total += wres.samples_written
                if m0_runtime.enabled:
                    m0_runtime.telemetry.sample("audio.write_call_ms", (now_ns - write_start_ns) / 1_000_000)
            else:
                normal_output_samples_accepted_total += wres.samples_written
                arrival_ns = meta.get("arrival_mono_ns", write_start_ns)
                if m0_runtime.enabled:
                    m0_runtime.telemetry.sample("audio.write_call_ms", (now_ns - write_start_ns) / 1_000_000)
                    m0_runtime.telemetry.sample("audio.arrival_to_flush_return_ms", (now_ns - arrival_ns) / 1_000_000)
                blk_id = meta.get("block_id")
                if blk_id is not None and m0_runtime.should_trace(blk_id):
                    m0_runtime.telemetry.emit(
                        "audio_block",
                        generation_id=meta.get("generation_id", coordinator.audio_gen_id),
                        block_id=blk_id,
                        first_observed_sample_index=meta.get("first_sample_index", 0),
                        sample_count=chunk_size,
                        rate=rate,
                        signal_duration_ms=128.0,
                        timestamp_quality="ARRIVAL_ESTIMATE",
                        sink_presentation_latency_ms=None,
                    )
            return True

        write_or_flush_failure_total += 1
        sink_retry_policy.record_failure()
        if m0_runtime.enabled:
            m0_runtime.telemetry.emit(
                "audio_sink_error",
                error=wres.error,
                bytes_written=wres.bytes_written,
                partial_discarded_bytes=wres.partial_discarded_bytes,
            )
        if running:
            logging.warning("PipeWire playback process pipe closed or stalled, restarting...")
            _fail_and_close_out_proc(reason="sink_error")
        return False

    def send_silence():
        nonlocal write_attempt_total
        if not _ensure_out_proc():
            return
        if output_writer.has_pending:
            wres_pend = output_writer.write_step(out_proc)
            _handle_output_result(wres_pend, time.monotonic_ns())
            if output_writer.has_pending or out_proc is None:
                return
        write_attempt_total += 1
        t_w_s = time.monotonic_ns()
        wres = output_writer.write_step(
            out_proc,
            silence_chunk,
            metadata={
                "kind": "silence",
                "generation_id": coordinator.audio_gen_id,
                "epoch": coordinator.epoch,
                "block_id": None,
                "first_sample_index": None,
                "arrival_mono_ns": t_w_s,
                "write_start_ns": t_w_s,
            },
        )
        t_w_e = time.monotonic_ns()
        _handle_output_result(wres, t_w_e)

    out_proc = spawn_out_proc()
    in_proc = spawn_in_proc(hw_mic) if hw_mic else None
    if in_proc is None:
        coordinator.mark_recovering()

    # Standby automatico: libera o microfone fisico (LED apagado) quando nenhum
    # aplicativo estiver de fato consumindo o microfone virtual da NPU.
    last_consumer_check = 0.0
    last_active_time = time.monotonic()
    in_standby = False

    try:
        while running:
            now_m = time.monotonic()

            # Dynamic config and hot-plug check (every 200 ms)
            if now_m - last_config_check > 0.2:
                last_config_check = now_m
                if CONFIG_PATH.exists():
                    try:
                        mtime = CONFIG_PATH.stat().st_mtime
                        if mtime != last_config_mtime:
                            candidate_cfg, is_valid_cfg = load_config(last_valid=last_applied_cfg, return_status=True)
                            if is_valid_cfg:
                                last_config_mtime = mtime
                                cfg = candidate_cfg
                                if cfg != last_applied_cfg:
                                    last_applied_cfg = copy.deepcopy(cfg)
                                    config_version += 1
                                    m0_runtime.update_config(cfg.get("audio", cfg))
                            else:
                                cfg = last_applied_cfg
                    except Exception:
                        cfg = last_applied_cfg

                desired_input = cfg.get("input_device", "auto")
                best_mic = select_hardware_microphone(cfg)

                switch_requested = coordinator.request_source_switch(desired_input)
                mic_changed = (best_mic != hw_mic)

                if switch_requested or mic_changed:
                    if not switch_requested:
                        coordinator.notify_source_change()
                    audio_input_switch_total += 1
                    if m0_runtime.enabled:
                        m0_runtime.telemetry.emit(
                            "audio_source_switch_requested",
                            old_source=hw_mic,
                            new_desired=desired_input,
                            target_mic=best_mic,
                            epoch=coordinator.epoch,
                        )
                    logging.info(f"Audio source switch: desired='{desired_input}', target='{best_mic}', previous='{hw_mic}'")

                    # Invalidate active reader epoch and discard partial bytes
                    discarded = capture_reader.set_epoch(coordinator.epoch)
                    capture_partial_discard_bytes_total += discarded

                    if in_proc is not None:
                        discarded = capture_reader.detach()
                        capture_partial_discard_bytes_total += discarded
                        stop_process(in_proc)
                        in_proc = None

                    if output_writer.has_pending:
                        cancel_res = output_writer.cancel_pending(reason="source_switch")
                        if m0_runtime.enabled:
                            m0_runtime.telemetry.emit(
                                "audio_output_pending_cancelled",
                                reason="source_switch",
                                bytes_written=cancel_res.bytes_written,
                                partial_discarded_bytes=cancel_res.partial_discarded_bytes,
                                requires_reconnect=cancel_res.requires_reconnect,
                            )
                        if cancel_res.requires_reconnect:
                            _reconnect_out_proc_for_alignment()

                    effects_chain.reset_state()
                    states = {n: np.zeros(inp_shapes[n], dtype=np.float32) for n in state_names}
                    model_state_reset_total += 1

                    hw_mic = best_mic
                    if hw_mic and not in_standby:
                        in_proc = spawn_in_proc(hw_mic)
                    elif not hw_mic:
                        coordinator.mark_recovering()
                        logging.warning(f"Requested microphone '{desired_input}' unavailable (SOURCE_UNAVAILABLE). Emitting silence...")

                # Maintain virtual microphone as default source (re-assert if stolen by hotplug)
                cur_def = run_cmd("pactl get-default-source")
                if cur_def and cur_def != source_name and not is_virtual_source(cur_def):
                    original_default_source = cur_def
                    run_cmd(f"pactl set-default-source {source_name}")

            # Voice zoom check from video
            voice_zoom_requested = False
            try:
                if CONFIG_PATH.exists():
                    with open(CONFIG_PATH, "r") as f_cfg:
                        full_c = json.load(f_cfg)
                        if isinstance(full_c, dict) and isinstance(full_c.get("video"), dict):
                            voice_zoom_requested = bool(full_c["video"].get("framing_voice_zoom", False))
            except Exception:
                pass

            if voice_zoom_requested and in_standby:
                logging.info("Zoom por Voz ativado no vídeo. Saindo do standby de áudio para monitorar fala...")
                in_standby = False
                last_active_time = now_m
                if in_proc is None and hw_mic:
                    in_proc = spawn_in_proc(hw_mic)
                    effects_chain.reset_state()
                    states = {n: np.zeros(inp_shapes[n], dtype=np.float32) for n in state_names}
                    model_state_reset_total += 1

            # Auto-standby check
            if cfg.get("auto_standby", True) and not voice_zoom_requested:
                check_interval = 0.2 if in_standby else 0.25
                if now_m - last_consumer_check > check_interval:
                    last_consumer_check = now_m
                    active_consumers = count_active_source_consumers(source_name)
                    if active_consumers > 0:
                        last_active_time = now_m
                        if in_standby:
                            logging.info(f"Consumidor detectado no microfone virtual ({active_consumers} ativo(s)). Saindo do standby de audio...")
                            in_standby = False
                            if in_proc is None and hw_mic:
                                in_proc = spawn_in_proc(hw_mic)
                                effects_chain.reset_state()
                                states = {n: np.zeros(inp_shapes[n], dtype=np.float32) for n in state_names}
                                model_state_reset_total += 1

                    standby_timeout = float(cfg.get("standby_timeout", 1.0))
                    if active_consumers == 0 and not in_standby and (now_m - last_active_time) > standby_timeout:
                        logging.info(f"Nenhum consumidor do microfone virtual ha {standby_timeout:.1f}s. Entrando em standby de audio (liberando microfone fisico)...")
                        in_standby = True
                        coordinator.mark_standby()
                        if in_proc is not None:
                            discarded = capture_reader.detach()
                            capture_partial_discard_bytes_total += discarded
                            stop_process(in_proc)
                            in_proc = None

            # Safeguard: prevent illegal loopback (our capture node connected to npu_clearvoice)
            if in_proc and not in_standby and (now_m - last_loopback_check > 3.0):
                last_loopback_check = now_m
                loop_check = run_cmd(f"pw-link -l | grep -A 1 '^{rec_node_name}:input' | grep npu_clearvoice || true")
                if loop_check:
                    logging.warning(f"Detected illegal loopback routing ({rec_node_name} connected to npu_clearvoice). Resetting capture...")
                    discarded = capture_reader.detach()
                    capture_partial_discard_bytes_total += discarded
                    stop_process(in_proc)
                    in_proc = None

            # Advance any in-flight incremental output write before standby or capture read
            if output_writer.has_pending and _ensure_out_proc():
                wres_pend = output_writer.write_step(out_proc)
                _handle_output_result(wres_pend, time.monotonic_ns())

            # Handle standby or temporarily unavailable physical microphone before read step
            if in_standby or in_proc is None:
                if not in_standby and in_proc is None and retry_policy.can_retry():
                    new_mic = select_hardware_microphone(cfg)
                    if new_mic:
                        hw_mic = new_mic
                        in_proc = spawn_in_proc(hw_mic)
                        effects_chain.reset_state()
                        states = {n: np.zeros(inp_shapes[n], dtype=np.float32) for n in state_names}
                        model_state_reset_total += 1
                    else:
                        retry_policy.record_failure()

                if in_standby or in_proc is None:
                    now_ns_idle = time.monotonic_ns()
                    wait_ns = max(0, m0_runtime._next_silence_ns - now_ns_idle)
                    if wait_ns > 0:
                        time.sleep(min(wait_ns / 1_000_000_000.0, 0.025))
                    now_ns_idle = time.monotonic_ns()
                    if now_ns_idle >= m0_runtime._next_silence_ns:
                        send_silence()
                        m0_runtime._last_silence_ns = now_ns_idle
                        m0_runtime._next_silence_ns += frame_duration_ns
                        if m0_runtime._next_silence_ns <= now_ns_idle:
                            m0_runtime._next_silence_ns = now_ns_idle + frame_duration_ns
                    if m0_runtime.enabled and (now_ns_idle - m0_runtime._last_health_ns) >= 5_000_000_000:
                        m0_runtime._last_health_ns = now_ns_idle
                        m0_runtime.telemetry.emit("audio_capture_health", **_build_health_payload(now_ns_idle))
                    continue

            # Read audio chunk from physical microphone via non-blocking reader
            t_read_start = time.monotonic_ns()
            result = capture_reader.read_step(timeout_sec=0.025)
            t_read_end = time.monotonic_ns()
            if m0_runtime.enabled:
                m0_runtime.telemetry.sample("audio.read_call_ms", (t_read_end - t_read_start) / 1_000_000)

            if not running:
                break

            if result.status == CaptureReadStatus.PENDING:
                pass
            elif result.status == CaptureReadStatus.STALL:
                capture_stall_total += 1
                capture_partial_discard_bytes_total += result.discarded_bytes
                coordinator.mark_recovering()
                if m0_runtime.enabled:
                    m0_runtime.telemetry.emit(
                        "audio_capture_stall",
                        reason=result.reason,
                        discarded_bytes=result.discarded_bytes,
                        epoch=coordinator.epoch,
                        generation_id=coordinator.audio_gen_id,
                    )
                logging.warning("Audio capture stall detected (deadline exceeded). Resetting capture...")
                discarded = capture_reader.detach()
                capture_partial_discard_bytes_total += discarded
                stop_process(in_proc)
                in_proc = None
                retry_policy.record_failure()
            elif result.status == CaptureReadStatus.EOF:
                short_read_total += 1
                capture_partial_discard_bytes_total += result.discarded_bytes
                coordinator.mark_recovering()
                if m0_runtime.enabled:
                    m0_runtime.telemetry.emit(
                        "audio_short_read",
                        requested_samples=chunk_size,
                        actual_samples=result.discarded_bytes // 4,
                        actual_bytes=result.discarded_bytes,
                    )
                logging.warning(f"Physical microphone '{hw_mic}' stream ended (EOF). Waiting for reconnection...")
                discarded = capture_reader.detach()
                capture_partial_discard_bytes_total += discarded
                stop_process(in_proc)
                in_proc = None
                retry_policy.record_failure()
            elif result.status == CaptureReadStatus.ERROR:
                capture_partial_discard_bytes_total += result.discarded_bytes
                coordinator.mark_recovering()
                logging.warning(f"Audio capture error: {result.reason}. Resetting capture...")
                discarded = capture_reader.detach()
                capture_partial_discard_bytes_total += discarded
                stop_process(in_proc)
                in_proc = None
                retry_policy.record_failure()
            elif result.status == CaptureReadStatus.CANCELLED:
                pass
            elif not coordinator.is_block_valid(result):
                audio_stale_block_rejected_total += 1
            elif (
                result.first_byte_time is not None
                and getattr(m0_runtime, "_last_silence_ns", 0) > 0
                and int(result.first_byte_time * 1_000_000_000) < getattr(m0_runtime, "_last_silence_ns", 0)
            ):
                # Obsolescence rule: a silence slot was already emitted after this block began forming,
                # so discarding the late block prevents output latency backlog accumulation.
                audio_stale_block_rejected_total += 1
                capture_partial_discard_bytes_total += len(result.data) if result.data else 0
            else:
                if coordinator.state != AudioCaptureState.ACTIVE:
                    coordinator.mark_active()
                    if m0_runtime.enabled:
                        m0_runtime.telemetry.emit(
                            "audio_capture_resumed",
                            source=hw_mic,
                            generation_id=coordinator.audio_gen_id,
                            epoch=coordinator.epoch,
                        )

                retry_policy.record_success()
                read_attempt_total += 1
                received_bytes_total += len(result.data)
                full_input_blocks_total += 1
                input_samples_observed_total += chunk_size
                audio_block_id += 1
                arrival_mono_ns = t_read_end
                m0_runtime._last_complete_block_ns = t_read_end

                data = np.frombuffer(result.data, dtype=np.float32)

                if cfg.get("noise_suppression", True):
                    # Feed chunk to NPU
                    t_prep_start = time.monotonic_ns()
                    chunk = data[None, :]  # shape (1, 2048)
                    feed = {"input": chunk}
                    feed.update(states)
                    t_prep_end = time.monotonic_ns()
                    if m0_runtime.enabled:
                        m0_runtime.telemetry.sample("audio.feed_prepare_ms", (t_prep_end - t_prep_start) / 1_000_000)

                    t_infer_start = time.monotonic_ns()
                    infer_request.infer(feed)
                    t_infer_end = time.monotonic_ns()
                    submitted_model_samples_total += chunk_size
                    if m0_runtime.enabled:
                        m0_runtime.telemetry.sample("audio.infer_call_ms", (t_infer_end - t_infer_start) / 1_000_000)

                    t_fetch_start = time.monotonic_ns()
                    cleaned = infer_request.get_tensor("output").data  # shape (1, 2048)

                    # Update state tensors for seamless next iteration
                    for n in state_names:
                        out_name = n.replace("inp", "out")
                        states[n] = infer_request.get_tensor(out_name).data
                    t_fetch_end = time.monotonic_ns()
                    if m0_runtime.enabled:
                        m0_runtime.telemetry.sample("audio.output_and_state_fetch_ms", (t_fetch_end - t_fetch_start) / 1_000_000)

                    audio_for_dsp = cleaned[0]
                else:
                    audio_for_dsp = data

                # Apply studio DSP chain (Low-cut, De-reverb, Studio EQ, De-esser, Gate/VAD, Compressor/AGC)
                t_dsp_start = time.monotonic_ns()
                processed = effects_chain.process(audio_for_dsp, cfg)
                t_dsp_end = time.monotonic_ns()
                if m0_runtime.enabled:
                    m0_runtime.telemetry.sample("audio.dsp_ms", (t_dsp_end - t_dsp_start) / 1_000_000)

                t_ser_start = time.monotonic_ns()
                out_bytes = processed.tobytes()
                t_ser_end = time.monotonic_ns()
                if m0_runtime.enabled:
                    m0_runtime.telemetry.sample("audio.serialization_ms", (t_ser_end - t_ser_start) / 1_000_000)

                if output_writer.has_pending and _ensure_out_proc():
                    wres_pend = output_writer.write_step(out_proc)
                    _handle_output_result(wres_pend, time.monotonic_ns())
                    if not running:
                        break

                t_write_start = time.monotonic_ns()
                m0_runtime._next_silence_ns = t_write_start + getattr(m0_runtime, "_frame_duration_ns", 128_000_000)

                if output_writer.has_pending:
                    if m0_runtime.enabled:
                        m0_runtime.telemetry.emit(
                            "audio_output_backpressure_drop",
                            generation_id=coordinator.audio_gen_id,
                            epoch=coordinator.epoch,
                            block_id=audio_block_id,
                            pending_remaining_bytes=output_writer.pending_remaining_bytes,
                        )
                elif _ensure_out_proc():
                    write_attempt_total += 1
                    wres = output_writer.write_step(
                        out_proc,
                        out_bytes,
                        metadata={
                            "kind": "normal",
                            "generation_id": coordinator.audio_gen_id,
                            "epoch": coordinator.epoch,
                            "block_id": audio_block_id,
                            "first_sample_index": first_sample_index,
                            "arrival_mono_ns": arrival_mono_ns,
                            "write_start_ns": t_write_start,
                        },
                    )
                    t_write_end = time.monotonic_ns()
                    m0_runtime._next_silence_ns = t_write_end + getattr(m0_runtime, "_frame_duration_ns", 128_000_000)
                    _handle_output_result(wres, t_write_end)
                    if not running:
                        break

                first_sample_index += chunk_size

            # Centralized monotonic silence pacing (128 ms) & periodic health reporting (5.0 s)
            _next_sil_ns = getattr(m0_runtime, "_next_silence_ns", None)
            if _next_sil_ns is None:
                _next_sil_ns = t_read_start
                m0_runtime._next_silence_ns = _next_sil_ns
            if t_read_end >= _next_sil_ns:
                send_silence()
                m0_runtime._last_silence_ns = t_read_end
                _frame_ns = getattr(m0_runtime, "_frame_duration_ns", 128_000_000)
                m0_runtime._next_silence_ns = _next_sil_ns + _frame_ns
                if m0_runtime._next_silence_ns <= t_read_end:
                    m0_runtime._next_silence_ns = t_read_end + _frame_ns

            if m0_runtime.enabled:
                _last_hlth_ns = getattr(m0_runtime, "_last_health_ns", None)
                if _last_hlth_ns is None:
                    _last_hlth_ns = t_read_start
                    m0_runtime._last_health_ns = _last_hlth_ns
                if (t_read_end - _last_hlth_ns) >= 5_000_000_000:
                    m0_runtime._last_health_ns = t_read_end
                    _hlth_fn = getattr(m0_runtime, "_health_payload_fn", None)
                    _hlth_kw = _hlth_fn(t_read_end) if _hlth_fn is not None else {
                        "capture_state": "CONNECTING",
                        "read_pending": getattr(capture_reader, "is_pending", True),
                        "partial_bytes": getattr(capture_reader, "buffer_len", 0),
                        "pending_block_age_ms": getattr(capture_reader, "pending_age_ms", lambda: 0.0)(),
                        "total_discarded_bytes": getattr(capture_reader, "total_discarded_bytes", 0),
                    }
                    m0_runtime.telemetry.emit("audio_capture_health", **_hlth_kw)

    except Exception as e:
        logging.error(f"Error in audio streaming loop: {e}")
    finally:
        coordinator.mark_stopping()
        try:
            m0_runtime.stop()
        except Exception as e:
            logging.debug(f"Telemetry shutdown ignored error: {e}")
        if in_proc is not None:
            capture_reader.detach()
            stop_process(in_proc)
        output_writer.reset(reason="shutdown")
        stop_process(out_proc)
        cleanup()
        logging.info("Audio daemon stopped.")

if __name__ == "__main__":
    main()
