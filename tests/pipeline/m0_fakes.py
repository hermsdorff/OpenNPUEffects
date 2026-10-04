"""Deterministic test fixtures and fakes for M0 video, audio, and runtime tests."""
import io
import threading
import numpy as np


class FakeClock:
    def __init__(self, start_ns=1_000_000_000):
        self.now = start_ns

    def __call__(self):
        return self.now

    def advance_ms(self, ms):
        self.now += int(ms * 1_000_000)

    def advance_ns(self, ns):
        self.now += int(ns)


class FakeCamera:
    """Simulates physical camera cap.read()."""

    def __init__(self, frames=None, clock=None):
        self.frames = list(frames) if frames else []
        self.read_index = 0
        self.clock = clock
        self.released = False
        self.read_calls = 0

    def read(self):
        self.read_calls += 1
        if self.clock:
            self.clock.advance_ms(5.0)  # simulate decode duration
        if self.read_index < len(self.frames):
            frame = self.frames[self.read_index]
            self.read_index += 1
            if frame is None:
                return False, None
            return True, frame
        return False, None

    def release(self):
        self.released = True

    def isOpened(self):
        return not self.released


class FakeVirtualCam:
    """Simulates pyvirtualcam vcam."""

    def __init__(self, clock=None, fail_on_send=False):
        self.clock = clock
        self.fail_on_send = fail_on_send
        self.sent_frames = []
        self.sleep_calls = 0
        self.closed = False

    def send(self, frame):
        if self.fail_on_send:
            raise RuntimeError("vcam send simulated failure")
        if self.clock:
            self.clock.advance_ms(1.0)
        self.sent_frames.append(frame.copy() if hasattr(frame, "copy") else frame)

    def sleep_until_next_frame(self):
        self.sleep_calls += 1
        if self.clock:
            self.clock.advance_ms(10.0)

    def close(self):
        self.closed = True


class FakeTensor:
    def __init__(self, data):
        self.data = data


class FakeInferRequest:
    """Simulates OpenVINO infer request."""

    def __init__(self, output_dict=None):
        self.output_dict = output_dict or {}
        self.last_inputs = None
        self.infer_count = 0

    def infer(self, inputs):
        self.infer_count += 1
        self.last_inputs = inputs
        return self.output_dict

    def get_tensor(self, name):
        val = self.output_dict.get(name)
        if isinstance(val, FakeTensor):
            return val
        if hasattr(val, "data") and not isinstance(val, np.ndarray):
            return val
        return FakeTensor(val)


class FakeAsyncQueue:
    """Simulates AsyncInferQueue with deterministic completion ordering."""

    def __init__(self, jobs=2):
        self.jobs = jobs
        self.callback = None
        self.pending = []  # list of (inputs, userdata)
        self._ready = True

    def set_callback(self, cb):
        self.callback = cb

    def is_ready(self):
        return len(self.pending) < self.jobs and self._ready

    def start_async(self, inputs, userdata):
        if not self.is_ready():
            raise RuntimeError("Queue not ready")
        self.pending.append((inputs, userdata))

    def finish_job(self, index, output_data, out_name="output"):
        req = FakeInferRequest({out_name: output_data})
        _, userdata = self.pending.pop(index)
        if self.callback:
            self.callback(req, userdata)


class FakeAudioPipe:
    """Simulates stdin / stdout pipes with controlled chunk delivery and failure."""

    def __init__(self, data_bytes=b"", fail_after_bytes=None, raise_on_flush=False):
        self.buffer = io.BytesIO(data_bytes)
        self.written_bytes = bytearray()
        self.fail_after_bytes = fail_after_bytes
        self.raise_on_flush = raise_on_flush
        self.write_calls = 0
        self.flush_calls = 0

    def read(self, size=-1):
        return self.buffer.read(size)

    def write(self, b):
        self.write_calls += 1
        if self.fail_after_bytes is not None and len(self.written_bytes) + len(b) > self.fail_after_bytes:
            raise BrokenPipeError("Simulated broken pipe on write")
        self.written_bytes.extend(b)
        return len(b)

    def flush(self):
        self.flush_calls += 1
        if self.raise_on_flush:
            raise OSError("Simulated flush error")


class FakeAudioProc:
    """Simulates subprocess.Popen for audio in_proc / out_proc."""

    def __init__(self, in_data=b"", fail_after_bytes=None, raise_on_flush=False):
        self.stdout = FakeAudioPipe(in_data)
        self.stdin = FakeAudioPipe(fail_after_bytes=fail_after_bytes, raise_on_flush=raise_on_flush)
        self.terminated = False

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True
