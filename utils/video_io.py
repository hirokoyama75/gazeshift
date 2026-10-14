"""Bounded latest-frame capture and clean, replaceable output interfaces."""
import os
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import platform
import threading
import time
import cv2


class Capture:
    def __init__(self, camera=0, width=1280, height=720, fps=30, factory=None, system=None, clock=time.monotonic):
        system = system or platform.system()
        self.clock = clock
        self.device = (factory or cv2.VideoCapture)(camera, cv2.CAP_DSHOW if system == "Windows" else cv2.CAP_AVFOUNDATION)
        if not self.device.isOpened():
            self.device.release()
            raise RuntimeError("Physical Camera could not be opened")
        try:
            if system == "Windows":
                self.device.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.device.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.device.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.device.set(cv2.CAP_PROP_FPS, fps)
        except BaseException:
            self.device.release()
            raise
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.latest = None
        self.sequence, self.consumed, self.last_ms = 0, 0, -1
        self.done, self.error = False, None
        self.thread = threading.Thread(target=self._worker, name="gazeshift-capture", daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.device.read()
                if not ok:
                    break
                timestamp = max(self.last_ms + 1, int(self.clock() * 1000))
                self.last_ms = timestamp
                with self.condition:
                    self.sequence += 1
                    self.latest = (frame, timestamp, self.sequence)
                    self.condition.notify_all()
        except BaseException as error:
            self.error = error
        finally:
            self.device.release()
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def read(self, timeout=2):
        with self.condition:
            ready = self.condition.wait_for(lambda: self.sequence > self.consumed or self.done, timeout)
            if not ready:
                raise TimeoutError("Capture stalled")
            if self.sequence > self.consumed:
                self.consumed = self.sequence
                return self.latest[:2]
            if self.error:
                raise self.error
            return None

    def close(self):
        self.stop.set()
        self.device.release()  # Unblock a pending backend read before joining.
        self.thread.join(2)
        if self.thread.is_alive():
            raise TimeoutError("Capture worker did not stop within 2 seconds")


class VirtualCamera:
    def __init__(self, width=1280, height=720, fps=30):
        import pyvirtualcam
        self.camera = pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR, backend="obs")

    def send(self, bgr_frame):
        self.camera.send(bgr_frame)

    def close(self):
        self.camera.close()


def discover_virtual_camera(width, height):
    try:
        return VirtualCamera(width, height)
    except (ImportError, RuntimeError):
        return None


def render_hud(frame, text):
    result = frame.copy()
    cv2.putText(result, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
    return result


def calibration_canvas(bounds, state):
    import numpy as np
    from utils.calibration import monitor_center
    x, y, width, height = bounds
    image = np.zeros((height, width, 3), np.uint8)
    if state == "screen":
        center = monitor_center(bounds)
        cv2.drawMarker(image, (int(center[0]-x), int(center[1]-y)), (255, 255, 255), cv2.MARKER_CROSS, 32, 2)
    return image


class Preview:
    def __init__(self, name="gazeshift", mirror=False, monitors=None):
        self.name, self.mirror, self.monitors = name, mirror, monitors
        self.fullscreen = False
        self.bounds = None
        from utils.monitors import enable_dpi_awareness
        enable_dpi_awareness()
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    def enter_calibration(self):
        from utils.monitors import get_monitors, choose_monitor
        monitors = self.monitors() if self.monitors else get_monitors()
        monitor = choose_monitor(cv2.getWindowImageRect(self.name), monitors)
        self.bounds = (monitor.x, monitor.y, monitor.width, monitor.height)
        cv2.moveWindow(self.name, monitor.x, monitor.y)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.fullscreen = True

    def exit_calibration(self):
        if self.fullscreen:
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            self.fullscreen = False

    def show(self, frame, text="", calibration=None):
        image = cv2.flip(frame, 1) if self.mirror else frame.copy()
        if calibration and calibration.active:
            if not self.fullscreen:
                self.enter_calibration()
            image = calibration_canvas(self.bounds, calibration.state)
            text = calibration.message
        else:
            self.exit_calibration()
        cv2.imshow(self.name, render_hud(image, text))
        return cv2.waitKeyEx(1)

    send = show

    def close(self):
        cv2.destroyWindow(self.name)
