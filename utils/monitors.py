"""Native monitor bounds without a PyObjC dependency; injectable at the UI seam."""
import ctypes
from dataclasses import dataclass
import platform


@dataclass(frozen=True)
class Monitor:
    x: int
    y: int
    width: int
    height: int


def enable_dpi_awareness():
    if platform.system() == "Windows":
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))


def get_monitors():
    system = platform.system()
    if system == "Windows":
        from ctypes import wintypes
        monitors = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE, wintypes.HDC,
                                           ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
        def collect(monitor, dc, rect, data):
            r = rect.contents
            monitors.append(Monitor(r.left, r.top, r.right-r.left, r.bottom-r.top))
            return True
        callback = callback_type(collect)
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.EnumDisplayMonitors.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT), callback_type, wintypes.LPARAM]
        if not user32.EnumDisplayMonitors(None, None, callback, 0):
            raise OSError(ctypes.get_last_error(), "Cannot enumerate monitor bounds")
    elif system == "Darwin":
        class Point(ctypes.Structure):
            _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]
        class Size(ctypes.Structure):
            _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]
        class Rect(ctypes.Structure):
            _fields_ = [("origin", Point), ("size", Size)]
        core = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        core.CGGetActiveDisplayList.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
        core.CGDisplayBounds.argtypes = [ctypes.c_uint32]
        core.CGDisplayBounds.restype = Rect
        displays = (ctypes.c_uint32 * 32)()
        count = ctypes.c_uint32()
        if core.CGGetActiveDisplayList(32, displays, ctypes.byref(count)) != 0 or count.value > 32:
            raise OSError("Cannot enumerate monitor bounds")
        monitors = []
        for display in displays[:count.value]:
            r = core.CGDisplayBounds(display)
            monitors.append(Monitor(round(r.origin.x), round(r.origin.y), round(r.size.width), round(r.size.height)))
    else:
        raise RuntimeError("Supported platforms are Windows 11 x64 and macOS 14+ Apple Silicon")
    if not monitors:
        raise RuntimeError("No monitor bounds available")
    return monitors


def choose_monitor(window_rect, monitors):
    x, y, width, height = window_rect
    cx, cy = x + width/2, y + height/2
    return next((m for m in monitors if m.x <= cx < m.x+m.width and m.y <= cy < m.y+m.height), monitors[0])
