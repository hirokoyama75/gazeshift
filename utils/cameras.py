"""Read DirectShow camera names without opening a video stream."""
import ctypes
import platform
import uuid


def list_cameras():
    if platform.system() != "Windows":
        # shortcut: generic indices on macOS; use native names when adding a Mac launcher.
        return [(i, f"Camera {i}") for i in range(4)]
    ole = ctypes.OleDLL("ole32")
    auto = ctypes.OleDLL("oleaut32")
    guid = lambda value: (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le)
    cls = guid("62BE5D10-60EB-11D0-BD3B-00A0C911CE86")
    iid = guid("29840822-5B84-11D0-BD3B-00A0C911CE86")
    category = guid("860BB310-5D01-11D0-BD3B-00A0C911CE86")
    bag_iid = guid("55272A00-42CB-11CE-8135-00AA004BB851")
    def call(pointer, slot, types, *values):
        table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *types)(table[slot])(pointer, *values)
    def release(pointer):
        if pointer: call(pointer, 2, [])
    ole.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    # HRESULT must remain signed so RPC_E_CHANGED_MODE can be handled explicitly.
    ole.CoInitializeEx.restype = ctypes.c_long
    ole.CoUninitialize.argtypes = []
    ole.CoUninitialize.restype = None
    result = ole.CoInitializeEx(None, 2)
    initialized = result in (0, 1)
    if result < 0 and result != -2147417850:
        raise OSError("Cannot enumerate cameras")
    device, iterator = ctypes.c_void_p(), ctypes.c_void_p()
    try:
        ole.CoCreateInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                                        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        ole.CoCreateInstance.restype = ctypes.c_long
        if ole.CoCreateInstance(cls, None, 1, iid, ctypes.byref(device)) < 0:
            raise OSError("Cannot enumerate cameras")
        result = call(device, 3, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong],
                      category, ctypes.byref(iterator), 0)
        if result == 1: return []
        if result < 0: raise OSError("Cannot enumerate cameras")
        cameras, index = [], 0
        while True:
            moniker, bag = ctypes.c_void_p(), ctypes.c_void_p()
            if call(iterator, 3, [ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
                    1, ctypes.byref(moniker), None) != 0:
                break
            try:
                name = f"Camera {index}"
                result = call(moniker, 9, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_void_p)], None, None, bag_iid, ctypes.byref(bag))
                if result >= 0:
                    variant = (ctypes.c_ubyte * (24 if ctypes.sizeof(ctypes.c_void_p) == 8 else 16))()
                    try:
                        result = call(bag, 3, [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p],
                                      "FriendlyName", variant, None)
                        if result >= 0 and ctypes.c_ushort.from_buffer(variant).value == 8:
                            address = ctypes.c_void_p.from_buffer(variant, 8).value
                            if address: name = ctypes.wstring_at(address)
                    finally:
                        auto.VariantClear.argtypes = [ctypes.c_void_p]
                        auto.VariantClear.restype = ctypes.c_long
                        auto.VariantClear(variant)
                if name.casefold() != "obs virtual camera":
                    cameras.append((index, name))
                index += 1
            finally:
                release(bag)
                release(moniker)
        return cameras
    finally:
        release(iterator)
        release(device)
        if initialized: ole.CoUninitialize()
