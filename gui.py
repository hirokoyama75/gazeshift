"""Gazeshift desktop: camera access starts only after the user presses Start."""
import argparse
from pathlib import Path
import queue
import tkinter as tk
from tkinter import ttk

import cv2
import onnxruntime as ort
from engine.pipeline import Correction
from utils.cameras import list_cameras
from utils.desktop import DesktopSettings, DesktopWorker, load_settings, save_settings
from utils.monitors import enable_dpi_awareness, get_monitors, choose_monitor


class Desktop:
    def __init__(self, root, settings_path, config_path):
        self.root, self.settings_path, self.config_path = root, settings_path, config_path
        self.worker, self.calibration_window, self.save_job = None, None, None
        self.closing, self.failed, self.photo = False, False, None
        warning = ""
        try:
            settings = load_settings(settings_path)
        except (OSError, ValueError, TypeError) as error:
            settings = DesktopSettings()
            warning = "Saved controls could not be loaded; using defaults. " + str(error)
        self.camera = tk.StringVar(value=str(settings.camera))
        self.providers = {"CPU": "cpu"}
        for label, key, provider in (("AMD / Intel GPU", "dml", "DmlExecutionProvider"),
                                     ("NVIDIA GPU", "cuda", "CUDAExecutionProvider"),
                                     ("Apple GPU", "coreml", "CoreMLExecutionProvider")):
            if provider in ort.get_available_providers(): self.providers[label] = key
        selected = next((label for label, key in self.providers.items() if key == settings.provider), "CPU")
        self.provider = tk.StringVar(value=selected)
        self.virtual = tk.BooleanVar(value=settings.virtual_camera)
        self.mirror = tk.BooleanVar(value=settings.mirror)
        self.enabled, self.original = tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        self.amount = tk.DoubleVar(value=settings.correction.amount * 100)
        self.horizontal = tk.DoubleVar(value=settings.correction.horizontal)
        self.vertical = tk.DoubleVar(value=settings.correction.vertical)
        self.status = tk.StringVar(value="Ready when you are")
        self.detail = tk.StringVar(value=warning or "Your camera stays off until you press Start.")
        self.output = tk.StringVar(value="Call output is off")
        self.stats = tk.StringVar(value="Camera off")
        self.scale_labels = []
        root.title("gazeshift")
        root.geometry("1160x780")
        root.minsize(980, 720)
        root.configure(bg="#f2f4f7")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#ffffff")
        style.configure("TLabel", background="#ffffff", foreground="#172633", font=("Segoe UI", 11))
        style.configure("TButton", padding=(12, 9), font=("Segoe UI", 10))
        style.configure("TCheckbutton", background="#ffffff", font=("Segoe UI", 10))
        style.configure("TCombobox", padding=6, font=("Segoe UI", 10))
        style.configure("TScale", background="#ffffff", troughcolor="#d9e1e8")
        style.configure("Primary.TButton", background="#176959", foreground="#ffffff", font=("Segoe UI", 11, "bold"))
        style.map("Primary.TButton", background=[("active", "#115246"), ("disabled", "#8a9994")])
        style.configure("Muted.TLabel", foreground="#536575", font=("Segoe UI", 9))
        header = tk.Frame(root, bg="#f2f4f7")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=26, pady=(22, 16))
        tk.Label(header, text="gazeshift", font=("Segoe UI", 27, "bold"), fg="#172633", bg="#f2f4f7").pack(side="left")
        tk.Label(header, text="A little closer to eye contact", font=("Segoe UI", 11), fg="#536575", bg="#f2f4f7").pack(side="left", padx=22)
        tk.Label(header, textvariable=self.status, font=("Segoe UI", 10, "bold"), fg="#176959", bg="#f2f4f7").pack(side="right")
        controls = ttk.Frame(root, padding=20)
        controls.grid(row=1, column=0, sticky="ns", padx=(26, 16))
        controls.columnconfigure(0, weight=1)
        ttk.Label(controls, text="YOUR CAMERA", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        camera_row = ttk.Frame(controls)
        camera_row.grid(row=1, column=0, sticky="ew", pady=(8, 12))
        self.camera_menu = ttk.Combobox(camera_row, textvariable=self.camera, state="readonly", width=27)
        self.camera_menu.pack(side="left", fill="x", expand=True)
        self.refresh_button = ttk.Button(camera_row, text="Refresh", command=self.refresh_cameras)
        self.refresh_button.pack(side="right", padx=(6, 0))
        self.start_button = ttk.Button(controls, text="Start camera", style="Primary.TButton", command=self.toggle)
        self.start_button.grid(row=2, column=0, sticky="ew", pady=(0, 18))
        tabs = ttk.Notebook(controls)
        tabs.grid(row=3, column=0, sticky="nsew")
        adjustments, setup = ttk.Frame(tabs, padding=12), ttk.Frame(tabs, padding=12)
        tabs.add(adjustments, text="Correction")
        tabs.add(setup, text="Call output & setup")
        adjustments.columnconfigure(0, weight=1)
        setup.columnconfigure(0, weight=1)
        ttk.Checkbutton(adjustments, text="Eye correction", variable=self.enabled, command=self.toggle_correction).grid(row=0, column=0, sticky="w", pady=8)
        self.slider(adjustments, 1, "Strength", self.amount, 0, 150, "%")
        self.slider(adjustments, 3, "Horizontal aim", self.horizontal, -10, 10, "°")
        self.slider(adjustments, 5, "Vertical aim", self.vertical, -10, 10, "°")
        ttk.Label(adjustments, text="100% is the original correction.\nBoost keeps the same angle limits.", style="Muted.TLabel").grid(row=7, column=0, sticky="w", pady=(6, 12))
        ttk.Button(adjustments, text="Reset adjustments", command=self.reset).grid(row=8, column=0, sticky="ew", pady=(0, 8))
        self.calibrate_button = ttk.Button(adjustments, text="Calibrate · two quick looks", command=self.calibrate, state="disabled")
        self.calibrate_button.grid(row=9, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(setup, text="Processing").grid(row=0, column=0, sticky="w")
        self.provider_menu = ttk.Combobox(setup, textvariable=self.provider, values=list(self.providers), state="readonly")
        self.provider_menu.grid(row=1, column=0, sticky="ew", pady=(6, 16))
        self.virtual_check = ttk.Checkbutton(setup, text="Send video to call apps", variable=self.virtual, command=self.save_soon)
        self.virtual_check.grid(row=2, column=0, sticky="w")
        ttk.Label(setup, text="Choose OBS Virtual Camera in your\ncall app. Leave OBS's own virtual-\ncamera output stopped.", style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=(6, 18))
        ttk.Label(setup, text="Close other apps using your webcam.\nStop the camera before changing\nyour camera or output settings.", style="Muted.TLabel").grid(row=4, column=0, sticky="w")
        self.device_controls = [self.camera_menu, self.provider_menu, self.refresh_button, self.virtual_check]
        self.camera_menu.bind("<<ComboboxSelected>>", lambda event: self.save_soon())
        self.provider_menu.bind("<<ComboboxSelected>>", lambda event: self.save_soon())
        viewer = ttk.Frame(root, padding=16)
        viewer.grid(row=1, column=1, sticky="nsew", padx=(0, 26))
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(1, weight=1)
        top = ttk.Frame(viewer)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(top, text="LIVE PREVIEW", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(top, textvariable=self.stats, style="Muted.TLabel").pack(side="right")
        self.preview = tk.Label(viewer, text="Your preview will appear here\n\nPress Start camera to begin", font=("Segoe UI", 14),
                                bg="#172633", fg="#d5e2ec", anchor="center")
        self.preview.grid(row=1, column=0, sticky="nsew")
        bottom = ttk.Frame(viewer)
        bottom.grid(row=2, column=0, sticky="ew", pady=(12, 4))
        ttk.Checkbutton(bottom, text="Mirror preview", variable=self.mirror, command=self.save_soon).pack(side="left")
        ttk.Checkbutton(bottom, text="Show original", variable=self.original).pack(side="right")
        ttk.Label(viewer, textvariable=self.output, style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=4)
        self.notice = tk.Label(root, textvariable=self.detail, font=("Segoe UI", 10), fg="#536575", bg="#f2f4f7",
                               anchor="w", justify="left", wraplength=1050, pady=16)
        self.notice.grid(row=2, column=0, columnspan=2, sticky="ew", padx=26)
        root.bind("<Configure>", lambda event: self.notice.configure(wraplength=max(300,event.width-52)) if event.widget==root else None)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda event: self.cancel_calibration())
        self.refresh_cameras()
        self.poll_job = root.after(30, self.poll)

    def slider(self, parent, row, title, variable, minimum, maximum, unit):
        line = ttk.Frame(parent)
        line.grid(row=row, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(line, text=title).pack(side="left")
        label = ttk.Label(line, text=f"{variable.get():.0f}{unit}", style="Muted.TLabel")
        label.pack(side="right")
        self.scale_labels.append((variable, label, unit))
        ttk.Scale(parent, from_=minimum, to=maximum, variable=variable, command=self.controls_changed).grid(row=row+1, column=0, sticky="ew", pady=(2, 6))

    def refresh_cameras(self):
        try:
            devices = list_cameras()
        except OSError:
            devices = []
        current = int(self.camera.get().split(":")[0])
        values = [f"{index}: {name}" for index, name in devices] or [f"{i}: Camera {i}" for i in range(4)]
        self.camera_menu.configure(values=values)
        self.camera.set(next((v for v in values if int(v.split(":")[0]) == current), values[0]))

    def settings(self):
        return DesktopSettings(int(self.camera.get().split(":")[0]), self.providers[self.provider.get()],
                               self.virtual.get(), self.mirror.get(),
                               Correction(self.amount.get()/100, self.horizontal.get(), self.vertical.get())).validate()

    def save_soon(self):
        if self.save_job: self.root.after_cancel(self.save_job)
        self.save_job = self.root.after(300, self.save)

    def save(self):
        self.save_job = None
        try: save_settings(self.settings_path, self.settings())
        except (OSError, ValueError, TypeError) as error:
            self.detail.set("Your controls could not be saved: " + str(error))

    def controls_changed(self, value=None):
        for variable, label, unit in self.scale_labels:
            label.configure(text=f"{variable.get():.0f}{unit}")
        if self.worker:
            self.worker.commands.put(("correction", self.settings().correction))
        self.save_soon()

    def toggle_correction(self):
        if self.worker: self.worker.commands.put(("enabled", self.enabled.get()))

    def reset(self):
        self.amount.set(100)
        self.horizontal.set(0)
        self.vertical.set(0)
        self.controls_changed()

    def toggle(self):
        if self.worker:
            self.worker.stop()
            self.status.set("Stopping…")
            self.start_button.configure(state="disabled")
            self.calibrate_button.configure(state="disabled")
            self.cancel_calibration()
            return
        settings = self.settings()
        self.save()
        self.failed = False
        self.status.set("Starting…")
        self.detail.set("Preparing the camera. The face model downloads on first use.")
        self.notice.configure(fg="#536575")
        self.worker = DesktopWorker(settings, self.config_path)
        self.worker.commands.put(("enabled", self.enabled.get()))
        for widget in self.device_controls: widget.configure(state="disabled")
        self.start_button.configure(text="Stop camera")
        self.worker.start()

    def calibrate(self):
        if self.worker:
            self.worker.commands.put(("calibrate", None))

    def cancel_calibration(self):
        if self.worker: self.worker.commands.put(("cancel", None))
        self.hide_calibration()

    def hide_calibration(self):
        if self.calibration_window:
            self.calibration_window.destroy()
            self.calibration_window = None

    def calibration(self, state, message):
        if state not in ("lens", "screen"):
            self.hide_calibration()
            self.detail.set(message)
            return
        self.calibrate_button.configure(state="disabled")
        if not self.calibration_window:
            bounds = (self.root.winfo_rootx(), self.root.winfo_rooty(), self.root.winfo_width(), self.root.winfo_height())
            monitor = choose_monitor(bounds, get_monitors())
            window = self.calibration_window = tk.Toplevel(self.root)
            window.overrideredirect(True)
            # A leading '+' makes even negative coordinates absolute in Tk.
            window.geometry(f"{monitor.width}x{monitor.height}+{monitor.x}+{monitor.y}")
            window.attributes("-topmost", True)
            self.calibration_canvas = tk.Canvas(window, bg="#101820", highlightthickness=0)
            self.calibration_canvas.pack(fill="both", expand=True)
            self.calibration_size = monitor.width, monitor.height
            window.bind("<Escape>", lambda event: self.cancel_calibration())
            ttk.Button(window, text="Cancel calibration", command=self.cancel_calibration).place(relx=.5, rely=.85, anchor="center")
            window.focus_force()
        canvas = self.calibration_canvas
        canvas.delete("all")
        width, height = self.calibration_size
        canvas.create_text(width/2, height/2+85, text=message+"\nKeep your head still.", fill="white", font=("Segoe UI", 23), justify="center")
        if state == "screen":
            canvas.create_line(width/2-20, height/2, width/2+20, height/2, fill="white", width=3)
            canvas.create_line(width/2, height/2-20, width/2, height/2+20, fill="white", width=3)
        self.calibration_progress = canvas.create_text(width/2, height/2+165, text="", fill="#a8c9d8", font=("Segoe UI", 13))

    def show_frame(self, packet):
        original, corrected, found, fps, lens, screen = packet
        frame = original if self.original.get() else corrected
        width, height = self.preview.winfo_width(), self.preview.winfo_height()
        scale = min(width/frame.shape[1], height/frame.shape[0], 1.)
        if scale <= 0: return
        frame = cv2.resize(frame, (max(1, int(frame.shape[1]*scale)), max(1, int(frame.shape[0]*scale))))
        if self.mirror.get(): frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        header = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode()
        self.photo = tk.PhotoImage(data=header+rgb.tobytes(), format="PPM", master=self.root)
        self.preview.configure(image=self.photo, text="")
        self.stats.set(f"{fps:.0f} fps · {'Face found' if found else 'Looking for a face'}")
        if self.calibration_window:
            self.calibration_canvas.itemconfigure(self.calibration_progress, text=f"Lens {lens}/15  ·  Screen {screen}/15")

    def poll(self):
        worker = self.worker
        if worker:
            while True:
                try: kind, value = worker.events.get_nowait()
                except queue.Empty: break
                if kind in ("error", "warning"):
                    self.failed = True
                    self.detail.set(value)
                    self.notice.configure(fg="#a33b25")
                elif kind == "started":
                    self.status.set("Camera running")
                    self.calibrate_button.configure(state="normal")
                    self.output.set("Call output on · choose OBS Virtual Camera" if value else "Call output is off · preview only")
                    if not self.failed: self.detail.set("Adjust strength while you watch. Use Calibrate for your camera position.")
                elif kind == "calibration":
                    self.calibration(*value)
                    if value[0] not in ("lens", "screen"): self.calibrate_button.configure(state="normal")
                elif kind == "stopped":
                    self.hide_calibration()
                    self.worker = None
                    self.status.set("Camera stopped")
                    self.stats.set("Camera off")
                    self.output.set("Call output is off")
                    self.preview.configure(image="", text="Camera is off\n\nPress Start camera to begin")
                    self.photo = None
                    self.start_button.configure(text="Start camera", state="normal")
                    self.calibrate_button.configure(state="disabled")
                    for widget in self.device_controls: widget.configure(state="normal")
                    self.camera_menu.configure(state="readonly")
                    self.provider_menu.configure(state="readonly")
                    if not self.failed: self.detail.set("Camera released. Your adjustments are saved.")
            if self.worker:
                try: self.show_frame(worker.frames.get_nowait())
                except queue.Empty: pass
        if self.closing and not self.worker:
            self.root.destroy()
            return
        self.poll_job = self.root.after(30, self.poll)

    def close(self):
        self.closing = True
        if self.save_job: self.root.after_cancel(self.save_job)
        self.save()
        self.cancel_calibration()
        if self.worker:
            self.worker.stop()
            self.status.set("Releasing the camera…")
            self.start_button.configure(state="disabled")
        else:
            self.root.after_cancel(self.poll_job)
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Gazeshift desktop")
    parser.add_argument("--settings", type=Path, default=Path.home()/".gaze_cam/desktop.json")
    parser.add_argument("--config", type=Path, default=Path.home()/".gaze_cam/config.json")
    args = parser.parse_args()
    enable_dpi_awareness()
    root = tk.Tk()
    Desktop(root, args.settings, args.config)
    root.mainloop()


if __name__ == "__main__":
    main()
