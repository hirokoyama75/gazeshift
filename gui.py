"""Gazeshift desktop: camera access starts only after the user presses Start."""
import argparse
from pathlib import Path
import queue
import tkinter as tk
from tkinter.font import nametofont
from tkinter import ttk

import cv2
import onnxruntime as ort
from engine.pipeline import Correction
from utils.cameras import list_cameras
from utils.desktop import DesktopSettings, DesktopWorker, load_settings, save_settings
from utils.monitors import enable_dpi_awareness, get_monitors, choose_monitor

THEMES = {
    "dark": dict(app="#111113", panel="#1b1b1f", field="#28282f", text="#f4f4f5",
                 muted="#aaaab5", accent="#b6b2ff", primary="#6258d5", hover="#746ae5",
                 border="#36363f", danger="#ffb69f", preview="#09090b"),
    "light": dict(app="#f5f5f7", panel="#ffffff", field="#eeeef3", text="#222229",
                  muted="#666674", accent="#564bc0", primary="#6258d5", hover="#5147bd",
                  border="#dddde5", danger="#9f351e", preview="#101013"),
}


class Desktop:
    def __init__(self, root, settings_path, config_path, provider=None):
        self.root, self.settings_path, self.config_path = root, settings_path, config_path
        self.font = nametofont("TkDefaultFont", root=root).actual("family")
        self.worker, self.calibration_window, self.save_job = None, None, None
        self.closing, self.failed, self.photo = False, False, None
        warning = ""
        try:
            settings = load_settings(settings_path)
        except (OSError, ValueError, TypeError) as error:
            settings = DesktopSettings()
            warning = "Saved controls could not be loaded; using defaults. " + str(error)
        self.camera = tk.StringVar(value=str(settings.camera))
        self.providers = {"Automatic": "auto", "CPU": "cpu"}
        for label, key, ep in (("AMD / Intel GPU", "dml", "DmlExecutionProvider"),
                                     ("NVIDIA GPU", "cuda", "CUDAExecutionProvider"),
                                     ("Apple Core ML", "coreml", "CoreMLExecutionProvider")):
            if ep in ort.get_available_providers(): self.providers[label] = key
        preferred = provider or (settings.provider if Path(settings_path).exists() else "auto")
        selected = next((label for label, key in self.providers.items() if key == preferred), "Automatic")
        self.provider = tk.StringVar(value=selected)
        self.active_provider = selected
        if selected == "Automatic" and preferred != "auto":
            warning = f"Saved processing option {preferred} is unavailable; Automatic will compare installed options."
        self.theme = tk.StringVar(value=settings.theme.title())
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
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)
        self.style = ttk.Style(root)
        self.style.theme_use("clam")
        header = ttk.Frame(root, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=26, pady=(22, 16))
        ttk.Label(header, text="gazeshift", style="Brand.TLabel").pack(side="left")
        ttk.Label(header, text="A more natural connection", style="Subtitle.TLabel").pack(side="left", padx=20)
        self.theme_menu = ttk.Combobox(header, textvariable=self.theme, values=("Dark", "Light"), state="readonly", width=7)
        self.theme_menu.pack(side="right")
        self.theme_menu.bind("<<ComboboxSelected>>", lambda event: self.change_theme())
        ttk.Label(header, textvariable=self.status, style="Status.TLabel").pack(side="right", padx=16)
        controls = ttk.Frame(root, padding=20)
        controls.grid(row=1, column=0, sticky="ns", padx=(26, 16))
        controls.columnconfigure(0, weight=1)
        ttk.Label(controls, text="CAMERA", font=(self.font, 10, "bold")).grid(row=0, column=0, sticky="w")
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
        tabs.add(setup, text="Setup")
        adjustments.columnconfigure(0, weight=1)
        setup.columnconfigure(0, weight=1)
        ttk.Checkbutton(adjustments, text="Eye correction", variable=self.enabled, command=self.toggle_correction).grid(row=0, column=0, sticky="w", pady=8)
        self.slider(adjustments, 1, "Strength", self.amount, 0, 150, "%")
        self.slider(adjustments, 3, "Left / right", self.horizontal, -10, 10, "°")
        self.slider(adjustments, 5, "Up / down", self.vertical, -10, 10, "°")
        ttk.Label(adjustments, text="100% is the original correction.\nAim fine-tunes both eyes; it doesn't\nlock your gaze. Angle limits still apply.", style="Muted.TLabel").grid(row=7, column=0, sticky="w", pady=(6, 12))
        ttk.Button(adjustments, text="Reset adjustments", command=self.reset).grid(row=8, column=0, sticky="ew", pady=(0, 8))
        self.calibrate_button = ttk.Button(adjustments, text="Calibrate · two quick looks", command=self.calibrate, state="disabled")
        self.calibrate_button.grid(row=9, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(setup, text="Processing").grid(row=0, column=0, sticky="w")
        self.provider_menu = ttk.Combobox(setup, textvariable=self.provider, values=list(self.providers), state="readonly")
        self.provider_menu.grid(row=1, column=0, sticky="ew", pady=(6, 16))
        self.virtual_check = ttk.Checkbutton(setup, text="Send video to call apps", variable=self.virtual, command=self.save_soon)
        self.virtual_check.grid(row=2, column=0, sticky="w")
        ttk.Label(setup, text="Choose OBS Virtual Camera in your\ncall app. Leave OBS's own virtual-\ncamera output stopped.", style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=(6, 18))
        ttk.Label(setup, text="Automatic compares available options\nand chooses the fastest.\n\nStop the camera before changing\nprocessing or call output.", style="Muted.TLabel").grid(row=4, column=0, sticky="w")
        self.device_controls = [self.camera_menu, self.provider_menu, self.refresh_button, self.virtual_check]
        self.camera_menu.bind("<<ComboboxSelected>>", lambda event: self.save_soon())
        self.provider_menu.bind("<<ComboboxSelected>>", lambda event: self.save_soon())
        viewer = ttk.Frame(root, padding=16)
        viewer.grid(row=1, column=1, sticky="nsew", padx=(0, 26))
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(1, weight=1)
        top = ttk.Frame(viewer)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(top, text="PREVIEW", font=(self.font, 10, "bold")).pack(side="left")
        ttk.Label(top, textvariable=self.stats, style="Muted.TLabel").pack(side="right")
        self.preview = tk.Label(viewer, text="Your camera is ready\n\nStart camera to see your preview", font=(self.font, 14),
                                bg="#172633", fg="#d5e2ec", anchor="center")
        self.preview.grid(row=1, column=0, sticky="nsew")
        bottom = ttk.Frame(viewer)
        bottom.grid(row=2, column=0, sticky="ew", pady=(12, 4))
        ttk.Checkbutton(bottom, text="Mirror preview", variable=self.mirror, command=self.save_soon).pack(side="left")
        ttk.Checkbutton(bottom, text="Show original", variable=self.original).pack(side="right")
        ttk.Label(viewer, textvariable=self.output, style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=4)
        self.notice = tk.Label(root, textvariable=self.detail, font=(self.font, 10), fg="#536575", bg="#f2f4f7",
                               anchor="w", justify="left", wraplength=1050, pady=16)
        self.notice.grid(row=2, column=0, columnspan=2, sticky="ew", padx=26)
        self.apply_theme()
        root.bind("<Configure>", lambda event: self.notice.configure(wraplength=max(300,event.width-52)) if event.widget==root else None)
        root.protocol("WM_DELETE_WINDOW", self.close)
        if root.tk.call("tk", "windowingsystem") == "aqua":
            root.createcommand("::tk::mac::Quit", self.close)
        root.bind("<Escape>", lambda event: self.cancel_calibration())
        self.refresh_cameras()
        self.poll_job = root.after(250, self.poll)

    def apply_theme(self):
        p = self.palette = THEMES[self.theme.get().lower()]
        style = self.style
        self.root.configure(bg=p["app"])
        style.configure("TFrame", background=p["panel"])
        style.configure("App.TFrame", background=p["app"])
        style.configure("TLabel", background=p["panel"], foreground=p["text"], font=(self.font, 11))
        style.configure("Muted.TLabel", foreground=p["muted"], font=(self.font, 9))
        style.configure("Brand.TLabel", background=p["app"], font=(self.font, 27, "bold"))
        style.configure("Subtitle.TLabel", background=p["app"], foreground=p["muted"], font=(self.font, 10))
        style.configure("Status.TLabel", background=p["field"], foreground=p["accent"], padding=(12,8), font=(self.font, 10, "bold"))
        style.configure("TButton", padding=(12,9), font=(self.font,10), background=p["field"],
                        foreground=p["text"], bordercolor=p["border"], lightcolor=p["field"], darkcolor=p["field"], relief="flat")
        style.map("TButton", background=[("active",p["border"])], foreground=[("disabled",p["muted"])])
        style.configure("Primary.TButton", background=p["primary"], foreground="#ffffff", bordercolor=p["primary"],
                        lightcolor=p["primary"], darkcolor=p["primary"], font=(self.font,11,"bold"))
        style.map("Primary.TButton", background=[("disabled",p["field"]),("active",p["hover"])])
        style.configure("TCheckbutton", background=p["panel"], foreground=p["text"], font=(self.font,10),
                        indicatorbackground=p["field"], indicatorforeground=p["accent"], bordercolor=p["border"])
        style.map("TCheckbutton", background=[("active",p["panel"])], foreground=[("disabled",p["muted"])])
        style.configure("TCombobox", padding=7, font=(self.font,10), background=p["field"],
                        fieldbackground=p["field"], foreground=p["text"], arrowcolor=p["muted"], bordercolor=p["border"],
                        lightcolor=p["field"], darkcolor=p["field"])
        style.map("TCombobox", fieldbackground=[("readonly",p["field"])],
                  foreground=[("disabled",p["muted"]),("readonly",p["text"])], selectbackground=[("readonly",p["field"])],
                  selectforeground=[("readonly",p["text"])])
        style.configure("TScale", background=p["accent"], troughcolor=p["field"], bordercolor=p["panel"])
        style.configure("TNotebook", background=p["panel"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(10,9), background=p["panel"], foreground=p["muted"],
                        bordercolor=p["panel"], lightcolor=p["panel"], font=(self.font,10))
        style.map("TNotebook.Tab", background=[("selected",p["field"])], foreground=[("selected",p["accent"])])
        self.root.option_add("*TCombobox*Listbox.background",p["field"])
        self.root.option_add("*TCombobox*Listbox.foreground",p["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground",p["primary"])
        self.root.option_add("*TCombobox*Listbox.selectForeground","#ffffff")
        for menu in (self.theme_menu,self.camera_menu,self.provider_menu):
            popdown = self.root.tk.call("ttk::combobox::PopdownWindow",menu)
            self.root.tk.call(f"{popdown}.f.l","configure","-background",p["field"],"-foreground",p["text"],
                              "-selectbackground",p["primary"],"-selectforeground","#ffffff")
        self.preview.configure(bg=p["preview"],fg=p["muted"],highlightthickness=1,highlightbackground=p["border"])
        self.notice.configure(bg=p["app"],fg=p["danger"] if self.failed else p["muted"])

    def change_theme(self):
        self.apply_theme()
        self.save_soon()

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
                               Correction(self.amount.get()/100, self.horizontal.get(), self.vertical.get()),
                               theme=self.theme.get().lower()).validate()

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
        self.detail.set("Preparing the camera and eye correction.")
        self.notice.configure(fg=self.palette["muted"])
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
        canvas.create_text(width/2, height/2+85, text=message+"\nKeep your head still.", fill="white", font=(self.font, 23), justify="center")
        if state == "screen":
            canvas.create_line(width/2-20, height/2, width/2+20, height/2, fill="white", width=3)
            canvas.create_line(width/2, height/2-20, width/2, height/2+20, fill="white", width=3)
        self.calibration_progress = canvas.create_text(width/2, height/2+165, text="", fill="#a8c9d8", font=(self.font, 13))

    def show_frame(self, packet):
        original, corrected, found, fps, lens, screen = packet
        frame = original if self.original.get() else corrected
        width, height = self.preview.winfo_width(), self.preview.winfo_height()
        scale = min(width/frame.shape[1], height/frame.shape[0], 1.)
        if scale <= 0: return
        if scale < 1:
            frame = cv2.resize(frame, (max(1, int(frame.shape[1]*scale)), max(1, int(frame.shape[0]*scale))),
                               interpolation=cv2.INTER_AREA)
        if self.mirror.get(): frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        header = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode()
        data = header + rgb.tobytes()
        if self.photo and (self.photo.width(), self.photo.height()) == (rgb.shape[1], rgb.shape[0]):
            self.photo.configure(data=data, format="PPM")
        else:
            self.photo = tk.PhotoImage(data=data, format="PPM", master=self.root)
        self.preview.configure(image=self.photo, text="")
        self.stats.set(f"{fps:.0f} fps · {self.active_provider} · {'Face found' if found else 'No face'}")
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
                    self.notice.configure(fg=self.palette["danger"])
                elif kind == "status":
                    self.detail.set(value)
                elif kind == "provider":
                    self.active_provider = next(label for label, key in self.providers.items() if key == value[0])
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
                try:
                    packet = worker.frames.get_nowait()
                    if not self.closing and self.root.state() != "iconic":
                        self.show_frame(packet)
                except queue.Empty: pass
        if self.closing and not self.worker:
            self.root.destroy()
            return
        self.poll_job = self.root.after(30 if self.worker else 250, self.poll)

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
    parser.add_argument("--provider", choices=("auto", "cpu", "dml", "cuda", "coreml"))
    args = parser.parse_args()
    enable_dpi_awareness()
    root = tk.Tk()
    Desktop(root, args.settings, args.config, provider=args.provider)
    root.mainloop()


if __name__ == "__main__":
    main()
