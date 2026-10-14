"""Real Tk widget smoke test; explicit opt-in, no camera or virtual output."""
import queue
import tkinter as tk
import numpy as np
import gui
from utils.desktop import load_settings
from utils.monitors import Monitor


def test_desktop_window_controls_and_lifecycle(tmp_path,monkeypatch):
    workers=[]
    class Worker:
        def __init__(self,settings,path):
            self.settings=settings
            self.events,self.frames,self.commands=queue.Queue(),queue.Queue(),queue.Queue()
            workers.append(self)
        def start(self):
            self.events.put(('provider', ('dml', {'dml': 1000})))
            self.events.put(('started',False))
        def stop(self):self.events.put(('stopped',None))
    real_window=tk.Toplevel
    class HiddenWindow(real_window):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.attributes('-alpha',0)
        def focus_force(self):pass
    monkeypatch.setattr(gui,'DesktopWorker',Worker)
    monkeypatch.setattr(gui,'list_cameras',lambda:[(0,'Logi C270'),(2,'Other camera')])
    monkeypatch.setattr(gui.ort,'get_available_providers',lambda:['CPUExecutionProvider','DmlExecutionProvider'])
    monkeypatch.setattr(gui,'get_monitors',lambda:[Monitor(-1280,0,1280,720)])
    monkeypatch.setattr(gui.tk,'Toplevel',HiddenWindow)
    root=tk.Tk();root.withdraw();root.attributes('-alpha',0)
    app=None
    try:
        app=gui.Desktop(root,tmp_path/'desktop.json',tmp_path/'config.json',provider='dml')
        root.deiconify();root.update()
        assert app.worker is None and app.camera.get()=='0: Logi C270'
        assert app.provider.get()=='AMD / Intel GPU'
        assert app.theme.get()=='Dark' and root.cget('bg')==gui.THEMES['dark']['app']
        app.theme.set('Light');app.theme_menu.event_generate('<<ComboboxSelected>>');root.update()
        assert root.cget('bg')==gui.THEMES['light']['app']
        assert app.preview.cget('bg')==gui.THEMES['light']['preview']
        popdown=root.tk.call('ttk::combobox::PopdownWindow',app.theme_menu)
        assert root.tk.call(f'{popdown}.f.l','cget','-background')==gui.THEMES['light']['field']
        app.save()
        assert load_settings(tmp_path/'desktop.json').theme=='light'
        app.theme.set('Dark');app.change_theme()
        assert app.style.lookup('TLabel','foreground')==gui.THEMES['dark']['text']
        assert root.tk.call(f'{popdown}.f.l','cget','-background')==gui.THEMES['dark']['field']
        for size in ('1160x780','980x720'):
            root.geometry(size);root.update()
            assert app.calibrate_button.winfo_rooty()+app.calibrate_button.winfo_height()<root.winfo_rooty()+root.winfo_height()
        def poll():
            root.after_cancel(app.poll_job)
            app.poll()
            root.update_idletasks()
        app.provider.set('Automatic')
        app.start_button.invoke();poll()
        worker=workers[0]
        assert app.worker is worker and app.status.get()=='Camera running'
        assert worker.settings.provider == 'auto' and app.active_provider == 'AMD / Intel GPU'
        assert app.camera_menu.instate(['disabled'])
        for value in (200,250):
            app.amount.set(value);app.controls_changed()
            assert list(worker.commands.queue)[-1][1].amount==value/100
        app.save()
        assert load_settings(tmp_path/'desktop.json').correction.amount==2.5
        app.reset()
        assert app.amount.get()==100
        raw=np.full((2,4,3),10,np.uint8)
        corrected=np.full((2,4,3),200,np.uint8)
        app.original.set(True);app.show_frame((raw,corrected,True,30.,0,0))
        assert app.photo.get(0,0)==(10,10,10)
        previous_photo = app.photo
        app.original.set(False);app.show_frame((raw,corrected,True,30.,0,0))
        assert app.photo.get(0,0)==(200,200,200)
        assert app.photo is previous_photo
        assert 'Automatic' in app.providers
        assert 'AMD / Intel GPU' in app.stats.get() and 'Automatic' not in app.stats.get()
        with monkeypatch.context() as patch:
            patch.setattr(root, 'state', lambda: 'iconic')
            worker.frames.put((raw,raw,True,30.,0,0))
            poll()
            assert worker.frames.empty() and app.photo.get(0,0)==(200,200,200)
        worker.events.put(('calibration',('lens','Look at the lens')));poll()
        assert app.calibration_window.winfo_rootx()==-1280
        worker.events.put(('calibration',('screen','Look at the crosshair')));poll()
        canvas=app.calibration_canvas
        lines=[canvas.coords(item) for item in canvas.find_all() if canvas.type(item)=='line']
        assert lines==[[620.,360.,660.,360.],[640.,340.,640.,380.]]
        app.cancel_calibration()
        assert app.calibration_window is None and list(worker.commands.queue)[-1]==('cancel',None)
        app.start_button.invoke();poll()
        assert app.worker is None and app.preview.cget('image')==''
        assert app.camera_menu.instate(['readonly'])
        app.save()
        assert load_settings(tmp_path/'desktop.json').correction.amount==1.
        assert load_settings(tmp_path/'desktop.json').theme=='dark'
        assert load_settings(tmp_path/'desktop.json').provider == 'auto'
    finally:
        if app:
            app.hide_calibration()
            if app.save_job:root.after_cancel(app.save_job)
            root.after_cancel(app.poll_job)
        root.destroy()
