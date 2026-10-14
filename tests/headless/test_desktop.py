from dataclasses import replace
import queue
from types import SimpleNamespace
import numpy as np
import pytest

from engine.geometry import CameraGeometry, load_config, save_config
from engine.onnx_warper import OnnxWarper
from engine.pipeline import Correction, Pipeline
from scripts.replay import synthetic_sample
from utils.desktop import DesktopSettings, DesktopWorker, load_settings, save_settings


def test_desktop_correction_controls(monkeypatch, metrics):
    import engine.pipeline as module
    frame, face = synthetic_sample(0)
    face.facing = np.array([0., 0., 1.])
    face.centers = np.array([[675., 360.], [605., 360.]])
    angles, strengths = [], []
    class Engine:
        def run(self, inputs):
            angles.append(inputs['ang_l'].copy())
            return {f'{kind}_{side}': np.zeros((1,48,64,2), np.float32)
                    for kind in ('flow','lcm') for side in ('l','r')}
    def compose(image, crop, flow, light, strength):
        strengths.append(strength)
        return image
    monkeypatch.setattr(module, 'compose', compose)
    pipeline = Pipeline(Engine(), CameraGeometry(camera_pos=(0.,0.,0.)), Correction(amount=0))
    assert np.array_equal(pipeline.process(frame,face),frame) and not angles
    pipeline.correction=Correction(.5,10.,-10.)
    pipeline.process(frame,face)
    np.testing.assert_allclose(angles[-1],[[-10.,10.]],atol=1e-6)
    assert strengths[-2:]==[.5,.5]
    pipeline.correction=Correction(1.5,10.,10.)
    pipeline.process(frame,face)
    np.testing.assert_allclose(angles[-1],[[15.,15.]],atol=1e-6)
    assert strengths[-2:]==[1.,1.]
    face.blink=0
    count=len(angles)
    assert np.array_equal(pipeline.process(frame,face),frame) and len(angles)==count
    for kwargs in ({'amount':-1},{'amount':2},{'amount':np.nan},{'horizontal':11},{'vertical':np.inf}):
        with pytest.raises(ValueError): Correction(**kwargs)
    metrics.update(zero_skips_inference=True,boost_angle_limit_deg=15,opacity_limit=1,blink_preserved=True)


def test_desktop_settings_roundtrip(tmp_path,monkeypatch,metrics):
    import utils.desktop as module
    path=tmp_path/'desktop.json'
    assert load_settings(path)==DesktopSettings()
    settings=DesktopSettings(camera=2,virtual_camera=True,mirror=False,correction=Correction(1.25,-3.,2.))
    save_settings(path,settings)
    assert load_settings(path)==settings
    before=path.read_bytes()
    path.write_text('[]')
    with pytest.raises(ValueError): load_settings(path)
    path.write_bytes(before)
    for settings in (replace(settings,camera=-1),replace(settings,provider='invalid'),replace(settings,mirror='false')):
        with pytest.raises(ValueError): save_settings(path,settings)
        assert path.read_bytes()==before
    def fail(*args): raise OSError('injected write failure')
    monkeypatch.setattr(module.os,'replace',fail)
    with pytest.raises(OSError): save_settings(path,DesktopSettings())
    assert path.read_bytes()==before and not list(tmp_path.glob('.desktop-*'))
    metrics.update(roundtrip=True,invalid_preserves_bytes=True,failed_write_preserves_bytes=True)


def test_desktop_worker_controls_and_cleanup(tmp_path,metrics):
    frame,face=synthetic_sample(0)
    closed,outputs,engines=[],[],[]
    class Input:
        i=0
        def __init__(self,*args): pass
        def read(self):
            if self.i==4: return None
            self.i+=1
            return frame,self.i*34
        def close(self):closed.append('capture')
    class Tracker:
        i=0
        def __init__(self,geometry):self.geometry=geometry
        def track(self,*args):
            actions=[('correction',Correction(1.5)),('enabled',False),('enabled',True),('stop',None)]
            action,value=actions[self.i];self.i+=1
            if action=='stop':worker.stop()
            else:worker.commands.put((action,value))
            return face
        def close(self):closed.append('tracker')
    class Virtual:
        def __init__(self,width,height):assert (width,height)==(1280,720)
        def send(self,image):outputs.append(image.copy())
        def close(self):closed.append('virtual')
    def engine(**kwargs):
        model=OnnxWarper(**kwargs);engines.append(model);return model
    worker=DesktopWorker(DesktopSettings(virtual_camera=True,correction=Correction(0)),tmp_path/'config.json',
                         capture_factory=Input,tracker_factory=Tracker,virtual_factory=Virtual,engine_factory=engine)
    worker.start();worker.join(5)
    assert not worker.is_alive()
    assert engines[0].calls==2 and len(outputs)==4
    assert np.array_equal(outputs[0],frame) and np.array_equal(outputs[2],frame)
    assert not np.array_equal(outputs[1],frame) and not np.array_equal(outputs[3],frame)
    assert sorted(closed)==['capture','tracker','virtual'] and worker.frames.qsize()==1
    events=list(worker.events.queue)
    assert not any(kind=='error' for kind,value in events) and events[-1][0]=='stopped'
    metrics.update(frames=4,real_inference_calls=2,latest_preview_only=True,cleanup=closed,stopped_within_seconds=5)


def test_desktop_worker_failure_paths(tmp_path,metrics):
    checked=[]
    for failure in ('camera','empty','output','processing','stopped'):
        closed=[]
        class Input:
            i=0
            def __init__(self,*args):
                if failure=='camera': raise RuntimeError('busy')
            def read(self):
                if failure=='empty' or self.i==1:return None
                self.i+=1
                return np.zeros((240,320,3),np.uint8),34
            def close(self):closed.append('capture')
        class Tracker:
            def __init__(self,geometry):self.geometry=geometry
            def track(self,*args):
                if failure=='processing':raise RuntimeError('injected processing failure')
                return None
            def close(self):closed.append('tracker')
        def virtual(width,height):
            assert (width,height)==(320,240)
            raise RuntimeError('virtual output busy')
        worker=DesktopWorker(DesktopSettings(virtual_camera=True),tmp_path/'config.json',capture_factory=Input,
                             tracker_factory=Tracker,engine_factory=lambda **kw:object(),virtual_factory=virtual)
        if failure=='stopped':worker.stop()
        worker.start();worker.join(2)
        assert not worker.is_alive()
        events=list(worker.events.queue)
        assert events[-1][0]=='stopped'
        if failure=='camera':
            assert closed==['tracker']
            assert any(kind=='error' and 'Close OBS' in value for kind,value in events)
        elif failure=='stopped':assert not closed
        else:assert sorted(closed)==['capture','tracker']
        if failure=='output':
            assert any(kind=='warning' and 'preview still works' in value for kind,value in events)
            assert worker.frames.qsize()==1
        checked.append(failure)
    metrics.update(failure_paths=checked,all_workers_stopped=True)


def test_desktop_worker_calibration(tmp_path,metrics):
    from test_m3 import forward_observations
    truth,lens,screen,_=forward_observations(camera=(-8.,14.,-1.),roll=17.5)
    path=tmp_path/'config.json'
    save_config(path,replace(truth,camera_pos=(1.,-2.,-1.)))
    now=[0.]
    class Input:
        i=0
        def __init__(self,*args):pass
        def read(self):
            if self.i==30:return None
            self.i+=1;now[0]=self.i/30
            return np.zeros((720,1280,3),np.uint8),self.i*34
        def close(self):pass
    class Tracker:
        i=0
        def __init__(self,geometry):self.geometry=geometry
        def track(self,*args):
            assert self.calibrating
            observation=lens if self.i<15 else screen
            self.i+=1
            return SimpleNamespace(observation=observation)
        def close(self):pass
    worker=DesktopWorker(DesktopSettings(),path,capture_factory=Input,tracker_factory=Tracker,
                         engine_factory=lambda **kw:object(),clock=lambda:now[0])
    worker.commands.put(('enabled',False));worker.commands.put(('calibrate',None))
    worker.start();worker.join(2)
    assert not worker.is_alive()
    result=load_config(path)
    np.testing.assert_allclose(result.camera_pos,truth.camera_pos,atol=1e-3)
    assert result.focal_length==truth.focal_length and result.ipd==truth.ipd and result.provider=='cpu'
    events=list(worker.events.queue)
    assert not any(kind=='error' for kind,value in events),events
    assert [value[0] for kind,value in events if kind=='calibration']==['lens','screen','success']
    metrics.update(raw_observations=30,shared_calibration=True,known_camera_recovered=True)
