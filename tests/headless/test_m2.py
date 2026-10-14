import platform
import time
from types import SimpleNamespace
import cv2
import numpy as np
import onnxruntime as ort
import pytest
from engine.onnx_warper import OnnxWarper, fixed_inputs
from engine.geometry import CameraGeometry
from engine.pipeline import Pipeline, run_stream
from scripts.assets import ROOT, sha256
from scripts.replay import synthetic_sample
from utils.video_io import Capture, render_hud


@pytest.mark.timeout(300)
def test_real_cpu_benchmark(metrics):
    engine=OnnxWarper(provider='cpu',threads=6)
    assert engine.session.get_providers()==['CPUExecutionProvider']
    inputs=fixed_inputs()
    for _ in range(50):engine.session.run(None,inputs)
    durations=[]
    for _ in range(500):
        start=time.perf_counter_ns()
        outputs=engine.session.run(None,inputs)
        durations.append((time.perf_counter_ns()-start)/1e6)
    assert len(outputs)==4 and all(v.shape==(1,48,64,2) for v in outputs)
    options=engine.session.get_session_options()
    metrics.update(warmups=50,timed_runs=500,p95_limit_ms=9.0,median_ms=float(np.median(durations)),p95_ms=float(np.percentile(durations,95)),max_ms=float(max(durations)),
        providers=engine.session.get_providers(),intra_op_threads=options.intra_op_num_threads,inter_op_threads=options.inter_op_num_threads,
        execution_mode=str(options.execution_mode),python=platform.python_version(),numpy=np.__version__,onnxruntime=ort.__version__,
        host=platform.platform(),cpu=platform.processor(),model_sha256=sha256(ROOT/'weights/flx_12.onnx'))
    assert metrics['p95_ms']<metrics['p95_limit_ms'], metrics


def test_dshow_order_latest_frame_and_bounded_shutdown(metrics):
    import threading
    calls=[]
    class Device:
        def __init__(self):self.i=0;self.released=False
        def isOpened(self):return True
        def set(self,prop,value):calls.append((prop,value));return True
        def read(self):
            if self.i==20:return False,None
            self.i+=1
            return True,np.full((720,1280,3),self.i,np.uint8)
        def release(self):self.released=True
    device=Device()
    constructor=[]
    def factory(camera,backend):constructor.append((camera,backend));return device
    capture=Capture(factory=factory,system='Windows',clock=lambda:1.)
    capture.thread.join(1)
    assert not capture.thread.is_alive()
    sample=capture.read()
    assert sample[0].shape==(720,1280,3) and np.all(sample[0]==20)
    assert sample[1]==1019
    assert capture.read() is None
    start=time.monotonic();capture.close();elapsed=time.monotonic()-start
    assert elapsed<2 and device.released
    assert constructor==[(0,cv2.CAP_DSHOW)]
    assert [p for p,v in calls]==[cv2.CAP_PROP_FOURCC,cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS]
    assert calls[0][1]==cv2.VideoWriter_fourcc(*'MJPG')
    # A blocked read must be unblocked by release before join.
    event=threading.Event()
    class Blocking(Device):
        def read(self):event.wait(1);return False,None
        def release(self):event.set();self.released=True
    block=Blocking();capture=Capture(factory=lambda *a:block,system='Darwin')
    start=time.monotonic();capture.close()
    assert time.monotonic()-start<2 and not capture.thread.is_alive()
    class Broken(Device):
        def read(self):raise RuntimeError('injected worker read failure')
    broken=Broken();capture=Capture(factory=lambda *a:broken,system='Windows')
    capture.thread.join(1)
    with pytest.raises(RuntimeError,match='worker read failure'):capture.read()
    start=time.monotonic();capture.close()
    assert time.monotonic()-start<2 and not capture.thread.is_alive() and broken.released
    metrics.update(frames_produced=20,latest_frame=20,timestamps_strict=True,shutdown_seconds=elapsed,dshow_mjpg_first=True)


@pytest.mark.parametrize('failure',['none','capture','tracker','preview','virtual'])
def test_injected_720p_pipeline_and_cleanup(failure,metrics):
    frames=[synthetic_sample(i)[0] for i in range(5)]
    faces=[synthetic_sample(i)[1] for i in range(5)]
    faces[1]=None
    faces[2].blink=0
    closed=[];clean=[];hud=[];timestamps=[]
    class Input:
        i=0
        def read(self):
            if self.i==5:return None
            if failure=='capture' and self.i==3:raise RuntimeError('injected capture')
            i=self.i;self.i+=1
            return frames[i],i*34
        def close(self):closed.append('capture')
    class Tracker:
        i=0
        def track(self,frame,timestamp):
            timestamps.append(timestamp)
            if failure=='tracker' and self.i==3:raise RuntimeError('injected tracker')
            result=faces[self.i];self.i+=1;return result
        def close(self):closed.append('tracker')
    class Virtual:
        def send(self,frame):
            if failure=='virtual' and len(clean)==3:raise RuntimeError('injected virtual')
            clean.append(frame.copy())
        def close(self):closed.append('virtual')
    class Preview:
        def show(self,frame,text):
            if failure=='preview' and len(hud)==3:raise RuntimeError('injected preview')
            # Deliberately mutate the input copy to expose feed aliasing mistakes.
            hud.append(render_hud(frame,text));return -1
        def close(self):closed.append('preview')
    engine=OnnxWarper()
    pipeline=Pipeline(engine,CameraGeometry())
    if failure=='none':
        assert run_stream(Input(),Tracker(),pipeline,Virtual(),Preview())==5
        assert engine.calls==3
        assert np.array_equal(clean[1],frames[1]) and np.array_equal(clean[2],frames[2])
        assert all(not np.array_equal(a,b) for a,b in zip(clean,hud))
        assert not np.array_equal(clean[0],frames[0])
    else:
        with pytest.raises(RuntimeError,match='injected'):
            run_stream(Input(),Tracker(),pipeline,Virtual(),Preview())
    assert sorted(closed)==['capture','preview','tracker','virtual']
    assert all(a<b for a,b in zip(timestamps,timestamps[1:]))
    metrics.update(failure=failure,inference_calls=engine.calls,outputs=len(clean),cleanup=closed,timestamps=timestamps)


def test_tracker_rgb_raw_blink_and_calibration_only_iris(monkeypatch,metrics):
    import engine.landmark_tracker as tracking
    # Adapter injection still exercises color conversion, raw blink, anchor ordering,
    # 1-Euro, centers, and timestamp validation through the production adapter.
    points=[SimpleNamespace(x=.5,y=.5) for _ in range(478)]
    for indices,cx in ((tracking.LEFT,.56),(tracking.RIGHT,.44)):
        outline=np.array([[-.02,0],[-.01,-.006],[.01,-.006],[.02,0],[.01,.006],[-.01,.006]])
        for index,p in zip(indices,outline):points[index]=SimpleNamespace(x=cx+p[0],y=.45+p[1])
    points[473]=SimpleNamespace(x=.56,y=.45);points[468]=SimpleNamespace(x=.44,y=.45)
    result=SimpleNamespace(face_landmarks=[points],face_blendshapes=[[SimpleNamespace(category_name='eyeBlinkLeft',score=0.),SimpleNamespace(category_name='eyeBlinkRight',score=.9)]])
    seen=[]
    class Detector:
        def detect_for_video(self,image,timestamp):seen.append((image.copy(),timestamp));return result
        def close(self):pass
    monkeypatch.setattr(tracking,'head_pose',lambda *a:np.array([0.,0.,0.]))
    tracker=tracking.LandmarkTracker(CameraGeometry(),Detector())
    frame=np.full((720,1280,3),(10,20,30),np.uint8)
    face=tracker.track(frame,0)
    assert np.array_equal(seen[0][0][0,0],[30,20,10])
    assert face.blink==0 and face.observation is None
    np.testing.assert_allclose(face.centers,face.anchors[:,[0,3]].mean(axis=1))
    tracker.calibrating=True
    assert tracker.track(frame,34).observation is not None
    with pytest.raises(ValueError):tracker.track(frame,34)
    tracker.close()
    metrics.update(bgr_to_rgb=True,calibration_only_iris=True,same_frame_blink=True)


def test_projected_head_pose_matches_screen_facing_direction(metrics):
    from engine.landmark_tracker import head_pose, facing_from_pose
    from engine.geometry import pose_strength
    # Independent camera projection: nose is nearest; eyes/chin/mouth are deeper.
    model=np.array([[0,0,0],[0,330,65],[225,-170,135],[-225,-170,135],[-150,150,125],[150,150,125]],float)
    records=[]
    for pitch,yaw,roll in ((0.,0.,0.),(12.,-9.,17.5),(-14.,8.,-23.),(np.degrees(np.arctan2(21,60)),0.,0.)):
        p,y,r=np.radians([pitch,yaw,roll])
        rx=np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
        ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
        rz=np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])
        rotation=rz@ry@rx
        camera_points=model@rotation.T+[0,0,600]
        pixels=camera_points[:,:2]/camera_points[:,2:]*650+[640,360]
        pose=head_pose(pixels,(1280,720),650)
        error=np.abs((pose-[pitch,yaw,roll]+180)%360-180)
        assert np.all(error<1e-4),(pose,error)
        expected=np.diag([-1,1,-1])@(rotation@np.array([0.,0.,-1.]))
        actual=facing_from_pose(pose)
        np.testing.assert_allclose(actual,expected,atol=1e-6)
        records.append(dict(pitch=pitch,yaw=yaw,roll=roll,error_deg=error.tolist()))
    assert np.isclose(pose_strength(actual,np.array([0.,-21.,-60.])),1.)
    metrics.update(projected_cases=records,top_camera_screen_facing_strength=1.)


def test_cli_calibration_keys_persistence_and_clean_output(monkeypatch,tmp_path,metrics):
    import app
    from engine.geometry import save_config,load_config
    from utils.calibration import Observation
    config=tmp_path/'config.json'
    geometry=CameraGeometry()
    save_config(config,geometry)
    frame,face=synthetic_sample(0)
    centers=face.centers.copy()
    depth=650*6.3/np.linalg.norm(centers[0]-centers[1])
    pixel=centers.mean(axis=0)-[640,360]
    relative=np.array([-depth*pixel[0]/650,depth*pixel[1]/650])
    camera=np.array([0.,-21.])
    difference=np.arctan2(-relative-camera,depth)+np.arctan2(relative,depth-1)
    screen_iris=centers+difference*1.2*650/depth
    closed=[];output=[];texts=[];trackers=[];clock=[0.]
    class Input:
        i=0
        def __init__(self,*args):pass
        def read(self):
            if self.i==35:return None
            self.i+=1
            clock[0]+=1/30
            return frame.copy(),self.i*34
        def close(self):closed.append('capture')
    class Tracker:
        calibrating=False
        accepted=0
        frames=0
        def __init__(self,geometry):self.geometry=geometry;trackers.append(self)
        def track(self,frame,timestamp):
            self.frames+=1
            if self.frames==1:return None
            observation=None
            if self.calibrating:
                irises=centers if self.accepted<15 else screen_iris
                observation=Observation(centers.copy(),irises.copy(),np.zeros(3),(1280,720))
                self.accepted+=1
            return SimpleNamespace(anchors=face.anchors,centers=centers,pose=np.zeros(3),facing=face.facing,blink=1.,observation=observation)
        def close(self):closed.append('tracker')
    class Virtual:
        def __init__(self,*args):pass
        def send(self,image):output.append(image.copy())
        def close(self):closed.append('virtual')
    class Preview:
        i=0
        def __init__(self,**kwargs):pass
        def show(self,image,text,calibration):
            keys={0:ord(' '),31:ord('c'),32:2555904,33:ord('g'),34:ord('q')}
            texts.append(text)
            key=keys.get(self.i,-1);self.i+=1;return key
        def close(self):closed.append('preview')
    monkeypatch.setattr(app,'Capture',Input)
    monkeypatch.setattr(app,'LandmarkTracker',Tracker)
    monkeypatch.setattr(app,'VirtualCamera',Virtual)
    monkeypatch.setattr(app,'Preview',Preview)
    app.main(['--config',str(config),'--preview','--virtual-cam'],clock=lambda:clock[0])
    result=load_config(config)
    assert trackers[0].accepted==30
    assert result.provider=='cpu'
    np.testing.assert_allclose(result.camera_pos,[.5,-21.,-1.],atol=1e-3)
    assert result.focal_length==650 and result.ipd==6.3
    assert sorted(closed)==['capture','preview','tracker','virtual']
    assert len(output)==35
    assert np.array_equal(output[0],frame) and np.array_equal(output[-1],frame)
    assert any('Camera ' in text for text in texts)
    assert any('Calibration saved' in text for text in texts)
    metrics.update(frames=len(output),calibration_observations=trackers[0].accepted,provider=result.provider,camera_pos=list(result.camera_pos),cleanup=closed)
