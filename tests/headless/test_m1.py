import pickle
import numpy as np
from engine.compositor import EyeCrop, compose, full_coordinates, remap_patch
from engine.geometry import CameraGeometry
from engine.pipeline import Pipeline
from scripts.replay import synthetic_sample
from scripts.assets import ROOT
from scripts.sampler_reference import coordinates
from utils.filters import OneEuro, Blink


def test_zero_strength_exact_and_inference_bypass(metrics):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (720, 1280, 3), np.uint8)
    _, tracked = synthetic_sample(0)
    tracked.blink = 0
    class ForbiddenEngine:
        def run(self, inputs): raise AssertionError("Inference called at zero strength")
    pipeline = Pipeline(ForbiddenEngine(), CameraGeometry())
    assert np.array_equal(pipeline.process(frame, tracked), frame)
    assert np.array_equal(pipeline.process(frame, None), frame)
    crop = EyeCrop.from_points(tracked.anchors[0], 17.3)
    flow = rng.uniform(-.2, .2, (48, 64, 2)).astype(np.float32)
    light = np.full((48, 64, 2), .5, np.float32)
    assert np.array_equal(compose(frame, crop, flow, light, 0), frame)
    metrics.update(bit_exact=True, inference_calls=0, nonzero_flow_lighting=True)


def test_complete_coordinate_fade_and_rolled_fractional_crops(metrics):
    rng = np.random.default_rng(0)
    # Smooth ramp permits a meaningful continuity bound after uint8 quantization.
    yy, xx = np.indices((180, 240))
    frame = np.stack((xx % 256, yy % 256, (xx+yy)//2), axis=-1).astype(np.uint8)
    points = np.array([[90.25, 85.4], [100.3, 81.7], [116.7, 81.7], [126.25, 85.4], [116.7, 89.7], [100.3, 89.7]])
    flow = np.full((48, 64, 2), (.08, -.04), np.float32)
    light = np.full((48, 64, 2), (.9, .1), np.float32)
    errors = []
    for roll in (0., 13.7, -21.2):
        crop = EyeCrop.from_points(points, roll)
        zero = compose(frame, crop, flow, light, 0)
        tiny = compose(frame, crop, flow, light, 1e-7)
        half = compose(frame, crop, flow, light, .5)
        full = compose(frame, crop, flow, light, 1.)
        assert np.array_equal(zero, frame)
        assert np.max(np.abs(tiny.astype(int)-frame)) <= 1
        assert not np.array_equal(full, frame)
        assert not np.array_equal(half, full)
        steps = [compose(frame, crop, flow, light, float(s)) for s in np.linspace(0, 1, 21)]
        jump = max(np.abs(b.astype(int)-a.astype(int)).max() for a,b in zip(steps, steps[1:]))
        assert jump <= 3
        errors.append(int(jump))
    # Zero learned flow still carries the inherited TF grid displacement.
    full = full_coordinates(np.zeros((48, 64, 2), np.float32))
    assert abs(full[20,30,0]-30*64/63) < 1e-5
    assert abs(full[20,30,1]-20*48/47) < 1e-5
    # At full strength and mask 1 an axis-aligned crop equals the production sampler.
    image = rng.random((48,64,3), dtype=np.float32)
    uint = np.rint(image*255).astype(np.uint8)
    crop = EyeCrop(np.array([[1.,0.,0.],[0.,1.,0.]]), (64,48), points)
    neutral = np.zeros((48,64,2),np.float32); neutral[...,0]=1
    actual = compose(uint,crop,flow,neutral,1,np.ones((48,64),np.float32))
    expected = np.rint(remap_patch(uint,flow)).astype(np.uint8)
    assert np.array_equal(actual,expected)
    metrics.update(max_fade_step_uint8=errors, rolled_fractional_identity=True)


def test_production_against_real_tf_interior(metrics):
    fixture=pickle.loads((ROOT/'.cache/parity/dirl_flx12.pkl').read_bytes())
    reference=pickle.loads((ROOT/'.cache/parity/restored_tf.pkl').read_bytes())
    records=[]
    for case, ref in zip(fixture['cases'],reference['cases']):
        for side in ('l','r'):
            image=case['inputs'][f'img_{side}'][0]
            flow=ref['outputs'][f'flow_{side}'][0]
            expected=ref['outputs'][f'sampler_{side}'][0]
            x,y=coordinates(flow)
            valid=(x>=0)&(x<63)&(y>=0)&(y<47)
            assert valid.sum()>0
            delta=np.abs(remap_patch(image,flow)-expected)
            error=float(delta[valid].max())
            assert error<2/255
            records.append(dict(case=case['label'],eye=side,coverage=float(valid.mean()),interior_max=error,
                                boundary_max=float(delta[~valid].max())))
    metrics['interior_parity']=records


def test_rolled_sampling_map_against_independent_frame_bilinear(metrics):
    rng=np.random.default_rng(0)
    frame=rng.integers(0,256,(160,240,3),np.uint8)
    flow=np.full((48,64,2),(.08,-.04),np.float32)
    light=np.full((48,64,2),(.9,.1),np.float32)
    records=[]
    for roll in (13.7,-21.2):
        angle=np.radians(roll)
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        origin=np.array([73.25,51.4])
        crop=EyeCrop(np.column_stack([rotation,origin]),(64,48),np.zeros((6,2)))
        # Destination pixels are chosen from geometry, independently of errors.
        pixels=[]
        for y in range(50,110,7):
            for x in range(65,150,7):
                local=np.linalg.solve(rotation,np.array([x,y])-origin)
                if 4<local[0]<59 and 4<local[1]<43:pixels.append((x,y,local))
        assert pixels
        for strength in (0.,.25,.5,1.):
            actual=compose(frame,crop,flow,light,strength,np.ones((48,64),np.float32))
            errors=[]
            for x,y,local in pixels:
                full=local*np.array([64/63,48/47])+np.array([.08*32,-.04*24])
                sampling=np.clip(local+strength*(full-local),[0,0],[63,47])
                fx,fy=rotation@sampling+origin
                x0,y0=int(np.floor(fx)),int(np.floor(fy));dx,dy=fx-x0,fy-y0
                expected=sum(frame[yy,xx].astype(float)*weight for xx,yy,weight in (
                    (x0,y0,(1-dx)*(1-dy)),(x0+1,y0,dx*(1-dy)),(x0,y0+1,(1-dx)*dy),(x0+1,y0+1,dx*dy)))
                expected=np.clip(np.rint(expected*(1-.1*strength)+255*.1*strength),0,255).astype(np.uint8)
                errors.append(int(np.abs(actual[y,x].astype(int)-expected).max()))
            assert max(errors)<=1,(roll,strength,max(errors))
            records.append(dict(roll=roll,strength=strength,pixels=len(pixels),max_error_uint8=max(errors)))
    metrics['independent_sampling']=records


def test_one_euro_stationary_variance(metrics):
    raw=np.random.default_rng(0).normal(0,1,(600,2))
    filt=OneEuro()
    smooth=np.array([filt(p,i/30) for i,p in enumerate(raw)])
    raw_variance=raw[60:].var(axis=0)
    filtered_variance=smooth[60:].var(axis=0)
    assert np.all(filtered_variance < .5*raw_variance)
    metrics.update(samples=600,warmup=60,hz=30,raw_variance=raw_variance.tolist(),filtered_variance=filtered_variance.tolist())


def test_blink_shared_min_immediate_close_linear_reopen(metrics):
    blink=Blink()
    def update(t,left,right):return blink.update(t,dict(eyeBlinkLeft=left,eyeBlinkRight=right))
    assert update(0,0,0)==1
    assert update(.01,0,.8)==0
    assert np.isclose(update(.06,0,0),.5)
    assert np.isclose(update(.11,0,0),1)
    assert np.isclose(update(.12,.375,0),.5)
    assert update(.13,.9,0)==0
    assert np.isclose(update(.18,0,0),.5)
    # EAR equivalent for both asymmetric closures and partial strength.
    def eye(ear):return np.array([[0,0],[2,-ear*5],[8,-ear*5],[10,0],[8,ear*5],[2,ear*5]],float)
    fallback=Blink()
    assert fallback.update(0,anchors=[eye(.25),eye(.25)])==1
    assert fallback.update(.01,anchors=[eye(.25),eye(.17)])==0
    assert np.isclose(fallback.update(.06,anchors=[eye(.25),eye(.25)]),.5)
    assert np.isclose(fallback.update(.11,anchors=[eye(.25),eye(.25)]),1)
    assert np.isclose(fallback.update(.12,anchors=[eye(.215),eye(.25)]),.5)
    metrics.update(asymmetric_close='same_frame',reopen_50ms=.5,reopen_100ms=1,ear_fallback=True)
