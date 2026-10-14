from dataclasses import replace
import numpy as np
import pytest
from engine.geometry import CameraGeometry,load_config,save_config,gaze_offset,pose_strength
from utils.calibration import Calibration,Observation,monitor_center


def forward_observations(camera=(0.,-21.,-1.),depth=60.,roll=0.):
    """Independent analytic forward projection, never calls production inverse/offset.

    Choose camera-relative eye position (a,b,-D), project eyelid corners and
    iris displacements with the declared fixed 1.2 cm eye approximation.
    """
    focal,ipd=712.,6.7
    a,b=3.4,8.2
    pixel=np.array([640-focal*a/depth,360+focal*b/depth])
    r=np.radians(roll)
    rotation=np.array([[np.cos(r),-np.sin(r)],[np.sin(r),np.cos(r)]])
    half=np.array([focal*ipd/(2*depth),0.])@rotation.T
    centers=np.array([pixel+half,pixel-half])
    # Angular difference from the two distinct rays in WangWilly coordinates.
    screen_xy=np.degrees(np.arctan2(-np.array([a+camera[0],b+camera[1]]),depth))
    lens_term_xy=np.degrees(np.arctan2(np.array([a,b]),depth+camera[2]))
    difference_xy=screen_xy+lens_term_xy
    aligned_xy=rotation.T@difference_xy
    # Arbitrary lens fixation plus the independently predicted screen difference.
    lens_direction=np.array([1.2,-2.4])
    screen_direction=lens_direction+aligned_xy[::-1]
    def observation(direction):
        qxy=np.radians(direction[::-1])*1.2
        displacement=rotation@qxy*focal/depth
        return Observation(centers.copy(),centers+displacement,np.array([2.,-1.,roll]),(1280,720))
    truth=CameraGeometry(focal,ipd,camera)
    return truth,observation(lens_direction),observation(screen_direction),difference_xy[::-1]


@pytest.mark.parametrize('camera,depth,roll',[
    ((0.,-21.,-1.),60.,0.),((18.,0.,-3.),45.,0.),((-16.,9.,2.),85.,0.),
    ((5.,-18.,-1.),55.,17.5),((-8.,14.,-2.),100.,-23.),((0.,0.,-1.),70.,8.)])
def test_independent_two_look_recovery(camera,depth,roll,tmp_path,metrics):
    truth,lens,screen,difference=forward_observations(camera,depth,roll)
    initial=replace(truth,camera_pos=(1.,-2.,camera[2]))
    path=tmp_path/'config.json';save_config(path,initial)
    clock=[0.]
    calibration=Calibration(initial,path,clock=lambda:clock[0]);calibration.start()
    for observation in [lens]*15+[screen]*15:
        clock[0]+=.033
        calibration.update(observation)
    assert calibration.state=='success',calibration.message
    fitted=load_config(path)
    error=np.abs(np.asarray(fitted.camera_pos[:2])-camera[:2])
    assert np.all(error<1e-3)
    assert fitted.focal_length==truth.focal_length and fitted.ipd==truth.ipd and fitted.camera_pos[2]==camera[2]
    actual,_=gaze_offset(lens.centers,lens.frame_size,fitted)
    residual=np.abs(actual-difference)
    assert np.all(residual<1e-4)
    metrics.update(camera=list(camera),depth=depth,roll=roll,xy_error_cm=error.tolist(),angular_residual_deg=residual.tolist(),reload=True)


@pytest.mark.parametrize('failure',['cancel','timeout','pose','nan','coincident','iris_domain','negative_focal','negative_ipd','depth','missing_face','closed','frame_size'])
def test_failures_preserve_config_bytes(failure,tmp_path,metrics):
    truth,lens,screen,_=forward_observations()
    path=tmp_path/'config.json';save_config(path,truth)
    before=path.read_bytes();clock=[0.]
    geometry=truth
    if failure=='negative_focal':geometry=replace(truth,focal_length=-1)
    if failure=='negative_ipd':geometry=replace(truth,ipd=-1)
    if failure=='depth':geometry=replace(truth,camera_pos=(0.,-21.,-1000.))
    cal=Calibration(geometry,path,clock=lambda:clock[0]);cal.start()
    if failure=='cancel':cal.cancel()
    elif failure=='timeout':clock[0]=8.;cal.update(lens)
    elif failure=='pose':
        cal.update(lens)
        cal.update(replace(lens,pose=lens.pose+np.array([3.0001,0,0])))
    else:
        observation=lens
        if failure=='nan':observation=replace(lens,irises=np.full((2,2),np.nan))
        if failure=='coincident':observation=replace(lens,centers=np.ones((2,2)))
        if failure=='iris_domain':observation=replace(lens,irises=lens.centers+1000)
        if failure=='missing_face':observation=None
        if failure=='closed':observation=replace(lens,blink=0.)
        if failure=='frame_size':observation=replace(lens,frame_size=(0,720))
        cal.update(observation)
        if failure in ('missing_face','closed'):
            assert len(cal.lens)==0
            clock[0]=8.;cal.update(observation)
    assert cal.state in ('failed','cancelled','timeout')
    assert path.read_bytes()==before
    assert cal.result is None
    metrics.update(failure=failure,config_byte_identical=True,state=cal.state)


def test_three_degree_wrapped_boundary_and_secondary_monitor(tmp_path,metrics):
    truth,lens,screen,_=forward_observations(roll=179.)
    path=tmp_path/'config.json';save_config(path,truth)
    cal=Calibration(truth,path,clock=lambda:0.);cal.start();cal.update(lens)
    cal.update(replace(lens,pose=lens.pose+np.array([3.,-3.,-358.])))
    assert cal.state=='lens' and len(cal.lens)==2
    assert monitor_center((0,0,1920,1080))==(960,540)
    assert monitor_center((-2560,-200,2560,1440))==(-1280,520)
    assert monitor_center((1920,100,3840,2160))==(3840,1180)
    from utils.monitors import choose_monitor, Monitor
    from utils.video_io import calibration_canvas
    monitors = [Monitor(0,0,1920,1080), Monitor(-2560,-200,2560,1440)]
    chosen = choose_monitor((-2300,100,800,600), monitors)
    assert chosen == monitors[1]
    image = calibration_canvas((chosen.x,chosen.y,chosen.width,chosen.height),'screen')
    assert image.shape==(1440,2560,3)
    assert np.all(image[720,1280]==255)
    assert np.count_nonzero(calibration_canvas((chosen.x,chosen.y,chosen.width,chosen.height),'lens'))==0
    metrics.update(pose_boundary_deg=3,secondary_crosshair=[-1280,520])


@pytest.mark.parametrize('degrees,expected',[(0,1),(12,1),(15,.5),(18,0),(30,0)])
def test_screen_relative_pose_cone(degrees,expected,metrics):
    eye=np.array([0.,21.,-60.])
    screen=-eye/np.linalg.norm(eye)
    r=np.radians(degrees)
    rotation=np.array([[np.cos(r),0,np.sin(r)],[0,1,0],[-np.sin(r),0,np.cos(r)]])
    # Rotation about a perpendicular axis gives the exact declared departure.
    perpendicular=np.cross(screen,[1.,0.,0.]);perpendicular/=np.linalg.norm(perpendicular)
    facing=screen*np.cos(r)+perpendicular*np.sin(r)
    value=pose_strength(facing,eye)
    assert np.isclose(value,expected,atol=1e-12)
    assert pose_strength(screen,eye)==1
    metrics.update(angle_deg=degrees,strength=value,top_camera_axis_separation_deg=float(np.degrees(np.arctan2(21,60))))
