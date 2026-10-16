import numpy as np
import pytest

from sim.envs.robodojo.deformable_surface import (
    DeformableCoverageError, LocalPatchMonitor, MaterialSurface,
)


def sheet():
    p=np.array([[x*.01,y*.01,0.] for x in range(9) for y in range(9)])
    f=[]
    for x in range(8):
        for y in range(8):
            a=x*9+y; f.extend([[a,a+1,a+9],[a+1,a+10,a+9]])
    return p,np.array(f)


def surface():
    p,f=sheet()
    return MaterialSurface(p,f,contact_offset_m=.008)


def test_replacing_frame_does_not_rewrite_previous_collision_snapshot():
    s=surface(); before=s.frame; p=before.vertices.copy(); p[:,2]+=.1
    after=s.update(p,simulation_time_s=.004)
    assert after.revision==1 and np.all(before.vertices[:,2]==0)
    assert np.all(after.vertices[:,2]==.1)
    p[:,2]=99
    assert np.all(after.vertices[:,2]==.1)
    with pytest.raises(ValueError):after.vertices[0]=0
    with pytest.raises(ValueError):after.triangles[0]=0


@pytest.mark.parametrize('kind',['nan','missing','topology','clock','same_clock'])
def test_bad_live_read_is_unknown_without_replacing_last_snapshot(kind):
    s=surface(); original=s.frame; p=original.vertices.copy(); kw={'simulation_time_s':.004}
    if kind=='nan':p[1,1]=np.nan
    if kind=='missing':p=p[:-1]
    if kind=='topology':kw['triangles']=original.triangles[:,::-1]
    if kind=='clock':kw['simulation_time_s']=-.001
    if kind=='same_clock':kw['simulation_time_s']=0; p[0,2]+=.01
    with pytest.raises(DeformableCoverageError):s.update(p,**kw)
    assert s.frame is original


def test_identical_read_at_same_clock_does_not_create_false_physics_progress():
    s=surface(); frame=s.frame
    assert s.update(frame.vertices,simulation_time_s=0) is frame


def test_far_cloth_bending_is_not_whole_body_grasp_slip():
    s=surface(); patch=s.material_patch([0,0,0]); wrist=np.eye(4)
    monitor=LocalPatchMonitor(patch,s.frame,wrist)
    p=s.frame.vertices.copy(); far=np.linalg.norm(p[:,:2],axis=1)>.05
    p[far,2]=.2
    frame=s.update(p,simulation_time_s=.004)
    r=monitor.measure(frame,wrist,finger_clearances_m=[.001,.001])
    assert r['median_relative_motion_m']==0 and r['p95_relative_motion_m']==0
    assert r['proximity']=='both_fingers_near_patch'
    assert r['grasp_confirmed'] is False and r['whole_object_rigid_pose_checked'] is False
    assert 'vertex_ids' not in r and 'vertices' not in r


def test_following_patch_vs_sliding_patch_use_material_identities():
    s=surface(); patch=s.material_patch([0,0,0]); wrist=np.eye(4)
    monitor=LocalPatchMonitor(patch,s.frame,wrist)
    p=s.frame.vertices.copy()+[0,0,.02]; wrist[2,3]=.02
    r=monitor.measure(s.update(p,simulation_time_s=.004),wrist,finger_clearances_m=[.001,.001])
    assert r['p95_relative_motion_m']<1e-10
    p[patch.vertex_ids,0]+=.01
    r=monitor.measure(s.update(p,simulation_time_s=.008),wrist,finger_clearances_m=[.02,.001])
    assert r['median_relative_motion_m']==pytest.approx(.01)
    assert r['proximity']=='two_finger_proximity_not_observed'


@pytest.mark.parametrize('distances',[[None,.001],[float('nan'),0],[],[.001]])
def test_missing_contact_measurement_does_not_mean_confirmed_grasp(distances):
    s=surface(); patch=s.material_patch([0,0,0])
    m=LocalPatchMonitor(patch,s.frame,np.eye(4))
    r=m.measure(s.frame,np.eye(4),finger_clearances_m=distances)
    assert r['proximity']=='unknown' and not r['grasp_confirmed']


def test_folded_layers_near_same_pixel_are_not_implicitly_authorized():
    s=surface(); p=s.frame.vertices.copy()
    p[8*9:]=p[:9]+[0,0,.001]
    s.update(p,simulation_time_s=.004)
    with pytest.raises(DeformableCoverageError,match='ambiguous'):
        s.material_patch([0,0,0])


def test_patch_does_not_follow_new_material_after_reset():
    s=surface(); patch=s.material_patch([0,0,0]); other=surface()
    with pytest.raises(DeformableCoverageError):patch.relative_points(other.frame,np.eye(4))


def test_stale_read_after_observed_motion_is_rejected():
    s=surface(); patch=s.material_patch([0,0,0]); old=s.frame
    m=LocalPatchMonitor(patch,old,np.eye(4))
    m.measure(s.update(old.vertices,simulation_time_s=.004),np.eye(4),finger_clearances_m=[0,0])
    with pytest.raises(DeformableCoverageError,match='Stale'):
        m.measure(old,np.eye(4),finger_clearances_m=[0,0])


def test_invalid_triangle_indices_and_degenerate_material_rejected():
    p,f=sheet()
    with pytest.raises(DeformableCoverageError):MaterialSurface(p,f+.5,contact_offset_m=.008)
    with pytest.raises(DeformableCoverageError):MaterialSurface(p,f+999,contact_offset_m=.008)
    p[0]=p[1]
    with pytest.raises(DeformableCoverageError):MaterialSurface(p,f,contact_offset_m=.008)


def test_explicit_local_layers_selects_nearby_layers_but_not_far_depth():
    p,f=sheet();points=np.concatenate([p,p+[0,0,-.005],p+[0,0,-.03]])
    faces=np.concatenate([f,f+len(p),f+2*len(p)])
    s=MaterialSurface(points,faces,contact_offset_m=.008)
    patch=s.local_layers([.04,.04,0],[0,0,-1])
    assert any(i<len(p) for i in patch.vertex_ids)
    assert any(len(p)<=i<2*len(p) for i in patch.vertex_ids)
    assert not any(i>=2*len(p) for i in patch.vertex_ids)
    before=patch.vertex_ids.copy();moved=points.copy();moved[2*len(p):,2]=-.001
    s.update(moved,simulation_time_s=.004)
    assert np.array_equal(before,patch.vertex_ids)


def test_local_layers_bounded_by_axis_radius_and_total_area():
    p,f=sheet();s=MaterialSurface(p,f,contact_offset_m=.008)
    with pytest.raises(ValueError):s.local_layers([0,0,0],[0,0,2])
    with pytest.raises(ValueError):s.local_layers([0,0,0],[0,0,1],half_depth_m=.1)
    many=np.concatenate([p+[0,0,-i*.001] for i in range(10)])
    faces=np.concatenate([f+i*len(p) for i in range(10)])
    s=MaterialSurface(many,faces,contact_offset_m=.008)
    with pytest.raises(DeformableCoverageError,match='Too much'):
        s.local_layers([.04,.04,0],[0,0,1])
