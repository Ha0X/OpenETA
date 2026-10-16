from copy import deepcopy
import json

import numpy as np
import pytest

from sim.envs.robodojo.coordination import (JointTrack, combined_state, coordinate,
                                          execution_samples, validate_schedule)
from sim.envs.robodojo.collision_geometry import CollisionScene
from test_robodojo_joint_planning import dual_crossing


def tracks(end=.3):
    return [JointTrack(role, index, [0.,2.], [[-.3],[end]])
            for index, role in enumerate(('left','right'))]


def certify_result(scene, paths, result):
    import time
    assert result['status'] in ('safe','repaired')
    assert validate_schedule(scene, [[-.3],[-.3]], paths, result['delays_s'], [],
                             time.monotonic()+5)['status'] == 'clear'


def test_disjoint_paths_execute_concurrently_without_adjustment():
    scene = dual_crossing(); paths = tracks(end=-.6)
    result = coordinate(scene, [[-.3],[-.3]], paths)
    assert result['status'] == 'safe' and result['mode'] == 'concurrent'
    assert result['adjustments'] == [] and result['attempts']['temporal_shift'] == 0
    certify_result(scene, paths, result)


def test_crossing_paths_are_retimed_before_serialization_and_hold_endpoints():
    scene = dual_crossing(); paths = tracks()
    result = coordinate(scene, [[-.3],[-.3]], paths)
    assert result['mode'] == 'temporal_shift'
    assert result['attempts']['serialize'] == 0
    assert result['delays_s'][1] > 0
    assert result['adjustments'][0]['optimality'] == 'first_verified_grid_delay'
    assert combined_state([[-.3],[-.3]], paths, result['delays_s'], 0.)[1] == [-.3]
    assert combined_state([[-.3],[-.3]], paths, result['delays_s'], 10.)[1] == [.3]
    certify_result(scene, paths, result)


def test_serialization_is_fallback_when_offset_budget_is_insufficient():
    scene = dual_crossing(); paths = tracks()
    result = coordinate(scene, [[-.3],[-.3]], paths, max_delay_s=.1)
    assert result['mode'] == 'serialize' and result['delays_s'] == [0.,2.]
    assert result['attempts']['temporal_shift'] == 2
    certify_result(scene, paths, result)


def test_spatial_repair_precedes_serialization_and_is_rechecked():
    from sim.envs.robodojo.joint_planning import JointPlanner
    scene=dual_crossing();paths=tracks();called=[]
    def replan(original,remaining):
        called.append(True)
        q,detail=JointPlanner(scene,[0,1]).plan([[-.3],[-.3]],[[[.3],[.3]]],timeout_s=min(5,remaining))
        assert q is not None
        return [JointTrack(t.role,t.index,np.arange(len(q))*.5,[p[t.index] for p in q]) for t in original],detail
    result=coordinate(scene,[[-.3],[-.3]],paths,max_delay_s=0,spatial_replanner=replan)
    assert called and result['mode']=='spatial_replanning'
    assert result['attempts']['serialize']==0
    certify_result(scene,paths,result)


def test_unsafe_spatial_candidate_never_bypasses_full_scene_check():
    scene=dual_crossing();paths=tracks()
    result=coordinate(scene,[[-.3],[-.3]],paths,max_delay_s=0,
                      spatial_replanner=lambda original,remaining:(original,{}))
    assert result['mode']=='serialize'
    certify_result(scene,paths,result)


def test_second_serial_order_is_checked_when_first_parks_in_the_other_path():
    scene = dual_crossing()
    paths = tracks()
    paths[0] = JointTrack('left', 0, [0.,2.], [[-.3],[0.]])
    result = coordinate(scene, [[-.3],[-.3]], paths, max_delay_s=0.)
    assert result['mode'] == 'serialize' and result['delays_s'] == [2.,0.]
    assert result['attempts']['serialize'] == 2
    certify_result(scene, paths, result)


def test_colliding_final_configuration_is_rejected_without_time_search():
    result = coordinate(dual_crossing(), [[-.3],[-.3]], tracks(end=0.))
    assert result['status'] == 'rejected'
    assert result['reason'] == 'candidate_final_configuration_not_certified'
    assert not any(result['attempts'].values())
    assert result['search_exhaustion_is_infeasibility_proof'] is False


def test_shared_object_cannot_be_independently_retimed():
    scene = dual_crossing()
    scene.snapshot['closed_chain_constraints'] = [{'owner':0,'follower':1}]
    result = coordinate(scene, [[-.3],[-.3]], tracks())
    assert result['reason'] == 'shared_body_requires_coupled_planning'
    assert not any(result['attempts'].values())


def test_attached_object_pair_is_checked_and_identified_without_asset_paths():
    source = deepcopy(dual_crossing().snapshot)
    cube = np.array(source['colliders'][0]['vertices'])
    source['colliders'][0]['vertices'] = (cube+[-.8,0,0]).tolist()
    source['colliders'][2]['vertices'] = (cube+[0,-.8,0]).tolist()
    # Bare robots clear the crossing; held objects extend into the intersection.
    assert CollisionScene(source).check_path([[-.3],[-.3]], [[.3],[.3]])['status'] == 'clear'
    for i,robot in enumerate(source['robots']):
        body = f'held_geometry_{i}'
        robot['body_frames'][body] = deepcopy(robot['body_frames']['tool'])
        robot['joints'].append({'name':body+'_fixed','parent':'tool','child':body,'kind':'fixed',
            'axis':'X','parent_frame':np.eye(4).tolist(),'child_frame':np.eye(4).tolist(),
            'collision_enabled':True})
        source['colliders'].append({'id':f'/World/private/held_{i}', 'robot_index':i, 'body':body,
                                    'approximation':'convexHull','vertices':cube.tolist(),'triangles':[]})
    scene = CollisionScene(source)
    result = coordinate(scene, [[-.3],[-.3]], tracks())
    assert result['mode'] == 'temporal_shift'
    assert [p['kind'] for p in result['conflict']['participants']] == ['attached_object']*2
    assert '/World' not in json.dumps(result)
    certify_result(scene, tracks(), result)


def test_execution_sampling_preserves_every_corner_and_the_certified_curve():
    scene = dual_crossing()
    paths = [JointTrack('left',0,[0.,.11,.2],[[-.3],[-.5],[-.6]]),
             JointTrack('right',1,[0.,.15],[[-.3],[-.6]])]
    result = coordinate(scene, [[-.3],[-.3]], paths)
    samples = list(execution_samples([[-.3],[-.3]], paths, result['delays_s'], .04))
    assert {0.11,0.15,0.2} <= {t for t,_ in samples}
    previous = [[-.3],[-.3]]
    for _,q in samples:
        assert scene.check_path(previous,q)['status'] == 'clear'
        previous = q


def test_zero_budget_cannot_claim_a_safe_schedule():
    result = coordinate(dual_crossing(), [[-.3],[-.3]], tracks(), timeout_s=0.)
    assert result['status'] == 'rejected' and result['reason'] == 'coordination_search_budget'


@pytest.mark.parametrize('times', [[1.,2.], [0.,0.], [0.,float('nan')]])
def test_invalid_phase_clock_is_rejected(times):
    with pytest.raises(ValueError):
        JointTrack('left',0,times,[[-.3],[.3]])
