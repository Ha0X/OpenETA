import pytest
from sim.envs.robodojo.native_surface import candidate_bound_ancestor
from sim.envs.robodojo.voxel_geometry import CoverUnavailable


def test_nested_usd_body_can_only_propose_a_bound_parent_for_native_verification():
    assert candidate_bound_ancestor(['/object/nested','/object'],{'/object':object()})=='/object'
    assert candidate_bound_ancestor(['/object/nested','/object'],{'/object/nested':object(),'/object':object()})=='/object/nested'
    assert candidate_bound_ancestor([], {}) is None
    with pytest.raises(CoverUnavailable):candidate_bound_ancestor(['/unbound'],{'/unrelated':object()})


def test_callback_wait_is_read_only_and_bounded():
    from sim.envs.robodojo.native_surface import await_query
    calls=[]
    await_query(lambda:len(calls)==3,lambda:calls.append(True),lambda:0)
    assert len(calls)==3
    with pytest.raises(CoverUnavailable,match='advanced physics'):
        await_query(lambda:False,lambda:calls.append(True),lambda:len(calls))
    times=iter([0.,0.,7.])
    with pytest.raises(CoverUnavailable,match='timed out'):
        await_query(lambda:False,lambda:None,lambda:0,monotonic=lambda:next(times))
    await_query(lambda:True,lambda:pytest.fail('must not pump completed query'),lambda:0)


def test_large_static_cover_is_explicit_bounded_and_never_applied_to_movable_geometry():
    from sim.envs.robodojo.native_surface import cover_resolution
    bounds=[[-2.081,-.401,.05],[1.884,.776,.975]]
    assert cover_resolution(bounds,.0005,static=True)==.004
    assert cover_resolution([[0,0,0],[.1,.1,.1]],.0005,static=False)==.0005
    with pytest.raises(CoverUnavailable,match='budget'):
        cover_resolution(bounds,.0005,static=False)
    with pytest.raises(CoverUnavailable,match='budget'):
        cover_resolution([[-20]*3,[20]*3],.0005,static=True)
