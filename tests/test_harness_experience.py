import copy
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
from PIL import Image
import pytest

from tools.harness_experience import ExperienceLibrary, SCHEMA, PROTOCOL, digest, embodiment_digest
from tools.harness_config import ConfigRegistry
from tools.codex_robodojo_host import RoboDojoHost
from test_codex_robodojo_host import host


def bundle(tmp_path):
    config=ConfigRegistry().resolve('robodojo_dual_x5_checked_v1')
    root=tmp_path/'bundle';root.mkdir();b=BytesIO();Image.new('RGB',(20,12),'red').save(b,format='PNG');data=b.getvalue()
    (root/'frame.png').write_bytes(data)
    state={'action_index':2,'simulation_time_s':1.,'robot':[
        {'role':r,'tcp_xyz_m':[0,0,1],'tcp_approach':[0,0,-1],'tcp_jaw':[1,0,0],
         'joint_positions_rad':[0]*6,'gripper_openness':1.,'commanded_openness':1.} for r in ['left','right']],
        'images':[{'camera':'cam_head','path':'frame.png','sha256':digest(data)}]}
    value={'schema':SCHEMA,'public_only_reviewed':True,'label':'Historical lift',
        'review':{'kind':'human_visual_review','text':'Lift visible, full task later failed.'},
        'source':{'episode_id':'old','env_id':'openeta/robodojo_dual_x5_general_pickup-v0','seed':0,'task':'Pick up',
                  'config_sha256':'a'*64,'embodiment_sha256':embodiment_digest(config.value['components']['embodiment']),
                  'official_success_recorded':False},
        'segments':[{'id':1,'call_id':'000001','tool':'move_to','arguments':{'role':'left','target':{'delta_xyz':[0,0,.02],'frame':'world'}},
                     'before':state,'after':copy.deepcopy(state),'feedback':{'native_actions':1,'physics_steps':10,'reached_target':True,'stop_reason':None,'error_code':None}}],
        'provenance':[{'path':'host/calls/000001/request.json','sha256':'b'*64}]}
    value['segments'][0]['after']['action_index']=3
    path=root/'manifest.json';path.write_text(json.dumps(value))
    return path,value,config


def save(path,value):
    path.write_text(json.dumps(value));return digest(path.read_bytes())


def test_library_catalog_real_pixels_cached_snapshot_and_no_false_success(tmp_path):
    path,value,config=bundle(tmp_path);lib=ExperienceLibrary(path,digest(path.read_bytes()),embodiment=config.value['components']['embodiment'])
    catalog,images=lib.read();assert not images and catalog['source']['official_success_recorded'] is False
    snap=lib.freeze(tmp_path/'frozen');path.unlink();(path.parent/'frame.png').unlink()
    result,images=lib.read(1)
    assert len(images)==4 and images[1].mimeType=='image/png'
    assert result['segment']['arguments']==value['segments'][0]['arguments']
    assert 'path' not in result['segment']['before']['images'][0]
    assert len(lib.read(1)[1])==0 and lib.record()['independent_pass_at_k'] is False
    assert snap.exists()
    catalog['source']['official_success_recorded']=True
    assert not lib.read()[0]['source']['official_success_recorded']


@pytest.mark.parametrize('mutation',['private','nan','bad_hash','escape','symlink','depth','role','frame','clock'])
def test_invalid_or_unscoped_evidence_rejected(tmp_path,mutation):
    path,v,c=bundle(tmp_path);im=v['segments'][0]['before']['images'][0]
    if mutation=='private':v['segments'][0]['before']['hidden_object_pose']=[1,2,3]
    if mutation=='nan':v['segments'][0]['arguments']['target']['delta_xyz'][0]=float('nan')
    if mutation=='bad_hash':im['sha256']='c'*64
    if mutation=='escape':im['path']='../outside.png'
    if mutation=='symlink':
        outside=tmp_path/'outside.png';outside.write_bytes((path.parent/'frame.png').read_bytes());(path.parent/'frame.png').unlink();(path.parent/'frame.png').symlink_to(outside)
    if mutation=='depth':
        b=BytesIO();Image.new('I;16',(2,2)).save(b,format='PNG');(path.parent/'frame.png').write_bytes(b.getvalue());im['sha256']=digest(b.getvalue())
    if mutation=='role':v['source']['embodiment_sha256']='d'*64
    if mutation=='frame':v['segments'][0]['arguments']['target']['frame']='hidden_object'
    if mutation=='clock':v['segments'][0]['after']['action_index']=0
    sha=save(path,v)
    with pytest.raises((ValueError,jsonschema.ValidationError)):
        ExperienceLibrary(path,sha,embodiment=c.value['components']['embodiment'])


def test_manifest_hash_and_missing_review_rejected(tmp_path):
    path,v,c=bundle(tmp_path)
    with pytest.raises(ValueError):ExperienceLibrary(path,'0'*64)
    v['public_only_reviewed']=False
    with pytest.raises(jsonschema.ValidationError):ExperienceLibrary(path,save(path,v))


def test_host_optional_reader_is_readonly_and_archives_actual_images(tmp_path):
    path,v,c=bundle(tmp_path)
    original,remote=host(tmp_path/'baseline',dual=True)
    assert 'review_experience' not in original.schemas
    original.close('test')
    args=copy.copy(original.args);args.output=tmp_path/'enabled';args.experience_bundle=path;args.experience_sha256=digest(path.read_bytes())
    h=RoboDojoHost(args,original.config,original.binding,transport=remote)
    try:
        calls=len(remote.calls);packet=copy.deepcopy(h.packet)
        result=h.call('review_experience',{'segment_id':1});value=json.loads(result[0].text)
        assert value['error'] is None and value['receipt']['historical']
        assert len(remote.calls)==calls and h.packet==packet and h.tool_calls==1
        assert len([x for x in result if x.type=='image'])==2
        evidence=json.loads((h.output/'calls/000001/response.json').read_text())
        assert len([x for x in evidence['content'] if x['type']=='image'])==2
        assert not [x for x in h.call('review_experience',{'segment_id':1}) if x.type=='image']
        bad=json.loads(h.call('review_experience',{'path':'/etc/passwd'})[0].text)
        assert bad['error']['code']=='request_rejected'
        # Reviewing history is optional: ordinary operations continue without a review gate.
        h.call('observe',{});assert len(remote.calls)>calls
    finally:h.close('test')


def test_bad_bundle_fails_before_creating_environment(tmp_path):
    original,remote=host(tmp_path/'baseline',dual=True);original.close('test');calls=len(remote.calls)
    args=copy.copy(original.args);args.output=tmp_path/'bad';args.experience_bundle=tmp_path/'missing';args.experience_sha256='a'*64
    with pytest.raises(FileNotFoundError):RoboDojoHost(args,original.config,original.binding,transport=remote)
    assert len(remote.calls)==calls


def test_non_robodojo_does_not_silently_ignore_experience(tmp_path):
    from tools.codex_host import build_host
    with pytest.raises(ValueError,match='RoboDojo'):
        build_host(SimpleNamespace(experience_bundle=tmp_path/'x',experience_sha256='a'*64,harness_config=None,expected_controller='mink'))


@pytest.mark.parametrize('combined', [False, True])
def test_launcher_freezes_bundle_and_stamps_protocol_before_operator(tmp_path,monkeypatch,combined):
    from scripts import codex_plugin_smoke as launcher
    import shutil
    path,v,c=bundle(tmp_path);sha=digest(path.read_bytes());out=tmp_path/'run';seen={}
    def prepared(output,host_args,*args,**kwargs):
        seen['args']=host_args
        frozen=Path(host_args[host_args.index('--experience-bundle')+1]);assert frozen!=path and frozen.exists()
        assert digest(frozen.read_bytes())==sha
        home=output/'codex-home';home.mkdir();(home/'auth.json').write_text('synthetic')
        skill=output/'marketplace/plugins/openeta/skills/openeta-robodojo/SKILL.md';skill.parent.mkdir(parents=True)
        shutil.copyfile(launcher.REPO/'plugins/openeta/skills/openeta-robodojo/SKILL.md',skill)
        (output/'host').mkdir();(output/'host/host-status.json').write_text(json.dumps({'closed':True,'cleanup':{'ok':True},'tool_calls':2,'official_task_success':False}))
        return home,output,{}
    monkeypatch.setattr(launcher,'prepare',prepared)
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(pid=99999999,returncode=0,wait=lambda **kw:0))
    monkeypatch.setattr(launcher.os,'killpg',lambda *a:None)
    args=launcher.parser().parse_args(['--sim-url','http://unused','--task','Pick up','--env-id',v['source']['env_id'],
        '--seed','0','--output',str(out),'--controller','robodojo_checked','--tool-profile','atomic',
        '--experience-bundle',str(path),'--experience-sha256',sha])
    args.harness_config=c.id;args.harness_config_sha256=c.sha256
    if combined:
        source=tmp_path/'prior.jsonl'
        source.write_text(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'The grasp slipped during a turn.'}})+'\n'+json.dumps({'type':'turn.completed'})+'\n')
        selection=tmp_path/'selection.json'
        selection.write_text(json.dumps({'schema':'openeta.public_episode_experience.v1','entries':[
            {'source':str(source),'source_sha256':digest(source.read_bytes()),'event_index':0}]}))
        args.public_experience=selection;args.combine_public_visual_experience=True
    assert launcher.run(args)==0
    summary=json.loads((out/'summary.json').read_text())
    expected='cross_attempt_visual_and_public_experience' if combined else PROTOCOL
    assert summary['evaluation_protocol']==expected
    assert summary['experiment_factors']['conditioning']==expected
    if combined:
        assert summary['experiment_factors']['visual_experience_sha256']==sha
        assert summary['experiment_factors']['public_experience_sha256']==digest(selection.read_bytes())
        assert not summary['independent_pass_at_k']
        assert 'The grasp slipped during a turn.' in (out/'prompt.txt').read_text()
    assert summary['visual_action_experience']['manifest_sha256']==sha
    assert not (out/'codex-home/auth.json').exists()
    assert 'review_experience' in (out/'prompt.txt').read_text()


@pytest.mark.parametrize('combined', [False, True])
def test_matrix_separates_visual_experience_and_preserves_bundle_identity(tmp_path,monkeypatch,combined):
    from scripts import update_robodojo_experiment_table as table
    monkeypatch.setattr(table,'ROOT',tmp_path)
    run=tmp_path/'model-v88-icl';run.mkdir()
    (run/'experiment.json').write_text(json.dumps({'env_id':'openeta/robodojo_dual_x5_general_pickup-v0'}))
    summary={'model':'gpt-6-astra','reasoning_effort':'high','status':'completed','evaluation_protocol':PROTOCOL,
             'visual_action_experience':{'manifest_sha256':'a'*64},'host':{'tool_calls':3,'outcome':{'status':'running'}}}
    if combined:
        summary['evaluation_protocol']='cross_attempt_visual_and_public_experience'
        summary['public_experience']={'selection_sha256':'b'*64}
    path=run/'summary.json';path.write_text(json.dumps(summary));entry=table.record(path,summary,{'general_pickup'})
    assert entry['column']==('v88 · 跨轮图像＋原文经验' if combined else 'v88 · 跨轮图像动作经验')
    if combined:
        assert entry['public_experience_sha256']=='b'*64
    assert entry['visual_action_experience_sha256']=='a'*64 and entry['success']==0


def source_fixture(tmp_path):
    path,v,c=bundle(tmp_path);source=tmp_path/'source';call=source/'host/calls/000001';call.mkdir(parents=True)
    (source/'summary.json').write_text(json.dumps({'status':'completed','host':{'closed':True,'outcome':{'status':'running','terminal':False,'benchmark_success':None}}}))
    (source/'host/harness-binding.json').write_text(json.dumps({'resolved_config':c.value,'config_sha256':c.sha256,
        'task_instance':{'env_id':v['source']['env_id'],'seed':0,'task':'Pick up'}}))
    (call/'request.json').write_text(json.dumps({'call_id':'000001','tool':'move_to','arguments':v['segments'][0]['arguments'],'before_action_index':2}))
    image=(path.parent/'frame.png').read_bytes()
    for phase,index in [('before',2),('after',3)]:
        (call/f'{phase}-000-rgb.png').write_bytes(image)
        packet={'action_index':index,'control':{'observation_clock':{'simulation_time_s':float(index)},
            'tcp_poses':{r:[[1,0,0,0],[0,-1,0,0],[0,0,-1,1],[0,0,0,1]] for r in ['left','right']},'private_geometry':'DO_NOT_EXPORT'},
            'observation':{'objects':[{'private_pose':'DO_NOT_EXPORT'}],
            'robot':{'arms':{r:{'joint_positions':[0]*6,'gripper_state':{'openness':1,'commanded_openness':1,'measurement_available':True}} for r in ['left','right']}},
            'cameras':[{'frame_id':'cam_head','rgb_artifact':{'path':f'{phase}-000-rgb.png','sha256':digest(image)}}]}}
        (call/f'{phase}-observation.json').write_text(json.dumps(packet))
    payload={'action_index':3,'receipt':{'native_actions':1,'physics_steps':10,'reached_target':True,'private_evaluator':'DO_NOT_EXPORT'}}
    (call/'response.json').write_text(json.dumps({'call_id':'000001','content':[{'type':'text','text':json.dumps(payload)}]}))
    return source


def test_exporter_projects_public_fields_and_keeps_episode_failure(tmp_path):
    from scripts.build_robodojo_experience import build
    source=source_fixture(tmp_path);out=tmp_path/'export'
    record=build(source,out,first=1,last=1,cameras=['cam_head'],label='lift',review_kind='unreviewed',review_text='')
    text=(out/'manifest.json').read_text()
    assert 'DO_NOT_EXPORT' not in text and 'private_pose' not in text
    assert not record['source']['official_success_recorded']
    lib=ExperienceLibrary(out/'manifest.json',record['manifest_sha256']);receipt,images=lib.read(1)
    assert receipt['segment']['arguments']['target']['delta_xyz']==[0,0,.02]
    assert len([x for x in images if x.type=='image'])==2


@pytest.mark.parametrize('fault',['not_closed','missing_after','corrupt_rgb','unsupported_argument'])
def test_exporter_rejects_unfinished_or_incomplete_or_unscoped_sources(tmp_path,fault):
    from scripts.build_robodojo_experience import build
    source=source_fixture(tmp_path);call=source/'host/calls/000001'
    if fault=='not_closed':(source/'summary.json').write_text(json.dumps({'status':'running'}))
    if fault=='missing_after':(call/'after-observation.json').unlink()
    if fault=='corrupt_rgb':(call/'before-000-rgb.png').write_bytes(b'bad')
    if fault=='unsupported_argument':
        p=call/'request.json';v=json.loads(p.read_text());v['arguments']['hidden_target']=[1,2,3];p.write_text(json.dumps(v))
    with pytest.raises((ValueError,FileNotFoundError,jsonschema.ValidationError)):
        build(source,tmp_path/'out',first=1,last=1,cameras=['cam_head'],label='lift',review_kind='unreviewed',review_text='')
    assert not (tmp_path/'out').exists()


def test_single_arm_experience_has_no_dual_role_assumption(tmp_path):
    path,v,_=bundle(tmp_path);c=ConfigRegistry().resolve('robodojo_single_x5_left_mount_v1')
    v['source']['embodiment_sha256']=embodiment_digest(c.value['components']['embodiment'])
    v['segments'][0]['arguments']['role']='primary'
    for phase in ('before','after'):
        arm=v['segments'][0][phase]['robot'][0];arm['role']='primary';v['segments'][0][phase]['robot']=[arm]
    lib=ExperienceLibrary(path,save(path,v),embodiment=c.value['components']['embodiment'])
    assert lib.read(1)[0]['segment']['arguments']['role']=='primary'


def test_visual_public_mix_still_requires_explicit_combined_condition(tmp_path, monkeypatch):
    from scripts import codex_plugin_smoke as launcher
    path, v, c = bundle(tmp_path)
    out = tmp_path/'rejected-run'
    args = launcher.parser().parse_args(['--sim-url','http://unused','--task','Pick up',
        '--env-id',v['source']['env_id'],'--output',str(out),'--controller','robodojo_checked',
        '--tool-profile','atomic','--experience-bundle',str(path),
        '--experience-sha256',digest(path.read_bytes()),'--public-experience',str(tmp_path/'not-read.json')])
    args.harness_config=c.id;args.harness_config_sha256=c.sha256
    monkeypatch.setattr(launcher, 'prepare', lambda *a, **kw: pytest.fail('must reject before private runtime'))
    assert launcher.run(args) != 0
    assert 'Do not mix visual action experience' in (out/'summary.json').read_text()
