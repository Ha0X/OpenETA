"""Export a reviewed call range into a public-only visual/action experience bundle.

This offline command reads an explicitly selected completed episode. It never
reads simulator assets, invokes a model, or labels a clip as a task success.
"""
from pathlib import Path
import argparse
import json
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.harness_experience import SCHEMA, ExperienceLibrary, contained, digest, embodiment_digest


def build(source, output, *, first, last, cameras, label, review_kind, review_text):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Experience output must be a new directory')
    if not 1<=first<=last or last-first>59:raise ValueError('Select a bounded call range')
    if not cameras or len(cameras)>3 or len(cameras)!=len(set(cameras)):raise ValueError('Select 1–3 unique public cameras')
    provenance=[]
    def read(relative):
        raw=contained(source,relative).read_bytes()
        provenance.append({'path':relative,'sha256':digest(raw)})
        return json.loads(raw)
    summary=read('summary.json');binding=read('host/harness-binding.json')
    if summary.get('status')!='completed' or not summary.get('host',{}).get('closed'):
        raise ValueError('Experience requires a completed, closed prior episode')
    config=binding['resolved_config'];parts=config['components'];embodiment=parts['embodiment']
    if parts['benchmark']['backend']!='robodojo':raise ValueError('Only RoboDojo public call evidence is supported')
    roles=list(embodiment['effectors']);assets={}
    def snapshot(relative, phase, call_id):
        packet=read(relative);obs=packet['observation'];control=packet['control']
        robot=[]
        for role in roles:
            t=control['tcp_poses'][role];arm=obs['robot']['arms'][role];g=arm.get('gripper_state',{})
            robot.append({'role':role,'tcp_xyz_m':[row[3] for row in t[:3]],
                'tcp_approach':[row[2] for row in t[:3]],'tcp_jaw':[row[0] for row in t[:3]],
                'joint_positions_rad':arm['joint_positions'],
                'gripper_openness':g.get('openness') if g.get('measurement_available') else None,
                'commanded_openness':g.get('commanded_openness')})
        images=[]
        for camera in cameras:
            matches=[f for f in obs['cameras'] if f['frame_id']==camera]
            if len(matches)!=1:raise ValueError('Selected public camera unavailable or ambiguous')
            artifact=matches[0]['rgb_artifact'];f=contained(source, str(Path(relative).parent/artifact['path']))
            data=f.read_bytes()
            if digest(data)!=artifact['sha256']:raise ValueError('Source public RGB hash mismatch')
            name=f'images/{call_id}-{phase}-{len(images)}.png';assets[name]=data
            images.append({'camera':camera,'path':name,'sha256':digest(data)})
            provenance.append({'path':str(f.relative_to(source)),'sha256':digest(data)})
        return {'action_index':packet['action_index'],'simulation_time_s':control.get('observation_clock',{}).get('simulation_time_s'),
                'robot':robot,'images':images}
    segments=[]
    for index in range(first,last+1):
        call_id=f'{index:06d}';folder=f'host/calls/{call_id}'
        request=read(folder+'/request.json')
        if request['tool'] not in ('move_to','gripper_control','wait'):continue
        response=read(folder+'/response.json')
        if response['call_id']!=call_id or request['call_id']!=call_id:raise ValueError('Mismatched call evidence')
        value=json.loads(next(x['text'] for x in response['content'] if x['type']=='text'))
        receipt=value.get('receipt') or {};error=value.get('error') or {}
        # Required after snapshots: never invent post-action evidence or substitute a future frame.
        before=snapshot(folder+'/before-observation.json','before',call_id)
        after=snapshot(folder+'/after-observation.json','after',call_id)
        if request['before_action_index']!=before['action_index'] or value.get('action_index')!=after['action_index']:
            raise ValueError('Source action index mismatch')
        segments.append({'id':len(segments)+1,'call_id':call_id,'tool':request['tool'],
            'arguments':request['arguments'],'before':before,'after':after,
            'feedback':{'native_actions':receipt.get('native_actions',0),'physics_steps':receipt.get('physics_steps',0),
                'reached_target':receipt.get('reached_target'),'stop_reason':(receipt.get('stop') or {}).get('reason'),
                'error_code':error.get('code')}})
    outcome=summary['host'].get('outcome') or {};instance=binding['task_instance']
    success=(outcome.get('status')=='success' and outcome.get('benchmark_success') is True
             and outcome.get('terminal') is True and outcome.get('source')=='robodojo.eval_env.terminal_flags')
    value={'schema':SCHEMA,'public_only_reviewed':True,'label':label,'review':{'kind':review_kind,'text':review_text},
        'source':{'episode_id':source.name,'env_id':instance['env_id'],'seed':instance['seed'],'task':instance['task'],
            'config_sha256':binding['config_sha256'],'embodiment_sha256':embodiment_digest(embodiment),
            'official_success_recorded':success},'segments':segments,'provenance':provenance}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='experience-build-',dir=output.parent) as temp:
        stage=Path(temp)
        for relative,data in assets.items():
            f=stage/relative;f.parent.mkdir(exist_ok=True);f.write_bytes(data)
        raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        manifest=stage/'manifest.json';manifest.write_bytes(raw)
        library=ExperienceLibrary(manifest,digest(raw),embodiment=embodiment)
        library.freeze(output)
        record=library.record()
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--first-call',type=int,required=True);p.add_argument('--last-call',type=int,required=True)
    p.add_argument('--cameras',nargs='+',default=['cam_head','cam_left_wrist'])
    p.add_argument('--label',required=True);p.add_argument('--review-kind',choices=['unreviewed','agent_report','human_visual_review'],default='unreviewed')
    p.add_argument('--review-text',default='')
    p.add_argument('--public-only-reviewed',action='store_true',required=True,help='Attest review of selected public imagery and annotation; no hidden-state diagnosis')
    a=p.parse_args();print(json.dumps(build(a.source,a.output,first=a.first_call,last=a.last_call,cameras=a.cameras,
        label=a.label,review_kind=a.review_kind,review_text=a.review_text),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
