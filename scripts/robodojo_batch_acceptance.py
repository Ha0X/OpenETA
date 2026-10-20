"""Sequential, owned MCP admission canaries; never a model or task-success trial.

Bootstrap from the official seeded observation, then pass the exact real
create/reset receipts into the ordinary Host. No task instruction is guessed,
no pose comes from an object annotation, and no motion bypasses the Host.
"""
import argparse
import base64
import hashlib
import io
import json
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from tools.codex_robodojo_host import RoboDojoHost
from tools.harness_config import ConfigRegistry
from tools.harness_transport import StreamableHttpSimulatorMcpTransport

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT/'configs/harness/robodojo_rigid32_inventory.json'
PROFILE = 'robodojo_dual_x5_rigid32_acceptance_v1'


def verification_levels():
    return {name: {'status':'not_run'} for name in ('layout','protocol','motion','task')}


class OwnedBootstrap:
    """Adopt only this canary's real, already-reset environment, exactly once."""
    def __init__(self, transport, output):
        self.transport, self.output = transport, output
        self.owned = {}
        self.cleanup = None
        self.cached = {}
        self.unknown_create = False

    def call_tool(self, name, arguments, *, timeout_s=None):
        if name in self.cached:
            expected, result = self.cached.pop(name)
            if arguments != expected:
                raise ValueError('Owned bootstrap arguments differ from the real request')
            return result
        try:
            result = self.transport.call_tool(name, arguments, timeout_s=timeout_s)
        except BaseException:
            if name == 'create_env':
                self.unknown_create = True
            raise
        if name == 'create_env' and result.get('handle'):
            self.owned = {k:result[k] for k in ('session_id','handle')}
            (self.output/'owned.json').write_text(json.dumps(self.owned)+'\n')
        if name == 'close_env':
            self.cleanup = {'ok':not result.get('error') and result.get('ok',True) is not False,'receipt':result}
        return result

    def bootstrap(self, env_id, seed):
        create_args = {'env_id':env_id,'seed':seed,'include_objects':False}
        created = self.call_tool('create_env', create_args, timeout_s=240)
        if created.get('error'):
            raise RuntimeError('create_env: '+str(created))
        reset_args = {**self.owned,'seed':seed}
        observations = []
        for index in (1,2):
            result = self.call_tool('reset_env', reset_args, timeout_s=180)
            (self.output/f'reset-{index}.json').write_text(json.dumps(result)+'\n')
            if result.get('error'):
                raise RuntimeError('reset_env: '+str(result))
            observations.append(result)
        self.cached = {'create_env':(create_args,created), 'reset_env':(reset_args,observations[-1])}
        return observations


def seed_evidence(observations):
    def brief(obs):
        return {'task':obs.get('task'), 'robot':obs.get('robot'),
                'depth_hashes':{c['frame_id']:hashlib.sha256(base64.b64decode(c.get('depth_base64',''))).hexdigest()
                                for c in obs.get('cameras',[])}}
    first, second = map(brief, observations)
    return {'requested_seed':0, 'same_task':first['task']==second['task'],
            'same_robot':first['robot']==second['robot'], 'same_depth_images':first['depth_hashes']==second['depth_hashes'],
            'note':'Two actual resets. Image equality is diagnostic, not a complete physical-state determinism proof.'}


def run_task(task, output, url):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'task':task,'seed':0,'purpose':'admission_canary_not_benchmark_trial','passed':False,
              'phase':'bootstrap','checks':{},'verification':verification_levels()}
    active_level = 'layout'
    def save():
        report['elapsed_s'] = round(time.monotonic()-started,3)
        (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    transport = OwnedBootstrap(StreamableHttpSimulatorMcpTransport(url), output)
    host = None
    try:
        save()
        env_id = f'openeta/robodojo_dual_x5_{task}-v0'
        config = ConfigRegistry().resolve(PROFILE)
        config.require_executable()
        observations = transport.bootstrap(env_id, 0)
        report['checks']['seed_reset'] = seed_evidence(observations)
        official_task = observations[-1].get('task')
        if not isinstance(official_task,str) or not official_task.strip():
            raise ValueError('Official seeded observation has no task instruction')
        report['official_instruction'] = official_task
        binding = config.bind_task(env_id=env_id, task=official_task, seed=0)
        report['config_sha256'] = config.sha256
        report['phase'] = 'host_binding_and_geometry'; save()
        args = argparse.Namespace(output=output/'host',sim_url=url,env_id=env_id,task=official_task,
                                  seed=0,timeout=600,max_requests=30)
        host = RoboDojoHost(args,config,binding,transport=transport)
        report['verification']['layout'] = {'status':'passed', 'evidence':'host/harness-binding.json'}
        active_level = 'protocol'
        report['checks']['host_binding_and_geometry'] = {'passed':True,'roles':host.packet['control']['roles'],
                                              'collision_policy':host.packet['control']['collision_policy']}
        def call(name, arguments):
            result=json.loads(host.call(name,arguments)[0].text)
            if result.get('error'):
                raise RuntimeError(json.dumps(result['error']))
            return result
        report['phase'] = 'observation_and_mark'; save()
        initial = call('observe',{})
        assert initial['status']['outcome']['status'] != 'unknown', initial['status']['outcome']
        assert not host.packet['observation'].get('objects')
        frames=host.packet['observation']['cameras']
        assert frames and all(c.get('rgb_base64') and c.get('depth_base64') for c in frames)
        selected = None
        for frame in frames:
            depth=np.array(Image.open(io.BytesIO(base64.b64decode(frame['depth_base64']))))
            pixels=np.argwhere((depth>20)&(depth<3000))
            if len(pixels):
                center=np.array(depth.shape)/2
                v,u=pixels[np.argmin(np.sum((pixels-center)**2,axis=1))]
                selected={'camera':frame['frame_id'],'pixel':[int(u),int(v)]}; break
        assert selected is not None, 'No valid depth pixel in any public camera'
        mark=call('mark_point',selected)
        assert len(mark['receipt']['mark']['xyz'])==3
        report['checks']['observation_and_mark']={'passed':True,'cameras':[c['frame_id'] for c in frames],
                                               'mark':mark['receipt']['mark'],'initial_outcome':initial['status']['outcome']}
        report['verification']['protocol'] = {'status':'passed', 'evidence':'host/calls/'}
        active_level = 'motion'
        roles=list(host.packet['control']['roles'])
        assert roles==['left','right'], roles
        initial_tcp={r:np.array(t) for r,t in host.packet['control']['tcp_poses'].items()}
        def move(label, motions):
            report['phase']=label; save()
            result=call('move_to',motions)
            report['checks'][label]=result['receipt']; save()
            assert result['receipt']['reached_target'], json.dumps(result['receipt'].get('stop'))
        xyz=initial_tcp['left'][:3,3].copy();xyz[2]+=.02
        move('left_20mm',{'role':'left','target':{'xyz':xyz.tolist()}})
        held=np.array(host.packet['control']['tcp_poses']['right'])
        delta=float(np.max(abs(held-initial_tcp['right'])))
        report['checks']['unselected_right_hold']={'max_matrix_delta':delta,'passed':delta<.002}
        assert delta<.002, report['checks']['unselected_right_hold']
        motions=[]
        for role in roles:
            xyz=np.array(host.packet['control']['tcp_poses'][role])[:3,3].copy();xyz[2]+=.01
            motions.append({'role':role,'target':{'xyz':xyz.tolist()}})
        move('coordinated_10mm',{'motions':motions})
        assert report['checks']['coordinated_10mm']['coordination']['status'] in ('safe','repaired')
        motions=[]
        for role in roles:
            xyz=np.array(host.packet['control']['tcp_poses'][role])[:3,3].copy();xyz[2]+=.005
            motions.append({'role':role,'target':{'xyz':xyz.tolist()}})
        move('explicit_synchronized_5mm',{'motions':motions,'coordination':'synchronized'})
        for role in roles:
            for openness in (0.,1.):
                label=f'{role}_gripper_{int(openness)}';report['phase']=label;save()
                result=call('gripper_control',{'role':role,'openness':openness})
                receipt=result['receipt'];report['checks'][label]=receipt;save()
                assert receipt.get('stop') is None, receipt.get('stop')
                measured=receipt['measured']['openness']
                assert abs(measured-openness)<.02, f'Empty gripper target {openness}, measured {measured}'
        report['checks']['final_outcome']=call('episode_status',{})['status']['outcome']
        assert report['checks']['final_outcome']['status']!='unknown'
        report['verification']['motion'] = {'status':'passed', 'scope':'checked_micro_motion_and_empty_grippers',
                                             'evidence':'host/calls/'}
        active_level = None
        report['passed']=True;report['phase']='complete'
    except Exception:
        report['error']=traceback.format_exc()
        if active_level:
            report['verification'][active_level] = {'status':'failed', 'phase':report['phase']}
    finally:
        try:
            if host is not None:
                host.close('admission_canary_complete')
            elif transport.owned and transport.cleanup is None:
                transport.call_tool('close_env',transport.owned,timeout_s=90)
            report['cleanup']=transport.cleanup or {
                'ok':not transport.owned and not transport.unknown_create,
                'no_owned_handle':not transport.owned,
                'create_outcome_unknown':transport.unknown_create}
        except Exception:
            report['cleanup']={'ok':False,'error':traceback.format_exc()}
        report['passed']=report['passed'] and report['cleanup']['ok']
        save()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sim-url',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',help='Subset of the pinned 32 candidates; default all')
    args=p.parse_args()
    inventory={r['task']:r for r in json.loads(INVENTORY.read_text())['tasks'] if r['candidate']}
    candidates=list(inventory)
    names=args.tasks or candidates
    if len(set(names))!=len(names) or not set(names)<=set(candidates):p.error('Use distinct registered candidates')
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name in names:
        print(json.dumps({'event':'start','task':name}),flush=True)
        blocker=inventory[name].get('admission_blocker')
        if blocker:
            report={'task':name,'passed':False,'phase':'native_task_contract','elapsed_s':0,
                    'error':blocker,'cleanup':{'ok':True,'no_environment_created':True},
                    'verification':verification_levels()}
            report['verification']['layout'] = {'status':'blocked', 'reason':blocker}
            (args.output/name).mkdir()
            (args.output/name/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        else:
            report=run_task(name,args.output/name,args.sim_url)
        rows.append({k:report.get(k) for k in ('task','passed','phase','elapsed_s','error','cleanup','verification')})
        (args.output/'summary.json').write_text(json.dumps({'purpose':'admission_not_task_success','planned':names,'results':rows},indent=2)+'\n')
        print(json.dumps({'event':'complete',**rows[-1]}),flush=True)
        if not report['cleanup']['ok']:
            print('Stopping batch because owned cleanup is unresolved',flush=True);return 2
    return 0 if all(r['passed'] for r in rows) else 1


if __name__=='__main__':
    raise SystemExit(main())
