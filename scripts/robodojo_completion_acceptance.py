"""Read-only completion-checker admission. Never a model/task-success trial.

Use the actual service catalog and official seeded task text. Keep controller
admission intact; unavailable motion profiles are not bypassed for a pass.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.robodojo_batch_acceptance import OwnedBootstrap, INVENTORY
from tools.codex_robodojo_host import RoboDojoHost
from tools.harness_config import ConfigRegistry
from tools.harness_transport import StreamableHttpSimulatorMcpTransport

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'robodojo_native_coarse_v1'
PREVIOUSLY_CHECKED = {'build_tower', 'stack_bowls'}


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def make_config(task, output):
    components=json.loads((ROOT/'configs/harness/components.json').read_text())
    presets=json.loads((ROOT/'configs/harness/presets.json').read_text())
    base_id=('robodojo_dual_x5_visual_cloth_v75' if 'Garment' in task['scene_types']
             else 'robodojo_dual_x5_coarse_completion_v80')
    preset=copy.deepcopy(next(p for p in presets['presets'] if p['id']==base_id))
    benchmark=copy.deepcopy(components['components']['benchmark'][preset['components']['benchmark']])
    key='checker_'+task['task'].lower()+'_v80'
    env_id='openeta/robodojo_dual_x5_'+task['task']+'-v0'
    benchmark.update(id=key,env_pattern=env_id,completion_feedback=PROFILE)
    benchmark.pop('episode_budget_override',None)
    components['components']['benchmark'][key]=benchmark
    preset.update(id=key,description='Read-only checker admission; not execution or task-success certification.')
    preset['components']['benchmark']=key
    presets['presets'].append(preset)
    directory=output/'catalog';directory.mkdir()
    write(directory/'components.json',components);write(directory/'presets.json',presets)
    config=ConfigRegistry(directory).resolve(key);config.require_executable()
    return config,env_id


def state_evidence(host):
    p=host.packet
    return {'action_index':p['action_index'],
            'observation_sha256':fingerprint(p['observation']),
            'budget':copy.deepcopy(p['control']['episode_budget']),
            'outcome':copy.deepcopy(host._status()['outcome'])}


def classify_error(phase, error):
    if phase=='bootstrap':return 'environment_bootstrap_failed'
    if phase=='host_binding':return 'host_admission_failed'
    if phase=='checks':return 'completion_checker_failed'
    return 'acceptance_failed'


def run_task(task, output, url, registered, progress=None):
    output.mkdir(parents=True,exist_ok=False)
    name=task['task'];env_id='openeta/robodojo_dual_x5_'+name+'-v0'
    result={'task':name,'env_id':env_id,'seed':0,'kind':'read_only_checker_acceptance_not_agent_trial',
            'status':'running','phase':'catalog','scene_types':task['scene_types'],
            'native_robot_config':task['native_robot_config'],'native_actions_requested':0,
            'checker_status':'not_run','registered':env_id in registered}
    started=time.monotonic()
    def save():
        result['elapsed_s']=round(time.monotonic()-started,3)
        write(output/'summary.json',result)
        if progress:progress(copy.deepcopy(result))
    save()
    if env_id not in registered:
        result.update(status='blocked_not_registered',phase='complete',
            reason='The existing service catalog does not expose this native task; no create or motion was attempted.',
            known_admission_blocker=task.get('admission_blocker'),cleanup={'ok':True,'no_environment_created':True})
        save();return result
    transport=OwnedBootstrap(StreamableHttpSimulatorMcpTransport(url),output)
    host=None
    try:
        config,env_id=make_config(task,output)
        result['config_sha256']=config.sha256
        result['phase']='bootstrap';save()
        create_args={'env_id':env_id,'seed':0,'include_objects':False}
        created=transport.call_tool('create_env',create_args,timeout_s=240)
        if created.get('error'):raise RuntimeError('create_env: '+str(created['error']))
        reset_args={**transport.owned,'seed':0}
        observation=transport.call_tool('reset_env',reset_args,timeout_s=180)
        write(output/'reset.json',observation)
        if observation.get('error'):raise RuntimeError('reset_env: '+str(observation['error']))
        instruction=observation.get('task')
        if not isinstance(instruction,str) or not instruction.strip():raise ValueError('No official task instruction')
        result['official_instruction']=instruction
        transport.cached={'create_env':(create_args,created),'reset_env':(reset_args,observation)}
        result['phase']='host_binding';save()
        args=argparse.Namespace(output=output/'host',sim_url=url,env_id=env_id,task=instruction,
                                seed=0,timeout=600,max_requests=12)
        host=RoboDojoHost(args,config,config.bind_task(env_id=env_id,task=instruction,seed=0),transport=transport)
        assert 'check_task_completion' in host.schemas
        assert not host.packet['observation'].get('objects')
        assert host.packet['control']['completion_feedback']==PROFILE
        initial=state_evidence(host)
        assert initial['action_index']==0
        result['phase']='checks';save()
        receipts=[]
        for _ in range(2):
            response=json.loads(host.call('check_task_completion',{})[0].text)
            if response.get('error'):raise RuntimeError(str(response['error']))
            receipt=response['receipt'];receipts.append(receipt)
            assert receipt['status']=='checked',receipt
            assert receipt['profile']==PROFILE and receipt['feedback_level']=='coarse'
            assert receipt['source_action_index']==0 and receipt['native_actions']==receipt['physics_steps']==0
            assert receipt['current_conditions_satisfied'] is None
            assert state_evidence(host)==initial,'Read-only check altered measured state, budget or outcome'
        assert receipts[0]==receipts[1],'Repeated checks differed without a physical action'
        result.update(status='passed',phase='complete',checker_status='checked',
            repeated_receipts_identical=True,measured_state_unchanged=True,
            initial_state=initial,receipt=receipts[0],
            limits='Initial-state read-only integration only; no motion, success-state or full task validation.')
    except Exception as exc:
        result['status']=classify_error(result['phase'],exc)
        result['error_type']=type(exc).__name__
        result['error']=str(exc)
        (output/'operator-error.txt').write_text(traceback.format_exc())
    finally:
        try:
            if host is not None:host.close('completion_checker_acceptance_complete')
            elif transport.owned and transport.cleanup is None:
                transport.call_tool('close_env',transport.owned,timeout_s=90)
            result['cleanup']=transport.cleanup or {'ok':not transport.owned and not transport.unknown_create,
                'no_owned_handle':not transport.owned,'create_outcome_unknown':transport.unknown_create}
        except Exception as exc:
            result['cleanup']={'ok':False,'error_type':type(exc).__name__,'error':str(exc)}
        if not result['cleanup']['ok']:result['status']='cleanup_failed'
        save()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:9889/mcp')
    parser.add_argument('--tasks',nargs='*',help='Default: remaining 40 base tasks, excluding the two prior preflights.')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    tasks=json.loads(INVENTORY.read_text())['tasks']
    chosen=set(args.tasks) if args.tasks else {t['task'] for t in tasks}-PREVIOUSLY_CHECKED
    assert chosen<={t['task'] for t in tasks}
    tasks=[t for t in tasks if t['task'] in chosen]
    if args.tasks:tasks.sort(key=lambda t:args.tasks.index(t['task']))
    transport=StreamableHttpSimulatorMcpTransport(args.url)
    catalog=transport.call_tool('list_envs',{'env_type':'robodojo'},timeout_s=30)
    write(args.output/'service-catalog.json',catalog)
    registered={e['id'] for e in catalog['envs']}
    manifest={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'components_sha256':hashlib.sha256((ROOT/'configs/harness/components.json').read_bytes()).hexdigest(),
        'task_count':len(tasks),'tasks':[t['task'] for t in tasks],'profile':PROFILE,
        'seed':0,'purpose':'initial_state_read_only_checker_acceptance','model_runs':0,'native_actions_requested':0}
    write(args.output/'manifest.json',manifest)
    results=[]
    # Record all catalog blockers immediately, then run real environments sequentially.
    tasks.sort(key=lambda t:('openeta/robodojo_dual_x5_'+t['task']+'-v0') in registered)
    for task in tasks:
        def progress(row):write(args.output/'progress.json',{'completed':len(results),'total':len(tasks),'current':row,'results':results})
        result=run_task(task,args.output/task['task'],args.url,registered,progress)
        results.append(result);write(args.output/'results.json',results)
        print(json.dumps({'completed':len(results),'total':len(tasks),'task':task['task'],'status':result['status'],'elapsed_s':result['elapsed_s']}),flush=True)
        if not result['cleanup']['ok']:
            write(args.output/'stopped.json',{'reason':'cleanup_not_confirmed','task':task['task']})
            return 2
    write(args.output/'complete.json',{'total':len(tasks),'counts':{s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})},'full_agent_task_runs':0})
    return 0


if __name__=='__main__':raise SystemExit(main())
