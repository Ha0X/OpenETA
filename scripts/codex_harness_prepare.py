"""Run an isolated ChatGPT-authenticated configuration selection session."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from scripts import codex_campaign as ownership
from tools.harness_config import ConfigRegistry
from tools.harness_setup import read_selection


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    home, work, market = output/'codex-home', output/'operator-workspace', output/'marketplace'
    home.mkdir(mode=0o700)
    home.chmod(0o700)
    work.mkdir()
    source = Path(os.environ.get('CODEX_HOME', str(Path.home()/'.codex')))/'auth.json'
    auth = json.loads(source.read_text())
    if not auth.get('tokens') or auth.get('auth_mode') == 'apikey':
        raise RuntimeError('Preparation requires saved ChatGPT login')
    fd = os.open(home/'auth.json', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(auth, stream)
    shutil.copytree(ROOT/'plugins/openeta-setup', market/'plugins/openeta-setup')
    (market/'.agents/plugins').mkdir(parents=True)
    # Fresh, run-owned marketplace; never update a user's installed catalog.
    catalog = {'name':'openeta-codex-smoke', 'interface':{'displayName':'OpenETA Preparation'},
        'plugins':[{'name':'openeta-setup','source':{'source':'local','path':'./plugins/openeta-setup'},
                    'policy':{'installation':'AVAILABLE','authentication':'ON_USE'},'category':'Productivity'}]}
    (market/'.agents/plugins/marketplace.json').write_text(json.dumps(catalog, indent=2)+'\n')
    setup_args = ['--env-id',args.env_id,'--task',args.task,'--seed',str(args.seed),'--output',str(output/'selection.json')]
    mcp = {'openeta-setup':{'command':str(ROOT/'.venv/bin/python'),'args':['-m','tools.harness_setup'],
        'env':{'PYTHONPATH':str(ROOT),'PYTHONDONTWRITEBYTECODE':'1','OPENETA_SETUP_ARGS':json.dumps(setup_args)},
        'startup_timeout_sec':30,'tool_timeout_sec':30}}
    (market/'plugins/openeta-setup/.mcp.json').write_text(json.dumps(mcp,indent=2)+'\n')
    config = (f'model = {json.dumps(args.model)}\nmodel_reasoning_effort = {json.dumps(args.effort)}\n'
        'model_provider = "openai"\nforced_login_method = "chatgpt"\nmcp_optional_startup_grace_ms = 0\n'
        'web_search = "disabled"\n[features]\nmemories = false\nshell_tool = false\nmulti_agent = false\n'
        '[history]\npersistence = "none"\n'
        '[plugins."openeta-setup@openeta-codex-smoke".mcp_servers.openeta-setup]\n'
        'enabled = true\ndefault_tools_approval_mode = "approve"\n')
    if args.model_catalog_json:
        models = json.loads(Path(args.model_catalog_json).read_text())
        if not any(m.get('slug') == args.model for m in models.get('models',[]) if isinstance(m,dict)):
            raise RuntimeError('Pinned model catalog does not contain the selected model')
        target = home/'model-catalog.json'
        target.write_text(json.dumps(models)+'\n')
        config = f'model_catalog_json = {json.dumps(str(target))}\n' + config
    (home/'config.toml').write_text(config)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('OPENAI_','OPENETA_LLM_','CODEX_'))}
    env.update(CODEX_HOME=str(home), PYTHONDONTWRITEBYTECODE='1')
    with (output/'plugin-install.log').open('w') as log:
        for command in ([args.codex,'plugin','marketplace','add',str(market)],
                        [args.codex,'plugin','add','openeta-setup@openeta-codex-smoke','--json']):
            subprocess.run(command,env=env,cwd=work,stdout=log,stderr=log,timeout=60,check=True)
    return home, work, env


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Preparation output must be a new directory')
    registry = ConfigRegistry()
    if not registry.catalog(env_id=args.env_id):
        raise ValueError('No registered configuration matches this environment')
    processes, owned = [], {}
    start = time.monotonic()
    summary = {'phase':'preparation','model':args.model,'effort':args.effort,
               'authentication':'chatgpt','status':'preparing','integration_passed':False}
    try:
        _,work,env = prepare(args)
        semantics = (ROOT/'plugins/openeta-setup/skills/openeta-configure/SKILL.md').read_text().split('---',2)[2].strip()
        prompt = (f'Assigned environment: {args.env_id}\nTask: {args.task}\nSeed: {args.seed}\n\n'+semantics+
            '\n\nUse only the two OpenETA configuration tools. They may be exposed through functions.exec; '
            'you may discover/invoke them via tools/ALL_TOOLS. Do not use shell, filesystem, web or other tools. '
            'After a successful selection, report the ID and finish; do not attempt to operate a robot.')
        (output/'prompt.txt').write_text(prompt+'\n')
        command = [args.codex,'exec','-C',str(work),'--skip-git-repo-check','--ephemeral','--json',
                   '-s','read-only','-m',args.model,'-o',str(output/'final.txt'),prompt]
        with (output/'codex-events.jsonl').open('w') as stdout, (output/'codex-stderr.log').open('w') as stderr:
            process = subprocess.Popen(command,env=env,cwd=work,stdin=subprocess.DEVNULL,
                                       stdout=stdout,stderr=stderr,start_new_session=True)
            processes.append(process)
            deadline = time.monotonic()+args.timeout
            while process.poll() is None and time.monotonic() < deadline:
                ownership.track([process.pid], owned)
                try:
                    process.wait(timeout=min(2.,max(.01,deadline-time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            summary.update(returncode=process.poll(), status='completed' if process.poll()==0 else
                           'launcher_timeout' if process.poll() is None else 'codex_failed')
        summary['non_mcp_actions'] = []
        for line in (output/'codex-events.jsonl').read_text().splitlines():
            event = json.loads(line)
            item = event.get('item') or {}
            if item.get('type') in {'command_execution','web_search','file_change'}:
                summary['non_mcp_actions'].append(item['type'])
        config = read_selection(output/'selection.json',env_id=args.env_id,task=args.task,seed=args.seed)
        summary['selected_config'] = {'id':config.id,'sha256':config.sha256}
        audit = output/'selection.mcp.jsonl'
        calls = [json.loads(line) for line in audit.read_text().splitlines()] if audit.exists() else []
        summary['mcp_calls'] = calls
        summary['selection_verified'] = bool(summary['status']=='completed' and not summary['non_mcp_actions']
            and any(c['tool']=='list_harness_configs' and c['code']=='ok' for c in calls)
            and any(c['tool']=='select_harness_config' and c['code']=='ok' for c in calls))
    except Exception as exc:
        summary.update(status='preparation_failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        summary['remaining_owned_pids'] = ownership.cleanup(processes, owned)
        (output/'codex-home/auth.json').unlink(missing_ok=True)
        summary['private_auth_removed'] = not (output/'codex-home/auth.json').exists()
        summary['elapsed_s'] = round(time.monotonic()-start,3)
        summary['integration_passed'] = bool(summary.get('selection_verified') and not summary['remaining_owned_pids']
                                             and summary['private_auth_removed'])
        output.mkdir(parents=True,exist_ok=True)
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return 0 if summary['integration_passed'] else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-id',required=True)
    p.add_argument('--task',required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--effort',default='high')
    p.add_argument('--model-catalog-json',type=Path)
    p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--codex',default='codex')
    return p


if __name__ == '__main__':
    raise SystemExit(run(parser().parse_args()))
