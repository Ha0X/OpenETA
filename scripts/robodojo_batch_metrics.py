"""Report CLI usage and Host counts without estimating missing token usage."""
import json
from pathlib import Path

FIELDS = ('input_tokens','cached_input_tokens','cache_write_input_tokens',
          'output_tokens','reasoning_output_tokens')


def read_usage(path):
    path=Path(path)
    result={'source':'codex_exec_jsonl.turn.completed.usage','available':False,
            'complete':False,'completed_turns':0,'started_turns':0,'invalid_lines':0,
            'observed_usage':{k:None for k in FIELDS}}
    if not path.exists():return result
    reports=[];last=None;active=False;duplicate=False;missing=False
    for line in path.read_text().splitlines():
        if not line.strip():continue
        try:event=json.loads(line)
        except (ValueError,TypeError):result['invalid_lines']+=1;continue
        if not isinstance(event,dict):result['invalid_lines']+=1;continue
        last=event.get('type')
        if last=='turn.started':result['started_turns']+=1;active=True
        if last!='turn.completed':continue
        if not active:duplicate=True;continue
        active=False;result['completed_turns']+=1
        usage=event.get('usage')
        if not isinstance(usage,dict) or any(type(usage.get(k)) is not int or usage[k]<0 for k in ('input_tokens','output_tokens')):
            missing=True;continue
        if any(k in usage and (type(usage[k]) is not int or usage[k]<0) for k in FIELDS):missing=True;continue
        reports.append(usage)
    result['available']=bool(reports)
    for k in FIELDS:
        if reports and all(k in r for r in reports):result['observed_usage'][k]=sum(r[k] for r in reports)
    result['complete']=bool(reports and not missing and not duplicate and not active and not result['invalid_lines']
                            and last=='turn.completed' and result['completed_turns']==result['started_turns'])
    result['note']='Input includes cached tokens. Output and reasoning are reported separately, never added together. Incomplete usage is retained as observed partial data, not a complete total.'
    return result


def trial_metrics(run):
    run=Path(run);summary=json.loads((run/'summary.json').read_text());host=summary.get('host') or {};outcome=host.get('outcome') or {}
    usage=read_usage(run/'codex-events.jsonl')
    requests=run/'host/requests.jsonl';skills=[]
    if requests.exists():
        for line in requests.read_text().splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            if r.get('tool')=='read_skill':skills.append({'request':r.get('index'),'arguments':r.get('arguments'),'error':r.get('error')})
    return {'official_success':bool(outcome.get('source')=='robodojo.eval_env.terminal_flags' and outcome.get('terminal') is True and outcome.get('benchmark_success') is True),
            'outcome':outcome,'launcher_status':summary.get('status'),'elapsed_s':summary.get('elapsed_s'),
            'tool_calls':host.get('tool_calls'),'native_actions':outcome.get('action_index'),
            'usage':usage,'input_tokens':usage['observed_usage']['input_tokens'] if usage['complete'] else None,
            'output_tokens':usage['observed_usage']['output_tokens'] if usage['complete'] else None,
            'skills_read':skills,'conditioning':summary.get('experiment_factors',{}).get('conditioning'),
            'initialization':host.get('initialization'),'cleanup':host.get('cleanup') or summary.get('launcher_cleanup')}
