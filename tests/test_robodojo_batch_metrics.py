import json
from scripts.robodojo_batch_metrics import read_usage


def log(tmp_path,events):
 p=tmp_path/'events.jsonl';p.write_text('\n'.join(json.dumps(e) for e in events)+'\n');return p


def turn(inp=100,out=10):
 return [{'type':'turn.started'},{'type':'turn.completed','usage':{'input_tokens':inp,'output_tokens':out,'cached_input_tokens':80,'reasoning_output_tokens':3}}]


def test_input_cache_and_output_reasoning_not_double_counted(tmp_path):
 r=read_usage(log(tmp_path,turn()));assert r['complete'];assert r['observed_usage']['input_tokens']==100;assert r['observed_usage']['output_tokens']==10
 assert r['observed_usage']['cache_write_input_tokens'] is None


def test_multiple_completed_turns_sum_once(tmp_path):
 r=read_usage(log(tmp_path,turn()+turn(200,20)));assert r['complete'];assert r['observed_usage']['input_tokens']==300


def test_failure_without_usage_is_missing_not_zero(tmp_path):
 r=read_usage(log(tmp_path,[{'type':'turn.started'},{'type':'turn.failed'}]));assert not r['available'];assert r['observed_usage']['input_tokens'] is None


def test_partial_usage_not_a_complete_total(tmp_path):
 r=read_usage(log(tmp_path,turn()+[{'type':'turn.started'},{'type':'turn.failed'}]));assert r['available'] and not r['complete'];assert r['observed_usage']['input_tokens']==100


def test_duplicate_completion_does_not_inflate(tmp_path):
 r=read_usage(log(tmp_path,turn()+[turn()[-1]]));assert not r['complete'];assert r['observed_usage']['input_tokens']==100


def test_bad_lines_or_negative_values_are_not_totals(tmp_path):
 p=log(tmp_path,turn());p.write_text(p.read_text()+'truncated');assert not read_usage(p)['complete']
 assert not read_usage(log(tmp_path,turn(-1)))['available']
