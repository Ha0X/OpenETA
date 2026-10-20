import json
from pathlib import Path

import pytest

from scripts.robodojo_batch_acceptance import INVENTORY, OwnedBootstrap, PROFILE
from tools.harness_config import ConfigError, ConfigRegistry


def test_exact_candidate_scope_preserves_unsupported_boundaries():
    rows=json.loads(INVENTORY.read_text())['tasks']
    assert len(rows)==42
    candidates=[r for r in rows if r['candidate']]
    assert len(candidates)==32
    c=ConfigRegistry().resolve(PROFILE)
    c.require_executable()
    for row in rows:
        env=f"openeta/robodojo_dual_x5_{row['task']}-v0"
        if row['candidate'] and not row.get('admission_blocker'):
            c.bind_task(env_id=env,task='Official instruction',seed=0)
        else:
            with pytest.raises(ConfigError):c.bind_task(env_id=env,task='Official instruction',seed=0)
    assert ConfigRegistry().resolve('libero_panda_v1').sha256=='a386669816d589be328f0890f1ccd27e9d4098e48aacc4a909e4207fb2defb52'


def test_bootstrap_adopts_exact_owned_receipts_once(tmp_path):
    class Transport:
        def __init__(self):self.calls=[]
        def call_tool(self,name,args,timeout_s=None):
            self.calls.append((name,args))
            if name=='create_env':return {'handle':'owned','session_id':'ours'}
            if name=='reset_env':return {'task':'Native instruction','robot':{},'cameras':[]}
            return {'ok':True}
    remote=Transport();t=OwnedBootstrap(remote,tmp_path)
    obs=t.bootstrap('env',0)
    assert len(remote.calls)==3
    assert t.call_tool('create_env',{'env_id':'env','seed':0,'include_objects':False})['handle']=='owned'
    assert t.call_tool('reset_env',{'handle':'owned','session_id':'ours','seed':0}) is obs[-1]
    assert len(remote.calls)==3
    t.call_tool('close_env',t.owned)
    assert t.cleanup['ok']
    assert remote.calls[-1][1]=={'handle':'owned','session_id':'ours'}


def test_bootstrap_cannot_adopt_a_different_seed(tmp_path):
    t=OwnedBootstrap(None,tmp_path)
    t.cached={'reset_env':({'handle':'owned','seed':0},{'task':'Native'})}
    with pytest.raises(ValueError,match='differ'):
        t.call_tool('reset_env',{'handle':'owned','seed':1})


def test_support_robot_cannot_be_silently_removed():
    from sim.envs.robodojo.task_contract import validate_task_robot
    settings={'common':{'robot_config':'dual_x5'},'tasks':{'opponent':{'robot_config':'dual_x5_and_franka_competition'}}}
    assert validate_task_robot('ordinary',settings,'dual_x5')=='dual_x5'
    with pytest.raises(ValueError,match='support-robot'):
        validate_task_robot('opponent',settings,'dual_x5')
    with pytest.raises(ValueError,match='support-robot'):
        validate_task_robot('unknown',{},'dual_x5')


def test_lost_create_reply_does_not_mean_no_environment_exists(tmp_path):
    class LostReply:
        def call_tool(self,*args,**kwargs):raise TimeoutError('reply lost')
    transport=OwnedBootstrap(LostReply(),tmp_path)
    with pytest.raises(TimeoutError):transport.bootstrap('env',0)
    assert transport.unknown_create
    assert not transport.owned
