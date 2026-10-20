"""Bounded, frozen historical visual/action evidence; never executable replay."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import jsonschema
from PIL import Image
from mcp.types import ImageContent, TextContent

PROTOCOL = 'cross_attempt_visual_action_experience'
SCHEMA = 'openeta.visual_action_experience.v1'
MAX_BYTES = 48_000_000


def obj(properties, required=None):
    return {'type':'object', 'properties':properties, 'required':list(properties if required is None else required), 'additionalProperties':False}


TEXT = {'type':'string','minLength':1,'maxLength':1500}
NUM = {'type':'number'}
INT = {'type':'integer','minimum':0}
HASH = {'type':'string','pattern':'^[0-9a-f]{64}$'}
ROLE = {'type':'string','enum':['primary','left','right']}
VEC = {'type':'array','items':NUM,'minItems':3,'maxItems':3}
TARGET = obj({'xyz':VEC,'delta_xyz':VEC,'frame':{'enum':['world','tcp']},'approach':VEC,'jaw':VEC,'point_in_tcp':VEC}, [])
TARGET['oneOf'] = [{'required':['xyz'],'not':{'required':['delta_xyz']}}, {'required':['delta_xyz'],'not':{'required':['xyz']}}]
MOTION = obj({'role':ROLE,'target':TARGET,'waypoints':{'type':'array','items':TARGET,'maxItems':4}}, ['role','target'])
ARGS = obj({**MOTION['properties'],'motions':{'type':'array','items':MOTION,'minItems':1,'maxItems':2},
            'coordination':{'enum':['independent','synchronized']},'max_duration_s':{'type':'number','exclusiveMinimum':0,'maximum':2},
            'openness':{'type':'number','minimum':0,'maximum':1},'duration_s':{'type':'number','exclusiveMinimum':0,'maximum':2}}, [])
ARM = obj({'role':ROLE,'tcp_xyz_m':VEC,'tcp_approach':VEC,'tcp_jaw':VEC,
           'joint_positions_rad':{'type':'array','items':NUM,'maxItems':32},
           'gripper_openness':{'type':['number','null'],'minimum':0,'maximum':1},
           'commanded_openness':{'type':['number','null'],'minimum':0,'maximum':1}})
IMAGE = obj({'path':TEXT,'sha256':HASH,'camera':TEXT})
STATE = obj({'action_index':INT,'simulation_time_s':{'type':['number','null'],'minimum':0},
             'robot':{'type':'array','items':ARM,'minItems':1,'maxItems':2},
             'images':{'type':'array','items':IMAGE,'minItems':1,'maxItems':3}})
FEEDBACK = obj({'native_actions':INT,'physics_steps':INT,'reached_target':{'type':['boolean','null']},
                'stop_reason':{'type':['string','null'],'maxLength':160},'error_code':{'type':['string','null'],'maxLength':160}})
REVIEW = obj({'kind':{'enum':['unreviewed','agent_report','human_visual_review']},'text':{'type':'string','maxLength':1500}})
SEGMENT = obj({'id':{'type':'integer','minimum':1,'maximum':12},'call_id':{'type':'string','pattern':'^[0-9]{6}$'},
               'tool':{'enum':['move_to','gripper_control','wait']},'arguments':ARGS,'before':STATE,'after':STATE,
               'feedback':FEEDBACK})
MOVE_ARGS = obj({k: ARGS['properties'][k] for k in ('role','target','waypoints','motions','coordination','max_duration_s')}, [])
MOVE_ARGS['oneOf'] = [
    {'required':['role','target'],'not':{'required':['motions']}},
    {'required':['motions'],'not':{'anyOf':[{'required':[k]} for k in ('role','target','waypoints')]}}]
SEGMENT['allOf'] = [
    {'if':{'properties':{'tool':{'const':tool}}},'then':{'properties':{'arguments':schema}}}
    for tool,schema in [
        ('move_to',MOVE_ARGS),
        ('gripper_control',obj({k:ARGS['properties'][k] for k in ('role','openness','max_duration_s')}, ['role','openness'])),
        ('wait',obj({'duration_s':ARGS['properties']['duration_s']}))]]


MANIFEST = obj({'schema':{'const':SCHEMA},'public_only_reviewed':{'const':True},'label':TEXT,'review':REVIEW,
    'source':obj({'episode_id':TEXT,'env_id':TEXT,'seed':INT,'task':TEXT,'config_sha256':HASH,
                  'embodiment_sha256':HASH,'official_success_recorded':{'type':'boolean'}}),
    'segments':{'type':'array','items':SEGMENT,'minItems':1,'maxItems':12},
    'provenance':{'type':'array','minItems':1,'maxItems':160,'items':obj({'path':TEXT,'sha256':HASH})}})


def digest(data):
    return hashlib.sha256(data).hexdigest()


def embodiment_digest(embodiment):
    return digest(json.dumps(embodiment,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def contained(root, relative):
    rel = Path(relative)
    target = (root/rel).resolve()
    if rel.is_absolute() or '..' in rel.parts or not target.is_relative_to(root.resolve()):
        raise ValueError('Experience file escapes its bundle')
    return target


class ExperienceLibrary:
    def __init__(self, path, expected_sha256, *, embodiment=None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        if len(raw)>192_000 or not re.fullmatch(r'[0-9a-f]{64}',expected_sha256 or '') or digest(raw)!=expected_sha256:
            raise ValueError('Experience manifest size/hash mismatch')
        value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite experience value')))
        # allow_nan=False also catches 1e999 parsed as infinity.
        json.dumps(value,allow_nan=False)
        jsonschema.validate(value,MANIFEST)
        if embodiment is not None and value['source']['embodiment_sha256']!=embodiment_digest(embodiment):
            raise ValueError('Experience embodiment/calibration mismatch')
        if [s['id'] for s in value['segments']]!=list(range(1,len(value['segments'])+1)):
            raise ValueError('Experience segment IDs must be sequential')
        if len({s['call_id'] for s in value['segments']})!=len(value['segments']):
            raise ValueError('Duplicate experience call')
        self.assets = {}; total = 0
        previous_action = -1
        for segment in value['segments']:
            for phase in ('before','after'):
                state=segment[phase]
                roles=[x['role'] for x in state['robot']]
                if len(roles)!=len(set(roles)):
                    raise ValueError('Duplicate historical role')
                if embodiment is not None and set(roles)!=set(embodiment['effectors']):
                    raise ValueError('Historical role set differs from embodiment')
                for entry in state['images']:
                    relative=entry['path']; data=contained(path.parent,relative).read_bytes()
                    if len(data)>8_000_000 or digest(data)!=entry['sha256']:
                        raise ValueError('Experience image size/hash mismatch')
                    with Image.open(BytesIO(data)) as im:
                        if im.format!='PNG' or im.mode not in ('RGB','RGBA') or max(im.size)>4096 or getattr(im,'n_frames',1)!=1:
                            raise ValueError('Experience requires bounded public RGB PNG images')
                        im.verify()
                    if relative not in self.assets:total+=len(data)
                    if total>MAX_BYTES:raise ValueError('Experience image budget exceeded')
                    self.assets[relative]=data
            before,after=segment['before'],segment['after']
            state_roles={x['role'] for x in before['robot']}
            args=segment['arguments']
            requested_roles=([m['role'] for m in args['motions']] if 'motions' in args else [args['role']] if 'role' in args else [])
            if (set(requested_roles)-state_roles or len(requested_roles)!=len(set(requested_roles))
                    or {x['role'] for x in after['robot']}!=state_roles):
                raise ValueError('Historical action role differs from measured robot')
            if before['action_index']<previous_action or after['action_index']-before['action_index']!=segment['feedback']['native_actions']:
                raise ValueError('Historical action clock inconsistent')
            previous_action=after['action_index']
            if before['simulation_time_s'] is not None and after['simulation_time_s'] is not None and after['simulation_time_s']<before['simulation_time_s']:
                raise ValueError('Historical simulation clock reversed')
        self.value=value;self.raw=raw;self.sha256=expected_sha256;self.delivered=set()

    def freeze(self, destination):
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
        for relative,data in self.assets.items():
            target=contained(destination,relative);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(data)
        path=destination/'manifest.json'
        with path.open('xb') as f:f.write(self.raw)
        return path

    def record(self):
        return {'manifest_sha256':self.sha256,'evaluation_protocol':PROTOCOL,'independent_pass_at_k':False,
                'public_only_review':'operator_attested_plus_structured_allowlist; not_machine_semantic_verification',
                'source':deepcopy(self.value['source']),'segment_count':len(self.value['segments']),
                'image_count':len(self.assets),'image_bytes':sum(map(len,self.assets.values()))}

    def read(self, segment_id=None):
        receipt={'kind':'historical_visual_action_experience','historical':True,'physics_advanced':False,
                 'native_actions':0,'consumes_tool_request':True,'capabilities_changed':False,
                 'manifest_sha256':self.sha256,'source':deepcopy(self.value['source']),
                 'label':self.value['label'],'clip_review':deepcopy(self.value['review']),
                 'conventions':{'position':'world metres','axes':'calibrated TCP +Z approach, +X jaw',
                     'joint_positions':'arm radians','openness':'0 closed, 1 open; not grasp confirmation',
                     'time':'native observation clock in seconds; not network/wall time'},
                 'scope':'Historical observations and recorded requests, not current evidence or executable replay. Reobserve and make fresh marks. Do not reuse old coordinates or assume episode/clip success transfers.'}
        if segment_id is None:
            receipt['segments']=[{'id':s['id'],'call_id':s['call_id'],'tool':s['tool'],
                'action_range':[s['before']['action_index'],s['after']['action_index']],
                'feedback':deepcopy(s['feedback'])} for s in self.value['segments']]
            return receipt,[]
        if type(segment_id)!=int or not 1<=segment_id<=len(self.value['segments']):
            raise ValueError('Unknown experience segment')
        if segment_id in self.delivered:
            receipt.update(segment_id=segment_id,delivery='already_delivered; refer to earlier historical tool result')
            return receipt,[]
        segment=deepcopy(self.value['segments'][segment_id-1]);blocks=[]
        for phase in ('before','after'):
            references=[]
            for entry in segment[phase]['images']:
                label=f'HISTORICAL segment {segment_id} {phase} {entry["camera"]}; NOT current episode'
                blocks.extend([TextContent(type='text',text=label),ImageContent(type='image',data=base64.b64encode(self.assets[entry['path']]).decode(),mimeType='image/png')])
                references.append({'label':label,'sha256':entry['sha256']})
            segment[phase]['images']=references
        receipt.update(segment=segment,delivery='historical_images_delivered')
        self.delivered.add(segment_id)
        return receipt,blocks
