"""Write-once public sensor evidence. Incomplete calls must never be replayed."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path


def write_once(path: Path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class CallEvidence:
    def __init__(self, output, index, tool, arguments, config_sha256, packet):
        self.call_id = f'{index:06d}'
        self.path = Path(output) / 'calls' / self.call_id
        self.path.mkdir(parents=True, exist_ok=False)
        write_once(self.path / 'request.json', {
            'schema': 'openeta.call_evidence.v1', 'call_id': self.call_id,
            'tool': tool, 'arguments': arguments, 'config_sha256': config_sha256,
            'before_action_index': (packet or {}).get('action_index'),
            'recovery': 'Missing response means incomplete evidence; never replay automatically.',
        })
        self.snapshot('before', packet)

    def snapshot(self, label, packet):
        if packet is None:
            return
        value = copy.deepcopy(packet)
        # Only public sensor packets enter this journal; never archive object annotations.
        if value.get('observation', {}).get('objects'):
            raise ValueError('Privileged object annotations cannot enter call evidence')
        frames = value.get('observation', {}).get('cameras', [])
        for index, frame in enumerate(frames):
            for channel in ('rgb', 'depth'):
                encoded = frame.pop(channel + '_base64', None)
                if not encoded:
                    continue
                data = base64.b64decode(encoded, validate=True)
                # Camera names are metadata, never filesystem paths.
                filename = f'{label}-{index:03d}-{channel}.png'
                with (self.path / filename).open('xb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                frame[channel + '_artifact'] = {
                    'path': filename, 'sha256': hashlib.sha256(data).hexdigest(),
                    'bytes': len(data),
                }
        write_once(self.path / f'{label}-observation.json', value)

    def complete(self, result, packet, *, received_observation):
        if received_observation:
            self.snapshot('after', packet)
        # Archive exact returned images too, including derived overlays and crops.
        blocks = []
        for index, block in enumerate(result):
            if block.type == 'image':
                data = base64.b64decode(block.data, validate=True)
                suffix = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp'}.get(block.mimeType, '.bin')
                filename = f'response-image-{index:03d}{suffix}'
                with (self.path / filename).open('xb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                blocks.append({'type': 'image', 'mimeType': block.mimeType, 'path': filename,
                               'sha256': hashlib.sha256(data).hexdigest()})
            else:
                blocks.append(block.model_dump(exclude_none=True))
        write_once(self.path / 'response.json', {'call_id': self.call_id, 'content': blocks})
