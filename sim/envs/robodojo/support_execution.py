"""Lossless pause of task-owned commands at the confirmed physics boundary.

A lease spans one native action, including its reward/phase callbacks. The
native planner and trajectory values remain authoritative. No actor trajectory
or task labels are included in feedback. This is not collision admission.
"""
from contextlib import contextmanager
from copy import deepcopy

import numpy as np


def _same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return bool(np.array_equal(a, b))


class SupportExecution:
    def __init__(self, native, ownership):
        if len(ownership.support) != 1:
            raise ValueError('Native support queue currently requires exactly one task-owned actor')
        self.native = native
        self.keys = ownership.support_keys(native.robot_manager)
        self.active = False
        self.invalid = False
        self.last_feedback = None

    def _source(self):
        source = self.native.support_arm_action
        if not isinstance(source, list) or len(source) != 1 or not isinstance(source[0], list):
            raise ValueError('Support execution requires one owned environment queue')
        return source[0]

    @contextmanager
    def action(self):
        if self.active or self.invalid:
            raise RuntimeError('Support execution is active or uncertain; reset before retry')
        native = self.native
        manager = native.robot_manager.control_manager
        if len(manager.control_queue) != 1 or not manager.control_queue[0].is_empty():
            raise RuntimeError('Support lease requires an empty owned native control queue')
        self._source()
        self.active = True
        self.reserved = []
        self.sequence_length = None
        self.committed = 0
        self.popped = 0
        self.pending_previous = None
        self.in_physics = False
        self.paused = False
        self.last_feedback = None
        original_process = native.process_control_info
        original_pop = manager.pop
        process_override = vars(native).get('process_control_info')
        pop_override = vars(manager).get('pop')
        had_process = 'process_control_info' in vars(native)
        had_pop = 'pop' in vars(manager)

        def process(info, env_idx):
            if env_idx != 0 or self.sequence_length is not None:
                raise RuntimeError('Support lease permits one native sequence in environment zero')
            if set(info).intersection(self.keys):
                raise ValueError('Agent action may not command task-owned support actors')
            before = deepcopy(self._source())
            try:
                sequence = original_process(info, env_idx)
                remaining = self._source()
                count = len(before) - len(remaining)
                if (not isinstance(sequence, list) or not sequence or count < 0
                        or count > len(sequence) or not _same(remaining, before[count:])):
                    raise RuntimeError('Native support reservation contract changed')
                for i, entry in enumerate(sequence):
                    actual = {k: entry[k] for k in self.keys if k in entry}
                    expected = before[i] if i < count else {}
                    if (set(expected) - set(self.keys)) or not _same(actual, expected):
                        raise RuntimeError('Native sequence changed or omitted a support command')
                self.reserved = deepcopy(before[:count])
                self.sequence_length = len(sequence)
                return sequence
            except Exception:
                # No sequence has been returned/dispatched: consumption is
                # known to be zero even if native expansion raised midway.
                self._source()[:] = before
                raise

        def pop(env_idx_list=None):
            if env_idx_list not in (None, [0]):
                raise ValueError('Support lease may only pop owned environment zero')
            if (self.paused or self.in_physics or self.pending_previous is not None
                    or self.sequence_length is None or self.popped >= self.sequence_length):
                raise RuntimeError('Unconfirmed native support step or unexpected queue pop')
            previous = manager.prev_control[0]
            self.pending_previous = {k: deepcopy(previous[k]) for k in self.keys if k in previous}
            result = original_pop(env_idx_list)
            self.popped += 1
            return result

        native.process_control_info = process
        manager.pop = pop
        try:
            yield self
            if not self.paused and self.sequence_length is not None:
                if self.committed != self.sequence_length or self.pending_previous is not None:
                    raise RuntimeError('Native action returned with unconfirmed support steps')
        except Exception:
            if self.in_physics:
                self.uncertain()
            elif not self.invalid:
                self.pause()
            raise
        finally:
            if had_process:
                native.process_control_info = process_override
            else:
                del native.process_control_info
            if had_pop:
                manager.pop = pop_override
            else:
                del manager.pop
            self.active = False

    def begin_physics(self):
        if (not self.active or self.paused or self.invalid or self.in_physics
                or self.pending_previous is None or self.popped != self.committed + 1):
            raise RuntimeError('Physics step has no exclusive native support reservation')
        self.in_physics = True

    def commit_physics(self):
        if not self.in_physics:
            raise RuntimeError('Cannot commit a support frame without physics completion')
        self.in_physics = False
        self.committed += 1
        self.pending_previous = None

    def uncertain(self):
        self.invalid = True
        self.in_physics = False
        self.last_feedback = {'status': 'unknown', 'reason': 'support_physics_outcome_unknown',
                              'automatic_replay': False, 'reset_required': True}
        return dict(self.last_feedback)

    def pause(self):
        if self.invalid:
            return dict(self.last_feedback)
        if not self.active:
            return None
        if self.in_physics:
            return self.uncertain()
        if self.paused:
            return dict(self.last_feedback)
        remaining = deepcopy(self.reserved[self.committed:])
        self._source()[:0] = remaining
        if self.pending_previous is not None:
            # ControlManager.pop updates its hold cache before sim_step. Undo
            # only task-owned cache keys for a known-unexecuted current frame.
            previous = self.native.robot_manager.control_manager.prev_control[0]
            for key in self.keys:
                if key in self.pending_previous:
                    previous[key] = deepcopy(self.pending_previous[key])
                else:
                    previous.pop(key, None)
        self.paused = True
        self.last_feedback = {'status': 'paused', 'reason': 'host_stopped_owned_physics',
            'unexecuted_frames_restored': len(remaining),
            'includes_popped_unexecuted_frame': self.pending_previous is not None,
            'trajectory_values_and_order': 'unchanged', 'support_drive_targets': 'not_rewritten_by_host',
            'automatic_resume': False, 'resume_on': 'next_accepted_native_action'}
        return dict(self.last_feedback)
