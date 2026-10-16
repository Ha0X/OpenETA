"""Private, measured cloth surfaces and material-local contact tracking.

These helpers do not enable a task, create an attachment, change physics, or
predict deformation. Vertex identities stay private; callers may publish only
aggregate contact diagnostics. A fresh immutable surface replaces each frame.
"""
from dataclasses import dataclass
import heapq
import math

import numpy as np


class DeformableCoverageError(ValueError):
    """A measured surface is unavailable; it must not become empty space."""


def _points(value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 3 or not np.isfinite(value).all():
        raise DeformableCoverageError('Invalid live cloth vertices')
    return value


def _triangles(value, count):
    raw = np.asarray(value)
    if (raw.ndim != 2 or raw.shape[1] != 3 or not len(raw)
            or raw.dtype.kind not in 'iu' or np.any(raw < 0) or np.any(raw >= count)):
        raise DeformableCoverageError('Invalid cloth triangle topology')
    return raw.astype(np.intp, copy=True)


@dataclass(frozen=True)
class SurfaceFrame:
    material_id: object
    vertices: np.ndarray
    triangles: np.ndarray
    revision: int
    simulation_time_s: float
    contact_offset_m: float


class MaterialSurface:
    """Stable material topology with atomic updates, never cached verdicts."""

    def __init__(self, vertices, triangles, *, contact_offset_m, simulation_time_s=0.):
        points = _points(vertices)
        faces = _triangles(triangles, len(points))
        if not math.isfinite(contact_offset_m) or not 0 < contact_offset_m <= .1:
            raise DeformableCoverageError('Invalid native cloth contact offset')
        self.material_id = object()
        self.contact_offset_m = float(contact_offset_m)
        self.triangles = faces
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        self.edges = np.unique(np.sort(edges, axis=1), axis=0)
        self.reference_lengths = np.linalg.norm(points[self.edges[:, 0]]-points[self.edges[:, 1]], axis=1)
        if np.any(self.reference_lengths <= 1e-9):
            raise DeformableCoverageError('Degenerate reference material edges')
        self.adjacency = [[] for _ in points]
        for (a, b), length in zip(self.edges, self.reference_lengths):
            self.adjacency[a].append((int(b), float(length)))
            self.adjacency[b].append((int(a), float(length)))
        self.frame = None
        self.update(points, simulation_time_s=simulation_time_s)
        self.triangles.setflags(write=False)

    def update(self, vertices, *, simulation_time_s, triangles=None):
        points = _points(vertices)
        if len(points) != len(self.adjacency):
            raise DeformableCoverageError('Cloth vertex identities changed')
        if triangles is not None and not np.array_equal(_triangles(triangles, len(points)), self.triangles):
            raise DeformableCoverageError('Cloth topology changed; contact references are invalid')
        if (not math.isfinite(simulation_time_s) or simulation_time_s < 0
                or self.frame is not None and simulation_time_s < self.frame.simulation_time_s):
            raise DeformableCoverageError('Cloth clock regressed; reset requires a new material surface')
        if self.frame is not None and simulation_time_s == self.frame.simulation_time_s:
            if not np.array_equal(points, self.frame.vertices):
                raise DeformableCoverageError('Cloth changed without a physics timestamp advance')
            return self.frame
        points = points.copy()
        points.setflags(write=False)
        frame = SurfaceFrame(self.material_id, points, self.triangles, 0 if self.frame is None else self.frame.revision+1,
                             float(simulation_time_s), self.contact_offset_m)
        self.frame = frame
        return frame

    def material_patch(self, point, *, radius_m=.025, surface_tolerance_m=.011):
        """Select connected material, not every overlapping fold near a pixel.

        Ambiguous nearby layers are refused. Radius uses reference material
        edge lengths; a later fold cannot silently authorize a second layer.
        """
        point = np.asarray(point, dtype=float)
        if (point.shape != (3,) or not np.isfinite(point).all()
                or not math.isfinite(radius_m) or not 0 < radius_m <= .05
                or not math.isfinite(surface_tolerance_m) or not 0 < surface_tolerance_m <= .02):
            raise ValueError('Invalid marked cloth patch')
        distances = np.linalg.norm(self.frame.vertices-point, axis=1)
        seed = int(np.argmin(distances))
        if distances[seed] > surface_tolerance_m:
            raise DeformableCoverageError('Marked pixel is not near measured cloth material')
        reached = {seed: 0.}
        pending = [(0., seed)]
        while pending:
            distance, vertex = heapq.heappop(pending)
            if distance != reached[vertex]:
                continue
            for other, length in self.adjacency[vertex]:
                candidate = distance+length
                if candidate <= radius_m and candidate < reached.get(other, math.inf):
                    reached[other] = candidate
                    heapq.heappush(pending, (candidate, other))
        ids = np.array(sorted(reached), dtype=np.intp)
        # Do not pick an invisible material layer merely because its vertex
        # wins a submillimetre nearest-neighbour tie after folding.
        near = np.flatnonzero(distances <= min(surface_tolerance_m, distances[seed]+.002))
        if any(int(index) not in reached for index in near):
            raise DeformableCoverageError('Marked cloth layers are ambiguous; select an exposed edge')
        if len(ids) < 3:
            raise DeformableCoverageError('Insufficient material samples for a local contact patch')
        ids.setflags(write=False)
        return MaterialPatch(self.material_id, ids, self.frame.revision, radius_m)

    def local_layers(self, point, approach, *, radius_m=.05, half_depth_m=.012):
        """Explicit local stack intent; freeze current material, not a moving volume.

        The cylinder is centred on the measured surface nearest the public
        mark and oriented by the proposed TCP approach. Other cloth arriving
        later is never added. Neither layer identities nor geometry are public.
        """
        point, axis = np.asarray(point, dtype=float), np.asarray(approach, dtype=float)
        if (point.shape != (3,) or axis.shape != (3,) or not np.isfinite(point).all()
                or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1.)
                or not 0 < radius_m <= .05 or not 0 < half_depth_m <= .012):
            raise ValueError('Invalid explicit local-layer contact volume')
        p = self.frame.vertices
        seed = int(np.argmin(np.linalg.norm(p-point, axis=1)))
        if np.linalg.norm(p[seed]-point) > .011:
            raise DeformableCoverageError('Marked pixel is not near measured cloth material')
        delta = p-p[seed]; depth = delta@axis
        lateral = np.linalg.norm(delta-depth[:, None]*axis, axis=1)
        ids = np.flatnonzero((lateral <= radius_m) & (abs(depth) <= half_depth_m))
        faces = self.triangles[np.isin(self.triangles, ids).all(axis=1)]
        if not len(faces):
            raise DeformableCoverageError('Insufficient local-layer surface coverage')
        triangles = p[faces]
        area = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                                      triangles[:,2]-triangles[:,0]), axis=1).sum()/2
        if area > .025:
            raise DeformableCoverageError('Too much folded material in local-layer contact volume')
        ids.setflags(write=False)
        return MaterialPatch(self.material_id, ids, self.frame.revision, radius_m)


@dataclass(frozen=True)
class MaterialPatch:
    material_id: object
    vertex_ids: np.ndarray
    source_revision: int
    radius_m: float

    def relative_points(self, frame, wrist):
        wrist = np.asarray(wrist, dtype=float)
        if (wrist.shape != (4, 4) or not np.isfinite(wrist).all()
                or not np.allclose(wrist[3], [0, 0, 0, 1])
                or not np.allclose(wrist[:3, :3].T@wrist[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(wrist[:3, :3]), 1., atol=1e-6)
                or frame.material_id is not self.material_id or frame.revision < self.source_revision):
            raise DeformableCoverageError('Invalid patch frame or wrist transform')
        return (frame.vertices[self.vertex_ids]-wrist[:3, 3])@wrist[:3, :3]


class LocalPatchMonitor:
    """Measure local following, without a whole-object rigid-pose assumption.

    Distance diagnostics are geometry-based, not force sensing or grasp proof.
    Missing samples are unknown, never a zero-motion or safe-holding result.
    """

    def __init__(self, patch, frame, wrist):
        self.patch = patch
        self.reference = patch.relative_points(frame, wrist).copy()
        self.last_time = frame.simulation_time_s
        self.last_revision = frame.revision

    def measure(self, frame, wrist, *, finger_clearances_m):
        if frame.revision < self.last_revision or frame.simulation_time_s < self.last_time:
            raise DeformableCoverageError('Stale local contact measurement')
        points = self.patch.relative_points(frame, wrist)
        residual = np.linalg.norm(points-self.reference, axis=1)
        distances = list(finger_clearances_m)
        if len(distances) != 2 or any(x is None or not math.isfinite(x) for x in distances):
            proximity = 'unknown'
        else:
            proximity = 'both_fingers_near_patch' if max(distances) <= .002 else 'two_finger_proximity_not_observed'
        self.last_time, self.last_revision = frame.simulation_time_s, frame.revision
        return {'model': 'material_local_patch', 'proximity': proximity,
                'median_relative_motion_m': float(np.median(residual)),
                'p95_relative_motion_m': float(np.quantile(residual, .95)),
                'whole_object_rigid_pose_checked': False, 'grasp_confirmed': False,
                'force_or_tension_measured': False,
                'guidance': 'Local following is evidence only. Inspect images after lifting; normal cloth bending is not rigid-object slip.'}
