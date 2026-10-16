"""Read-only native cloth binding. Not an execution admission adapter yet."""
import numpy as np

from .deformable_surface import DeformableCoverageError, MaterialSurface
from .scene_geometry import array


class NativeClothReader:
    def __init__(self, garment, *, use_fabric, simulation_time_s):
        from pxr import UsdGeom, Usd
        self.garment = garment
        self.mesh = UsdGeom.Mesh(garment.prim)
        self.points_attribute = self.mesh.GetPointsAttr()
        self.xform = UsdGeom.Xformable(garment.prim)
        self.time_code = Usd.TimeCode.Default()
        device = self.device = str(garment._device)
        if device == 'cpu' and use_fabric:
            raise DeformableCoverageError('CPU cloth requires native USD physics writeback')
        if device != 'cpu':
            view = garment._cloth_prim_view
            if not hasattr(view, 'is_physics_handle_valid') or not view.is_physics_handle_valid():
                raise DeformableCoverageError('Cloth has no valid live particle view')
        counts = np.asarray(self.mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        if not len(counts) or np.any(counts != 3):
            raise DeformableCoverageError('Native cloth must supply triangle topology')
        indices = np.asarray(self.mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        if len(indices) != 3*len(counts):
            raise DeformableCoverageError('Incomplete native cloth triangle topology')
        system = garment.particle_system.prim
        for attribute in ('particleSystemEnabled', 'nonParticleCollisionEnabled'):
            if system.GetAttribute(attribute).Get() is not True:
                raise DeformableCoverageError('Native cloth collisions are not enabled')
        offset = system.GetAttribute('contactOffset').Get()
        if offset is None:
            raise DeformableCoverageError('Native cloth collision thickness is unavailable')
        points = self._read_vertices()
        canonical = array(garment.sample_mesh_vertices()[0])
        if points.shape != canonical.shape or not np.allclose(points, canonical, rtol=0., atol=2e-6):
            raise DeformableCoverageError('Direct cloth writeback differs from native world vertices')
        self.surface = MaterialSurface(points, indices.reshape(-1, 3), contact_offset_m=float(offset),
                                       simulation_time_s=simulation_time_s)
        self.source = 'native_cpu_usd_physics_writeback' if device == 'cpu' else 'live_physx_particle_view'

    def refresh(self, *, simulation_time_s):
        # No fallback to an initial mesh/root pose if a live read fails.
        points = self._read_vertices()
        counts = np.asarray(self.mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(self.mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        if len(indices) != 3*len(counts) or np.any(counts != 3):
            raise DeformableCoverageError('Native cloth topology changed')
        return self.surface.update(points, simulation_time_s=simulation_time_s,
                                   triangles=indices.reshape(-1, 3))

    def _read_vertices(self):
        if self.device == 'cpu':
            # Read the same live USD writeback used by sample_mesh_vertices,
            # avoiding repeated generic XformPrim tensor/property conversions.
            # Cache handles only: points and the full world transform are fresh.
            points = array(self.points_attribute.Get())
            transform = array(self.xform.ComputeLocalToWorldTransform(self.time_code)).T
            if transform.shape != (4,4) or not np.isfinite(transform).all():
                raise DeformableCoverageError('Invalid live cloth world transform')
            if points.ndim != 2 or points.shape[1] != 3:
                raise DeformableCoverageError('Missing live cloth USD points')
            return points@transform[:3,:3].T+transform[:3,3]
        if not self.garment._cloth_prim_view.is_physics_handle_valid():
            raise DeformableCoverageError('Cloth live particle handle became invalid')
        return array(self.garment.sample_mesh_vertices()[0])
