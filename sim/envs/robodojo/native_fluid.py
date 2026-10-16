"""Operator-private fluid state. Never a volume, grasp, or force observation.

The native render positions may be smoothed. A smoothed particle system must
supply its separate simulationPoints; missing physics data fails closed.
"""
import numpy as np


class FluidCoverageError(ValueError):
    pass


def physical_points(render_points, simulation_points, *, smoothing_enabled):
    render = np.asarray(render_points, dtype=float)
    if render.ndim != 2 or render.shape[1] != 3 or not len(render) or not np.isfinite(render).all():
        raise FluidCoverageError('Invalid fluid particle directory')
    if not isinstance(smoothing_enabled, (bool, np.bool_)):
        raise FluidCoverageError('Unknown particle smoothing state')
    if smoothing_enabled:
        points = np.asarray(simulation_points, dtype=float)
        if points.shape != render.shape or not np.isfinite(points).all():
            raise FluidCoverageError('Smoothed fluid has no complete physical particle positions')
        return points.copy(), 'physx_particle_simulation_points'
    return render.copy(), 'unsmoothed_physx_usd_positions'


class NativeFluidReader:
    def __init__(self, fluid, native):
        from pxr import UsdGeom, Usd, PhysxSchema
        self.fluid, self.native = fluid, native
        self.instancer = fluid.point_instancer
        self.prim = self.instancer.GetPrim()
        self.system = fluid.particle_system
        self.particle = PhysxSchema.PhysxParticleAPI(self.prim)
        self.particle_set = PhysxSchema.PhysxParticleSetAPI(self.prim)
        self.xform = UsdGeom.Xformable(self.prim)
        self.time_code = Usd.TimeCode.Default()
        if UsdGeom.GetStageMetersPerUnit(self.prim.GetStage()) != 1.:
            raise FluidCoverageError('Fluid safety requires verified metre units')
        self.count = None
        self.previous_time = None
        self.refresh()

    def refresh(self):
        import carb
        from pxr import PhysxSchema
        if str(self.native.device) != 'cpu' or self.native.use_fabric:
            raise FluidCoverageError('Fluid reader requires validated CPU USD writeback')
        settings = carb.settings.get_settings()
        if any(settings.get(k) is not True for k in ('/physics/updateToUsd','/physics/updateParticlesToUsd')):
            raise FluidCoverageError('Fluid physical writeback is disabled')
        if not self.prim.IsValid() or not self.prim.IsActive() or not self.system.GetPrim().IsValid():
            raise FluidCoverageError('Fluid particle binding is invalid')
        if (not self.particle or not self.particle_set or
                self.particle.GetParticleEnabledAttr().Get() is not True or
                self.particle_set.GetFluidAttr().Get() is not True or
                self.system.GetParticleSystemEnabledAttr().Get() is not True or
                self.system.GetNonParticleCollisionEnabledAttr().Get() is not True):
            raise FluidCoverageError('Native fluid collision system is not enabled')
        if list(self.particle.GetParticleSystemRel().GetTargets()) != [self.system.GetPrim().GetPath()]:
            raise FluidCoverageError('Fluid physical system ownership changed')
        smoothing = PhysxSchema.PhysxParticleSmoothingAPI(self.system.GetPrim())
        enabled = smoothing.GetParticleSmoothingEnabledAttr().Get() if smoothing else False
        points, self.source = physical_points(self.instancer.GetPositionsAttr().Get(),
            self.particle_set.GetSimulationPointsAttr().Get(), smoothing_enabled=enabled)
        if self.count is not None and len(points) != self.count:
            raise FluidCoverageError('Fluid particle count changed')
        if self.instancer.GetInvisibleIdsAttr().Get() or self.prim.GetMetadata('inactiveIds'):
            raise FluidCoverageError('Masked fluid particles are not supported')
        transform = np.asarray(self.xform.ComputeLocalToWorldTransform(self.time_code), dtype=float).T
        if (transform.shape != (4,4) or not np.isfinite(transform).all() or
                not np.allclose(transform[3], [0,0,0,1]) or abs(np.linalg.det(transform[:3,:3])) < 1e-12):
            raise FluidCoverageError('Invalid physical particle coordinate frame')
        contact = float(self.system.GetContactOffsetAttr().Get())
        if not np.isfinite(contact) or contact <= 0:
            raise FluidCoverageError('Fluid/rigid contact offset is unavailable')
        now = float(self.native.sim.sim.current_time)
        if not np.isfinite(now) or (self.previous_time is not None and now < self.previous_time):
            raise FluidCoverageError('Fluid physical clock moved backwards')
        self.count, self.previous_time = len(points), now
        self.points = points@transform[:3,:3].T+transform[:3,3]
        if not np.isfinite(self.points).all():
            raise FluidCoverageError('Invalid world particle positions')
        self.contact_offset_m = contact
        return self
