"""Private fluid/robot exclusion; native fluid/container contact is unchanged.

A frozen particle envelope checks commanded robot motion. Every physical step
refreshes the particles. This is NOT prediction of fluid flow or contact force,
or a guarantee against transient contact between two measured physical states.
"""
import math
import time
import numpy as np
from .scene_geometry import forward_kinematics
from .native_fluid import NativeFluidReader, FluidCoverageError


def check_particles(scene, positions, clouds, *, motion_bounds=None, deadline=math.inf):
    if len(positions) != len(scene.snapshot['robots']):
        raise FluidCoverageError('Missing robot state for fluid exclusion')
    for robot,q in zip(scene.snapshot['robots'],positions):
        q=np.asarray(q,dtype=float)
        if q.shape!=(len(robot['joint_names']),) or not np.isfinite(q).all():
            raise FluidCoverageError('Invalid robot state for fluid exclusion')
    if not clouds:raise FluidCoverageError('Missing fluid snapshot')
    frames = [forward_kinematics(r,q) for r,q in zip(scene.snapshot['robots'],positions)]
    fcl = scene.fcl
    for i,item in enumerate(scene.items):
        robot=item['robot_index']
        if robot is None:continue  # Container/fluid dynamics belong to native physics.
        if item['approximation'] not in ('convexHull','convexDecomposition','sphere'):
            raise FluidCoverageError('Fluid exclusion requires solid robot geometry')
        transform=frames[robot][item['body']]
        scene.set_transform(i,transform)
        vertices=scene.vertices[i]@transform[:3,:3].T+transform[:3,3]
        low,high=vertices.min(0),vertices.max(0)
        bound=0. if motion_bounds is None else float(motion_bounds[i])
        if not math.isfinite(bound) or bound<0:raise FluidCoverageError('Invalid robot motion bound')
        for points,offset in clouds:
            if not math.isfinite(float(offset)) or float(offset)<=0:
                raise FluidCoverageError('Invalid fluid contact offset')
            radius=float(offset)+.0002+bound
            if (not math.isfinite(radius) or radius<=0 or points.ndim!=2 or points.shape[1]!=3
                    or not len(points) or not np.isfinite(points).all()):
                raise FluidCoverageError('Invalid fluid exclusion envelope')
            near=points[np.all((points>=low-radius)&(points<=high+radius),axis=1)]
            particle=fcl.CollisionObject(fcl.Sphere(radius))
            for point in near:
                if time.monotonic()>deadline:
                    return {'clear':False,'reason':'fluid_check_budget','evidence':'unknown_not_collision'}
                particle.setTransform(fcl.Transform(point))
                result=fcl.CollisionResult()
                fcl.collide(scene.objects[i],particle,fcl.CollisionRequest(),result)
                if result.is_collision:
                    return {'clear':False,'reason':'fluid_robot_proximity',
                            'role':scene.snapshot['robots'][robot]['role'],'part':item['body'],
                            'evidence':'private_geometric_exclusion_envelope_not_force_or_wetting_sensor'}
    return {'clear':True}


class FluidExecution:
    def __init__(self, control):
        self.control=control
        self.directory=dict(control.native.scene_manager._fluid_objects[0])
        self.readers=[NativeFluidReader(f,control.native) for f in self.directory.values()]
        if not self.readers:raise FluidCoverageError('Fluid profile has no native particles')
        self.checks=0
        self.refresh()

    def refresh(self):
        live=self.control.native.scene_manager._fluid_objects[0]
        if set(live)!=set(self.directory) or any(live[k] is not v for k,v in self.directory.items()):
            raise FluidCoverageError('Native fluid directory changed')
        self.clouds=[(r.refresh().points.copy(),r.contact_offset_m) for r in self.readers]

    def check(self, positions, **kwargs):
        self.checks+=1
        return check_particles(self.control.world.scene,positions,self.clouds,**kwargs)

    def check_segment(self, start, target):
        # Recursive robot sweep against a frozen measured liquid snapshot.
        scene=self.control.world.scene
        deadline=time.monotonic()+2.
        for phase,q in (('measured_start',start),('commanded_target',target)):
            result=self.check(q,deadline=deadline)
            if not result['clear']:
                return self.feedback({**result,'check_phase':phase})
        pending=[(start,target)];nodes=0
        while pending:
            if nodes>=128 or time.monotonic()>deadline:
                return self.feedback({'clear':False,'reason':'fluid_check_budget','evidence':'unknown_not_collision'})
            a,b=pending.pop();bounds=[]
            for item,chain in zip(scene.items,scene.relative_bounds.chains):
                index=item['robot_index'];value=0.
                if index is not None:
                    names=scene.snapshot['robots'][index]['joint_names']
                    value=sum((radius if kind=='revolute' else 1.)*abs(b[index][names.index(name)]-a[index][names.index(name)])
                              for name,kind,radius in chain if kind!='fixed')
                bounds.append(float(np.nextafter(value,np.inf)) if value else 0.)
            result=self.check(a,motion_bounds=bounds,deadline=deadline);nodes+=1
            if result['clear']:continue
            measured=self.check(a,deadline=deadline)
            if not measured['clear']:return self.feedback({**measured,'check_phase':'sweep_sample'})
            if max(bounds,default=0.)<1e-9:
                return self.feedback({'clear':False,'reason':'fluid_check_resolution','evidence':'unknown_not_collision'})
            middle=[(x+y)/2 for x,y in zip(a,b)]
            pending.extend([(a,middle),(middle,b)])
        return {'clear':True}

    @staticmethod
    def feedback(result):
        guidance = ('Liquid contact with the container is permitted, but direct robot/liquid proximity is restricted. '
                    'The host has not moved the target, rerouted, or inferred liquid volume, spill amount, force, or task success.')
        if result.get('reason') == 'fluid_robot_proximity' and result.get('check_phase') == 'measured_start':
            guidance += (' The current measured state already intersects the conservative liquid exclusion envelope. '
                         'This profile has no certified escape policy from that state: changing the endpoint or waiting '
                         'does not bypass its start-state check. No physics step is authorized by this check; '
                         'operator diagnosis is required. This is not proof of actual wetting.')
        elif result.get('check_phase') in ('commanded_target','sweep_sample'):
            guidance += ' Inspect public images and revise the approach; a target/sweep rejection does not establish contact in the current state.'
        else:
            guidance += ' Inspect public images and the reported check phase before revising the approach.'
        return {**result,'guidance':guidance}

    def policy(self):
        return {'deformables':'fluid_snapshot_and_per_physics_step_monitor', 'fluid_contact':{'profile':'fluid_robot_exclusion_v1',
                'container_contact':'native_physics_allowed','robot_contact':'restricted_including_gripper',
                'checks':'before_and_after_each_physics_step',
                'commanded_robot_sweep':'checked_against_current_particle_envelope',
                'future_fluid_motion_predicted':False,'continuous_fluid_safety_guarantee':False,
                'feedback':'coarse_proximity_or_unavailable; not_force_or_wetting',
                'private_particle_data_exposed':False}}
