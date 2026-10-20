"""Owned, private collision scene using live PhysX rigid-body poses.

Object identities and poses are never part of the public Agent observation.
Only fixed X5 bases on the pinned table receive a mount support exception.
"""
import numpy as np

from .collision_geometry import CollisionScene
from .scene_geometry import array, pose_matrix, snapshot_scene, forward_kinematics


def rigid_part(transform):
    value = array(transform).copy()
    lengths = np.linalg.norm(value[:3, :3], axis=0)
    if np.any(lengths < 1e-10):
        raise ValueError('Degenerate USD rigid root')
    value[:3, :3] /= lengths
    if (not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=2e-5)):
        raise ValueError('Unsupported shear/reflection in rigid root')
    return value


class NativeScene:
    def __init__(self, env, *, allow_cloth=False, allow_articulations=False, allow_fluid=False):
        from pxr import Usd, UsdGeom, UsdPhysics
        import omni.usd
        self.env = env
        self.stage = omni.usd.get_context().get_stage()
        self.scene = CollisionScene(snapshot_scene(env))
        if any(not r['fixed_base'] for r in self.scene.snapshot['robots']):
            raise ValueError('This execution profile requires fixed bases')
        self.world = {}
        self.dynamic = {}
        manager = env.scene_manager
        self.articulated = {}
        if any(manager._articulation_objects[0].values()) and not allow_articulations:
            raise ValueError('World articulations need a link-level physical adapter')
        from .articulation_bindings import bind_articulations
        articulated_roots = bind_articulations(env) if allow_articulations else {}
        if (manager._fluid_objects[0] and not allow_fluid) or (manager._garment_objects[0] and not allow_cloth):
            raise ValueError('Deformables need a live deforming collision adapter')
        from .physical_bindings import retain_rigid_bindings
        roots = {**retain_rigid_bindings(env), **articulated_roots}
        verified_convex_owners = {}
        for i, item in enumerate(self.scene.items):
            if item['robot_index'] is not None:
                continue
            prim = self.stage.GetPrimAtPath(item.get('source_prim_path', item['id']))
            parent = prim
            rigid = None
            verified_owner = item.get('native_verified_owner')
            if verified_owner is not None:
                source_path = str(prim.GetPath())
                if (item['approximation'] != 'nativeQueryVoxels'
                        or not (source_path == verified_owner or source_path.startswith(verified_owner+'/'))):
                    raise ValueError('Native verified owner is outside its collider hierarchy')
                parent = self.stage.GetPrimAtPath(verified_owner)
                if (not parent.IsValid() or not parent.HasAPI(UsdPhysics.RigidBodyAPI)
                        or not UsdPhysics.RigidBodyAPI(parent).GetRigidBodyEnabledAttr().Get()):
                    raise ValueError('Native verified owner has no enabled USD rigid body')
            while parent.IsValid() and not parent.IsPseudoRoot():
                if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                    if UsdPhysics.RigidBodyAPI(parent).GetRigidBodyEnabledAttr().Get():
                        rigid = str(parent.GetPath())
                        break
                parent = parent.GetParent()
            if rigid is not None:
                obj = roots.get(rigid)
                if obj is None and item.get('geometry_source') == 'physx_cooked_convex_part':
                    from .native_convex import verified_convex_owner
                    key=(item['source_prim_path'],item['collider_path'])
                    if key not in verified_convex_owners:
                        verified_convex_owners[key]=verified_convex_owner(env,self.stage,item)
                    rigid=verified_convex_owners[key]
                    parent=self.stage.GetPrimAtPath(rigid)
                    obj=roots.get(rigid)
                if obj is None:
                    import logging
                    logging.getLogger(__name__).error(
                        'Unbound physical rigid root %s for collider %s; known roots=%s',
                        rigid, item['id'], sorted(roots))
                    raise ValueError('World rigid body has no live PhysX pose binding')
                reference = np.array(UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                self.world[i] = (obj, np.linalg.inv(rigid_part(reference)))
                directory = self.articulated if rigid in articulated_roots else self.dynamic
                directory.setdefault(rigid, []).append(i)
        self.refresh()
        self.sphere_validation = self._verify_spheres()
        self.support_pairs = self._mount_support()

    def positions(self):
        from .collision_state import collision_positions
        return collision_positions(self.scene.snapshot['robots'], self.env.robot_manager.robot_key)

    def refresh(self):
        # Rigid object's get_world_pose reads its PhysX view while simulation
        # runs. USD/Fabric xforms alone may lag behind physical execution.
        poses = {}
        for i, (obj, inverse) in self.world.items():
            key = str(obj.prim_path)
            if key not in poses:
                p, q = obj.get_world_pose()
                poses[key] = pose_matrix([*array(p), *array(q)])
            delta = poses[key] @ inverse
            self.scene.world_transforms[i] = delta
            points = self.scene.vertices[i] @ delta[:3, :3].T + delta[:3, 3]
            self.scene.world_bounds[i] = (points.min(axis=0), points.max(axis=0))
            self.scene.set_transform(i, delta)
        for robot, entity, q in zip(self.scene.snapshot['robots'], self.env.robot_manager.robot_key, self.positions()):
            frames = forward_kinematics(robot, q)
            poses = array(entity.data.body_link_pose_w)[0]
            mismatches = []
            for j,name in enumerate(entity.body_names):
                measured = pose_matrix(poses[j])
                if not np.allclose(frames[name], measured, atol=2e-4):
                    mismatches.append({'link':name,
                        'translation_error_m':float(np.linalg.norm(frames[name][:3,3]-measured[:3,3])),
                        'rotation_matrix_max_error':float(abs(frames[name][:3,:3]-measured[:3,:3]).max())})
            if mismatches:
                # Robot-only telemetry in the operator worker log. Do not loosen
                # tolerance or silently fit the model to inconsistent live poses.
                import logging
                logging.getLogger(__name__).error('OPENETA_FK_MISMATCH role=%s joints=%s links=%s',
                    robot.get('role'), dict(zip(robot['joint_names'],q.tolist())), mismatches)
                # Failure-only evidence uses the same measured sample. It must
                # never replace the original failure or escape into tool output.
                try:
                    import json
                    from .fk_diagnostics import fk_mismatch_report
                    report = fk_mismatch_report(robot, q,
                        {name: pose_matrix(poses[j]) for j, name in enumerate(entity.body_names)})
                    logging.getLogger(__name__).error('OPENETA_FK_DIAGNOSTIC %s',
                        json.dumps(report, allow_nan=False, separators=(',', ':')))
                except Exception:
                    logging.getLogger(__name__).exception('OPENETA_FK_DIAGNOSTIC_UNAVAILABLE')
                raise ValueError('Runtime physical FK/base differs from collision model')

    def _verify_spheres(self):
        """Read actual cooked sphere dimensions; verify six physical diameters.

        USD boundingSphere cooking need not use the mesh surface radius. Never
        substitute that radius without inspecting the actual physical shape.
        """
        from itertools import product
        spheres = [i for i,x in enumerate(self.scene.items) if x['approximation'] == 'sphere']
        if not spheres:
            return {'count': 0, 'verified': True}
        import omni.physx
        query = omni.physx.get_physx_scene_query_interface()
        errors, calibrations = [], []
        for i in spheres:
            item = self.scene.items[i]
            if i not in self.world:
                raise ValueError('Cooked sphere requires a verified live rigid-body binding')
            center = array(self.scene.objects[i].getTranslation())
            v = self.scene.vertices[i]
            nominal = float((v.max(axis=0)-v.min(axis=0))[0]/2)
            expected_owner = str(self.world[i][0].prim_path)
            def diameter_samples(c):
                endpoints = []
                for axis in np.vstack([np.eye(3), -np.eye(3)]):
                    hits = []
                    def collect(hit):
                        if str(hit.collision) == item['collider_path']:
                            hits.append((float(hit.distance), str(hit.rigid_body)))
                        return True
                    origin = c+axis*3*nominal
                    query.raycast_all(tuple(origin), tuple(-axis), 6*nominal, collect)
                    if not hits:
                        raise ValueError('Native sphere geometry validation has no matching physical ray hit')
                    distance, owner = min(hits)
                    if owner != expected_owner:
                        raise ValueError('Native sphere physical owner differs from bound rigid body')
                    endpoints.append(origin-axis*distance)
                return array(endpoints)
            samples = diameter_samples(center)
            center = np.array([(samples[j,j]+samples[j+3,j])/2 for j in range(3)])
            samples = diameter_samples(center)
            radii = np.linalg.norm(samples-center,axis=1)
            radius = float(radii.max())
            error = float(np.max(abs(radii-radius)))
            midpoint_error = float(abs((samples[:3]+samples[3:])/2-center).max())
            if not 0 < radius < 3*nominal or max(error,midpoint_error) > 5e-5:
                raise ValueError('Native cooked sphere is not consistent with an analytic sphere')
            errors.append(max(error, midpoint_error))
            # Store calibrated geometry in the original reference-world frame,
            # keeping live rigid-body deltas and all subsequent FK unchanged.
            inverse = np.linalg.inv(self.scene.world_transforms[i])
            reference_center = inverse[:3,:3]@center+inverse[:3,3]
            self.scene.replace_world_sphere(i,reference_center,radius)
            calibrations.append({'mesh_radius_m': nominal, 'physical_radius_m': radius})
        return {'count': len(spheres), 'verified': True, 'maximum_ray_error_m': max(errors),
                'physical_owner_verified': True, 'calibrations': calibrations}

    def _mount_support(self):
        result = []
        # USD mesh suffixes depend on scene construction order. Match the
        # calibrated static tabletop's physical hull, not its material name.
        from .mount_geometry import calibrated_table_geometry
        layout = self.env.scene_manager.config.get('env0', {}).get('Table', {})
        native_tables = getattr(self.env.scene_manager, '_tables', None)
        if (not layout and native_tables is not None and len(native_tables) == 1
                and native_tables[0] is None
                and not any(i['id'] == '/World/envs/env_0/Table'
                            or i['id'].startswith('/World/envs/env_0/Table/') for i in self.scene.items)):
            # A fixed-base scene need not contain a tabletop. Grant NOTHING;
            # all actual base/world pairs remain protected by normal checks.
            self.table_id = None
            self.table_geometry_profile = 'absent_no_mount_exemption'
            return []
        self.table_geometry_profile, expected = calibrated_table_geometry(layout)
        tables = []
        for index,item in enumerate(self.scene.items):
            if (item['robot_index'] is not None or index in self.world
                    or not item['id'].startswith('/World/envs/env_0/Table/')
                    or item.get('approximation') != 'convexHull'):
                continue
            points = array(item['vertices'])
            if points.shape != (8,3):
                continue
            # Bijective corner match; duplicate vertices and changed geometry
            # do not inherit the mount exception merely through their bounds.
            distances = np.linalg.norm(points[:,None,:]-expected[None,:,:],axis=2)
            if np.all(distances.min(axis=0)<1e-5) and np.all(distances.min(axis=1)<1e-5):
                tables.append(item)
        if len(tables) != 1:
            raise ValueError('Calibrated static X5 tabletop geometry unavailable or ambiguous')
        self.table_id = table = tables[0]['id']
        for i, robot in enumerate(self.scene.snapshot['robots']):
            # This calibrated mount exception belongs only to target X5s.
            # Auxiliary fixed bases remain in ordinary collision checks.
            manager = getattr(self.env, 'robot_manager', None)
            if manager is not None and getattr(manager.robot_list[i], 'type', 'target') == 'support':
                continue
            base = array(robot['body_frames'][robot['base_link']])
            if (not np.isclose(abs(base[0, 3]), .3, atol=1e-4)
                    or not np.allclose(base[1:3, 3], [-.45, .765], atol=1e-4)):
                raise ValueError('Unverified X5 mount support location')
            for item in self.scene.items:
                if item['robot_index'] == i and item['body'] == robot['base_link']:
                    result.append((item['id'], table))
        return result

    def public_policy(self):
        covers=[i for i in self.scene.items if i['approximation']=='nativeQueryVoxels']
        auxiliary=any(r.get('control_owner')=='support' for r in self.scene.snapshot['robots'])
        return {'native_surface_cover': {
                    'count':len(covers), 'preserves_exterior_connected_cavities':True,
                    'cavity_resolution_scope':'resolved_voxel_connectivity_only; subresolution_openings_may_be_closed',
                    'conservative_static_fixture_count':sum(bool(i.get('cover_evidence',{}).get('conservative_static_fixture_fallback')) for i in covers),
                    'resolution_m':max((i['voxel_resolution_m'] for i in covers),default=None),
                    'meaning':'conservative_geometric_surface_cover; may_reject_subvoxel_clearance',
                    'exact_native_sdf_penetration':False,
                    'contact_depth':'verified_separating_translation_upper_bound'}, 'mount_support': {'allowed': bool(self.support_pairs), 'scope': 'fixed_base_to_pinned_table_only',
                                 'geometry_profile': self.table_geometry_profile,
                                 'pair_count': len(self.support_pairs)},
                'world_pose_source': 'live_physx_rigid_body_view',
                'geometry_model': self.scene.snapshot.get('geometry_model', 'usd_mesh_with_enclosing_convex_hulls'),
                'world_convex_decomposition': 'native_physx_cooked_parts',
                'robot_convex_decomposition': 'conservative_enclosing_hulls',
                'coverage': ('target_self_target_world_and_target_auxiliary; native_auxiliary_interactions_delegated'
                             if auxiliary else 'robot_self_and_robot_world'),
                'held_object_world_sweep': 'conditional_rigid_attachment_when_both_authorized_fingers_touch',
                'held_object_grasp_confirmation': False,
                'auxiliary_object_prediction': 'live_pose_each_physics_step_no_rigid_grasp_inference' if auxiliary else None,
                'supported_object_dynamics': 'rigid_bodies_only; tool_deformation_is_not_modeled',
                'ik_admission': 'curobo_candidates_checked_against_physical_fk_and_complete_scene',
                'joint_planner': 'requested_cartesian_knots_with_checked_ik_no_spatial_replanning',
                'world_articulations': 'live_link_poses_discrete_monitoring_no_future_prediction' if getattr(self,'articulated',{}) else 'none_admitted',
                'world_articulations_note': (
                    'Live world articulation links are monitored; their future motion is not predicted.'
                    if getattr(self,'articulated',{}) else
                    'No live world articulation links are bound in this scene. This is a coverage statement, '
                    'not a prohibition on finger contact with admitted rigid geometry. It does not establish '
                    'that a visually drawer-like object has a movable joint.'),
                'deformables': 'unsupported',
                **(self._fluid_policy() if getattr(self,'_fluid_policy',None) else {})}
