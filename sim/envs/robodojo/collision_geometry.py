"""Private FCL checks against a USD snapshot and calibrated physical FK.

The checker never changes the native scene. A recursive clearance certificate
bounds swept motion of all robot vertices along a linear joint segment. It
assumes the world snapshot is stationary; execution must refresh and monitor.
"""
import math
import time

import numpy as np

from .scene_geometry import array, forward_kinematics
from .collision_bounds import RelativeMotionBounds


def _geometry(item):
    import fcl
    from scipy.spatial import ConvexHull
    vertices = np.asarray(item['vertices'], dtype=np.float64)
    faces = np.asarray(item['triangles'], dtype=np.int32)
    if item['approximation'] == 'nativeQueryVoxels':
        import base64
        from .voxel_geometry import fcl_octree
        return fcl_octree(item['voxel_resolution_m'], base64.b64decode(item['octomap_binary'], validate=True))
    if item['approximation'] == 'sphere':
        return fcl.Sphere(float((vertices.max(axis=0)-vertices.min(axis=0))[0]/2))
    if item['approximation'] in {'convexHull', 'convexDecomposition'}:
        hull = ConvexHull(vertices)
        faces = hull.simplices.copy()
        normal = np.cross(vertices[faces[:,1]]-vertices[faces[:,0]], vertices[faces[:,2]]-vertices[faces[:,0]])
        flip = np.einsum('ij,ij->i', normal, hull.equations[:,:3]) < 0
        faces[flip] = faces[flip][:,[0,2,1]]
        packed = np.column_stack([np.full(len(faces),3),faces]).astype(np.int32).ravel()
        return fcl.Convex(vertices, len(faces), packed)
    if item['approximation'] != 'none':
        raise ValueError(f'Unsupported collision approximation: {item["approximation"]}')
    mesh = fcl.BVHModel()
    mesh.beginModel(len(vertices),len(faces));mesh.addSubModel(vertices,faces);mesh.endModel()
    return mesh


class CollisionScene:
    def __init__(self, snapshot, *, geometry_source=None, geometry_transforms=None):
        import fcl
        self.fcl = fcl
        if snapshot['unsupported']:
            raise ValueError(f'Incomplete collision coverage: {snapshot["unsupported"]}')
        self.relative_bounds = RelativeMotionBounds(snapshot)
        self.snapshot = snapshot
        self.items = snapshot['colliders']
        # Shapes are immutable; CollisionObjects/transforms must never be shared.
        # Attached snapshots retain transformed vertices for conservative bounds,
        # but can reuse the original shapes with an explicit rigid transform.
        if geometry_source is None:
            self.geometries = [_geometry(item) for item in self.items]
            self.shape_centers = [((array(item['vertices']).min(axis=0)+array(item['vertices']).max(axis=0))/2
                                   if item['approximation']=='sphere' else np.zeros(3)) for item in self.items]
            self.geometry_transforms = [np.eye(4) for _ in self.items]
        else:
            if ([i['id'] for i in self.items] != [i['id'] for i in geometry_source.items]
                    or geometry_transforms is None or len(geometry_transforms) != len(self.items)):
                raise ValueError('Geometry reuse requires the same immutable collider directory')
            self.geometries = geometry_source.geometries
            self.shape_centers = geometry_source.shape_centers
            self.geometry_transforms = [array(t).copy() for t in geometry_transforms]
        self.objects = [fcl.CollisionObject(shape) for shape in self.geometries]
        self.vertices = [array(item['vertices']) for item in self.items]
        self.world_transforms = [np.eye(4) for _ in self.items]
        self.world_bounds = [(v.min(axis=0),v.max(axis=0)) for v in self.vertices]
        for i in range(len(self.items)):
            self.set_transform(i, np.eye(4))
        self.allowed = set()
        for index,robot in enumerate(snapshot['robots']):
            frames = forward_kinematics(robot,robot['joint_positions'])
            error=max(float(abs(frames[name]-array(robot['body_frames'][name])).max()) for name in frames)
            if error>2e-4:
                raise ValueError(f'Physical FK disagreement for robot {index}: {error}')
            for joint in robot['joints']:
                if not joint['collision_enabled']:
                    self.allowed.add((index, frozenset((joint['parent'],joint['child']))))
        self.bounds = []
        for index,robot in enumerate(snapshot['robots']):
            length=sum(float(np.linalg.norm(array(j[k])[:3,3])) for j in robot['joints']
                       for k in ('parent_frame','child_frame'))
            length += sum(max(abs(a),abs(b)) for name,(a,b) in zip(robot['joint_names'],robot['joint_limits'])
                          if any(j['name']==name and j['kind']=='prismatic' for j in robot['joints']))
            radius=max((float(np.linalg.norm(array(item['vertices']),axis=1).max()) for item in self.items
                        if item['robot_index']==index),default=0.)
            self.bounds.append(length+radius)
        # Only topology is memoized. Poses, shapes, clearances and collision
        # verdicts remain fresh in this independent CollisionScene view.
        self._pair_topologies = (geometry_source._pair_topologies
                                 if geometry_source is not None else {})
        key = (tuple((a['id'], a['robot_index'], a['body']) for a in self.items),
               frozenset(self.allowed), tuple(tuple(p) for p in snapshot['filtered_pairs']),
               tuple(r.get('control_owner', 'target') for r in snapshot['robots']))
        cached = self._pair_topologies.get(key)
        self.pair_topology_reused = cached is not None
        if cached is None:
            cached = tuple((i,j) for i,a in enumerate(self.items) for j,b in enumerate(self.items[:i])
                           if not self._ignored(a,b))
            if len(self._pair_topologies) >= 8:
                self._pair_topologies.pop(next(iter(self._pair_topologies)))
            self._pair_topologies[key] = cached
        self.pairs = list(cached)  # callers cannot modify another view's cache

    def set_transform(self, index, transform):
        transform = transform @ self.geometry_transforms[index]
        position = transform[:3,3].copy()
        if self.items[index]['approximation'] == 'sphere':
            position += transform[:3,:3] @ self.shape_centers[index]
        self.objects[index].setTransform(self.fcl.Transform(transform[:3,:3], position))

    def replace_world_sphere(self, index, center, radius):
        """Install measured cooked geometry before any attachment views exist."""
        from itertools import product
        item=self.items[index]
        center=array(center)
        if (item['robot_index'] is not None or item['approximation']!='sphere'
                or center.shape!=(3,) or not np.isfinite(center).all()
                or not math.isfinite(radius) or radius<=0):
            raise ValueError('Invalid measured world sphere')
        vertices=center+array(list(product([-radius,radius],repeat=3)))
        self.vertices[index]=vertices
        item['vertices']=vertices.tolist()
        # Copy the directory on replacement: existing views retain their own
        # immutable snapshot instead of mixing old objects and new radii.
        self.geometries=list(self.geometries)
        self.shape_centers=list(self.shape_centers)
        self.geometries[index]=self.fcl.Sphere(radius)
        self.shape_centers[index]=center.copy()
        self.geometry_transforms[index]=np.eye(4)
        self.objects[index]=self.fcl.CollisionObject(self.geometries[index])
        transform=self.world_transforms[index]
        self.set_transform(index,transform)
        live_center=transform[:3,:3]@center+transform[:3,3]
        self.world_bounds[index]=(live_center-radius,live_center+radius)

    def refresh_world(self):
        from pxr import UsdGeom,Usd
        import omni.usd
        stage=omni.usd.get_context().get_stage()
        for i,item in enumerate(self.items):
            if item['robot_index'] is not None:
                continue
            reference=item.get('reference_world_transform')
            if reference is None:
                raise ValueError('Snapshot has no refreshable world transform')
            prim=stage.GetPrimAtPath(item.get('source_prim_path', item['id']))
            if not prim.IsValid():
                raise ValueError('World collision geometry was removed')
            current=np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            delta=current@np.linalg.inv(array(reference))
            if not np.allclose(delta[:3,:3].T@delta[:3,:3],np.eye(3),atol=2e-6):
                raise ValueError('World geometry changed scale or deformed')
            if not np.allclose(delta,self.world_transforms[i],atol=1e-12,rtol=0):
                self.world_transforms[i]=delta
                points=self.vertices[i]@delta[:3,:3].T+delta[:3,3]
                self.world_bounds[i]=(points.min(axis=0),points.max(axis=0))
                self.set_transform(i, delta)

    def _ignored(self, a, b):
        from .support_geometry import native_interaction
        if native_interaction(self.snapshot, a, b):
            return True
        ia,ib=a['robot_index'],b['robot_index']
        if ia is None and ib is None:
            return True
        if ia is not None and ia==ib:
            if a['body']==b['body'] or (ia,frozenset((a['body'],b['body']))) in self.allowed:
                return True
        for pa,pb in self.snapshot['filtered_pairs']:
            if ((a['id'].startswith(pa+'/') or a['id']==pa) and (b['id'].startswith(pb+'/') or b['id']==pb)
                    or (a['id'].startswith(pb+'/') or a['id']==pb) and (b['id'].startswith(pa+'/') or b['id']==pa)):
                return True
        return False

    def check(self, positions, *, allowed_pairs=(), clearance_limit=math.inf, clearance_limits=None, extra_pairs=()):
        fcl=self.fcl
        if len(positions)!=len(self.snapshot['robots']):
            raise ValueError('Missing physical articulation state')
        frames=[forward_kinematics(robot,q) for robot,q in zip(self.snapshot['robots'],positions)]
        violations = []
        for constraint in self.snapshot.get('closed_chain_constraints', []):
            from scipy.spatial.transform import Rotation
            a, b = constraint['owner'], constraint['follower']
            actual = np.linalg.inv(frames[a][constraint.get('owner_link', 'link6')]) @ frames[b][constraint.get('follower_link', 'link6')]
            target = array(constraint['relative_pose'])
            pe = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
            re = float(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude())
            if pe > .003 or re > .03:
                violations.append({'reason': 'shared_body_relative_wrist_constraint',
                    'roles': [self.snapshot['robots'][i]['role'] for i in (a,b)],
                    'position_error_m': pe, 'orientation_error_rad': re})
        bounds=[]
        for i,(item,obj) in enumerate(zip(self.items,self.objects)):
            if item['robot_index'] is None:
                bounds.append(self.world_bounds[i])
                continue
            transform=frames[item['robot_index']][item['body']]
            self.set_transform(i, transform)
            vertices=self.vertices[i]@transform[:3,:3].T+transform[:3,3]
            bounds.append((vertices.min(axis=0),vertices.max(axis=0)))
        allowed={frozenset(pair) for pair in allowed_pairs}
        closest=float(clearance_limit);pair=None;contacts=[];queries=0;uncertified=[]
        pairs = tuple(dict.fromkeys([*self.pairs, *extra_pairs]))
        # Batch only the AABB arithmetic. The narrow phase, pair order,
        # authorizations and pair-specific certificate thresholds stay intact.
        # Bounds are freshly measured/FK-derived on every call; no verdict cache.
        if pairs:
            slots = np.asarray(pairs, dtype=np.intp)
            boxes = np.asarray(bounds)
            gaps = np.maximum(np.maximum(boxes[slots[:, 0], 0]-boxes[slots[:, 1], 1],
                                         boxes[slots[:, 1], 0]-boxes[slots[:, 0], 1]), 0.)
            gap_distances = np.linalg.norm(gaps, axis=1)
            # Round towards overlap so batching cannot increase the claimed
            # lower bound at a floating-point threshold.
            gap_distances = np.nextafter(gap_distances, 0.)
        else:
            gap_distances = ()
        for (i,j), gap_distance in zip(pairs, gap_distances):
            a,b=self.items[i],self.items[j]
            if frozenset((a['id'],b['id'])) in allowed:
                continue
            limit = closest if clearance_limits is None else clearance_limits[(i,j)]
            gap_distance = float(gap_distance)
            if gap_distance>limit:
                # A pair-specific threshold may be smaller than the current
                # global minimum. Preserve a valid clearance lower bound for
                # skipped pairs using their AABB separation.
                closest = min(closest,gap_distance)
                continue
            queries+=1
            request=fcl.CollisionRequest(num_max_contacts=1,enable_contact=True)
            hit=fcl.CollisionResult();fcl.collide(self.objects[i],self.objects[j],request,hit)
            if hit.is_collision:
                contacts.append({'a':a['id'],'b':b['id'],'robot_a':a['robot_index'],'robot_b':b['robot_index']})
                closest=0.;pair=(a['id'],b['id']);uncertified.append((i,j))
            elif clearance_limits is not None or closest>0:
                distance=fcl.distance(self.objects[i],self.objects[j],fcl.DistanceRequest(),fcl.DistanceResult())
                if not np.isfinite(distance) or distance<0:
                    raise ValueError('FCL distance unavailable or inconsistent')
                if distance<closest:closest=float(distance);pair=(a['id'],b['id'])
                if distance<=limit:uncertified.append((i,j))
        return {'clear':not contacts and not violations,'clearance_m':closest,'closest_pair':pair,
                'contacts':contacts,'pair_queries':queries,'uncertified_pairs':uncertified,
                'constraint_violations': violations}

    def signed_distances(self, positions, pairs, *, contact_bounds=False, separation_bounds=False):
        """Convex penetration depths for recovery, never a fabricated clearance."""
        if separation_bounds and not contact_bounds:
            raise ValueError('Separation bounds are only for bounded contact monitoring')
        # Update every collision object's FK before requesting explicit pairs.
        self.check(positions, allowed_pairs=[(self.items[i]['id'], self.items[j]['id'])
                                           for i, j in self.pairs], clearance_limit=0.)
        ids = {item['id']: i for i, item in enumerate(self.items)}
        pairs = list(pairs)
        separated = {}
        if separation_bounds and pairs:
            # Fresh transforms, not the broad-phase cache: callers may supply
            # newly measured world transforms. Only strictly disjoint enclosing
            # boxes can skip FCL. All touching/overlapping boxes keep the same
            # narrow phase and disagreement fallback as the exact query.
            frames = [forward_kinematics(r, q) for r, q in zip(self.snapshot['robots'], positions)]
            boxes = []
            for i, item in enumerate(self.items):
                t = (self.world_transforms[i] if item['robot_index'] is None
                     else frames[item['robot_index']][item['body']])
                points = self.vertices[i] @ t[:3, :3].T + t[:3, 3]
                lo, hi = points.min(axis=0), points.max(axis=0)
                # Outward padding covers transform arithmetic roundoff; this
                # can only reduce pruning, never authorize extra penetration.
                pad = 64*np.finfo(float).eps*max(1., float(abs(points).max()))
                boxes.append((lo-pad, hi+pad))
            boxes = np.asarray(boxes)
            slots = np.asarray([(ids[a], ids[b]) for a, b in pairs], dtype=np.intp)
            gaps = np.maximum(np.maximum(boxes[slots[:, 0], 0]-boxes[slots[:, 1], 1],
                                         boxes[slots[:, 1], 0]-boxes[slots[:, 0], 1]), 0.)
            for pair, gap in zip(pairs, np.nextafter(np.linalg.norm(gaps, axis=1), 0.)):
                if gap > 0:
                    separated[frozenset(pair)] = float(gap)
        result = {}
        self.last_contact_distance_method=('fcl_penetration_with_aabb_separation_bounds'
                                           if separated else 'fcl_signed_distance')
        self.last_contact_distance_methods = {}
        for a, b in pairs:
            i, j = ids[a], ids[b]
            if any(self.items[k]['approximation']=='nativeQueryVoxels' for k in (i,j)):
                if not contact_bounds:
                    raise ValueError('Native voxel cover provides conservative bounds, not exact signed proximity')
                key=frozenset((a,b))
                if key in separated:
                    result[key]=separated[key]
                    self.last_contact_distance_methods[key]='aabb_separation_lower_bound'
                    continue
                hit=self.fcl.CollisionResult()
                self.fcl.collide(self.objects[i],self.objects[j],self.fcl.CollisionRequest(),hit)
                if hit.is_collision:
                    from .voxel_geometry import separation_translation_bound
                    distance=-separation_translation_bound(self,i,j)
                    method='verified_translation_upper_bound_on_native_voxel_cover'
                else:
                    distance=float(self.fcl.distance(self.objects[i],self.objects[j],
                        self.fcl.DistanceRequest(),self.fcl.DistanceResult()))
                    if not np.isfinite(distance) or distance<0:
                        raise ValueError('Native voxel cover distance unavailable')
                    method='native_voxel_cover_separation_lower_bound'
                result[key]=distance
                self.last_contact_distance_methods[key]=method
                self.last_contact_distance_method=method
                continue
            if any(self.items[k]['approximation'] not in {'convexHull', 'sphere'} for k in (i, j)):
                raise ValueError('Signed recovery requires convex collision geometry')
            key = frozenset((a, b))
            if key in separated:
                result[key] = separated[key]
                self.last_contact_distance_methods[key] = 'aabb_separation_lower_bound'
                continue
            hit = self.fcl.CollisionResult()
            self.fcl.collide(self.objects[i], self.objects[j],
                self.fcl.CollisionRequest(num_max_contacts=128, enable_contact=True), hit)
            if hit.is_collision:
                depths = [float(c.penetration_depth) for c in hit.contacts]
                distance = (-max(depths) if depths and np.isfinite(depths).all() and min(depths)>=-1e-8 else float('nan'))
                unavailable = not np.isfinite(distance)
            else:
                distance = float(self.fcl.distance(self.objects[i], self.objects[j],
                    self.fcl.DistanceRequest(enable_signed_distance=True), self.fcl.DistanceResult()))
                unavailable = not np.isfinite(distance) or distance < 0
            method = 'fcl_signed_distance'
            if unavailable:
                if not contact_bounds:
                    raise ValueError('Signed collision distance unavailable')
                from .contact_bounds import projection_contact_bound
                distance=projection_contact_bound(self,positions,i,j)
                method = 'conservative_projection_bound_on_fcl_disagreement'
                self.last_contact_distance_method = method
            result[frozenset((a, b))] = distance
            self.last_contact_distance_methods[frozenset((a, b))] = method
        return result

    def check_path(self, start, target, *, allowed_pairs=(), margin_m=.0002, timeout_s=15., max_nodes=2048):
        start=[array(q) for q in start];target=[array(q) for q in target]
        if len(start)!=len(self.snapshot['robots']) or len(target)!=len(start):
            raise ValueError('Path must specify each physical articulation')
        from .collision_state import stationary_measured_finger_slots
        for robot_index,(robot,a,b) in enumerate(zip(self.snapshot['robots'],start,target)):
            limits=array(robot['joint_limits'])
            if a.shape!=b.shape or b.shape!=(len(robot['joint_names']),) or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Invalid path joint vector')
            outside=(b<limits[:,0]-1e-7)|(b>limits[:,1]+1e-7)
            for slot in stationary_measured_finger_slots(robot,a,b):outside[slot]=False
            if np.any(outside):
                return {'status':'blocked','reason':'joint_limit','samples':0,
                        'robot_index':robot_index,'control_owner':robot.get('control_owner','target'),
                        'joint_names':[robot['joint_names'][i] for i in np.flatnonzero(outside)]}
        deadline=time.monotonic()+timeout_s;nodes=0;min_clearance=math.inf

        if self.snapshot.get('closed_chain_constraints'):
            count = max(2, math.ceil(max(float(abs(b-a).max()) for a,b in zip(start,target))/.01)+1)
            if count > max_nodes:
                return {'status':'unknown','reason':'closed_chain_sample_budget','samples':0}
            for alpha in np.linspace(0.,1.,count):
                if time.monotonic() > deadline:
                    return {'status':'unknown','reason':'closed_chain_sample_budget','samples':nodes}
                value = self.check([a+(b-a)*alpha for a,b in zip(start,target)], allowed_pairs=allowed_pairs, clearance_limit=0.)
                nodes += 1
                if value['constraint_violations']:
                    return {'status':'blocked','reason':'shared_body_relative_wrist_constraint',
                            'samples':nodes,'detail':value}

        stack=[(start,target,0.,1.)]
        last_uncertified = None
        while stack:
            if nodes>=max_nodes or time.monotonic()>deadline:
                feedback = {'status':'unknown','reason':'clearance_certificate_budget','samples':nodes}
                if last_uncertified is not None:
                    feedback.update(last_uncertified)
                return feedback
            a,b,lo,hi=stack.pop();pair_bounds=self.relative_bounds.for_segment(a,b,self.pairs);bound=max(pair_bounds.values(),default=0.)
            result=self.check(a,allowed_pairs=allowed_pairs,clearance_limit=bound+margin_m+1e-9,
                              clearance_limits={pair:bound+margin_m for pair,bound in pair_bounds.items()});nodes+=1
            min_clearance=min(min_clearance,result['clearance_m'])
            if not result['clear']:
                return {'status':'blocked','reason':'geometry_collision','fraction':lo,'samples':nodes,'detail':result}
            if not result['uncertified_pairs']:
                continue
            last_uncertified = {'detail': result, 'uncertified_segment_fraction': [lo, hi],
                                'evidence': 'last_uncertified_segment; not_a_measured_collision'}
            # A zero-margin separating proof can start at a positive submicron
            # support gap. The ordinary 1um subdivision cutoff would reject it
            # before the swept displacement bound can become smaller than the
            # measured gap. Refine further; never change the clearance test.
            resolution = 1e-9 if margin_m == 0. else 1e-6
            if bound<resolution:
                return {'status':'unknown' if margin_m == 0. else 'blocked',
                        'reason':'clearance_resolution_limit' if margin_m == 0. else 'insufficient_clearance',
                        'fraction':lo,'samples':nodes,'detail':result,
                        'refinement_floor_m':resolution}
            mid=[(x+y)/2 for x,y in zip(a,b)];fraction=(lo+hi)/2
            stack.extend([(mid,b,fraction,hi),(a,mid,lo,fraction)])
        return {'status':'clear','reason':'swept_clearance_bound','samples':nodes,
                'min_sample_clearance_lower_bound_m':min_clearance,'world_assumption':'stationary_snapshot',
                'geometry_model':self.snapshot.get('geometry_model','usd_mesh_with_enclosing_convex_hulls')}
