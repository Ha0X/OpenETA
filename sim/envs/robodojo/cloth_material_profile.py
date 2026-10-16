"""Operator-selected cloth friction experiment; native assets remain untouched."""
import math
import os

PROFILE = 'cloth_friction_2x_v1'
ENV = 'OPENETA_ROBODOJO_CLOTH_MATERIAL_PROFILE'


def selected_profile():
    profile = os.environ.get(ENV, '')
    if profile not in ('', PROFILE):
        raise ValueError('Unknown cloth material profile')
    return profile or None


def install_factory(module):
    """Wrap only the garment module's constructor alias, not every material."""
    original = module.ParticleMaterial
    if getattr(original, '_openeta_friction_profile', False):
        return

    def create(*args, **kwargs):
        if args or 'friction' not in kwargs:
            raise ValueError('Native garment material constructor contract changed')
        before = float(kwargs['friction'])
        after = before * 2.
        if not math.isfinite(before) or before <= 0 or not math.isfinite(after):
            raise ValueError('Cloth friction must be finite and positive for doubling')
        material = original(**dict(kwargs, friction=after))
        actual = float(material.get_friction())
        if not math.isclose(actual, after, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError('Cloth material friction readback differs from requested value')
        material._openeta_friction_record = {
            'native_constructor_friction': before, 'requested_friction': after,
            'measured_friction': actual, 'multiplier': 2.,
        }
        return material
    create._openeta_friction_profile = True
    module.ParticleMaterial = create


def install():
    if selected_profile():
        from env.scene_manager.objects import garment
        install_factory(garment)


def inspect_binding(native, requested):
    actual_profile = selected_profile()
    if requested != actual_profile:
        raise ValueError('Cloth material experiment differs from requested profile')
    if not actual_profile:
        return None
    garments = list(native.scene_manager._garment_objects[0].values())
    if not garments:
        raise ValueError('Cloth friction experiment requires native garments')
    values = []
    for garment in garments:
        material = garment.particle_material
        record = getattr(material, '_openeta_friction_record', None)
        if not record:
            raise ValueError('Garment was not created with the requested material profile')
        current = float(material.get_friction())
        if not math.isfinite(current) or not math.isclose(current, record['requested_friction'], rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError('Cloth material friction changed during the episode')
        values.append(dict(record, measured_friction=current))
    return {'profile':actual_profile, 'materials':values,
            'scope':'garment_particle_material; affects cloth contacts including table',
            'other_material_parameters_changed':False,
            'comparable_to_native_physics_baseline':False,
            'source':'native_material_constructor_and_get_friction_readback'}
