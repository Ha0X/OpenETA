"""Offline physics and camera smoke check; run inside a Slurm GPU allocation."""
from pathlib import Path
import json
import os
import sys
import traceback

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
output = Path(os.environ["OPENETA_SMOKE_OUTPUT"])
output.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 256, "height": 256,
                     "renderer": "RaytracedLighting", "multi_gpu": False})
try:
    import numpy as np
    import torch
    from PIL import Image
    from pxr import Gf, UsdLux
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, GroundPlane
    from isaacsim.sensors.camera import Camera

    cuda_value = float((torch.ones(4, device="cuda") * 2).sum().item())
    assert cuda_value == 8.0

    world = World(stage_units_in_meters=1.0)
    world.scene.add(GroundPlane(prim_path="/World/Ground", size=10))
    cube = world.scene.add(DynamicCuboid(
        prim_path="/World/Cube", name="cube", position=np.array([0., 0., 1.]),
        size=0.4, mass=1., color=np.array([0.9, 0.1, 0.1])))
    UsdLux.DomeLight.Define(world.stage, "/World/Light").CreateIntensityAttr(1000)
    camera = Camera(prim_path="/World/Camera", resolution=(128, 128))
    world.reset()
    camera.initialize()
    rotation = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(2, 2, 2), Gf.Vec3d(0, 0, 0.3), Gf.Vec3d(0, 0, 1)
    ).GetInverse().ExtractRotationQuat()
    camera.set_world_pose(position=np.array([2., 2., 2.]),
        orientation=np.array([rotation.GetReal(), *rotation.GetImaginary()]), camera_axes="usd")
    initial_z = float(cube.get_world_pose()[0][2])
    for _ in range(120):
        world.step(render=True)
    rgba = camera.get_rgba()
    final_z = float(cube.get_world_pose()[0][2])
    assert rgba.shape == (128, 128, 4), f"Unexpected camera frame: {rgba.shape}"
    rgb = rgba[:, :, :3]
    assert np.isfinite(rgb).all() and float(np.std(rgb)) > 1, "Camera frame is blank"
    assert 0.1 < final_z < 0.4 and final_z < initial_z - 0.3, (initial_z, final_z)
    Image.fromarray(rgba.astype(np.uint8)).save(output / "camera.png")
    stack = {}
    if os.environ.get("OPENETA_SMOKE_ROBODOJO") == "1":
        print("ROBODOJO_STACK_CHECK: imports", flush=True)
        import importlib.metadata
        import isaaclab.sim
        import isaaclab.assets
        import pinocchio
        import fcl
        import open3d
        from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.motion_planner import MotionPlanner
        from curobo.types import ToolPoseCriteria, JointState
        from curobo.kinematics import Kinematics, KinematicsCfg

        native_root = Path(__file__).resolve().parents[1] / "third_party" / "RoboDojo"
        sys.path.insert(0, str(native_root))
        from env.environment.base_env import BaseEnv

        print("ROBODOJO_STACK_CHECK: cuRobo GPU forward kinematics", flush=True)
        robot = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
        q = torch.zeros(1, robot.get_dof(), device="cuda", dtype=torch.float32)
        state = robot.compute_kinematics(JointState.from_position(q, joint_names=robot.joint_names))
        pose = state.tool_poses.get_link_pose(robot.tool_frames[0])
        assert torch.isfinite(pose.position).all(), "cuRobo FK returned nonfinite coordinates"
        stack = {"isaaclab_version": importlib.metadata.version("isaaclab"),
                 "curobo_version": importlib.metadata.version("nvidia-curobo"),
                 "native_base_env_import": True, "curobo_fk_position": pose.position.tolist()}
    result = {"ok": True, "initial_cube_z": initial_z, "final_cube_z": final_z,
              "image_shape": list(rgba.shape), "rgb_std": float(np.std(rgb)),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "cuda_tensor_check": cuda_value,
              "physics_backend": "numpy", "robodojo_stack": stack}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("ISAAC_SMOKE_PASS", json.dumps(result), flush=True)
except Exception:
    detail = traceback.format_exc()
    print(detail, flush=True)
    (output / "result.json").write_text(json.dumps({"ok": False, "error": detail}, indent=2) + "\n")
    raise
finally:
    app.close()
