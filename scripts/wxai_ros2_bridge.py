"""
WidowX AI - ROS2 Bridge (OmniGraph).

Publishing is handled entirely by OmniGraph ROS2 nodes — no rclpy needed for
topics going out. Subscribing to joint commands is also in-graph via
IsaacArticulationController.  rclpy is kept only for /wxai/target_pose
(PoseStamped) and degrades gracefully if the import fails.

Publishes  (OmniGraph):
  /wxai/joint_states     — sensor_msgs/JointState
  /tf                    — full robot TF tree (base_link → link_6 included)
  /wxai/camera/image_raw — sensor_msgs/Image

Subscribes (OmniGraph):
  /wxai/joint_commands   — sensor_msgs/JointState → applied directly to joints

Subscribes (rclpy, optional):
  /wxai/target_pose      — geometry_msgs/PoseStamped → Cartesian IK

Interactive: drag the blue cube in the viewport to move the robot.

Usage (from ~/trossen_ai_isaac):
    ~/isaacsim/_build/linux-x86_64/release/python.sh scripts/wxai_ros2_bridge.py

Verify:
    ros2 topic list
    ros2 topic echo /wxai/joint_states
    ros2 run tf2_tools view_frames

Joint command:
    ros2 topic pub --once /wxai/joint_commands sensor_msgs/JointState \
      "{name: ['joint_0','joint_1','joint_2','joint_3','joint_4','joint_5'], \
      position: [0.0, 0.3, -0.5, 0.0, 0.2, 0.0]}"

Cartesian command (requires rclpy):
    ros2 topic pub --once /wxai/target_pose geometry_msgs/PoseStamped \
      "{header: {frame_id: 'base_link'}, pose: {position: {x: 0.3, y: 0.0, z: 0.4}, \
      orientation: {w: 1.0}}}"
"""

from __future__ import annotations

import os
import sys

# ── ROS2 environment (needed even for OmniGraph bridge check binary) ─────────
_EXT_PATH = os.path.expanduser(
    "~/isaacsim/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge"
)
os.environ["LD_LIBRARY_PATH"] = (
    os.path.join(_EXT_PATH, "humble", "lib")
    + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ.setdefault("ROS_DISTRO", "humble")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

# Strip Python 3.10 ROS2 paths; insert bundled Python 3.11 paths up front
sys.path = [p for p in sys.path if "python3.10" not in p and "ros2_humble" not in p]
sys.path.insert(0, os.path.join(_EXT_PATH, "humble", "rclpy"))
sys.path.insert(1, os.path.join(_EXT_PATH, "humble"))
# ─────────────────────────────────────────────────────────────────────────────

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.kit.app

_ext_manager = omni.kit.app.get_app().get_extension_manager()
_ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
simulation_app.update()

import numpy as np
import omni.graph.core as og
import omni.replicator.core as rep
import usdrt.Sdf
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera

sys.path.append(os.path.dirname(__file__))
from controller import RobotType, TrossenAIController

# ── rclpy: joint commands + Cartesian pose commands ──────────────────────────
_rclpy_ok = False
_ros2_node = None

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import JointState as _JointState

    class _CommandNode(Node):
        def __init__(self):
            super().__init__("wxai_bridge")
            self.latest_pose: PoseStamped | None = None
            self.latest_joints: _JointState | None = None
            self.create_subscription(PoseStamped, "/wxai/target_pose", self._pose_cb, 10)
            self.create_subscription(_JointState, "/wxai/joint_commands", self._joint_cb, 10)

        def _pose_cb(self, msg: PoseStamped) -> None:
            self.latest_pose = msg

        def _joint_cb(self, msg: _JointState) -> None:
            self.latest_joints = msg

    rclpy.init()
    _ros2_node = _CommandNode()
    _rclpy_ok = True
    print("[INFO] rclpy loaded — /wxai/target_pose and /wxai/joint_commands active.")
except Exception as e:
    print(f"[WARN] rclpy unavailable ({e}). Cube-only control mode.")
# ─────────────────────────────────────────────────────────────────────────────

# --- Robot ---
ROBOT_USD_PATH = "./assets/robots/wxai/wxai_base.usd"
ROBOT_SCENE_PATH = "/World/wxai_robot"
WXAI_ARM_DOF_INDICES = [0, 1, 2, 3, 4, 5]
WXAI_GRIPPER_DOF_INDEX = 6
WXAI_DEFAULT_DOF_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044, 0.044]

# --- Interactive target cube ---
TARGET_SCENE_PATH = "/World/TargetCube"
TARGET_INITIAL_POSITION = np.array([0.3, 0.0, 0.2])
TARGET_INITIAL_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
TARGET_SIZE = 0.05

# --- Camera ---
CAMERA_PRIM_PATH = "/World/env_camera"
CAMERA_POSITION = np.array([0.8, 0.0, 1.2])
CAMERA_ORIENTATION = np.array([0.7071, 0.0, 0.7071, 0.0])  # [w, x, y, z]
CAMERA_RESOLUTION = (640, 480)
CAMERA_FREQUENCY = 30

# --- OmniGraph paths ---
ROS2_GRAPH_PATH = "/World/ROS2Graph"
CAMERA_GRAPH_PATH = "/World/CameraGraph"


def _setup_ros2_graph() -> None:
    """OmniGraph: publish joint states and full TF tree every simulation tick."""
    og.Controller.edit(
        {"graph_path": ROS2_GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishTF.inputs:execIn"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "PublishTF.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishJointState.inputs:topicName", "/wxai/joint_states"),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(ROBOT_SCENE_PATH)]),
                ("PublishTF.inputs:topicName", "/tf"),
                ("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(ROBOT_SCENE_PATH)]),
            ],
        },
    )


def _setup_camera_graph(render_product_path: str) -> None:
    """OmniGraph: publish camera RGB image."""
    og.Controller.edit(
        {"graph_path": CAMERA_GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("ROS2CameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "ROS2CameraHelper.inputs:execIn"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ROS2CameraHelper.inputs:renderProductPath", render_product_path),
                ("ROS2CameraHelper.inputs:topicName", "/wxai/camera/image_raw"),
                ("ROS2CameraHelper.inputs:frameId", "camera_link"),
                ("ROS2CameraHelper.inputs:type", "rgb"),
            ],
        },
    )


def _tick_ros2_graph() -> None:
    og.Controller.set(
        og.Controller.attribute(f"{ROS2_GRAPH_PATH}/OnImpulseEvent.state:enableImpulse"),
        True,
    )


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_SCENE_PATH)

    robot = TrossenAIController(
        robot_path=ROBOT_SCENE_PATH,
        robot_type=RobotType.WXAI,
        arm_dof_indices=WXAI_ARM_DOF_INDICES,
        gripper_dof_index=WXAI_GRIPPER_DOF_INDEX,
        default_dof_positions=WXAI_DEFAULT_DOF_POSITIONS,
    )

    target_cube = world.scene.add(
        VisualCuboid(
            prim_path=TARGET_SCENE_PATH,
            name="target_cube",
            position=TARGET_INITIAL_POSITION,
            orientation=TARGET_INITIAL_ORIENTATION,
            size=TARGET_SIZE,
            color=np.array([0.0, 0.0, 1.0]),
        )
    )

    camera = Camera(
        prim_path=CAMERA_PRIM_PATH,
        position=CAMERA_POSITION,
        orientation=CAMERA_ORIENTATION,
        frequency=CAMERA_FREQUENCY,
        resolution=CAMERA_RESOLUTION,
    )

    world.reset()
    camera.initialize()

    _setup_ros2_graph()
    render_product = rep.create.render_product(CAMERA_PRIM_PATH, CAMERA_RESOLUTION)
    _setup_camera_graph(render_product.path)

    # Start the simulation — physics tensors only exist while playing
    world.play()

    subs = "/wxai/joint_commands  /wxai/target_pose" if _rclpy_ok else "(rclpy unavailable)"
    print(
        "[INFO] ROS2 bridge active.\n"
        "       → Drag the blue cube in the viewport to move the robot.\n"
        f"       Publishes (OmniGraph): /wxai/joint_states  /tf  /wxai/camera/image_raw\n"
        f"       Subscribes (rclpy):   {subs}"
    )

    reset_needed = False

    while simulation_app.is_running():
        world.step(render=True)

        if world.is_stopped():
            reset_needed = True

        if not world.is_playing():
            continue

        if reset_needed:
            world.reset()
            reset_needed = False

        # OmniGraph tick: publish joint states + TF (only while physics is active)
        _tick_ros2_graph()

        if _rclpy_ok and _ros2_node is not None:
            rclpy.spin_once(_ros2_node, timeout_sec=0.0)  # type: ignore[possibly-unbound]

            # Joint position command (direct DOF targets)
            if _ros2_node.latest_joints is not None:
                msg = _ros2_node.latest_joints
                if len(msg.position) == len(WXAI_ARM_DOF_INDICES):
                    robot.set_dof_position_targets(
                        np.array([list(msg.position)]),
                        dof_indices=WXAI_ARM_DOF_INDICES,
                    )
                _ros2_node.latest_joints = None
                continue

            # Cartesian EE command (differential IK)
            if _ros2_node.latest_pose is not None:
                p = _ros2_node.latest_pose.pose.position
                o = _ros2_node.latest_pose.pose.orientation
                robot.set_end_effector_pose(
                    position=np.array([p.x, p.y, p.z]),
                    orientation=np.array([o.w, o.x, o.y, o.z]),
                )
                _ros2_node.latest_pose = None
                continue

        # Default: track the blue cube with differential IK
        cube_pos, cube_ori = target_cube.get_world_pose()
        robot.set_end_effector_pose(position=cube_pos, orientation=cube_ori)

    if _rclpy_ok and _ros2_node is not None:
        _ros2_node.destroy_node()
        rclpy.shutdown()  # type: ignore[possibly-unbound]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
