"""
WidowX AI - Environment Camera with ROS2 Publishing.

Spawns the WXAI robot and an overhead camera that publishes
RGB images as ROS2 topics.

Usage (run from ~/trossen_ai_isaac):
    ~/IsaacLab/isaaclab.sh -p scripts/wxai_camera.py
"""

from __future__ import annotations

import sys

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402

# --- Scene ---
ROBOT_USD_PATH = "./assets/robots/wxai/wxai_base.usd"
ROBOT_PRIM_PATH = "/World/wxai_robot"

# --- Camera ---
CAMERA_PRIM_PATH = "/World/env_camera"
CAMERA_POSITION = np.array([0.8, 0.0, 1.2])
CAMERA_ORIENTATION = np.array([0.7071, 0.0, 0.7071, 0.0])  # [w, x, y, z]
CAMERA_RESOLUTION = (640, 480)
CAMERA_FREQUENCY = 30

# --- ROS2 ---
ROS2_RGB_TOPIC = "/camera/image_raw"
ROS2_FRAME_ID = "camera_link"


def enable_ros2_bridge() -> bool:
    """Load the ROS2 bridge extension if not already active."""
    try:
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
        print("[INFO] isaacsim.ros2.bridge extension enabled.")
        return True
    except Exception as e:
        print(f"[WARN] Could not enable ROS2 bridge: {e}")
        return False


def setup_ros2_camera_graph(render_product_path: str) -> None:
    """Wire camera render product to ROS2 topic via Action Graph."""
    import omni.graph.core as og

    og.Controller.edit(
        {"graph_path": "/World/CameraROS2Graph", "evaluator_name": "execution"},
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
                ("ROS2CameraHelper.inputs:topicName", ROS2_RGB_TOPIC),
                ("ROS2CameraHelper.inputs:frameId", ROS2_FRAME_ID),
                ("ROS2CameraHelper.inputs:type", "rgb"),
            ],
        },
    )


def main():
    print("[INFO] Enabling ROS2 bridge...")
    ros2_ok = enable_ros2_bridge()

    print("[INFO] Building world...")
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_PRIM_PATH)

    camera = Camera(
        prim_path=CAMERA_PRIM_PATH,
        position=CAMERA_POSITION,
        orientation=CAMERA_ORIENTATION,
        frequency=CAMERA_FREQUENCY,
        resolution=CAMERA_RESOLUTION,
    )

    world.reset()
    camera.initialize()

    if ros2_ok:
        try:
            import omni.replicator.core as rep
            render_product = rep.create.render_product(CAMERA_PRIM_PATH, CAMERA_RESOLUTION)
            setup_ros2_camera_graph(render_product.path)
            print(f"[INFO] Camera publishing → {ROS2_RGB_TOPIC}")
            print("[INFO] Verify: ros2 topic echo /camera/image_raw --once")
        except Exception as e:
            print(f"[WARN] ROS2 graph setup failed: {e}")
    else:
        print("[INFO] ROS2 bridge unavailable. Camera data via camera.get_rgba().")

    print("[INFO] Running. Ctrl+C to stop.")
    while simulation_app.is_running():
        world.step(render=True)


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
