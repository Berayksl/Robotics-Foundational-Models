"""
Non-interactive test script for StretchController.
Pass a list of actions, get back states after each step.
"""
from __future__ import annotations
import time
import os
import json
import argparse
import prior
from wandb import controller

from environment.stretch_controller import StretchController
from utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
from utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
from utils.type_utils import THORActions

import matplotlib.pyplot as plt


from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from ai2thor.controller import Controller


# If you already have THORActions enum/constants, use them directly.
# This class supports both your THORActions values and simple strings.
MOVE_AHEAD_ALIASES = {"move_ahead", "m", "MoveAhead"}
MOVE_BACK_ALIASES = {"move_back", "b", "MoveBack"}
ROT_LEFT_ALIASES = {"rotate_left", "l", "RotateLeft"}
ROT_RIGHT_ALIASES = {"rotate_right", "r", "RotateRight"}


@dataclass
class NavConfig:
    ahead: float
    rot_deg: float
    grid_size: Optional[float] = None


class LightStretchController:
    """
    Minimal AI2-THOR controller wrapper for fast forward-propagation of base pose only.
    - No calibration
    - No extra cameras
    - No segmentation, depth, normals, flow
    - Small render size
    - Optional headless
    """

    def __init__(
        self,
        scene: Optional[Dict[str, Any]] = None,
        *,
        nav: NavConfig,
        headless: bool = True,
        width: int = 32,
        height: int = 32,
        server_timeout: int = 10,
        quality: str = "Very Low",
        controller_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.nav = nav

        base_kwargs: Dict[str, Any] = dict(
            width=width,
            height=height,
            server_timeout=server_timeout,
            renderDepthImage=False,
            renderNormalsImage=False,
            renderFlowImage=False,
            renderInstanceSegmentation=False,
            renderSemanticSegmentation=False,
            quality=quality,
        )

        # Some ai2thor builds support `headless`. If yours doesn't, remove it and use xvfb-run.
        base_kwargs["headless"] = headless

        if controller_kwargs:
            base_kwargs.update(controller_kwargs)

        self.controller = Controller(**base_kwargs)
        self.last_event = self.controller.last_event

        if scene is not None:
            self.reset(scene)

    def reset(self, scene: Dict[str, Any]) -> None:
        """
        Reset to a ProcTHOR/Objaverse scene JSON. Does not run any calibration.
        Ensures agent pose uses scene["metadata"]["agent"] if present.
        """
        self.last_event = self.controller.reset(scene=scene)

        agent_meta = scene.get("metadata", {}).get("agent", None)
        if agent_meta is not None:
            # If TeleportFull fails, we still proceed, but pose may be whatever reset produced.
            ev = self.controller.step(action="TeleportFull", **agent_meta)
            self.last_event = ev

        # Optional: cache reachable positions once (expensive, but only once)
        # if self.nav.grid_size is not None:
        #     self.controller.step(action="GetReachablePositions", gridSize=self.nav.grid_size)

    def stop(self) -> None:
        self.controller.stop()

    def get_pose_xzt(self) -> Tuple[float, float, float]:
        """
        Returns (x, z, yaw_degrees).
        """
        agent = self.controller.last_event.metadata["agent"]
        x = agent["position"]["x"]
        z = agent["position"]["z"]
        yaw = agent["rotation"]["y"]
        return (x, z, yaw)

    def step_nav(self, action: Union[str, Any]) -> Any:
        """
        Fast navigation-only step. Supports:
        - "m"/"b"/"l"/"r"
        - "move_ahead"/"move_back"/"rotate_left"/"rotate_right"
        - your THORActions values (since they often stringify to similar tokens)
        """
        a = str(action)

        if a in MOVE_AHEAD_ALIASES:
            ev = self.controller.step(
                action="MoveAgent",
                ahead=self.nav.ahead,
                renderImageSynthesis=False,
            )
        elif a in MOVE_BACK_ALIASES:
            ev = self.controller.step(
                action="MoveAgent",
                ahead=-self.nav.ahead,
                renderImageSynthesis=False,
            )
        elif a in ROT_LEFT_ALIASES:
            ev = self.controller.step(
                action="RotateAgent",
                degrees=-self.nav.rot_deg,
                renderImageSynthesis=False,
            )
        elif a in ROT_RIGHT_ALIASES:
            ev = self.controller.step(
                action="RotateAgent",
                degrees=self.nav.rot_deg,
                renderImageSynthesis=False,
            )
        else:
            raise ValueError(f"Unknown nav action: {action}")

        self.last_event = ev
        return ev




def load_houses(subset="val"):
    return prior.load_dataset(
        dataset="spoc-data",
        entity="spoc-robot",
        revision="local-objaverse-procthor-houses",
        path_to_splits=None,
        split_to_path={
            k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz")
            for k in ["train", "val", "test"]
        },
        max_houses_per_split=int(1e9),
    )[subset]


def get_state(ctrl: StretchController) -> dict:
    agent = ctrl.get_current_agent_full_pose()
    arm_rel = ctrl.get_relative_stretch_current_arm_state()
    return {
        "agent_position": agent["position"],
        "agent_rotation": agent["rotation"],
        "arm_relative": arm_rel,
        "wrist_rotation": ctrl.get_arm_wrist_rotation(),
        "wrist_position": ctrl.get_arm_wrist_position(),
        "held_objects": ctrl.get_held_objects(),
        "pickupable_nearby": ctrl.get_objects_in_hand_sphere(),
    }

def remove_objects_by_id(house_dict, ids_to_remove):
    ids_to_remove = set(ids_to_remove)
    house_dict = dict(house_dict)  # shallow copy

    new_objs = []
    for o in house_dict.get("objects", []):
        if o.get("id") in ids_to_remove:
            continue
        new_objs.append(o)

    house_dict["objects"] = new_objs
    return house_dict

def remove_windows_by_id(house_dict, ids_to_remove):
    ids_to_remove = set(ids_to_remove)
    house_dict = dict(house_dict)  # shallow copy
    house_dict["windows"] = [w for w in house_dict.get("windows", [])
                             if w.get("id") not in ids_to_remove]
    return house_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--house_index", type=int, default=9)
    parser.add_argument("--subset", type=str, default="val")
    parser.add_argument("--actions", nargs="+", default=[
        "move_ahead", "move_ahead", "rotate_right", "move_ahead",
        "move_arm_up", "move_arm_out", "pickup"
    ], help="List of action strings to execute")
    parser.add_argument("--output", type=str, default="action_results.json")
    args = parser.parse_args()

    # Load house
    houses_lazy = load_houses()
    houses = list(houses_lazy)          # materialize into a normal list of dicts

    houses[9] = remove_objects_by_id(houses[9], ["ObjaFoldingChair|2|2"])
    houses[9] = remove_windows_by_id(houses[9], ["window|2|1"])

    house = houses[args.house_index]

    # Init controller
    ctrl_args = STRETCH_ENV_ARGS.copy()
    ctrl_args['width'] = 1280
    ctrl_args['height'] = 720
    # ctrl_args["renderInstanceSegmentation"] = False
    # ctrl_args["headless"] = True

    start_pose = dict(
    x=5.0664897,
    y=0.9,          # agent base height in your scenes (often 0.9 for Stretch)
    z=8.38590813,
    rotation=dict(x=0.0, y=240.0, z=0.0),
    horizon=0.0,
    standing=True,  # IMPORTANT: TeleportFull requires this in your build
)

    ctrl = StretchController(scene=house, **ctrl_args)
    ev = ctrl.controller.step(action = "TeleportFull", **start_pose)
    print(ev.metadata["lastActionSuccess"])
    if not ev.metadata["lastActionSuccess"]:
        raise RuntimeError(ev.metadata.get("errorMessage", "TeleportFull failed"))

    trajectory = []
    actions_to_execute =['m'] *50
    start_time = time.time()
    for action in actions_to_execute:
        event = ctrl.agent_step(action)
        full_pose = ctrl.get_current_agent_full_pose() #(x, y, z) *y is height
        current_state = (full_pose['position']['x'], full_pose['position']['z'], full_pose['rotation']['y']) #(x, y, theta in degrees)
        trajectory.append((current_state[0], 0.9, current_state[1]))
        print(f"State after action: {current_state}")
    end_time = time.time()
    print(f"Total time for executing actions: {end_time - start_time:.2f} seconds")

    top_down = ctrl.get_top_down_path_view(
            agent_path=trajectory)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
        
    # Show top-down view
    ax.imshow(top_down)
    ax.set_title(f"House Top Down View")
    ax.axis('off')

    plt.show()

    ctrl.stop()


if __name__ == "__main__":
    main()