from __future__ import annotations
import time
import os
import json
import argparse
import prior
from wandb import controller
import math
from typing import Dict, Any, Tuple, Optional
from environment.stretch_controller import StretchController
from utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
from utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
from utils.type_utils import THORActions

import matplotlib.pyplot as plt


from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from ai2thor.controller import Controller


def _get_thor_controller_from_stretch(ctrl) -> "Controller":
    """
    Try common attribute names used by wrappers around ai2thor.controller.Controller.
    Adjust this list if your StretchController stores it differently.
    """
    for name in ["controller", "thor_controller", "_controller", "ai2thor_controller"]:
        c = getattr(ctrl, name, None)
        if c is not None:
            return c
    raise AttributeError(
        "Could not find underlying ai2thor Controller inside StretchController. "
        "Add your controller attribute name to _get_thor_controller_from_stretch()."
    )

def _pose_from_full_pose(full_pose: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert your get_current_agent_full_pose() output into TeleportFull kwargs.
    TeleportFull expects (x,y,z) + rotation + horizon in some builds.
    We'll include what we can; extra keys are ignored in many builds.
    """
    pos = full_pose["position"]
    rot = full_pose["rotation"]
    out = dict(
        x=float(pos["x"]),
        y=float(pos.get("y", 0.9)),   # y is height; your scenes often use ~0.9
        z=float(pos["z"]),
        rotation=dict(y=float(rot.get("y", 0.0))),
    )
    # If available, include horizon (some builds require it)
    if "cameraHorizon" in full_pose:
        out["horizon"] = float(full_pose["cameraHorizon"])
    return out

def _forward_unit_from_yaw_deg(yaw_deg: float) -> Tuple[float, float]:
    """
    AI2-THOR convention: yaw=0 faces +z, yaw=90 faces +x.
    So forward = (sin(yaw), cos(yaw)) in (x,z).
    """
    th = math.radians(float(yaw_deg))
    return (math.sin(th), math.cos(th))

def _teleport_full(thor: "Controller", base_pose: Dict[str, Any]) -> bool:
    """
    TeleportFull collision oracle: returns True if teleport succeeds.
    """
    ev = thor.step(action="TeleportFull", standing = True, **base_pose)
    return bool(ev.metadata.get("lastActionSuccess", False))

def _probe_max_free_displacement(
    thor: "Controller",
    start_pose: Dict[str, Any],
    dx: float,
    dz: float,
    *,
    max_dist: float = 2.0,
    tol: float = 1e-4,
    max_iter: int = 30,
) -> float:
    """
    Find maximum t in [0, max_dist] s.t. TeleportFull to start + t*(dx,dz) succeeds.
    Assumes start_pose itself is collision-free.
    """
    # quick sanity: start must be valid
    if not _teleport_full(thor, start_pose):
        raise RuntimeError("Start pose is already colliding; cannot probe clearance.")

    # 1) bracket [lo, hi] where lo succeeds and hi fails (or hi=max_dist succeeds)
    lo = 0.0
    hi = max_dist

    test_pose = dict(start_pose)
    test_pose["x"] = float(start_pose["x"] + hi * dx)
    test_pose["z"] = float(start_pose["z"] + hi * dz)

    if _teleport_full(thor, test_pose):
        # no collision within max_dist
        return hi

    # 2) binary search
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        test_pose["x"] = float(start_pose["x"] + mid * dx)
        test_pose["z"] = float(start_pose["z"] + mid * dz)

        ok = _teleport_full(thor, test_pose)
        if ok:
            lo = mid
        else:
            hi = mid

        if (hi - lo) <= tol:
            break

    return lo

def measure_thor_pose_collision_offset(
    ctrl,
    *,
    nav_step: float = 0.25,          # your MoveAgent "ahead" magnitude
    settle_back: float = 0.05,       # step back slightly from contact before probing
    drive_max_steps: int = 80,       # how many forward steps to try to find a wall
    max_dist: float = 0.8,           # probe distance each direction once near obstacle
    tol: float = 1e-4,
    max_iter: int = 30,
) -> Dict[str, float]:
    """
    Automatically moves the agent to a near-contact configuration, then measures
    directional clearance asymmetry along the agent's local forward axis.

    Procedure:
      1) Repeatedly MoveAgent ahead until action fails or pose stops changing.
      2) Move back by `settle_back` to avoid being exactly at contact.
      3) Binary-search (TeleportFull/Teleport) max free displacement forward/backward.

    Returns:
      {
        "t_forward_max": ...,
        "t_backward_max": ...,
        "asymmetry": t_forward_max - t_backward_max,
        "contact_pose_x": ...,
        "contact_pose_z": ...,
        "yaw": ...,
      }
    """
    thor = _get_thor_controller_from_stretch(ctrl)

    # --- helper: get current pose in TeleportFull format + yaw ---
    def current_base_pose_and_yaw():
        full_pose = ctrl.get_current_agent_full_pose()
        base = _pose_from_full_pose(full_pose)
        yaw = float(full_pose["rotation"]["y"])
        return base, yaw

    # --- 1) Drive forward until blocked / no motion ---
    last_xz = None
    blocked = False
    for _ in range(drive_max_steps):
        # Use MoveAgent (since that's what you used elsewhere)
        ev = thor.step(action="MoveAgent", ahead=float(nav_step), renderImageSynthesis=False)
        ok = bool(ev.metadata.get("lastActionSuccess", False))

        # read pose after
        agent = thor.last_event.metadata["agent"]
        x = float(agent["position"]["x"])
        z = float(agent["position"]["z"])
        xz = (round(x, 6), round(z, 6))

        if not ok:
            blocked = True
            break

        if last_xz is not None and xz == last_xz:
            # pose stopped changing => effectively blocked
            blocked = True
            break

        last_xz = xz

    # If we didn't get blocked, we can still probe by increasing max_dist,
    # but better to warn via return fields.
    # --- 2) Step back slightly from contact so teleport probing is stable ---
    if blocked and settle_back > 0:
        thor.step(action="MoveAgent", ahead=-float(settle_back), renderImageSynthesis=False)

    # --- 3) Probe forward/backward using Teleport oracle ---
    base, yaw = current_base_pose_and_yaw()
    fx, fz = _forward_unit_from_yaw_deg(yaw)

    t_fwd = _probe_max_free_displacement(
        thor, base, fx, fz, max_dist=max_dist, tol=tol, max_iter=max_iter
    )
    t_bwd = _probe_max_free_displacement(
        thor, base, -fx, -fz, max_dist=max_dist, tol=tol, max_iter=max_iter
    )

    return {
        "t_forward_max": float(t_fwd),
        "t_backward_max": float(t_bwd),
        "asymmetry": float(t_fwd - t_bwd),
        "contact_pose_x": float(base["x"]),
        "contact_pose_z": float(base["z"]),
        "yaw": float(yaw),
        "blocked_found": float(1.0 if blocked else 0.0),
    }


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--house_index", type=int, default=0)
    parser.add_argument("--subset", type=str, default="val")
    parser.add_argument("--actions", nargs="+", default=[
        "move_ahead", "move_ahead", "rotate_right", "move_ahead",
        "move_arm_up", "move_arm_out", "pickup"
    ], help="List of action strings to execute")
    parser.add_argument("--output", type=str, default="action_results.json")
    args = parser.parse_args()

    # Load house
    houses = list(load_houses(args.subset))
    house = houses[args.house_index]

    # Init controller
    ctrl_args = STRETCH_ENV_ARGS.copy()
    ctrl_args['width'] = 1280
    ctrl_args['height'] = 720

    ctrl = StretchController(scene=house, **ctrl_args)

    # Move to a spot where you KNOW you are close to an obstacle in both directions
    # (e.g., after a few steps of your "m" sequence), then call:
    info = measure_thor_pose_collision_offset(ctrl, max_dist=1.0, tol=1e-4, max_iter=30)
    print("OFFSET PROBE:", info)