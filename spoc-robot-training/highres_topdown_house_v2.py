# save as: highres_topdown_export.py
import os
import argparse
import numpy as np
import prior

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio

from environment.stretch_controller import StretchController
from utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
from utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR


def remove_objects_by_id(house_dict, ids_to_remove):
    ids_to_remove = set(ids_to_remove)
    house_dict = dict(house_dict)
    house_dict["objects"] = [o for o in house_dict.get("objects", []) if o.get("id") not in ids_to_remove]
    return house_dict


def remove_windows_by_id(house_dict, ids_to_remove):
    ids_to_remove = set(ids_to_remove)
    house_dict = dict(house_dict)
    house_dict["windows"] = [w for w in house_dict.get("windows", []) if w.get("id") not in ids_to_remove]
    return house_dict


def load_houses(split: str):
    split = split.lower()
    assert split in ("train", "val", "test")
    ds = prior.load_dataset(
        dataset="spoc-data",
        entity="spoc-robot",
        revision="local-objaverse-procthor-houses",
        path_to_splits=None,
        split_to_path={k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz") for k in ["train", "val", "test"]},
        max_houses_per_split=int(1e9),
    )
    return list(ds[split])


def init_controller_for_house(house_dict, res: int, timeout_s: int):
    controller_args = STRETCH_ENV_ARGS.copy()
    controller_args["renderInstanceSegmentation"] = False
    controller_args["server_timeout"] = int(timeout_s)
    controller_args["width"] = int(res)
    controller_args["height"] = int(res)
    return StretchController(scene=house_dict, **controller_args)


def teleport_agent(ctrl, x, z, y=0.9, yaw_deg=0.0, horizon_deg=30.0, standing=True):
    evt = ctrl.step(
        action="TeleportFull",
        position={"x": float(x), "y": float(y), "z": float(z)},
        rotation={"x": 0.0, "y": float(yaw_deg), "z": 0.0},
        horizon=float(horizon_deg),
        standing=bool(standing),
    )
    if not evt.metadata.get("lastActionSuccess", False):
        msg = evt.metadata.get("errorMessage", "Teleport failed (unknown reason)")
        raise RuntimeError(f"TeleportFull failed: {msg}")


def parse_goals_arg(goals_str: str):
    """
    Parse: "x,z,r; x,z,r; ..."
    Returns list[(x,z,r)]
    """
    goals = []
    if not goals_str:
        return goals
    chunks = [c.strip() for c in goals_str.split(";") if c.strip()]
    for c in chunks:
        parts = [p.strip() for p in c.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Bad goals chunk '{c}'. Expected 'x,z,r'.")
        goals.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return goals


def get_world_bounds_from_reachable(ctrl, pad=0.25):
    """
    Align overlay with the topdown map by using reachable-position bounds.
    """
    evt = ctrl.step(action="GetReachablePositions")
    if not evt.metadata.get("lastActionSuccess", False):
        return None

    pts = evt.metadata.get("actionReturn", None)
    if not pts:
        return None

    xs = [float(p["x"]) for p in pts]
    zs = [float(p["z"]) for p in pts]
    minx, maxx = min(xs) - pad, max(xs) + pad
    minz, maxz = min(zs) - pad, max(zs) + pad
    return (minx, maxx, minz, maxz)


def export_one_house(
    houses,
    house_index: int,
    outdir: str,
    res: int,
    timeout_s: int,
    show: bool,
    save_annotated: bool,
    goals_world,
    teleport_xyz=None,
    teleport_yaw=0.0,
    teleport_horizon=30.0,
    flip_z=False,
    swap_xz=False,
):
    os.makedirs(outdir, exist_ok=True)
    house = houses[house_index]

    # your special-case fixes
    if house_index == 9:
        house = remove_objects_by_id(house, ["ObjaFoldingChair|2|2"])
        house = remove_windows_by_id(house, ["window|2|1"])
    elif house_index == 152:
        house = remove_objects_by_id(house, ["FloorLamp|3|1"])
        house = remove_objects_by_id(house, ["ObjaWheelchair|2|3"])
        house = remove_objects_by_id(house, ["ObjaTrunk|3|3"])
        house = remove_objects_by_id(house, ["chair-diningtable-2|2|2|2"])
        house = remove_objects_by_id(house, ["Bowl|3|30"])
        house = remove_objects_by_id(house, ["SideTable|2|4"])

    ctrl = None
    try:
        ctrl = init_controller_for_house(house, res=res, timeout_s=timeout_s)

        if teleport_xyz is not None:
            tx, tz, ty = teleport_xyz
            teleport_agent(
                ctrl, x=tx, z=tz, y=ty,
                yaw_deg=teleport_yaw, horizon_deg=teleport_horizon
            )

        top_down = ctrl.get_top_down_path_view(agent_path=[])
        img = top_down if isinstance(top_down, np.ndarray) else np.array(top_down)

        if img.dtype != np.uint8:
            if img.max() <= 1.5:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                img = np.clip(img, 0.0, 255.0).astype(np.uint8)

        # Save raw (no overlays)
        raw_path = os.path.join(outdir, f"house_{house_index:04d}_topdown_raw_{img.shape[1]}x{img.shape[0]}.png")
        imageio.imwrite(raw_path, img)
        print(f"[OK] saved raw -> {raw_path}")

        if save_annotated:
            bounds = get_world_bounds_from_reachable(ctrl, pad=0.25)
            if bounds is None:
                raise RuntimeError("GetReachablePositions failed; cannot reliably align world coords to topdown.")

            minx, maxx, minz, maxz = bounds

            # We'll use origin="upper" (like normal images)
            # and extent with top=minz, bottom=maxz to keep the image visually correct.
            extent = [minx, maxx, maxz, minz]  # left, right, bottom, top

            # Borderless figure: pure image
            fig = plt.figure(figsize=(img.shape[1] / 200, img.shape[0] / 200), dpi=200)
            ax = fig.add_axes([0, 0, 1, 1])  # fill canvas
            ax.set_axis_off()

            ax.imshow(img, extent=extent, interpolation="nearest", origin="upper")
            ax.set_aspect("equal")

            # draw goals in "world coords", with optional transform knobs
            for (gx, gz, gr) in goals_world:
                xw, zw = gx, gz
                if swap_xz:
                    xw, zw = zw, xw
                if flip_z:
                    zw = (minz + maxz) - zw

                ax.add_patch(
                    Circle(
                        (xw, zw),
                        radius=gr,
                        facecolor="lime",
                        edgecolor="lime",
                        alpha=0.25,
                        linewidth=2.0,
                        zorder=10,
                    )
                )

            ann_path = os.path.join(outdir, f"house_{house_index:04d}_topdown_goals.png")
            fig.savefig(ann_path, dpi=200, bbox_inches="tight", pad_inches=0)
            print(f"[OK] saved annotated -> {ann_path}")

            if show:
                plt.show()
            else:
                plt.close(fig)

    finally:
        if ctrl is not None:
            ctrl.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--outdir", type=str, default="topdowns_highres")
    ap.add_argument("--res", type=int, default=1080)
    ap.add_argument("--timeout", type=int, default=30)

    ap.add_argument("--house", type=int, required=True)

    ap.add_argument("--teleport", nargs="+", type=float, default=None,
                    help="Teleport agent: x z [y]. Example: --teleport 7.0 2.0 0.9")
    ap.add_argument("--teleport_yaw", type=float, default=270.0)
    ap.add_argument("--teleport_horizon", type=float, default=30.0)

    ap.add_argument("--goals", type=str, default="",
                    help='World goals as "x,z,r; x,z,r". Example: --goals "7.5,5.5,0.45;1.0,1.75,0.45"')

    ap.add_argument("--flip_z", action="store_true",
                    help="Flip Z around reachable bounds: z_plot = (zmin+zmax) - z. Use if your 2D map's 'up' is larger coord.")
    ap.add_argument("--swap_xz", action="store_true",
                    help="Swap x and z for goals. Use if you accidentally provided goals as z,x,r.")

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save_annotated", action="store_true")

    args = ap.parse_args()

    houses = load_houses(args.split)
    goals_world = parse_goals_arg(args.goals)

    teleport_xyz = None
    if args.teleport is not None:
        if len(args.teleport) < 2:
            raise ValueError("--teleport requires at least x z (and optional y)")
        tx = float(args.teleport[0])
        tz = float(args.teleport[1])
        ty = float(args.teleport[2]) if len(args.teleport) >= 3 else 0.9
        teleport_xyz = (tx, tz, ty)

    export_one_house(
        houses,
        house_index=int(args.house),
        outdir=args.outdir,
        res=args.res,
        timeout_s=args.timeout,
        show=args.show,
        save_annotated=args.save_annotated,
        goals_world=goals_world,
        teleport_xyz=teleport_xyz,
        teleport_yaw=args.teleport_yaw,
        teleport_horizon=args.teleport_horizon,
        flip_z=args.flip_z,
        swap_xz=args.swap_xz,
    )


if __name__ == "__main__":
    main()