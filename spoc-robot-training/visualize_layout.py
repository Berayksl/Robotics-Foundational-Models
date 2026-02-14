#creates and visualizes the layout of the selected house
import os
import sys
import prior
import matplotlib.pyplot as plt
import numpy as np
from ai2thor.controller import Controller


# -------------------------
# THOR large object filtering + geometry
# -------------------------

# A practical allowlist for "large" household furniture/appliances.
# (You can add/remove objectType strings as you see in your metadata.)
LARGE_OBJECT_KEYWORDS = {
    # kitchen / appliances
    "fridge", "refrigerator", "microwave", "oven", "stove", "dishwasher", "trash", "garbage",
    "sink", "counter",

    # furniture
    "table", "sofa", "couch", "chair", "armchair", "bed", "stand",
    "dresser", "shelf", "bookshelf", "cabinet", "nightstand", "plant", "shelves",

    # electronics
    "tv", "television", "stand", "floorlamp",

    # bathroom / laundry
    "toilet", "bathtub", "shower", "washer", "dryer"
}

# Types you almost never want to paint as "obstacles" from THOR objects
IGNORE_OBJECT_TYPES = {
    "floor", "wall", "doorway", "room"  # you already render walls/doors from house_dict
}


def is_large_house_object(obj: dict) -> bool:
    t = obj.get("objectType", "")
    t_low = t.lower()

    if t_low in IGNORE_OBJECT_TYPES:
        return False

    return any(keyword in t_low for keyword in LARGE_OBJECT_KEYWORDS)

def _poly_area(xz: np.ndarray) -> float:
    """Polygon area for Nx2 points (assumed ordered, not necessarily closed)."""
    if xz is None or len(xz) < 3:
        return 0.0
    x = xz[:, 0]
    z = xz[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))))

def footprint_from_obb(obj: dict):
    """
    Return XZ footprint polygon (Nx2) from objectOrientedBoundingBox corner points.
    If missing, return None.
    """
    obb = obj.get("objectOrientedBoundingBox", None)
    if not obb or "cornerPoints" not in obb or not obb["cornerPoints"]:
        return None

    corners = np.array([[p[0], p[2]] for p in obb["cornerPoints"]], dtype=np.float32)  # (x,z)

    # OBB has 8 corners; footprint is the 4 lowest-y corners, but we only have x,z here.
    # Instead: pick the 4 points that form the convex hull in XZ.
    # Simple approach: use hull-like sorting by angle around centroid.
    c = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    order = np.argsort(angles)
    poly = corners[order]

    # Many corners repeat in angle sort; keep unique (within tolerance)
    uniq = []
    for p in poly:
        if not uniq or np.linalg.norm(p - uniq[-1]) > 1e-4:
            uniq.append(p)
    poly = np.array(uniq, dtype=np.float32)

    # Keep only up to 8; ideally hull would be 4. If we got more than 4,
    # we can downselect by taking the convex hull approx via removing near-collinear points.
    # For plotting, this is usually fine.
    if len(poly) < 3:
        return None
    return poly

def footprint_from_aabb(obj: dict):
    """
    Return XZ footprint rectangle (4x2) from axisAlignedBoundingBox.
    """
    aabb = obj.get("axisAlignedBoundingBox", None)
    if not aabb:
        return None
    c = aabb.get("center", None)
    s = aabb.get("size", None)
    if not c or not s:
        return None

    cx, cz = float(c["x"]), float(c["z"])
    sx, sz = float(s["x"]), float(s["z"])
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    z0, z1 = cz - sz / 2.0, cz + sz / 2.0

    return np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]], dtype=np.float32)

def get_large_object_footprint(obj: dict):
    """
    Prefer OBB footprint if available (more accurate under rotation),
    else fall back to AABB.
    Returns (poly_xz, source_str).
    """
    poly = footprint_from_obb(obj)
    if poly is not None and _poly_area(poly) > 1e-6:
        return poly, "OBB"
    poly = footprint_from_aabb(obj)
    if poly is not None and _poly_area(poly) > 1e-6:
        return poly, "AABB"
    return None, None

def draw_large_objects(ax, thor_meta, alpha=0.35):
    """
    Overlay large objects on ax using their footprints.
    """
    objs = thor_meta.get("objects", [])
    kept = []
    for o in objs:
        if not is_large_house_object(o):
            continue

        poly, src = get_large_object_footprint(o)
        if poly is None:
            continue

        kept.append((o, poly, src))

    # Plot
    for o, poly, src in kept:
        xs = poly[:, 0].tolist()
        zs = poly[:, 1].tolist()
        xs, zs = _close(xs, zs)

        ax.fill(xs, zs, alpha=alpha, zorder=3)   # no explicit color to respect your defaults
        ax.plot(xs, zs, linewidth=2, zorder=4)

        # label at centroid
        cx, cz = float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))
        label = o.get("objectType", "Obj")
        ax.text(cx, cz, label, fontsize=8, ha="center", va="center", zorder=5)

    return kept


def get_thor_metadata_for_house(house_dict, width=640, height=480):
    c = Controller(headless=True, width=width, height=height)
    # Initialize first (some builds require Initialize before CreateHouse; yours does CreateHouse after Initialize)
    c.reset(scene="Procedural")  # or just c.reset() in newer versions
    c.step(action="CreateHouse", house=house_dict)
    event = c.last_event
    return c, event.metadata



sys.path.append('/home/bera/Desktop/Codes/SPOC/spoc-robot-training/Evaluation')

def load_objaverse_houses():
    subset_to_load = "val"
    return prior.load_dataset(
        dataset="spoc-data",
        entity="spoc-robot",
        revision="local-objaverse-procthor-houses",
        path_to_splits=None,
        split_to_path={
            k: os.path.join(
                '/home/bera/Desktop/Codes/SPOC/spoc-robot-training/Evaluation/objaverse_houses',
                f"{k}.jsonl.gz"
            )
            for k in ["train", "val", "test"]
        },
        max_houses_per_split=int(1e9),
    )[subset_to_load]


# -------------------------
# Door logic helpers  (NEW)
# -------------------------
def is_door_open(door: dict, default_open_if_missing=True, thresh=0.5) -> bool:
    """
    Treat doors with missing 'openness' as OPEN (passable) by default.
    If 'openness' exists, openness > thresh => open.
    """
    if "openness" in door:
        try:
            return float(door.get("openness", 0.0)) > thresh
        except Exception:
            return False
    return bool(default_open_if_missing)


# -------------------------
# Geometry helpers
# -------------------------

def _xz(poly):
    """Convert [{'x','y','z'},...] -> Nx2 array [[x,z],...]"""
    return np.array([[p["x"], p["z"]] for p in poly], dtype=np.float32)

def _close(xs, zs):
    if len(xs) == 0:
        return xs, zs
    return xs + [xs[0]], zs + [zs[0]]

def _polygon_center(poly_xz):
    return float(np.mean(poly_xz[:, 0])), float(np.mean(poly_xz[:, 1]))

def parse_wall_id_endpoints(wall_id: str):
    """
    Wall ids encode world endpoints in their last 4 fields:
      'wall|2|2.17|2.17|6.52|2.17'  -> (2.17,2.17) to (6.52,2.17)
      'wall|exterior|0.00|0.00|6.52|0.00' -> (0,0) to (6.52,0)
    We interpret those as (x1,z1,x2,z2) in world coordinates (meters).
    """
    if not wall_id:
        return None
    parts = str(wall_id).split("|")
    if len(parts) < 5:
        return None
    try:
        x1, z1, x2, z2 = map(float, parts[-4:])
        return (x1, z1), (x2, z2)
    except Exception:
        return None

def wall_world_segment(wall_dict):
    """
    Prefer wall id parsing because it's consistent and avoids any ambiguity in polygon projection.
    Fallback to polygon->segment if needed.
    """
    wid = wall_dict.get("id", "")
    ends = parse_wall_id_endpoints(wid)
    if ends is not None:
        return wid, ends[0], ends[1]

    # Fallback: derive from wall polygon xz projection
    poly = wall_dict.get("polygon", None)
    if not poly:
        return None
    pts = _xz(poly)
    uniq = np.unique(pts, axis=0)
    if len(uniq) < 2:
        return None
    # farthest pair
    dmax, a, b = -1.0, None, None
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            d = float(np.sum((uniq[i] - uniq[j]) ** 2))
            if d > dmax:
                dmax, a, b = d, uniq[i], uniq[j]
    return wid, (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))

def door_opening_world_segment(door):
    """
    KEY FIX:
    door['holePolygon'] is in WALL-LOCAL coordinates:
      - holePolygon[*].x is distance ALONG the wall (meters)
      - holePolygon[*].y is vertical height
      - holePolygon[*].z is ~0 (on wall plane), NOT world z

    So: take wall endpoints from door.wall0 (or wall1), compute direction,
    then map along-wall distances into world XZ points.
    """
    hole = door.get("holePolygon", None)
    if not hole or len(hole) < 2:
        return None

    wall_id = door.get("wall0") or door.get("wall1")
    ends = parse_wall_id_endpoints(wall_id)
    if ends is None:
        return None
    (sx, sz), (ex, ez) = ends

    v = np.array([ex - sx, ez - sz], dtype=np.float32)
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return None
    d = v / L  # unit along-wall direction in world xz

    # along-wall distances (meters)
    u0 = float(hole[0]["x"])
    u1 = float(hole[1]["x"])

    p0 = (sx + d[0] * u0, sz + d[1] * u0)
    p1 = (sx + d[0] * u1, sz + d[1] * u1)

    return p0, p1, wall_id

def _project_param_on_segment(p, a, b):
    """Return t for projection of p onto segment a->b, where a,b,p are (x,z)."""
    ax, az = a
    bx, bz = b
    px, pz = p
    vx, vz = (bx - ax), (bz - az)
    denom = vx * vx + vz * vz
    if denom < 1e-9:
        return 0.0
    t = ((px - ax) * vx + (pz - az) * vz) / denom
    return float(t)

def _clip01(t):
    return max(0.0, min(1.0, t))

def _split_segment_by_open_intervals(a, b, intervals, eps=1e-6):
    """
    Split segment a->b by a list of open intervals in parameter t.
    intervals: list of (t0,t1) with t0<=t1 inside [0,1]
    Returns list of segments to draw as walls: [(p0,p1),...]
    """
    if not intervals:
        return [(a, b)]

    intervals = sorted([(max(0.0, t0), min(1.0, t1)) for t0, t1 in intervals], key=lambda x: x[0])
    merged = []
    for t0, t1 in intervals:
        if t1 - t0 < eps:
            continue
        if not merged or t0 > merged[-1][1] + eps:
            merged.append([t0, t1])
        else:
            merged[-1][1] = max(merged[-1][1], t1)

    keep = []
    cur = 0.0
    for t0, t1 in merged:
        if t0 > cur + eps:
            keep.append((cur, t0))
        cur = max(cur, t1)
    if cur < 1.0 - eps:
        keep.append((cur, 1.0))

    ax, az = a
    bx, bz = b
    segs = []
    for s0, s1 in keep:
        p0 = (ax + (bx - ax) * s0, az + (bz - az) * s0)
        p1 = (ax + (bx - ax) * s1, az + (bz - az) * s1)
        segs.append((p0, p1))
    return segs

def _build_wall_openings_from_doors(house_dict):
    """
    Map wall_id -> list of open intervals along that wall.
    CHANGED: if a door has no 'openness', treat it as OPEN.
    """
    openings = {}  # wall_id -> list[(t0,t1)]
    doors = house_dict.get("doors", [])

    for d in doors:
        # CHANGED HERE
        if not is_door_open(d, default_open_if_missing=True):
            continue

        out = door_opening_world_segment(d)
        if out is None:
            continue
        (p0, p1, wall_id) = out

        ends = parse_wall_id_endpoints(wall_id)
        if ends is None:
            continue
        a, b = ends

        t0 = _clip01(_project_param_on_segment(p0, a, b))
        t1 = _clip01(_project_param_on_segment(p1, a, b))
        if t1 < t0:
            t0, t1 = t1, t0

        openings.setdefault(wall_id, []).append((t0, t1))

    return openings


# -------------------------
# Visualization
# -------------------------
def visualize_house_structure(house_dict, house_index, thor_meta=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    ax1.set_title(f"House {house_index} - Rooms + Walls (Door Openings Applied)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Z (meters)")
    ax1.grid(True, alpha=0.25)

    # Rooms
    if "rooms" in house_dict and len(house_dict["rooms"]) > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, len(house_dict["rooms"])))
        for idx, room in enumerate(house_dict["rooms"]):
            if "floorPolygon" not in room:
                continue
            room_type = room.get("roomType", "Unknown")
            pts = _xz(room["floorPolygon"])
            xs = pts[:, 0].tolist()
            zs = pts[:, 1].tolist()
            xs, zs = _close(xs, zs)

            ax1.fill(xs, zs, color=colors[idx], alpha=0.35, label=room_type)
            ax1.plot(xs, zs, color=colors[idx], linewidth=2)

            cx, cz = _polygon_center(pts)
            ax1.text(cx, cz, room_type, ha="center", va="center",
                     fontsize=9, fontweight="bold")

        ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)

    # Build openings map to cut walls
    wall_openings = _build_wall_openings_from_doors(house_dict)

    # Walls (draw with gaps for OPEN doors)
    for w in house_dict.get("walls", []):
        out = wall_world_segment(w)
        if out is None:
            continue
        wid, a, b = out

        is_exterior = "exterior" in str(wid).lower()
        lw = 4 if is_exterior else 2.5
        alpha = 0.9 if is_exterior else 0.75

        intervals = wall_openings.get(wid, [])
        keep_segs = _split_segment_by_open_intervals(a, b, intervals)

        for (p0, p1) in keep_segs:
            ax1.plot([p0[0], p1[0]], [p0[1], p1[1]],
                     color="black", linewidth=lw, alpha=alpha, zorder=5)

    # Draw doors explicitly using WORLD coords (cyan=open, red=closed)
    for d in house_dict.get("doors", []):
        out = door_opening_world_segment(d)
        if out is None:
            continue
        (p0, p1, wall_id) = out

        # CHANGED HERE
        open_now = is_door_open(d, default_open_if_missing=True)
        color = "cyan" if open_now else "red"
        width = 6 if open_now else 4

        ax1.plot([p0[0], p1[0]], [p0[1], p1[1]],
                 color=color, linewidth=width, alpha=0.9, zorder=6)

    ax1.set_aspect("equal")

    kept = []
    if thor_meta is not None:
        kept = draw_large_objects(ax1, thor_meta, alpha=0.25)

    ax1.set_aspect("equal")

    # Right panel info
    ax2.axis("off")
    info_lines = [
        f"House Index: {house_index}",
        f"\n{'='*44}\n",
        "STRUCTURE:",
    ]
    if "rooms" in house_dict:
        info_lines.append(f"  Total Rooms: {len(house_dict['rooms'])}")
    if "walls" in house_dict:
        ext = sum(1 for w in house_dict["walls"] if "exterior" in str(w.get("id", "")).lower())
        info_lines.append(f"  Total Walls: {len(house_dict['walls'])} (exterior: {ext})")
    if "doors" in house_dict:
        open_ct = sum(1 for d in house_dict["doors"] if is_door_open(d, default_open_if_missing=True))
        info_lines.append(f"  Total Doors: {len(house_dict['doors'])} (open: {open_ct})")

    if thor_meta is not None:
        info_lines.append(f"  Large THOR Objects Plotted: {len(kept)}")
        # optional: show a compact type breakdown
        types = {}
        for o, _, src in kept:
            t = o.get("objectType", "Unknown")
            types[t] = types.get(t, 0) + 1
        info_lines.append("  Types: " + ", ".join([f"{k}({v})" for k, v in sorted(types.items())]))

    info_lines.append(f"\n{'='*44}")
    info_lines.append("\nAVAILABLE DATA:")
    info_lines.append(f"  Dictionary Keys: {list(house_dict.keys())}")
    if thor_meta is not None:
        info_lines.append(f"  THOR keys: {list(thor_meta.keys())}")

    ax2.text(0.05, 0.95, "\n".join(info_lines),
             transform=ax2.transAxes,
             fontsize=10, va="top", fontfamily="monospace")

    plt.tight_layout()
    return fig


# -------------------------
# Browser / CLI
# -------------------------
def browse_houses_interactive():
    print("Loading houses...")
    houses = list(load_objaverse_houses())
    print(f"Loaded {len(houses)} houses")

    current_index = 2

    print("\n" + "=" * 60)
    print("EVALUATION HOUSES BROWSER")
    print("=" * 60)
    print("Controls:")
    print("  n/Enter : Next house")
    print("  p       : Previous house")
    print("  s       : Show detailed info")
    print("  q       : Quit")
    print("  [number]: Jump to house index")
    print("=" * 60)

    plt.ion()
    fig = None

    while True:
        if fig is not None:
            plt.close(fig)

        try:
            print(f"\n--- Visualizing House {current_index} ---")
            house_dict = houses[current_index]

            controller, thor_meta = get_thor_metadata_for_house(house_dict)
            controller.stop()

            fig = visualize_house_structure(house_dict, current_index, thor_meta=thor_meta)

            # important: cleanly close controller so you don't leak Unity processes
            controller.stop()

            plt.show()
            plt.pause(0.1)
        except Exception as e:
            print(f"Error visualizing house {current_index}: {e}")
            import traceback
            traceback.print_exc()

        command = input(f"\nCurrent: House {current_index} | Command: ").strip().lower()

        if command in ["n", ""]:
            current_index = (current_index + 1) % len(houses)
        elif command == "p":
            current_index = (current_index - 1) % len(houses)
        elif command == "s":
            print_house_info(houses[current_index])
        elif command == "q":
            print("Exiting...")
            plt.close("all")
            return current_index
        elif command.isdigit():
            new_index = int(command)
            if 0 <= new_index < len(houses):
                current_index = new_index
            else:
                print(f"Invalid index. Must be 0-{len(houses)-1}")
        else:
            houses = list(load_objaverse_houses())
            house = houses[args.house_index]
            controller, thor_meta = get_thor_metadata_for_house(house)
            fig = visualize_house_structure(house, args.house_index, thor_meta=thor_meta)
            controller.stop()
            plt.show(block=True)


def print_house_info(house_dict):

    controller, thor_meta = get_thor_metadata_for_house(house_dict)

    print('House Detailed Info:')
    print("house dictionary:", house_dict, "\n")


    #print("THOR metadata keys:", thor_meta.keys())
    # print("Num objects:", len(thor_meta["objects"]))
    print("Objects:", thor_meta["objects"])

    print("\nHouse Dictionary Keys:", list(house_dict.keys()))
    if "doors" in house_dict:
        print(f"\nDoors ({len(house_dict['doors'])}):")
        for d in house_dict["doors"][:10]:
            out = door_opening_world_segment(d)
            seg = None if out is None else (out[0], out[1])

            # CHANGED HERE
            open_now = is_door_open(d, default_open_if_missing=True)
            print(
                f"  • {d.get('id','N/A')} | openness={d.get('openness','<missing>')} "
                f"| treated_open={open_now} | wall0={d.get('wall0','')} | world_seg={seg}"
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize evaluation houses with doors/walls (correct door coords)")
    parser.add_argument("--mode", choices=["browse", "specific"], default="browse")
    parser.add_argument("--house_index", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "browse":
        browse_houses_interactive()
    else:
        houses = list(load_objaverse_houses())
        house = houses[args.house_index]
        #if args.house_index == 9:
            
        fig = visualize_house_structure(house, args.house_index)
        plt.show(block=True)
