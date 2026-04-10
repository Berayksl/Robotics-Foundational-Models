#geometry helpers for collusion detection in the simulation
import numpy as np
from collections import defaultdict

def _dist_point_to_segment(p, a, b):
    """Euclidean distance from point p to segment a-b. p,a,b are (2,) arrays."""
    ap = p - a
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-12:
        return float(np.linalg.norm(ap))
    t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))

def _point_in_poly(p, poly):
    """
    Ray casting point-in-polygon.
    p: (2,) array, poly: (N,2) array.
    Works for convex/concave simple polygons.
    """
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = float(poly[i][0]), float(poly[i][1])
        x1, y1 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        # edge crosses horizontal ray?
        cond = ((y0 > y) != (y1 > y))
        if cond:
            x_at_y = x0 + (y - y0) * (x1 - x0) / (y1 - y0 + 1e-12)
            if x_at_y > x:
                inside = not inside
    return inside

def _segments_intersect(a, b, c, d):
    """Proper segment intersection test for a-b and c-d. All are (2,) arrays."""
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])

    def on_segment(p, q, r):
        # q on segment p-r
        return (min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12 and
                min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    # general case
    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True

    # collinear cases
    if abs(o1) < 1e-12 and on_segment(a, c, b): return True
    if abs(o2) < 1e-12 and on_segment(a, d, b): return True
    if abs(o3) < 1e-12 and on_segment(c, a, d): return True
    if abs(o4) < 1e-12 and on_segment(c, b, d): return True
    return False


def _circle_poly_collision(center, radius, poly):
    """
    center: (2,) array, radius: float, poly: (N,2)
    """
    # inside
    if _point_in_poly(center, poly):
        return True

    # edge distance
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if _dist_point_to_segment(center, a, b) <= radius:
            return True
    return False


def _circle_segment_collision(center, radius, seg_a, seg_b):
    return _dist_point_to_segment(center, seg_a, seg_b) <= radius


def _swept_circle_collision(p0, p1, radius, polys, wall_segments, step=0.02):
    """
    Sample along motion from p0->p1 every ~step meters.
    Returns True if any sampled pose collides.
    """
    dp = p1 - p0
    dist = float(np.linalg.norm(dp))
    if dist < 1e-9:
        # just check static
        return _static_circle_collision(p0, radius, polys, wall_segments)

    n = max(2, int(np.ceil(dist / step)) + 1)
    for i in range(n):
        alpha = i / (n - 1)
        p = p0 + alpha * dp
        if _static_circle_collision(p, radius, polys, wall_segments):
            return True
    return False

def _static_circle_collision(p, radius, polys, wall_segments):
    # objects
    for poly in polys:
        if _circle_poly_collision(p, radius, poly):
            return True
    # walls
    for (a, b) in wall_segments:
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        if _circle_segment_collision(p, radius, a, b):
            return True
    return False


def _compute_env_bounds_from_geom(geom):
    """
    Compute x/y bounds from wall segments + object polygons.
    Works even if coordinates start at 0 and vary per house.
    (used for funnel reward calculation)
    """
    xs, ys = [], []

    # wall_segments: list of (p0, p1) where each confirms 2D points
    for (p0, p1) in geom.get("wall_segments", []):
        xs.extend([p0[0], p1[0]])
        ys.extend([p0[1], p1[1]])

    # object_polys: list of polygons (Nx2 arrays)
    for poly in geom.get("object_polys", []):
        poly = np.asarray(poly)
        if poly.ndim == 2 and poly.shape[1] == 2:
            xs.extend(poly[:, 0].tolist())
            ys.extend(poly[:, 1].tolist())

    if len(xs) == 0 or len(ys) == 0:
        # fallback to config width/height if geom doesn't provide info
        return 0.0, float(self.width), 0.0, float(self.height)

    x_min, x_max = float(min(xs)), float(max(xs))
    y_min, y_max = float(min(ys)), float(max(ys))
    return x_min, x_max, y_min, y_max


def _swept_circle_clipped_translation(
    p0, p1, radius, polys, wall_segments, step=0.02, eps=1e-4
):
    """
    Fast 'clipped-at-contact' sweep:
    - samples along p0 -> p1 at ~step meters
    - returns (p_clipped, collided)
      where p_clipped is the farthest collision-free point along the segment
      (backs off by a tiny eps along the direction).
    This approximates: "move until right next to obstacle".
    """

    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)

    dp = p1 - p0
    dist = float(np.linalg.norm(dp))

    # No motion: just static check
    if dist < 1e-9:
        hit = _static_circle_collision(p0, radius, polys, wall_segments)
        return p0.copy(), bool(hit)

    # If already colliding at start, don't move
    if _static_circle_collision(p0, radius, polys, wall_segments):
        return p0.copy(), True

    # Unit direction
    u = dp / (dist + 1e-12)

    # How many samples (include endpoint)
    n = max(2, int(np.ceil(dist / step)) + 1)

    last_safe = p0.copy()
    collided = False

    # Start from i=1 (we already checked p0)
    for i in range(1, n):
        alpha = i / (n - 1)
        p = p0 + alpha * dp

        if _static_circle_collision(p, radius, polys, wall_segments):
            collided = True
            # back off slightly from the collision point toward last_safe
            # use last_safe + (some small retreat) to avoid "touching" numerically
            retreat = max(eps, 0.5 * step)
            p_back = p - retreat * u

            # Ensure p_back is not behind last_safe (numerical safety)
            # If p_back still collides, fall back to last_safe.
            if np.linalg.norm(p_back - p0) < np.linalg.norm(last_safe - p0):
                return last_safe.copy(), True

            if _static_circle_collision(p_back, radius, polys, wall_segments):
                return last_safe.copy(), True

            return p_back.astype(np.float32), True

        last_safe = p

    # No collision: full move
    return p1.copy(), False




#FOR FAST FORWARD PROP.

def _aabb_of_poly(poly):
    # poly: (N,2) array-like
    p = np.asarray(poly, dtype=np.float32)
    mn = p.min(axis=0)
    mx = p.max(axis=0)
    return float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])

def _aabb_of_segment(seg):
    # seg: ((x0,y0),(x1,y1)) or (2,2)
    s = np.asarray(seg, dtype=np.float32).reshape(2,2)
    mn = s.min(axis=0); mx = s.max(axis=0)
    return float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])

class SpatialHash2D:
    def __init__(self, cell_size: float):
        self.h = float(cell_size)
        self.wall_cells = defaultdict(list)   # (ix,iy) -> [segment indices]
        self.poly_cells = defaultdict(list)   # (ix,iy) -> [poly indices]
        self.wall_aabbs = None
        self.poly_aabbs = None

    def _cells_for_aabb(self, xmin, ymin, xmax, ymax):
        ix0 = int(np.floor(xmin / self.h))
        ix1 = int(np.floor(xmax / self.h))
        iy0 = int(np.floor(ymin / self.h))
        iy1 = int(np.floor(ymax / self.h))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                yield (ix, iy)

    def build(self, wall_segments, object_polys, inflate: float = 0.0):
        # cache AABBs
        self.wall_aabbs = []
        for seg in wall_segments:
            xmin, ymin, xmax, ymax = _aabb_of_segment(seg)
            xmin -= inflate; ymin -= inflate; xmax += inflate; ymax += inflate
            self.wall_aabbs.append((xmin, ymin, xmax, ymax))
            for cell in self._cells_for_aabb(xmin, ymin, xmax, ymax):
                self.wall_cells[cell].append(len(self.wall_aabbs) - 1)

        self.poly_aabbs = []
        for poly in object_polys:
            xmin, ymin, xmax, ymax = _aabb_of_poly(poly)
            xmin -= inflate; ymin -= inflate; xmax += inflate; ymax += inflate
            self.poly_aabbs.append((xmin, ymin, xmax, ymax))
            for cell in self._cells_for_aabb(xmin, ymin, xmax, ymax):
                self.poly_cells[cell].append(len(self.poly_aabbs) - 1)

    def query(self, xmin, ymin, xmax, ymax):
        # returns sets of candidate indices
        wall_ids = set()
        poly_ids = set()
        for cell in self._cells_for_aabb(xmin, ymin, xmax, ymax):
            wall_ids.update(self.wall_cells.get(cell, ()))
            poly_ids.update(self.poly_cells.get(cell, ()))
        return wall_ids, poly_ids