#geometry helpers for collusion detection in the simulation
import numpy as np

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


