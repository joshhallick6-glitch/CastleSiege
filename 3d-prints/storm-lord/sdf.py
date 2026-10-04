"""Tiny signed-distance-field sculpting kit (numpy).

Every primitive is a (fn, bmin, bmax) triple: fn(X, Y, Z) returns the signed
distance (negative inside, millimetres) for broadcastable coordinate arrays,
and bmin/bmax bound the surface so it is only evaluated where it matters.
Fields are dense float32 voxel grids that primitives are blended into.
"""
import numpy as np

BIG = np.float32(1e3)


# --------------------------------------------------------------------------- #
# Vector helpers
# --------------------------------------------------------------------------- #
def v3(p):
    return np.asarray(p, dtype=np.float64)


def unit(p):
    p = v3(p)
    return p / np.linalg.norm(p)


def frame(axis, hint=(0.0, 0.0, 1.0)):
    """Orthonormal (u, v, w) with w along axis."""
    w = unit(axis)
    hint = v3(hint)
    if abs(np.dot(hint, w)) > 0.95:
        hint = v3((1.0, 0.0, 0.0)) if abs(w[0]) < 0.9 else v3((0.0, 1.0, 0.0))
    u = unit(np.cross(hint, w))
    v = np.cross(w, u)
    return u, v, w


def rot(axis, deg):
    """Rotation matrix (Rodrigues)."""
    a = unit(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def local(X, Y, Z, c, M):
    """World -> local coordinates; rows of M are the local axes."""
    dx, dy, dz = X - c[0], Y - c[1], Z - c[2]
    return (M[0, 0] * dx + M[0, 1] * dy + M[0, 2] * dz,
            M[1, 0] * dx + M[1, 1] * dy + M[1, 2] * dz,
            M[2, 0] * dx + M[2, 1] * dy + M[2, 2] * dz)


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
class Prim:
    def __init__(self, fn, bmin, bmax):
        self.fn, self.bmin, self.bmax = fn, v3(bmin), v3(bmax)

    def __call__(self, X, Y, Z):
        return self.fn(X, Y, Z)


def sphere(c, r):
    c = v3(c)
    return Prim(lambda X, Y, Z: np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) - r,
                c - r, c + r)


def ellipsoid(c, radii, M=None):
    """Ellipsoid (iq bound).  M: optional rotation, rows = local axes."""
    c, r = v3(c), v3(radii)
    M = np.eye(3) if M is None else np.asarray(M)

    def fn(X, Y, Z):
        px, py, pz = local(X, Y, Z, c, M)
        k0 = np.sqrt((px / r[0]) ** 2 + (py / r[1]) ** 2 + (pz / r[2]) ** 2)
        k1 = np.sqrt((px / r[0] ** 2) ** 2 + (py / r[1] ** 2) ** 2 + (pz / r[2] ** 2) ** 2)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)

    if np.allclose(M, np.eye(3)):
        return Prim(fn, c - r, c + r)
    R = r.max()
    return Prim(fn, c - R, c + R)


def round_cone(a, b, r1, r2):
    """Cone between spheres (a, r1) and (b, r2) -- exact (iq)."""
    a, b = v3(a), v3(b)
    ba = b - a
    l2 = float(ba @ ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    if a2 <= 1e-9:                      # one sphere swallows the other
        return sphere(a, r1) if r1 >= r2 else sphere(b, r2)
    il2 = 1.0 / l2
    srr = np.sign(rr) * rr * rr

    def fn(X, Y, Z):
        pax, pay, paz = X - a[0], Y - a[1], Z - a[2]
        y = pax * ba[0] + pay * ba[1] + paz * ba[2]
        z = y - l2
        qx, qy, qz = pax * l2 - ba[0] * y, pay * l2 - ba[1] * y, paz * l2 - ba[2] * y
        x2 = qx * qx + qy * qy + qz * qz
        y2 = y * y * l2
        z2 = z * z * l2
        k = srr * x2
        d_tip = np.sqrt(x2 + z2) * il2 - r2
        d_base = np.sqrt(x2 + y2) * il2 - r1
        d_side = (np.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1
        return np.where(np.sign(z) * a2 * z2 > k, d_tip,
                        np.where(np.sign(y) * a2 * y2 < k, d_base, d_side))

    bmin = np.minimum(a - r1, b - r2)
    bmax = np.maximum(a + r1, b + r2)
    return Prim(fn, bmin, bmax)


def capsule(a, b, r):
    return round_cone(a, b, r, r)


def chain(points, radii, k=0.0):
    """Smoothly joined sequence of round cones (a 'lock', limb, strap...)."""
    prims = [round_cone(points[i], points[i + 1], radii[i], radii[i + 1])
             for i in range(len(points) - 1)]
    return union(prims, k)


def placed(prim, ref, at, s):
    """Uniformly scale a primitive by s about `ref` and move `ref` to `at`."""
    ref, at = v3(ref), v3(at)

    def fn(X, Y, Z):
        return s * prim((X - at[0]) / s + ref[0], (Y - at[1]) / s + ref[1],
                        (Z - at[2]) / s + ref[2])

    return Prim(fn, at + (prim.bmin - ref) * s, at + (prim.bmax - ref) * s)


def union(prims, k=0.0):
    bmin = np.min([p.bmin for p in prims], axis=0)
    bmax = np.max([p.bmax for p in prims], axis=0)

    def fn(X, Y, Z):
        d = prims[0](X, Y, Z)
        for p in prims[1:]:
            d = smin(d, p(X, Y, Z), k)
        return d

    return Prim(fn, bmin - k, bmax + k)


def flattened(prim, c, n, s):
    """Squash a primitive by factor s (>1 thins it) along unit normal n through c."""
    c, n = v3(c), unit(n)

    def fn(X, Y, Z):
        t = (X - c[0]) * n[0] + (Y - c[1]) * n[1] + (Z - c[2]) * n[2]
        f = (s - 1.0) * t
        return prim(X + f * n[0], Y + f * n[1], Z + f * n[2]) / s

    return Prim(fn, prim.bmin, prim.bmax)


def bezier(p0, p1, p2, n):
    p0, p1, p2 = v3(p0), v3(p1), v3(p2)
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def bezier3(p0, p1, p2, p3, n):
    p0, p1, p2, p3 = v3(p0), v3(p1), v3(p2), v3(p3)
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def lock(p0, p1, p2, r0, r1, n=5, flat=None, power=1.0, p3=None):
    """Tapered curved spike along a quadratic (or cubic, with p3) Bezier."""
    pts = bezier(p0, p1, p2, n) if p3 is None else bezier3(p0, p1, p2, p3, n)
    t = np.linspace(0, 1, n) ** power
    radii = r0 + (r1 - r0) * t
    pr = chain(pts, radii)
    if flat is not None:
        nrm, s = flat
        pr = flattened(pr, pts[0], nrm, s)
    return pr


def band(c, axis, R, half_w, thick, rnd=0.35):
    """Ring with a rounded-rectangle cross section (arm bands, cuffs)."""
    c = v3(c)
    u, v, w = frame(axis)

    def fn(X, Y, Z):
        dx, dy, dz = X - c[0], Y - c[1], Z - c[2]
        h = dx * w[0] + dy * w[1] + dz * w[2]
        rx, ry, rz = dx - h * w[0], dy - h * w[1], dz - h * w[2]
        rad = np.sqrt(rx * rx + ry * ry + rz * rz)
        qx = np.abs(rad - R) - (thick * 0.5 - rnd)
        qy = np.abs(h) - (half_w - rnd)
        out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
        return out + np.minimum(np.maximum(qx, qy), 0) - rnd

    e = R + thick + half_w
    return Prim(fn, c - e, c + e)


def torus(c, axis, R, r):
    c = v3(c)
    u, v, w = frame(axis)

    def fn(X, Y, Z):
        dx, dy, dz = X - c[0], Y - c[1], Z - c[2]
        h = dx * w[0] + dy * w[1] + dz * w[2]
        rx, ry, rz = dx - h * w[0], dy - h * w[1], dz - h * w[2]
        rad = np.sqrt(rx * rx + ry * ry + rz * rz)
        return np.sqrt((rad - R) ** 2 + h * h) - r

    e = R + r
    return Prim(fn, c - e, c + e)


def cylinder(a, b, r, rnd=0.0):
    """Capped cylinder between a and b with optional edge rounding."""
    a, b = v3(a), v3(b)
    ax = b - a
    L = np.linalg.norm(ax)
    w = ax / L
    m = (a + b) * 0.5

    def fn(X, Y, Z):
        dx, dy, dz = X - m[0], Y - m[1], Z - m[2]
        h = dx * w[0] + dy * w[1] + dz * w[2]
        rx, ry, rz = dx - h * w[0], dy - h * w[1], dz - h * w[2]
        rad = np.sqrt(rx * rx + ry * ry + rz * rz)
        qx = rad - (r - rnd)
        qy = np.abs(h) - (L * 0.5 - rnd)
        out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
        return out + np.minimum(np.maximum(qx, qy), 0) - rnd

    return Prim(fn, np.minimum(a, b) - r, np.maximum(a, b) + r)


def box(c, half, M=None, rnd=0.0):
    c, half = v3(c), v3(half)
    M = np.eye(3) if M is None else np.asarray(M)

    def fn(X, Y, Z):
        px, py, pz = local(X, Y, Z, c, M)
        qx = np.abs(px) - (half[0] - rnd)
        qy = np.abs(py) - (half[1] - rnd)
        qz = np.abs(pz) - (half[2] - rnd)
        out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
        return out + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - rnd

    R = np.linalg.norm(half)
    return Prim(fn, c - R, c + R)


def polygon2d(px, py, verts):
    """Signed distance to a closed 2D polygon (iq), vectorised."""
    V = np.asarray(verts, dtype=np.float64)
    d = (px - V[0, 0]) ** 2 + (py - V[0, 1]) ** 2
    s = np.ones_like(d)
    n = len(V)
    for i in range(n):
        j = (i - 1) % n
        ex, ey = V[j, 0] - V[i, 0], V[j, 1] - V[i, 1]
        wx, wy = px - V[i, 0], py - V[i, 1]
        t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        bx, by = wx - ex * t, wy - ey * t
        d = np.minimum(d, bx * bx + by * by)
        c1 = py >= V[i, 1]
        c2 = py < V[j, 1]
        c3 = ex * wy > ey * wx
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s = np.where(flip, -s, s)
    return s * np.sqrt(d)


def extrude(verts, c, u, v, thick, rnd=0.3, bevel=0.0):
    """Extrude a 2D polygon (in the u/v plane through c) to a slab."""
    c, u, v = v3(c), unit(u), unit(v)
    n = np.cross(u, v)
    V = np.asarray(verts, dtype=np.float64)

    def fn(X, Y, Z):
        dx, dy, dz = X - c[0], Y - c[1], Z - c[2]
        pu = dx * u[0] + dy * u[1] + dz * u[2]
        pv = dx * v[0] + dy * v[1] + dz * v[2]
        pn = dx * n[0] + dy * n[1] + dz * n[2]
        d2 = polygon2d(pu, pv, V) + rnd + bevel * np.abs(pn)
        qy = np.abs(pn) - (thick * 0.5 - rnd)
        out = np.sqrt(np.maximum(d2, 0) ** 2 + np.maximum(qy, 0) ** 2)
        return out + np.minimum(np.maximum(d2, qy), 0) - rnd

    R = np.max(np.linalg.norm(V, axis=1)) + thick
    return Prim(fn, c - R, c + R)


# --------------------------------------------------------------------------- #
# Voxel grid + fields
# --------------------------------------------------------------------------- #
class Grid:
    def __init__(self, lo, hi, h):
        self.lo = v3(lo)
        self.h = float(h)
        self.n = (np.ceil((v3(hi) - self.lo) / h).astype(int) + 1)
        self.axes = [(self.lo[i] + np.arange(self.n[i]) * h).astype(np.float32) for i in range(3)]

    def region(self, bmin, bmax):
        i0 = np.clip(np.floor((v3(bmin) - self.lo) / self.h).astype(int), 0, self.n)
        i1 = np.clip(np.ceil((v3(bmax) - self.lo) / self.h).astype(int) + 1, 0, self.n)
        if np.any(i1 <= i0):
            return None
        slc = tuple(slice(int(a), int(b)) for a, b in zip(i0, i1))
        X = self.axes[0][slc[0]][:, None, None]
        Y = self.axes[1][slc[1]][None, :, None]
        Z = self.axes[2][slc[2]][None, None, :]
        return slc, X, Y, Z

    def full(self):
        return self.region(self.lo, self.lo + self.n * self.h)


class Field:
    def __init__(self, grid):
        self.g = grid
        self.a = np.full(tuple(grid.n), BIG, dtype=np.float32)

    def _apply(self, prim, k, op):
        m = 1.5 * k + 3 * self.g.h
        reg = self.g.region(prim.bmin - m, prim.bmax + m)
        if reg is None:
            return
        slc, X, Y, Z = reg
        d = prim(X, Y, Z).astype(np.float32)
        cur = self.a[slc]
        if op == "add":
            self.a[slc] = smin(cur, d, k)
        elif op == "cut":
            self.a[slc] = smax(cur, -d, k)
        elif op == "paint":            # hard union, used for crisp details
            self.a[slc] = np.minimum(cur, d)

    def add(self, prim, k=0.0):
        self._apply(prim, k, "add")
        return self

    def cut(self, prim, k=0.0):
        self._apply(prim, k, "cut")
        return self

    def add_all(self, prims, k=0.0):
        for p in prims:
            self.add(p, k)
        return self
