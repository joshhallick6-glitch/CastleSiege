"""Storm Lord (Wizard101 fan art) -- procedural sculpt for 3D printing.

Builds the figure as signed distance fields, one per print colour, then
extracts watertight meshes with marching cubes:

    skin  - face, torso, arms, hands
    hair  - mane, beard, eyebrows, moustache (white)
    gold  - circlet, arm bands, bracers, brooch, lightning staff
    toga  - blue one-shoulder robe
    cloud - storm cloud base

Coordinates are millimetres, +Z up and the figure faces -Y.  The cloud is
sliced flat at Z = ZCUT and the exported meshes are dropped onto Z = 0.

Usage:  python3 storm_lord.py [voxel_mm] [out_dir]
"""
import os
import sys
import time

import numpy as np
from scipy import ndimage
from skimage import measure
import trimesh

from sdf import (BIG, Field, Grid, catmull_rom, chain, cylinder, ellipsoid, extrude, frame,
                 lock, placed, ribbon, rot, round_cone, smin, sphere, torus, union, unit, v3)

LO = (-66.0, -50.0, -1.0)
HI = (66.0, 50.0, 192.0)

STAFF_X, STAFF_Y = 51.0, -18.0          # staff axis (in the left hand)
STAFF_R = 3.0
ZCUT = 4.0                              # cloud is sliced flat here, then dropped to Z = 0


def lerp(a, b, t):
    return v3(a) + (v3(b) - v3(a)) * t


def look_rows(axis, hint=(0, -1, 0)):
    """Rotation rows with local z along `axis` (for limb-aligned ellipsoids)."""
    u, v, w = frame(axis, hint)
    return np.array([u, v, w])


# --------------------------------------------------------------------------- #
# Skin: torso, arms, head
# --------------------------------------------------------------------------- #
def torso_parts():
    parts = [
        ellipsoid((0, 3, 40), (23, 17, 13)),            # hips (inside cloud)
        ellipsoid((0, 0, 54), (24, 17, 15)),            # belly
        ellipsoid((0, 2, 73), (29, 20, 19)),            # rib cage
        ellipsoid((0, 10, 86), (22, 10, 13)),           # upper back
        round_cone((0, 3, 90), (0, 1, 107), 10.5, 8.5),  # neck
    ]
    for s in (-1, 1):
        parts += [
            ellipsoid((s * 12.5, -11, 81), (12.5, 8, 9.5),
                      rot((0, 1, 0), -s * 12)),          # pecs
            ellipsoid((s * 21, 6, 72), (10, 12, 17)),    # lats
            round_cone((s * 5, 4, 101), (s * 25, 4, 94), 8.5, 7.5),  # traps
            ellipsoid((s * 34, 0.5, 89), (12.8, 12.8, 13.2)),         # delts
            ellipsoid((s * 9, -13, 63), (7.5, 4, 6)),    # abs (under toga)
        ]
    return parts


def build_torso(g):
    T = Field(g)
    T.add_all(torso_parts(), k=5.0)
    return T


ARM = {
    # shoulder, elbow, wrist, fist centre
    "R": dict(S=(-35, 1, 87), E=(-45, -1, 60), W=(-38.5, -21, 44.5)),
    "L": dict(S=(35, 1, 87), E=(46, 6, 62), W=(47.0, -7.5, 78.5)),
}
FIST_L = v3((STAFF_X + 0.5, STAFF_Y + 1.5, 84.0))


def build_arm(g, side):
    a = ARM[side]
    S, E, W = v3(a["S"]), v3(a["E"]), v3(a["W"])
    up, fo = E - S, W - E
    out = np.sign(S[0])
    F = Field(g)
    parts = [
        round_cone(S, E, 10.6, 7.6),
        ellipsoid(lerp(S, E, 0.48) + (0, -4.0, 0), (7.3, 6.8, 11), look_rows(up)),  # biceps
        ellipsoid(lerp(S, E, 0.42) + (out * 1.0, 4.8, 0), (7.0, 6.6, 11.5), look_rows(up)),  # triceps
        round_cone(E, W, 7.6, 5.4),
        ellipsoid(lerp(E, W, 0.28) + (out * 1.5, 0, 0), (7.4, 6.8, 8.5), look_rows(fo)),  # forearm
    ]
    F.add_all(parts, k=3.0)
    hand = right_fist(W, unit(fo)) if side == "R" else staff_grip(W)
    F.add(union(hand, k=0.9), k=2.0)
    return F


def right_fist(W, d):
    """Clenched right fist resting on the cloud, knuckles forward."""
    l = unit(np.cross(d, (0, 0, 1)))          # across the knuckles
    n = unit(np.cross(l, d))                  # back of the hand
    if n[2] < 0:
        n = -n
    c = W + d * 5.0
    parts = [ellipsoid(c, (5.6, 4.6, 5.8), np.array([l, n, d])),
             round_cone(W - d * 2.0, c, 5.0, 4.8)]
    for i in range(4):
        k = c + d * 4.2 + l * (-4.1 + 2.75 * i) + n * 1.4
        r = (1.75, 1.85, 1.8, 1.55)[i]
        parts.append(chain([k, k + d * 1.7 - n * 2.4, k - d * 0.6 - n * 4.6], [r, r, r * 0.9]))
    t0 = c - l * 4.6 - d * 1.0
    parts.append(chain([t0, c - l * 5.2 + d * 2.6 - n * 1.6, c - l * 3.0 + d * 4.6 - n * 3.6],
                       [2.2, 1.9, 1.6]))                                  # thumb
    return parts


def staff_grip(W):
    """Left hand wrapped round the staff: palm behind, fingers round the front."""
    zc = FIST_L[2]

    def at(ang, R, z):
        a = np.radians(ang)
        return v3((STAFF_X + R * np.cos(a), STAFF_Y + R * np.sin(a), z))

    pa = np.radians(108)
    M = np.array([(-np.sin(pa), np.cos(pa), 0), (np.cos(pa), np.sin(pa), 0), (0, 0, 1)])
    palm = at(108, 5.4, zc)
    parts = [ellipsoid(palm, (4.8, 3.2, 6.8), M),
             round_cone(W, palm - v3((0, 0, 1.5)), 5.2, 4.6)]
    for i, z in enumerate(zc + np.array([3.9, 1.3, -1.3, -3.9])):
        r = (1.8, 1.85, 1.75, 1.5)[i]
        R = STAFF_R + 0.9 * r
        angs = [80, 35, -15, -65, -110, -145 if i < 3 else -120]
        pts = [at(a, R + (1.0 if a == 35 else 0.0), z) for a in angs]
        parts.append(chain(pts, [r] * (len(pts) - 1) + [r * 0.85]))
    parts.append(chain([at(150, 5.6, zc + 2.6), at(185, 5.0, zc + 5.0), at(-150, 4.8, zc + 5.4),
                        at(-122, 4.7, zc + 5.2)], [2.2, 2.0, 1.85, 1.6]))  # thumb
    return parts


# The head, hair and beard are modelled in a reference frame and then scaled
# and moved as one unit, so the head's size on the body is a single knob.
HEAD_REF = v3((0, 0, 114))
HEAD_C = v3((0, -3.5, 117.5))
HS = 1.25


def HX(prim):
    return placed(prim, HEAD_REF, HEAD_C, HS)


def w2r(p):
    """World point -> head reference frame."""
    return HEAD_REF + (v3(p) - HEAD_C) / HS


def hadd(F, prims, k=0.0):
    for p in prims:
        F.add(HX(p), k * HS)


def build_head(g):
    H = Field(g)
    parts = [
        ellipsoid((0, 2, 117), (10.2, 11.5, 11.2)),       # cranium
        ellipsoid((0, -4, 111), (8.8, 8.5, 10.5)),        # face
        ellipsoid((0, -4, 104.5), (8.6, 8, 6)),           # jaw
        sphere((0, -12.6, 116.6), 2.3),                   # glabella
        round_cone((0, -12.8, 115.4), (0, -15.8, 108.8), 1.6, 2.0),   # nose bridge
        sphere((0, -14.5, 112.6), 1.7),                   # hooked bump
        sphere((0, -15.6, 108.6), 2.15),                  # nose tip
    ]
    for s in (-1, 1):
        parts += [
            sphere((s * 5.7, -8.8, 111.6), 3.0),          # cheekbones
            round_cone((s * 7.8, -9.6, 118.9), (s * 1.5, -12.6, 116.6), 2.0, 2.45),  # brow
            sphere((s * 2.2, -13.4, 108.8), 1.55),        # nostril wings
        ]
    hadd(H, parts, k=1.6)
    for s in (-1, 1):
        H.cut(HX(sphere((s * 3.8, -12.0, 114.4), 2.5)), k=0.9 * HS)    # eye sockets
    for s in (-1, 1):
        hadd(H, [sphere((s * 3.8, -10.6, 114.2), 1.75)], k=0.3)       # eyeballs
        hadd(H, [round_cone((s * 1.8, -11.7, 115.5), (s * 5.9, -10.9, 116.4), 1.15, 0.9),
                 round_cone((s * 2.2, -11.6, 112.6), (s * 5.5, -10.8, 113.0), 0.6, 0.5)],
             k=0.5)                                                  # scowling lids
    for z in (120.2, 121.6):                                         # forehead creases
        H.cut(HX(lock((-5.5, -11.0, z - 0.4), (0, -12.6, z + 0.5), (5.5, -11.0, z - 0.4),
                      0.35, 0.35, n=5)), k=0.3)
    return H


# --------------------------------------------------------------------------- #
# Hair, beard and brows (head reference frame)
#
# Every lock is a flat, flame-shaped ribbon with carved strand grooves.  Locks
# are layered and overlap so the mane and beard read as sculpted masses of
# hair rather than separate spikes.
# --------------------------------------------------------------------------- #
SKULL_C = v3((0, 2, 117))
SKULL_R = v3((11.0, 12.3, 12.0))          # cranium + a little, the surface hair lies on


def skull(alpha, psi, lift):
    """Point on the skull: alpha = sideways angle, psi = 0 front / 90 top / 180 back."""
    a, p = np.radians(alpha), np.radians(psi)
    d = v3((np.sin(a), -np.cos(a) * np.cos(p), np.cos(a) * np.sin(p)))
    return SKULL_C + SKULL_R * d + lift * unit(d / SKULL_R)


def skull_normal(p, t):
    return unit((v3(p) - SKULL_C) / SKULL_R ** 2)


def hair_lock(ctrl, w0, t0, normal, rng, n=10, grooves=6, gdepth=0.17, wtip=0.75, ttip=0.42,
              taper=2.4, vtip=0.0):
    """Lock of hair through control points (head frame) -> world-space ribbon.

    w0 / t0 are the half-width and half-thickness at the root.  A high `taper`
    keeps the lock full most of its length and rounds it off at the end, so the
    hair reads as dense, combed strands rather than spikes.  vtip > 0 instead
    keeps full width until the last `vtip` of the length, then narrows it in a
    straight V, for hair ends lying flat on the body."""
    pts = catmull_rom(ctrl, n)
    s = np.linspace(0, 1, n)
    if vtip:
        widths = np.maximum(w0 * np.clip((1 - s) / vtip, 0, 1) ** 0.85, wtip)
    else:
        widths = np.maximum(w0 * (1 - s ** taper), wtip)
    thicks = np.maximum(t0 * (1 - 0.65 * s ** 1.5), ttip)
    normals = [normal(p, t) if callable(normal) else unit(normal) for p, t in zip(pts, s)]
    world = HEAD_C + HS * (pts - HEAD_REF)
    return ribbon(world, widths * HS, thicks * HS, normals, grooves, gdepth * HS,
                  rng.uniform(0, 2 * np.pi))


def around(theta, z, lift):
    """Point on a horizontal ring round the skull at height z (theta 0 = front, 90 = side)."""
    q = np.sqrt(max(1 - ((z - SKULL_C[2]) / SKULL_R[2]) ** 2, 0.0))
    rx, ry = max(SKULL_R[0] * q, 8.0), max(SKULL_R[1] * q, 8.5)
    t = np.radians(theta)
    n = unit((np.sin(t) / rx, -np.cos(t) / ry, 0))
    return v3((rx * np.sin(t), SKULL_C[1] - ry * np.cos(t), z)) + lift * n


NAPE = ellipsoid((0, 9.5, 112.5), (9.0, 6.0, 8.0))      # hair mass over the back of the neck


def build_hair(g):
    rng = np.random.default_rng(21)
    W = Field(g)
    hadd(W, [ellipsoid((0, 6.0, 119.5), (11.4, 11.8, 10.2)),     # skull cap under the locks
             NAPE], k=2.0)
    locks = []

    def lock(ctrl, w0, normal, n):
        locks.append(hair_lock(ctrl, w0 * rng.uniform(0.92, 1.08), 1.2, normal, rng, n=n,
                               grooves=int(rng.integers(4, 7)), gdepth=rng.uniform(0.13, 0.2)))

    # 1) crown: wavy clumps swept back over the top, down onto the nape
    for al in np.linspace(-58, 58, 11):
        ph = rng.uniform(0, 2 * np.pi)
        path = [(47, 0.3), (76, 1.4), (106, 2.0), (138, 2.3), (168, 2.6), (196, 2.8), (220, 2.4)]
        ctrl = [skull(np.clip(al * (1 + 0.2 * j / 6) + 5.0 * np.sin(0.95 * j + ph), -80, 80), ps,
                      lf + 0.8 * np.sin(1.4 * j + ph) * (j > 0))
                for j, (ps, lf) in enumerate(path)]
        ctrl.append(skull(np.clip(al * 1.2, -80, 80), 234, 1.6))
        lock(ctrl, 4.1, skull_normal, 13)

    # 2) a fuller top layer: lifts off the circlet, then combs back
    for al in np.linspace(-36, 36, 7):
        ph = rng.uniform(0, 2 * np.pi)
        path = [(53, 1.0), (80, 2.9), (112, 3.4), (146, 3.3), (178, 3.4), (204, 3.2)]
        ctrl = [skull(al * (1 + 0.2 * j / 5) + 4.0 * np.sin(0.95 * j + ph), ps,
                      lf + 0.6 * np.sin(1.4 * j + ph))
                for j, (ps, lf) in enumerate(path)]
        ctrl.append(skull(al * 1.25, 222, 2.6))
        lock(ctrl, 3.8, skull_normal, 12)

    # 3) sides: from the temples back and down over the ears to the nape,
    #    with a little volume and lobed, wavy edges
    for side in (-1, 1):
        for i, zs in enumerate(np.arange(107.0, 127.5, 2.3)):
            ph = rng.uniform(0, 2 * np.pi)
            v = 0.7 + 0.9 * np.sin(np.pi * (zs - 107) / 20) + rng.uniform(-0.5, 0.6)
            drop = min(12.0, zs - 106.5)
            path = [(62, 0.0, -0.3), (92, 0.25, 1.1 + v), (124, 0.55, 1.7 + v),
                    (154, 0.85, 1.5 + 0.6 * v), (172, 1.0, 0.8)]
            ctrl = [around(side * th, zs - drop * f + 0.7 * np.sin(2.2 * j + ph) * (0 < j < 4), lf)
                    for j, (th, f, lf) in enumerate(path)]
            lock(ctrl, 3.9, lambda p, t: skull_normal(p, t), 12)

    # 4) back of the head: one mass flowing down onto the neck
    for al in np.linspace(-70, 70, 11):
        ph = rng.uniform(0, 2 * np.pi)
        path = [(112, 1.2), (145, 2.2), (178, 2.9), (205, 3.3), (226, 3.1), (240, 2.4)]
        ctrl = [skull(np.clip(al * (1 + 0.05 * j) + 4.0 * np.sin(0.95 * j + ph), -80, 80), ps,
                      lf + 0.6 * np.sin(1.4 * j + ph) * (j > 0))
                for j, (ps, lf) in enumerate(path)]
        lock(ctrl, 4.0, skull_normal, 11)

    # 5) longer hair falling from the back of the head, lying flat down the
    #    neck and onto the upper back between the shoulders
    back = union(torso_parts() + [HX(NAPE)], k=5.0)

    def on_back(x, z, lift):
        ys = np.linspace(35, -5, 801)
        d = back(np.full_like(ys, x), ys, np.full_like(ys, z))
        inside = np.flatnonzero(d < 0)
        return v3((x, (ys[inside[0]] if len(inside) else 8.0) + lift, z))

    for layer, xs, drop, lift0 in [(0, np.linspace(-12.5, 12.5, 9), 0.0, 0.0),
                                   (1, np.linspace(-10.9, 10.9, 8), 4.5, 0.9)]:
        for x0 in xs:
            t = abs(x0) / 12.5
            ph = rng.uniform(0, 2 * np.pi)
            z_end = 91.0 + 6.5 * t ** 1.3 + drop + rng.uniform(-1.8, 1.8)
            ctrl = [skull(4.2 * x0, 186, 2.6 + lift0), skull(4.4 * x0, 214, 3.0 + lift0)]
            ctrl += [w2r(on_back(x0 * f + 0.8 * np.sin(2.1 * j + ph), z, lift + lift0))
                     for j, (f, z, lift) in enumerate([(0.8, 107.0, 2.7), (1.0, 101.5, 2.6),
                                                        (1.12, (101.5 + z_end) / 2, 2.4),
                                                        (1.2, z_end, 2.0)])]
            locks.append(hair_lock(ctrl, 3.9 * rng.uniform(0.9, 1.1), 1.2,
                                   lambda p, t, x0=x0: (0.03 * x0, 1.0, 0.3), rng, n=16,
                                   grooves=int(rng.integers(4, 7)), gdepth=rng.uniform(0.13, 0.2),
                                   vtip=0.3, wtip=0.28, ttip=0.32))

    for lk in locks:
        W.add(lk, k=0.6 * HS)

    # bushy, scowling eyebrows
    for side in (-1, 1):
        for t, rise, L, w in ((0.0, 0.2, 3.4, 1.35), (0.42, 0.4, 3.8, 1.4), (0.8, 0.5, 2.8, 1.25)):
            b = lerp((side * 1.8, -14.3, 118.0), (side * 7.4, -11.6, 119.4), t)
            d = unit((side, 0.3, rise))
            ctrl = [b - d * 0.8, b + d * L * 0.5 + v3((0, 0, 0.3)), b + d * L + v3((0, 0.4, 0.4))]
            W.add(hair_lock(ctrl, w, 0.8, (0, -1, 0.35), rng, n=6, grooves=3, gdepth=0.1,
                            wtip=0.45, ttip=0.32, taper=1.8), k=0.5 * HS)
    return W


def build_beard(g):
    rng = np.random.default_rng(22)
    W = Field(g)
    base = [
        ellipsoid((0, -12.0, 93), (13.5, 6.5, 10.5)),     # body of the beard under the locks
        ellipsoid((0, -12.5, 101), (8.2, 5.0, 4.6)),      # chin
    ]
    for s in (-1, 1):
        base += [ellipsoid((s * 9.0, -4.5, 106.5), (3.8, 5.6, 8.5)),  # sideburns
                 ellipsoid((s * 11.0, -8.5, 99), (5.4, 5.6, 7.0))]    # jowls
    hadd(W, base, k=3.0)

    locks = []
    for s in (-1, 1):
        # moustache: two heavy locks each side, sweeping out and down past the mouth
        locks.append(hair_lock([(s * 0.4, -15.9, 106.9), (s * 4.3, -17.3, 105.9),
                                (s * 8.0, -17.0, 102.0), (s * 10.2, -15.9, 96.5),
                                (s * 11.0, -15.0, 92.5)],
                               2.4, 1.15, (s * 0.2, -1, 0.35), rng, n=10, grooves=4,
                               gdepth=0.12, wtip=0.6, taper=2.0))
        locks.append(hair_lock([(s * 1.2, -15.7, 106.2), (s * 4.0, -17.9, 103.6),
                                (s * 6.2, -18.5, 98.5), (s * 7.0, -18.7, 93.0)],
                               2.0, 1.05, (s * 0.15, -1, 0.2), rng, n=8, grooves=4,
                               gdepth=0.12, wtip=0.6, taper=2.0))
        # cheek and jaw locks over the sideburns
        for (bx, by, bz), (ex, ey, ez) in [((9.3, -4.0, 110.0), (11.4, -7.5, 101.0)),
                                           ((9.6, -6.0, 108.0), (12.0, -9.5, 99.0)),
                                           ((10.0, -7.5, 106.0), (12.2, -11.0, 97.0)),
                                           ((9.0, -11.0, 103.0), (10.6, -14.5, 95.0))]:
            b, e = v3((s * bx, by, bz)), v3((s * ex, ey, ez))
            root = b + v3((-s * 1.6, 2.2, 2.0))          # buried in the head, no stub
            locks.append(hair_lock([root, b, lerp(b, e, 0.5) + v3((s * 0.4, -0.4, 0)), e],
                                   2.9, 1.15, (s * 0.6, -0.8, 0.1), rng, n=8, grooves=5))
        # sides of the beard: short and combed down, hugging the jowls
        for z, L, dz in [(104, 10, -1.2), (100, 12, -1.6), (95.5, 13, -2.2), (91, 12, -3.0)]:
            b = v3((s * 11.5, -9.5, z))
            d = unit((s * 0.8, -0.35, dz))
            locks.append(hair_lock([b, b + d * L * 0.5 + v3((s * 0.3, 0, 0)),
                                    b + d * L + v3((s * 0.4, 0, 0.5))],
                                   3.3, 1.2, (s * 0.35, -1, 0.1), rng, n=8, grooves=5))

    # the beard proper: three dense, overlapping rows hanging down the chest
    rows = [  # root y, root z, x positions, tip y (world), tip z centre/edge (world), w, t
        (-13.8, 100.5, np.linspace(-13, 13, 9), -20.0, 62, 76, 3.4, 1.3),
        (-16.6, 98.0, np.linspace(-11, 11, 8), -22.0, 66, 79, 3.2, 1.25),
        (-18.6, 100.0, np.linspace(-8.5, 8.5, 6), -24.0, 76, 84, 3.0, 1.2),
    ]
    for y0, z0, xs, yend, zc, ze, w0, t0 in rows:
        for x in xs:
            t = abs(x) / 13.0
            b = v3((x, y0, z0 - 0.04 * x * x + rng.uniform(-0.5, 0.5)))
            e = w2r((x * HS * 1.35 + rng.uniform(-0.8, 0.8), yend,
                     zc + (ze - zc) * t ** 1.5 + rng.uniform(-1.2, 1.2)))
            sway = rng.choice([-1, 1]) * rng.uniform(0.6, 1.2)
            ctrl = [b, lerp(b, e, 0.3) + v3((sway, 0, 0)), lerp(b, e, 0.65) - v3((sway, 0, 0)),
                    e, e + v3((x * 0.06 + sway * 0.3, -0.3, -1.2))]
            locks.append(hair_lock(ctrl, w0 * (1 - 0.15 * t), t0, (0.05 * x, -1, 0.12), rng,
                                   n=11, grooves=5, taper=2.0, wtip=0.7))
    for lk in locks:
        W.add(lk, k=0.6 * HS)
    return W


# --------------------------------------------------------------------------- #
# Gold: staff, circlet, bands
# --------------------------------------------------------------------------- #
BOLT = [(-2.6, 0.0), (2.6, 0.0), (1.4, 7.5), (6.8, 7.5), (0.6, 17.0), (4.6, 17.0),
        (-1.2, 30.0), (-0.6, 20.5), (-5.4, 20.5), (-0.8, 11.0), (-5.8, 11.0)]


def build_gold_static(g):
    G = Field(g)
    sx, sy = STAFF_X, STAFF_Y
    staff = [cylinder((sx, sy, 12), (sx, sy, 146), STAFF_R, rnd=0.8)]
    for z in (40, 66, 72.5, 95.5, 102, 128, 140):
        staff.append(torus((sx, sy, z), (0, 0, 1), STAFF_R + 0.15, 1.05))
    staff += [
        ellipsoid((sx, sy, 146.5), (4.6, 4.6, 4.0)),             # finial knob
        torus((sx, sy, 143.0), (0, 0, 1), 3.4, 1.0),
        torus((sx, sy, 150.0), (0, 0, 1), 3.0, 0.9),
        ellipsoid((sx, sy, 117), (3.8, 3.8, 7.0)),               # grip bulge
    ]
    G.add_all(staff, k=0.6)
    G.add(extrude([(x * 1.15, y * 1.1) for x, y in BOLT], (sx + 0.4, sy, 149.5), (1, 0, 0),
                  (0, 0, 1), 3.8, rnd=0.45,
                  bevel=0.35), k=1.2)

    # circlet emblem: a small bolt over the brow
    small = [(x * 0.55, y * 0.34) for x, y in BOLT]
    G.add(HX(extrude(small, (0, -10.6, 120.6), (1, 0, 0), unit((0, 0.45, 1)), 2.2, rnd=0.35,
                     bevel=0.25)), k=0.4)
    # shoulder brooch for the toga
    G.add(cylinder((26.0, -9.0, 95.0), (24.5, -11.2, 96.5), 3.9, rnd=0.6), k=0)
    G.add(sphere((24.4, -11.4, 96.6), 1.5), k=0.3)
    return G


def shell_band(target, src, c, axis, half_w, infl, ridge=True):
    """Gold band hugging a skin field: inflate src inside a slab across axis."""
    g = target.g
    c, w = v3(c), unit(axis)
    R = 16.0 + half_w
    reg = g.region(c - R, c + R)
    slc, X, Y, Z = reg
    h = (X - c[0]) * w[0] + (Y - c[1]) * w[1] + (Z - c[2]) * w[2]
    s = src.a[slc]
    d = np.maximum(s - infl, np.abs(h) - half_w)
    if ridge:
        for e in (-1, 1):
            d = np.minimum(d, np.maximum(s - infl - 0.7, np.abs(h - e * (half_w - 0.6)) - 0.6))
    target.a[slc] = np.minimum(target.a[slc], d.astype(np.float32))


def bracer(target, E, W, t, half_w):
    """Smooth gold cuff around the forearm cone, with raised rims."""
    g = target.g
    E, W = v3(E), v3(W)
    w = unit(W - E)
    c = lerp(E, W, t)
    cone = round_cone(E, W, 7.6 + 1.25, 5.4 + 1.25)
    slc, X, Y, Z = g.region(c - 16, c + 16)
    h = (X - c[0]) * w[0] + (Y - c[1]) * w[1] + (Z - c[2]) * w[2]
    d = cone(X, Y, Z)
    out = np.maximum(d, np.abs(h) - half_w)
    for e in (-1, 1):
        out = np.minimum(out, np.maximum(d - 0.75, np.abs(h - e * (half_w - 0.7)) - 0.7))
    out = np.minimum(out, np.maximum(d - 0.5, np.abs(h) - 0.6))   # centre bead
    target.a[slc] = np.minimum(target.a[slc], out.astype(np.float32))


def circlet(target):
    g = target.g
    c = v3((0, 2, 125.4))                       # head reference frame
    n = unit((0, -0.24, 1))
    slc, X, Y, Z = g.region(HEAD_C - 20, HEAD_C + 22)
    X, Y, Z = [(A - HEAD_C[i]) / HS + HEAD_REF[i] for i, A in enumerate((X, Y, Z))]
    h = (X - c[0]) * n[0] + (Y - c[1]) * n[1] + (Z - c[2]) * n[2]
    shell = ellipsoid((0, 2, 117), (11.4, 12.7, 12.4))(X, Y, Z)
    d = HS * np.maximum(shell, np.abs(h) - 1.5)
    target.a[slc] = np.minimum(target.a[slc], d.astype(np.float32))


# --------------------------------------------------------------------------- #
# Toga and cloud
# --------------------------------------------------------------------------- #
def toga_from(torso):
    g = torso.g
    B = Field(g)
    slc, X, Y, Z = g.region((-60, -40, 20), (60, 40, 112))
    # hem line runs from the right hip up over the left shoulder
    plane = (Z - 63.0 - 0.62 * (X + 30.0)) / np.sqrt(1 + 0.62 ** 2)
    folds = 0.5 * np.sin(2 * np.pi * (plane * 0.9 - 0.25 * X) / 6.0)
    hem = 1.0 * np.exp(-((plane + 1.0) / 1.1) ** 2)
    infl = 1.4 + folds * (1 - np.exp(-(plane / 3.0) ** 2)) + hem
    B.a[slc] = np.maximum(torso.a[slc] - infl, plane).astype(np.float32)
    return B


def build_cloud(g):
    RNG = np.random.default_rng(23)
    C = Field(g)
    puffs = [ellipsoid((0, 0, 13), (44, 31, 14))]
    for ring, (n, ax, ay, z, rr) in enumerate([(11, 42, 29, 12, (13, 17)),
                                                (9, 31, 22, 25, (11, 14)),
                                                (8, 23, 17, 37, (8.5, 11))]):
        ph = RNG.uniform(0, 2 * np.pi)
        for i in range(n):
            a = ph + 2 * np.pi * i / n + RNG.uniform(-0.15, 0.15)
            c = (ax * np.cos(a), ay * np.sin(a), z + RNG.uniform(-2.5, 2.5))
            puffs.append(sphere(c, RNG.uniform(*rr)))
    puffs += [
        sphere((STAFF_X, STAFF_Y, 15), 13),           # holds the staff
        sphere((STAFF_X - 9, STAFF_Y - 3, 27), 8),
        sphere((-35, -24, 29), 11.5),                 # under the right fist
        sphere((-25, -27, 37), 7.5),
        sphere((14, -24, 40), 7),                     # wisps round the waist
        sphere((-8, -25, 42), 6.5),
    ]
    C.add_all(puffs, k=4.0)
    slc, X, Y, Z = g.full()
    C.a[:] = np.maximum(C.a, ZCUT - Z).astype(np.float32)   # flat bed contact
    return C


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
PRIORITY = ["gold", "hair", "toga", "skin", "cloud"]


def build(h):
    t0 = time.time()
    g = Grid(LO, HI, h)
    print(f"grid {tuple(g.n)} = {np.prod(g.n) / 1e6:.0f}M voxels @ {h} mm")

    torso = build_torso(g)
    arm_r, arm_l = build_arm(g, "R"), build_arm(g, "L")
    head = build_head(g)
    print(f"  body {time.time() - t0:.0f}s")

    skin = Field(g)
    skin.a = smin(smin(torso.a, arm_r.a, 3.0), arm_l.a, 3.0)
    skin.a = smin(skin.a, head.a, 2.5).astype(np.float32)

    gold = build_gold_static(g)
    circlet(gold)
    for arm, side in ((arm_r, "R"), (arm_l, "L")):
        a = ARM[side]
        S, E, W = v3(a["S"]), v3(a["E"]), v3(a["W"])
        shell_band(gold, arm, lerp(S, E, 0.5), E - S, 2.8, 1.0)       # arm band
        bracer(gold, E, W, 0.78 if side == "L" else 0.74, 4.0)     # bracer
    print(f"  gold {time.time() - t0:.0f}s")

    toga = toga_from(torso)
    del torso, arm_r, arm_l, head

    hair = build_hair(g)
    beard = build_beard(g)
    hair.a = np.minimum(hair.a, beard.a)
    del beard
    print(f"  hair {time.time() - t0:.0f}s")

    cloud = build_cloud(g)
    print(f"  cloud {time.time() - t0:.0f}s")
    return g, {"gold": gold.a, "hair": hair.a, "toga": toga.a, "skin": skin.a, "cloud": cloud.a}


def cleanup(F):
    """Drop floating specks and fill sealed internal voids (voxel level)."""
    names = list(F)
    whole = F[names[0]].copy()
    for n in names[1:]:
        np.minimum(whole, F[n], out=whole)
    lab, num = ndimage.label(whole < 0, structure=np.ones((3, 3, 3)))
    if num > 1:
        sizes = ndimage.sum_labels(np.ones_like(lab, dtype=np.uint8), lab, range(1, num + 1))
        specks = (lab > 0) & (lab != 1 + int(np.argmax(sizes)))
        print(f"  removing {num - 1} speck(s), {int(specks.sum())} voxels")
        for n in names:
            F[n][specks] = np.maximum(F[n][specks], 1e-3)
    del lab
    lab, num = ndimage.label(whole >= 0)
    voids = (lab > 0) & (lab != lab[0, 0, 0])
    if num > 1:
        print(f"  filling {num - 1} void(s), {int(voids.sum())} voxels")
        stack = np.stack([F[n][voids] for n in names])
        nearest = np.argmin(stack, axis=0)
        for i, n in enumerate(names):
            sel = np.zeros(int(voids.sum()), bool)
            sel[nearest == i] = True
            vals = F[n][voids]
            vals[sel] = -1e-3
            F[n][voids] = vals


def extract(g, field, name):
    inside = np.argwhere(field < 0)
    if len(inside) == 0:
        return None
    i0 = np.maximum(inside.min(0) - 2, 0)
    i1 = np.minimum(inside.max(0) + 3, g.n)
    sub = field[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
    sub = np.pad(sub, 1, constant_values=BIG)
    # keep vertices off grid nodes (sign preserving, so no new pockets appear)
    sub = np.where(sub >= 0, np.maximum(sub, 1e-4), np.minimum(sub, -1e-4)).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(sub, level=0.0, spacing=(g.h,) * 3)
    verts += g.lo + (i0 - 1) * g.h
    verts[:, 2] -= ZCUT
    m = trimesh.Trimesh(verts, faces, process=True)
    if m.volume < 0:
        m.invert()
    pieces = m.split(only_watertight=False)
    if len(pieces) > 1:                     # drop zero-volume slivers, keep real pieces
        keep = [p for p in pieces if abs(p.volume) > 1.0]
        m = trimesh.util.concatenate(keep)
    print(f"  {name:6s} {len(m.faces):>8d} tris  vol {m.volume / 1000:7.1f} cm3  "
          f"watertight={m.is_watertight}")
    return m


def main():
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    out = sys.argv[2] if len(sys.argv) > 2 else "build"
    os.makedirs(out, exist_ok=True)
    g, F = build(h)
    cleanup(F)

    # carve each colour out of the higher-priority ones -> parts that tile exactly
    taken = None
    for name in PRIORITY:
        region = F[name] if taken is None else np.maximum(F[name], -taken)
        m = extract(g, region, name)
        if m is not None:
            m.export(os.path.join(out, f"part_{name}.ply"))
        taken = F[name] if taken is None else np.minimum(taken, F[name])
        del region
    m = extract(g, taken, "whole")
    m.export(os.path.join(out, "whole.ply"))
    comps = m.split(only_watertight=False)
    print(f"  whole: {len(comps)} connected piece(s), bbox {m.bounds.round(1).tolist()}")


if __name__ == "__main__":
    main()
