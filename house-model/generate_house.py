"""Generate a 1:100 printable 3MF model of a two-story colonial with an attached garage.

The house is modelled in feet (X = along the street, left to right; Y = front to
back, front wall at y=0; Z = up from grade) and scaled to millimetres on export.

Outputs (next to this script):
  house_assembled.3mf    one solid object, roof attached (print with supports, or just view)
  house_print_parts.3mf  body + main roof + garage roof laid flat on the bed, support-free;
                         roofs locate on the body with 1.75 mm filament pins
"""

import math
import os
import sys
import zipfile
from xml.sax.saxutils import escape

import numpy as np
from manifold3d import CrossSection, Manifold, OpType

SCALE_MM_PER_FT = 304.8 / 100.0  # 1:100

MATERIALS = {
    "siding": "#D9D2C1",
    "brick": "#8E4B3B",
    "trim": "#F4F3EE",
    "shutter": "#2F3A33",
    "door": "#6B1E22",
    "glass": "#3B4856",
    "roof": "#55524E",
    "concrete": "#A9A69E",
    "vent": "#3A3A3A",
    "flue": "#9B5A3C",
    "brass": "#C9A646",
}
MAT_INDEX = {name: i for i, name in enumerate(MATERIALS)}
ID2MAT = {}

# --- house dimensions (feet) -------------------------------------------------
HX0, HX1 = 22.0, 68.0      # main house, 46' wide
HY0, HY1 = 0.0, 28.5       # 28.5' deep  -> 46 x 28.5 x 2 floors = 2,622 sq ft
GX0, GX1 = 0.0, 22.0       # attached two-car garage
GY0, GY1 = -2.0, 22.0      # projects 2' in front of the house
FIRST = 2.0                # first floor level (exposed foundation below)
SECOND = 11.0
TOP = 20.0                 # top of house walls
GTOP = 10.0                # top of garage walls
PITCH = 8.0 / 12.0

SIDING_COURSE, SIDING_DEPTH = 0.6, 0.07
SIDING_PHASE = 0.017        # keeps course lines off the (round-number) trim heights
SHINGLE_COURSE, SHINGLE_DEPTH = 0.42, 0.07
BRICK = 0.1                # brick veneer stands proud of the framing line
EPS = 0.02                 # parts that meet always overlap by this much, never just touch


# --- primitives --------------------------------------------------------------
def tag(man, mat):
    man = man.as_original()
    ID2MAT[man.original_id()] = mat
    return man


def box(x0, x1, y0, y1, z0, z1, mat):
    return tag(Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0)), mat)


def _ccw(poly):
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))
    return poly if area > 0 else poly[::-1]


def prism_yz(poly, x0, x1, mat):
    """Polygon in the (y, z) plane extruded along x."""
    m = Manifold.extrude(CrossSection([_ccw(poly)]), x1 - x0)
    return tag(m.transform([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]]), mat)


def prism_xz(poly, y0, y1, mat):
    """Polygon in the (x, z) plane extruded along y."""
    m = Manifold.extrude(CrossSection([_ccw(poly)]), y1 - y0)
    return tag(m.transform([[1, 0, 0, 0], [0, 0, -1, y1], [0, 1, 0, 0]]), mat)


def union(parts):
    parts = [p for p in parts if p is not None]
    return Manifold.batch_boolean(parts, OpType.Add) if len(parts) > 1 else parts[0]


def subtract(base, cutters):
    return Manifold.batch_boolean([base] + list(cutters), OpType.Subtract) if cutters else base


def lbox(face, p, u0, u1, s0, s1, z0, z1, mat):
    """Box in wall-local coordinates: u along the wall, s outward from wall plane p."""
    if face == "y-":
        return box(u0, u1, p - s1, p - s0, z0, z1, mat)
    if face == "y+":
        return box(u0, u1, p + s0, p + s1, z0, z1, mat)
    if face == "x-":
        return box(p - s1, p - s0, u0, u1, z0, z1, mat)
    if face == "x+":
        return box(p + s0, p + s1, u0, u1, z0, z1, mat)
    raise ValueError(face)


def sawtooth(t0, t1, course, depth, e1=0.01, e2=0.05):
    """(t, n) outline of lap siding / shingle grooves: each course starts flush and
    recedes to `depth` before the next course steps back out."""
    n = max(1, int(math.ceil((t1 - t0) / course - 1e-9)))
    pts = [(t0, e2), (t0, e1)]
    for k in range(1, n + 1):
        t = t0 + k * course
        pts += [(t, -depth), (t, e1)]
    pts[-1] = (pts[-1][0], e2)
    return pts


def siding_grooves(face, p, u0, u1, z0, z1):
    pts = sawtooth(z0 + SIDING_PHASE, z1, SIDING_COURSE, SIDING_DEPTH)
    if face == "y-":
        return prism_yz([(p - s, z) for z, s in pts], u0, u1, "siding")
    if face == "y+":
        return prism_yz([(p + s, z) for z, s in pts], u0, u1, "siding")
    if face == "x-":
        return prism_xz([(p - s, z) for z, s in pts], u0, u1, "siding")
    if face == "x+":
        return prism_xz([(p + s, z) for z, s in pts], u0, u1, "siding")
    raise ValueError(face)


# --- openings ----------------------------------------------------------------
def window(face, p, u, z0, w, h, surf=0.0, cols=3, rows=2, shutters=False, rd=0.35):
    """Double-hung window with colonial grilles, casing, sill and optional shutters.
    Returns (cutters, additions)."""
    cut = lbox(face, p, u - w / 2, u + w / 2, -rd, 2.0, z0, z0 + h, "trim")
    adds = [lbox(face, p, u - w / 2 - EPS, u + w / 2 + EPS, -rd - 0.03, -rd + 0.05, z0 - EPS, z0 + h + EPS, "glass")]

    f, rail, mw = 0.17, 0.17, 0.13
    zm = z0 + h / 2
    panes = []
    for zb, zt in ((z0 + f, zm - rail / 2), (zm + rail / 2, z0 + h - f)):
        W, H = w - 2 * f, zt - zb
        pw, ph = (W - (cols - 1) * mw) / cols, (H - (rows - 1) * mw) / rows
        for c in range(cols):
            for r in range(rows):
                ua = u - w / 2 + f + c * (pw + mw)
                za = zb + r * (ph + mw)
                panes.append(lbox(face, p, ua, ua + pw, -rd - 0.1, 1.0, za, za + ph, "trim"))
    frame = lbox(face, p, u - w / 2 - EPS, u + w / 2 + EPS, -rd - 0.03, -rd + 0.16, z0 - EPS, z0 + h + EPS, "trim")
    adds.append(subtract(frame, panes))

    cw = 0.35
    adds += [
        lbox(face, p, u - w / 2 - cw, u - w / 2, -0.08, surf + 0.15, z0, z0 + h, "trim"),
        lbox(face, p, u + w / 2, u + w / 2 + cw, -0.08, surf + 0.15, z0, z0 + h, "trim"),
        lbox(face, p, u - w / 2 - cw - 0.08, u + w / 2 + cw + 0.08, -0.08, surf + 0.2, z0 + h - EPS, z0 + h + 0.45, "trim"),
        lbox(face, p, u - w / 2 - cw - 0.12, u + w / 2 + cw + 0.12, -0.08, surf + 0.3, z0 - 0.2, z0 + EPS, "trim"),
    ]
    if shutters:
        sw, gap = 1.4, 0.04
        a = u - w / 2 - cw - gap
        b = u + w / 2 + cw + gap
        adds += [
            lbox(face, p, a - sw, a, -0.08, surf + 0.12, z0, z0 + h, "shutter"),
            lbox(face, p, b, b + sw, -0.08, surf + 0.12, z0, z0 + h, "shutter"),
        ]
    return [cut], adds


def basement_window(face, p, u, w=2.6, z0=0.55, h=1.15):
    cut = lbox(face, p, u - w / 2, u + w / 2, -0.3, 2.0, z0, z0 + h, "trim")
    frame = lbox(face, p, u - w / 2 - EPS, u + w / 2 + EPS, -0.32, -0.18, z0 - EPS, z0 + h + EPS, "trim")
    pane = lbox(face, p, u - w / 2 + 0.15, u + w / 2 - 0.15, -0.5, 1.0, z0 + 0.15, z0 + h - 0.15, "trim")
    glass = lbox(face, p, u - w / 2 - EPS, u + w / 2 + EPS, -0.32, -0.26, z0 - EPS, z0 + h + EPS, "glass")
    return [cut], [subtract(frame, [pane]), glass]


def gable_vent(face, p, u, z0, w, h, slats=4):
    cut = lbox(face, p, u - w / 2, u + w / 2, -0.3, 2.0, z0, z0 + h, "vent")
    adds = [lbox(face, p, u - w / 2 - EPS, u + w / 2 + EPS, -0.32, -0.24, z0 - EPS, z0 + h + EPS, "vent")]
    pitch = h / slats
    for i in range(slats):
        zc = z0 + (i + 0.5) * pitch
        adds.append(lbox(face, p, u - w / 2 - EPS, u + w / 2 + EPS, -0.3, -0.08, zc - 0.07, zc + 0.07, "trim"))
    cw = 0.3
    adds += [
        lbox(face, p, u - w / 2 - cw, u - w / 2, -0.08, 0.14, z0, z0 + h, "trim"),
        lbox(face, p, u + w / 2, u + w / 2 + cw, -0.08, 0.14, z0, z0 + h, "trim"),
        lbox(face, p, u - w / 2 - cw, u + w / 2 + cw, -0.08, 0.14, z0 + h - EPS, z0 + h + cw, "trim"),
        lbox(face, p, u - w / 2 - cw, u + w / 2 + cw, -0.08, 0.2, z0 - cw, z0 + EPS, "trim"),
    ]
    return [cut], adds


def front_entry():
    """Six-panel door with sidelights and a crosshead on the brick front wall."""
    face, p, surf = "y-", HY0, BRICK
    z0, z1 = FIRST, FIRST + 6.67
    cuts = [lbox(face, p, 42.2, 47.8, -0.45, 2.0, z0, z1, "trim")]
    adds = [lbox(face, p, 43.5 - EPS, 46.5 + EPS, -0.47, -0.33, z0 - EPS, z1 + EPS, "door")]
    for ua, ub in ((43.78, 44.88), (45.12, 46.22)):
        for za, zb in ((z0 + 0.35, z0 + 2.2), (z0 + 2.55, z0 + 5.0), (z0 + 5.35, z1 - 0.3)):
            adds.append(lbox(face, p, ua, ub, -0.34, -0.29, za, zb, "door"))
    adds.append(lbox(face, p, 45.95, 46.15, -0.34, -0.2, z0 + 3.0, z0 + 3.2, "brass"))
    for ua, ub in ((42.2, 43.5), (46.5, 47.8)):
        frame = lbox(face, p, ua - EPS, ub + EPS, -0.47, -0.3, z0 - EPS, z1 + EPS, "trim")
        pane = lbox(face, p, ua + 0.17, ub - 0.17, -0.6, 1.0, z0 + 2.4, z1 - 0.2, "trim")
        adds.append(subtract(frame, [pane]))
        adds.append(lbox(face, p, ua - EPS, ub + EPS, -0.47, -0.4, z0 - EPS, z1 + EPS, "glass"))
    adds += [
        lbox(face, p, 41.65, 42.2, -0.08, surf + 0.16, z0, z1, "trim"),
        lbox(face, p, 47.8, 48.35, -0.08, surf + 0.16, z0, z1, "trim"),
        lbox(face, p, 41.5, 48.5, -0.08, surf + 0.3, z1 - EPS, z1 + 0.6, "trim"),
    ]
    return cuts, adds


def garage_door():
    face, p, surf = "y-", GY0, BRICK
    x0, x1, z1 = 3.0, 19.0, 7.0
    cuts = [lbox(face, p, x0, x1, -0.35, 2.0, 0.0, z1, "trim")]
    adds = [lbox(face, p, x0 - EPS, x1 + EPS, -0.37, -0.25, 0.0, z1 + EPS, "trim")]
    for r in range(4):
        for c in range(8):
            ua, za = x0 + c * 2.0, r * 1.75
            adds.append(lbox(face, p, ua + 0.22, ua + 1.78, -0.26, -0.2, za + 0.28, za + 1.47, "trim"))
    adds += [
        lbox(face, p, x0 - 0.4, x0, -0.08, surf + 0.16, 0.0, z1, "trim"),
        lbox(face, p, x1, x1 + 0.4, -0.08, surf + 0.16, 0.0, z1, "trim"),
        lbox(face, p, x0 - 0.4, x1 + 0.4, -0.08, surf + 0.2, z1 - EPS, z1 + 0.45, "trim"),
    ]
    return cuts, adds


def service_door(face, p, u):
    z1 = 6.67
    cuts = [lbox(face, p, u - 1.5, u + 1.5, -0.35, 2.0, 0.0, z1, "trim")]
    slab = lbox(face, p, u - 1.5 - EPS, u + 1.5 + EPS, -0.37, -0.25, 0.0, z1 + EPS, "trim")
    lite = lbox(face, p, u - 1.1, u + 1.1, -0.5, 1.0, 3.6, z1 - 0.4, "trim")
    adds = [subtract(slab, [lite]), lbox(face, p, u - 1.5 - EPS, u + 1.5 + EPS, -0.37, -0.31, 0.0, z1 + EPS, "glass")]
    for za, zb in ((0.4, 1.6), (1.9, 3.2)):
        adds.append(lbox(face, p, u - 1.1, u + 1.1, -0.26, -0.21, za, zb, "trim"))
    adds += [
        lbox(face, p, u - 1.85, u - 1.5, -0.08, 0.15, 0.0, z1, "trim"),
        lbox(face, p, u + 1.5, u + 1.85, -0.08, 0.15, 0.0, z1, "trim"),
        lbox(face, p, u - 1.93, u + 1.93, -0.08, 0.2, z1 - EPS, z1 + 0.45, "trim"),
    ]
    return cuts, adds


def slider(face, p, u0, u1):
    z0, z1 = FIRST, FIRST + 6.67
    cuts = [lbox(face, p, u0, u1, -0.35, 2.0, z0, z1, "trim")]
    frame = lbox(face, p, u0 - EPS, u1 + EPS, -0.37, -0.2, z0 - EPS, z1 + EPS, "trim")
    mid = (u0 + u1) / 2
    panes = [
        lbox(face, p, u0 + 0.22, mid - 0.08, -0.6, 1.0, z0 + 0.25, z1 - 0.22, "trim"),
        lbox(face, p, mid + 0.08, u1 - 0.22, -0.6, 1.0, z0 + 0.25, z1 - 0.22, "trim"),
    ]
    adds = [subtract(frame, panes), lbox(face, p, u0 - EPS, u1 + EPS, -0.37, -0.3, z0 - EPS, z1 + EPS, "glass")]
    adds += [
        lbox(face, p, u0 - 0.35, u0, -0.08, 0.15, z0, z1, "trim"),
        lbox(face, p, u1, u1 + 0.35, -0.08, 0.15, z0, z1, "trim"),
        lbox(face, p, u0 - 0.43, u1 + 0.43, -0.08, 0.2, z1 - EPS, z1 + 0.45, "trim"),
    ]
    return cuts, adds


# --- roofs ---------------------------------------------------------------------
def gable_roof(x0, x1, y0, y1, zb, pitch, eave, rake_l, rake_r, heel, tv, *,
               attic_mat="siding", groove_faces=("x-", "x+"), gutters=True, overlap=0.0,
               siding_base=None):
    """Side-gable roof with its ridge along X, sitting on walls x0..x1, y0..y1 whose
    top is zb. Includes the gable-end attic walls, so the piece has a flat bottom."""
    yr = (y0 + y1) / 2
    zr = zb + heel + (yr - y0) * pitch          # slab underside at the ridge
    ye0, ye1 = y0 - eave, y1 + eave
    drip = 0.05

    def zs(y):
        return zb + heel + (min(y - y0, y1 - y) * pitch)

    zlo = zb - overlap
    attic = prism_yz([(y0, zlo), (y1, zlo), (y1, zs(y1) + EPS), (yr, zr + EPS), (y0, zs(y0) + EPS)], x0, x1, attic_mat)
    base = siding_base if siding_base is not None else zb
    attic = subtract(attic, [siding_grooves(face, {"x-": x0, "x+": x1}[face], y0 - 0.3, y1 + 0.3, base, zr + 0.3)
                             for face in groove_faces])

    a, b = ye0 - drip, ye1 + drip
    slab = prism_yz([(a, zs(a)), (yr, zr), (b, zs(b)), (b, zs(b) + tv), (yr, zr + tv), (a, zs(a) + tv)],
                    x0 - rake_l - drip, x1 + rake_r + (drip if rake_r else 0.0), "roof")

    # shingle courses on both slopes
    L = math.hypot(yr - a, (yr - a) * pitch)
    ct, st = 1 / math.sqrt(1 + pitch ** 2), pitch / math.sqrt(1 + pitch ** 2)
    teeth = sawtooth(0.0, L - 0.35, SHINGLE_COURSE, SHINGLE_DEPTH)
    xa, xb = x0 - rake_l - 0.5, x1 + rake_r + 0.5
    front = [(a + t * ct - n * st, zs(a) + tv + t * st + n * ct) for t, n in teeth]
    back = [(b - t * ct + n * st, zs(b) + tv + t * st + n * ct) for t, n in teeth]
    slab = subtract(slab, [prism_yz(front, xa, xb, "roof"), prism_yz(back, xa, xb, "roof")])

    xe0, xe1 = x0 - rake_l, x1 + rake_r
    parts = [attic, slab]
    # soffit fill under the eaves, tucked just behind the fascia so their corners never coincide
    fa, fb = ye0 + 0.03, ye1 - 0.03
    zf = zlo if gutters else zlo + 0.03   # without a gutter the soffit and fascia bottoms would be coplanar
    parts.append(prism_yz([(fa, zf), (y0 + EPS, zf), (y0 + EPS, zs(y0) + EPS), (fa, zs(fa) + EPS)],
                          xe0 + 0.03, xe1 - 0.03, "trim"))
    parts.append(prism_yz([(y1 - EPS, zf), (fb, zf), (fb, zs(fb) + EPS), (y1 - EPS, zs(y1) + EPS)],
                          xe0 + 0.03, xe1 - 0.03, "trim"))
    parts.append(box(xe0, xe1, ye0, ye0 + 0.12, zlo, zs(ye0) + 0.3, "trim"))
    parts.append(box(xe0, xe1, ye1 - 0.12, ye1, zlo, zs(ye1) + 0.3, "trim"))
    for xr0, xr1, on in ((xe0, xe0 + 0.15, rake_l > 0), (xe1 - 0.15, xe1, rake_r > 0)):
        if on:
            prof = [(ye0, max(zlo + 0.04, zs(ye0) - 0.35)), (yr, zr - 0.35), (ye1, max(zlo + 0.04, zs(ye1) - 0.35)),
                    (ye1, zs(ye1) + 0.05), (yr, zr + 0.05), (ye0, zs(ye0) + 0.05)]
            parts.append(prism_yz(prof, xr0, xr1, "trim"))
    parts.append(box(x0 - rake_l - drip, x1 + rake_r + (drip if rake_r else 0.0),
                     yr - 0.32, yr + 0.32, zr + tv - 0.2, zr + tv + 0.12, "roof"))
    if gutters:
        parts.append(box(xe0, xe1, ye0 - 0.38, ye0 + 0.01, zlo, zb + 0.32, "trim"))
        parts.append(box(xe0, xe1, ye1 - 0.01, ye1 + 0.38, zlo, zb + 0.32, "trim"))
    return union(parts), zr + tv


def main_roof(mode):
    ov = 0.02 if mode == "assembled" else 0.0
    roof, ridge_top = gable_roof(HX0, HX1, HY0, HY1, TOP, PITCH, 1.0, 0.5, 0.5, 0.7, 0.5,
                                 overlap=ov, siding_base=FIRST)
    cuts, adds = gable_vent("x-", HX0, (HY0 + HY1) / 2, 23.6, 2.2, 1.6)
    roof = union([subtract(roof, cuts)] + adds)
    roof = subtract(roof, [box(67.95, 72.0, 12.05, 16.45, 0.0, 40.0, "roof")])  # chimney notch
    if mode == "parts":
        roof = subtract(roof, pin_holes(PINS_MAIN, TOP, up=True))
    return roof, ridge_top


def garage_roof(mode):
    x1 = GX1 + 0.1 if mode == "assembled" else GX1 - 0.02
    ov = 0.02 if mode == "assembled" else 0.0
    roof, _ = gable_roof(GX0, x1, GY0, GY1, GTOP, PITCH, 1.0, 0.5, 0.0, 0.5, 0.5,
                         overlap=ov, siding_base=GTOP, groove_faces=("x-",))
    cuts, adds = gable_vent("x-", GX0, (GY0 + GY1) / 2, 13.0, 2.0, 1.4)
    roof = union([subtract(roof, cuts)] + adds)
    if mode == "assembled":
        roof = subtract(roof, [box(GX1, GX1 + 1.0, -10.0, HY0, 0.0, 40.0, "roof")])
    else:
        roof = subtract(roof, pin_holes(PINS_GARAGE, GTOP, up=True))
    return roof


def portico_roof():
    """Front-gable portico roof; built along X then turned so the ridge runs along Y."""
    local, _ = gable_roof(-4.9, 0.08, 0.0, 9.0, 9.4, 5.0 / 12.0, 0.5, 0.5, 0.0, 0.45, 0.42,
                          attic_mat="trim", groove_faces=(), gutters=False)
    return local.transform([[0, -1, 0, 49.5], [1, 0, 0, 0], [0, 0, 1, 0]])


# --- locating pins (parts version) -------------------------------------------------
PIN_D_MM, PIN_DEPTH_MM = 2.0, 4.0
PINS_MAIN = [(31.0, 14.25), (59.0, 14.25)]
PINS_GARAGE = [(5.0, 10.0), (17.0, 10.0)]


def pin_holes(points, z, up):
    r = PIN_D_MM / 2 / SCALE_MM_PER_FT
    d = PIN_DEPTH_MM / SCALE_MM_PER_FT
    out = []
    for x, y in points:
        c = Manifold.cylinder(d + 0.05, r, circular_segments=24)
        c = c.translate((x, y, z - 0.05 if up else z - d))
        out.append(tag(c, "trim"))
    return out


# --- body ------------------------------------------------------------------------
def body(mode):
    base = union([
        box(HX0, HX1, HY0, HY1, 0.0, FIRST, "concrete"),
        box(HX0, HX1, HY0, HY1, FIRST, TOP, "siding"),
        box(GX0, GX1 + 0.05, GY0, GY1, 0.0, 0.5, "concrete"),
        box(GX0, GX1 + 0.05, GY0, GY1, 0.5, GTOP, "siding"),
        box(HX0, HX1 + 0.1, HY0 - BRICK, HY0 + 0.2, 0.0, SECOND, "brick"),
        box(GX0 - 0.1, GX1 + 0.05, GY0 - BRICK, GY0 + 0.2, 0.0, GTOP, "brick"),
    ])

    grooves = [
        siding_grooves("y-", HY0, HX0, HX1 + 0.2, SECOND, 19.3),
        siding_grooves("y+", HY1, HX0 - 0.2, HX1 + 0.2, FIRST, 19.3),
        siding_grooves("x+", HX1, HY0 + 0.2, HY1 + 0.2, FIRST, 19.3),
        siding_grooves("x-", HX0, GY1, HY1 + 0.2, FIRST, GTOP),
        siding_grooves("x-", HX0, HY0 - 0.2, HY1 + 0.2, FIRST + 14 * SIDING_COURSE, 19.3),
        siding_grooves("x-", GX0, GY0, GY1 + 0.2, 0.5, 9.4),
        siding_grooves("y+", GY1, GX0 - 0.2, GX1, 0.5, 9.4),
    ]
    base = subtract(base, grooves)

    cuts, adds = [], []

    def add(res):
        cuts.extend(res[0])
        adds.extend(res[1])

    # front: five-bay colonial, brick first floor
    for x in (27.5, 36.25, 53.75, 62.5):
        add(window("y-", HY0, x, 4.5, 3.0, 5.25, surf=BRICK, shutters=True))
    for x in (27.5, 36.25, 45.0, 53.75, 62.5):
        add(window("y-", HY0, x, 13.6, 3.0, 4.6, shutters=True))
    add(front_entry())
    add(garage_door())
    # rear
    for x in (27.5, 34.0, 62.5):
        add(window("y+", HY1, x, 4.5, 3.0, 5.25))
    add(window("y+", HY1, 53.0, 6.1, 4.0, 3.6, cols=4, rows=1))
    add(slider("y+", HY1, 39.5, 45.5))
    for x in (27.5, 36.25, 53.75, 62.5):
        add(window("y+", HY1, x, 13.6, 3.0, 4.6))
    add(window("y+", HY1, 45.0, 15.2, 2.5, 3.0, cols=2, rows=1))
    for x in (30.0, 58.0):
        add(basement_window("y+", HY1, x))
    # right side (chimney between the windows)
    for y in (5.0, 23.5):
        add(window("x+", HX1, y, 4.5, 3.0, 5.25))
        add(window("x+", HX1, y, 13.6, 3.0, 4.6))
        add(basement_window("x+", HX1, y + (3.0 if y < 10 else -3.0)))
    # left side, behind the garage
    add(window("x-", HX0, 25.5, 4.5, 3.0, 5.25))
    add(window("x-", HX0, 25.5, 13.6, 3.0, 4.6))
    # garage
    add(window("x-", GX0, 4.0, 3.5, 3.0, 3.5, cols=3, rows=1))
    add(service_door("x-", GX0, 16.0))
    add(window("y+", GY1, 11.0, 3.5, 3.0, 3.5, cols=3, rows=1))

    base = subtract(base, cuts)

    trim = [
        # brick rowlock ledge between brick and siding
        box(HX0, HX1 + 0.15, HY0 - 0.22, HY0 + 0.2, SECOND - 0.15, SECOND + 0.05, "brick"),
        # corner boards
        box(HX0, HX0 + 0.45, HY0 - 0.1, HY0 + 0.1, SECOND, 19.3 + EPS, "trim"),
        box(HX1 - 0.45, HX1 + 0.1, HY0 - 0.1, HY0 + 0.45, SECOND, 19.3 + EPS, "trim"),
        box(HX1 - 0.45, HX1 + 0.1, HY1 - 0.45, HY1 + 0.1, FIRST, 19.3 + EPS, "trim"),
        box(HX0 - 0.1, HX0 + 0.45, HY1 - 0.45, HY1 + 0.1, FIRST, 19.3 + EPS, "trim"),
        box(GX0 - 0.1, GX0 + 0.1, GY0, GY0 + 0.45, 0.5, 9.4 + EPS, "trim"),
        box(GX0 - 0.1, GX0 + 0.45, GY1 - 0.45, GY1 + 0.1, 0.5, 9.4 + EPS, "trim"),
        # frieze boards under the eaves
        box(HX0, HX1 + 0.1, HY0 - 0.1, HY0 + 0.1, 19.3, TOP, "trim"),
        box(HX0 - 0.1, HX1 + 0.1, HY1 - 0.1, HY1 + 0.1, 19.3, TOP, "trim"),
        box(HX1 - 0.1, HX1 + 0.1, HY0 - 0.1, HY1 + 0.1, 19.3, TOP, "trim"),
        box(HX0 - 0.1, HX0 + 0.1, HY0, HY1 + 0.1, 19.3, TOP, "trim"),
        box(GX0 - 0.1, GX0 + 0.1, GY0, GY1 + 0.1, 9.4, GTOP, "trim"),
        box(GX0 - 0.1, GX1, GY1 - 0.1, GY1 + 0.1, 9.4, GTOP, "trim"),
        # downspouts (held off the lap siding; they attach through the corner boards)
        box(67.5, 67.8, HY0 - 0.42, HY0 - 0.08, 0.0, TOP - 0.02, "trim"),
        box(67.5, 67.8, HY1 + 0.02, HY1 + 0.38, 0.0, TOP - 0.02, "trim"),
        box(22.2, 22.5, HY1 + 0.02, HY1 + 0.38, 0.0, TOP - 0.02, "trim"),
        box(0.2, 0.5, GY0 - 0.44, GY0 - 0.08, 0.0, GTOP - 0.02, "trim"),
        box(0.2, 0.5, GY1 + 0.02, GY1 + 0.38, 0.0, GTOP - 0.02, "trim"),
        # rear stoop under the slider
        box(38.5, 46.5, HY1 - 0.1, HY1 + 1.5, 0.0, 1.6, "concrete"),
        box(40.5, 44.5, HY1 + 1.4, HY1 + 2.3, 0.0, 0.8, "concrete"),
    ]

    # exterior brick chimney on the right gable end
    chim = [
        box(67.9, 70.0, 11.25, 17.25, 0.0, 15.0, "brick"),
        tag(Manifold.batch_hull([
            Manifold.cube((2.1, 6.0, 0.01)).translate((67.9, 11.25, 14.99)),
            Manifold.cube((2.1, 4.0, 0.01)).translate((67.9, 12.25, 16.5)),
        ]), "brick"),
        box(67.9, 70.0, 12.25, 16.25, 16.5, TOP - 0.02, "brick"),
        box(67.98, 70.0, 12.25, 16.25, TOP - 0.1, 33.0, "brick"),
        box(68.0, 70.15, 12.1, 16.4, 33.0 - EPS, 33.35, "concrete"),
        box(68.7, 69.3, 13.35, 13.95, 33.35 - EPS, 33.95, "flue"),
        box(68.7, 69.3, 14.55, 15.15, 33.35 - EPS, 33.95, "flue"),
    ]

    # front portico: slab, step, two columns, gable roof
    portico = [
        box(40.0, 50.0, -5.0, 0.1, 0.0, 1.6, "concrete"),
        box(41.5, 48.5, -5.8, -4.95, 0.0, 0.8, "concrete"),
        portico_roof(),
    ]
    for x in (41.0, 49.0):
        portico += [
            box(x - 0.45, x + 0.45, -4.9, -4.0, 1.6 - EPS, 1.95, "trim"),
            tag(Manifold.cylinder(7.25, 0.33, circular_segments=28).translate((x, -4.45, 1.9)), "trim"),
            box(x - 0.42, x + 0.42, -4.87, -4.03, 9.1, 9.42, "trim"),
        ]

    result = union([base] + adds + trim + chim + portico)
    if mode == "parts":
        result = subtract(result, pin_holes(PINS_MAIN, TOP, up=False) + pin_holes(PINS_GARAGE, GTOP, up=False))
    return result


# --- export ------------------------------------------------------------------------
def to_mm(man):
    # collapse boolean slivers far below print resolution so exported vertices stay distinct
    return man.scale((SCALE_MM_PER_FT,) * 3).simplify(0.002)


def mesh_arrays(man):
    mesh = man.to_mesh64()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    tris = np.asarray(mesh.tri_verts, dtype=np.int64)
    run_index = np.asarray(mesh.run_index, dtype=np.int64)
    run_ids = np.asarray(mesh.run_original_id, dtype=np.int64)
    mats = np.zeros(len(tris), dtype=np.int64)
    for i, oid in enumerate(run_ids):
        mats[run_index[i] // 3: run_index[i + 1] // 3] = MAT_INDEX[ID2MAT.get(int(oid), "siding")]
    return verts, tris, mats


def write_3mf(path, objects, title, thumbnail=None):
    """objects: list of (name, Manifold in mm)."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
        f'  <metadata name="Title">{escape(title)}</metadata>',
        '  <metadata name="Designer">generate_house.py</metadata>',
        '  <metadata name="Description">1:100 two-story colonial with attached two-car garage</metadata>',
        '  <resources>',
        '    <basematerials id="1">',
    ]
    for name, color in MATERIALS.items():
        lines.append(f'      <base name="{name}" displaycolor="{color}FF"/>')
    lines.append('    </basematerials>')
    for oid, (name, man) in enumerate(objects, start=2):
        verts, tris, mats = mesh_arrays(man)
        lines.append(f'    <object id="{oid}" name="{escape(name)}" type="model" pid="1" pindex="0">')
        lines.append('      <mesh>')
        lines.append('        <vertices>')
        lines.extend(f'          <vertex x="{v[0]:.4f}" y="{v[1]:.4f}" z="{v[2]:.4f}"/>' for v in verts)
        lines.append('        </vertices>')
        lines.append('        <triangles>')
        lines.extend(f'          <triangle v1="{t[0]}" v2="{t[1]}" v3="{t[2]}" pid="1" p1="{m}"/>'
                     for t, m in zip(tris, mats))
        lines.append('        </triangles>')
        lines.append('      </mesh>')
        lines.append('    </object>')
    lines.append('  </resources>')
    lines.append('  <build>')
    for oid in range(2, len(objects) + 2):
        lines.append(f'    <item objectid="{oid}"/>')
    lines.append('  </build>')
    lines.append('</model>')

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        '  <Default Extension="png" ContentType="image/png"/>\n'
        '</Types>\n')
    rels = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        '  <Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>',
    ]
    if thumbnail:
        rels.append('  <Relationship Target="/Metadata/thumbnail.png" Id="rel1" '
                    'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>')
    rels.append('</Relationships>')

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", "\n".join(rels) + "\n")
        z.writestr("3D/3dmodel.model", "\n".join(lines) + "\n")
        if thumbnail:
            z.write(thumbnail, "Metadata/thumbnail.png")


def drop_to_bed(man, dx=0.0, dy=0.0):
    x0, y0, z0, *_ = man.bounding_box()
    return man.translate((dx - x0, dy - y0, -z0))


def build(out_dir, thumbnail=None):
    house = union([body("assembled"), main_roof("assembled")[0], garage_roof("assembled")])
    house = drop_to_bed(to_mm(house))
    write_3mf(os.path.join(out_dir, "house_assembled.3mf"), [("Colonial house 1-100", house)],
              "Colonial house (assembled)", thumbnail)

    b = drop_to_bed(to_mm(body("parts")))
    mr = drop_to_bed(to_mm(main_roof("parts")[0]))
    gr = drop_to_bed(to_mm(garage_roof("parts")))
    gap = 6.0
    bx = b.bounding_box()
    mrx = mr.bounding_box()
    mr = mr.translate((0.0, bx[4] + gap, 0.0))
    gr = gr.translate((mrx[3] + gap, bx[4] + gap, 0.0))
    write_3mf(os.path.join(out_dir, "house_print_parts.3mf"),
              [("House body", b), ("Main roof", mr), ("Garage roof", gr)],
              "Colonial house (print parts)", thumbnail)
    return {"assembled": house, "body": b, "main_roof": mr, "garage_roof": gr}


if __name__ == "__main__":
    out = os.path.dirname(os.path.abspath(__file__))
    thumb = sys.argv[1] if len(sys.argv) > 1 else None
    parts = build(out, thumb)
    for name, man in parts.items():
        bb = man.bounding_box()
        print(f"{name:12s} {bb[3]-bb[0]:6.1f} x {bb[4]-bb[1]:6.1f} x {bb[5]-bb[2]:6.1f} mm  "
              f"tris={man.num_tri():7d}  genus={man.genus()}  vol={man.volume()/1000:7.1f} cm3")
