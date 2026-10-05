"""Render colour previews of the Storm Lord parts with Blender (Cycles, CPU).

<model_dir> is either a build folder (part_*.ply) or an export folder
(multicolor/<n>_<part>.stl).

Usage:  python3 render_preview.py <model_dir> <out_dir> [views...] [--grey]
Views: front, three_quarter, side, back, face, low, hand
"""
import math
import os
import sys

import bpy
from mathutils import Vector

COLORS = {  # base colour, metallic, roughness
    "skin": ((0.80, 0.52, 0.36, 1), 0.0, 0.55),
    "hair": ((0.90, 0.91, 0.94, 1), 0.0, 0.6),
    "gold": ((0.95, 0.68, 0.22, 1), 0.9, 0.3),
    "toga": ((0.10, 0.20, 0.58, 1), 0.0, 0.7),
    "cloud": ((0.48, 0.47, 0.66, 1), 0.0, 0.8),
}
VIEWS = {  # azimuth (deg, 0 = front), elevation, target, distance
    "front": (0, 8, (4, 0, 88), 520),
    "three_quarter": (-35, 12, (4, 0, 88), 520),
    "side": (-90, 8, (4, 0, 88), 520),
    "back": (180, 10, (4, 0, 88), 520),
    "face": (-15, 4, (0, -10, 108), 230),
    "low": (25, -4, (4, 0, 88), 520),
    "hand": (35, 10, (46, -12, 76), 200),
    "back_close": (165, 8, (0, 10, 100), 230),
}


def setup_scene(build_dir, grey):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    for name, (col, metal, rough) in COLORS.items():
        path = os.path.join(build_dir, f"part_{name}.ply")
        stl = [f for f in os.listdir(os.path.join(build_dir, "multicolor"))
               if f.endswith(f"_{name}.stl")] if not os.path.exists(path) else []
        if stl:
            bpy.ops.wm.stl_import(filepath=os.path.join(build_dir, "multicolor", stl[0]))
        elif os.path.exists(path):
            bpy.ops.wm.ply_import(filepath=path)
        else:
            continue
        ob = bpy.context.selected_objects[0]
        ob.name = name
        for p in ob.data.polygons:
            p.use_smooth = True
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        if grey:
            col, metal, rough = (0.62, 0.62, 0.64, 1), 0.0, 0.5
        bsdf.inputs["Base Color"].default_value = col
        bsdf.inputs["Metallic"].default_value = metal
        bsdf.inputs["Roughness"].default_value = rough
        ob.data.materials.append(mat)

    world = bpy.data.worlds.new("w")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.05, 0.07, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0

    def area(name, loc, energy, size, color=(1, 1, 1)):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.size, light.color = energy, size, color
        ob = bpy.data.objects.new(name, light)
        scene.collection.objects.link(ob)
        ob.location = loc
        d = Vector((0, 0, 90)) - Vector(loc)
        ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

    area("key", (-260, -320, 330), 4.5e6, 220, (1.0, 0.93, 0.85))
    area("fill", (330, -260, 150), 1.4e6, 260, (0.8, 0.85, 1.0))
    area("rim", (60, 380, 300), 3.5e6, 200, (0.9, 0.9, 1.0))

    floor = bpy.data.meshes.new("floor")
    floor.from_pydata([(-600, -600, 0), (600, -600, 0), (600, 600, 0), (-600, 600, 0)], [], [(0, 1, 2, 3)])
    fo = bpy.data.objects.new("floor", floor)
    scene.collection.objects.link(fo)
    fm = bpy.data.materials.new("floor")
    fm.use_nodes = True
    fm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.09, 0.09, 0.1, 1)
    floor.materials.append(fm)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 85
    cam_data.clip_end = 5000
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 40
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 900
    scene.render.resolution_y = 1200
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    return scene, cam


def place(cam, az, el, target, dist):
    t = Vector(target)
    a, e = math.radians(az), math.radians(el)
    off = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))) * dist
    cam.location = t + off
    cam.rotation_euler = (t - cam.location).to_track_quat("-Z", "Y").to_euler()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    grey = "--grey" in sys.argv
    build_dir, out_dir = args[0], args[1]
    views = args[2:] or ["front", "three_quarter", "side", "face"]
    os.makedirs(out_dir, exist_ok=True)
    scene, cam = setup_scene(build_dir, grey)
    for v in views:
        place(cam, *VIEWS[v])
        if v in ("face", "hand", "back_close"):
            scene.render.resolution_x, scene.render.resolution_y = 1000, 1000
        else:
            scene.render.resolution_x, scene.render.resolution_y = 900, 1200
        scene.render.filepath = os.path.join(out_dir, f"{v}{'_grey' if grey else ''}.png")
        bpy.ops.render.render(write_still=True)
        print("wrote", scene.render.filepath)


if __name__ == "__main__":
    main()
