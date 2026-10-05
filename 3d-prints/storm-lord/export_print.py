"""Decimate the marching-cubes meshes in Blender and export print-ready STLs.

Writes into <out_dir>:
    storm_lord_single_color.stl          one solid piece
    multicolor/<n>_<part>.stl            five colour parts that fit together exactly
    storm_lord.blend (with --blend)      coloured scene for further editing

Usage:  python3 export_print.py <build_dir> <out_dir> [decimate_ratio] [--blend]
"""
import os
import sys

import bpy
import trimesh

# load order = Bambu Studio part order; colours are suggestions for the AMS
PARTS = [
    ("cloud", "lavender / light grey", (0.48, 0.47, 0.66, 1)),
    ("toga", "royal blue", (0.10, 0.20, 0.58, 1)),
    ("skin", "tan / skin", (0.80, 0.52, 0.36, 1)),
    ("hair", "white", (0.90, 0.91, 0.94, 1)),
    ("gold", "gold / silk gold", (0.95, 0.68, 0.22, 1)),
]


def load(path, name):
    bpy.ops.wm.ply_import(filepath=path)
    ob = bpy.context.selected_objects[0]
    ob.name = name
    return ob


def decimate_and_export(ob, path, ratio):
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    mod = ob.modifiers.new("decimate", "DECIMATE")
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True, ascii_format=False,
                          apply_modifiers=True)


def check(path):
    m = trimesh.load(path)
    ok = m.is_watertight and m.is_winding_consistent and m.volume > 0
    print(f"  {os.path.basename(path):32s} {len(m.faces):>8d} tris  {m.volume / 1000:6.1f} cm3  "
          f"{os.path.getsize(path) / 1e6:5.1f} MB  {'OK' if ok else 'NOT WATERTIGHT'}")
    return ok


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    build_dir, out_dir = args[0], args[1]
    ratio = float(args[2]) if len(args) > 2 else 0.3
    os.makedirs(os.path.join(out_dir, "multicolor"), exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    paths = []
    for i, (name, _, color) in enumerate(PARTS, 1):
        ob = load(os.path.join(build_dir, f"part_{name}.ply"), name)
        path = os.path.join(out_dir, "multicolor", f"{i}_{name}.stl")
        decimate_and_export(ob, path, ratio)
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = color
        ob.data.materials.append(mat)
        paths.append(path)

    whole = load(os.path.join(build_dir, "whole.ply"), "single_color")
    path = os.path.join(out_dir, "storm_lord_single_color.stl")
    decimate_and_export(whole, path, ratio)
    paths.append(path)

    if "--blend" in sys.argv:
        whole.hide_set(True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "storm_lord.blend"),
                                    compress=True)

    print("exported:")
    if not all(check(p) for p in paths):
        sys.exit("some meshes are not watertight")


if __name__ == "__main__":
    main()
