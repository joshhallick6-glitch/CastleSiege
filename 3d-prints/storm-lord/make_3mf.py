"""Pack the five colour parts into one ready-to-print 3MF on a single plate.

The figure is written as ONE object made of five named parts (cloud, toga,
skin, hair, gold), already assembled and centred on a 256 x 256 mm plate
(Bambu Lab X2D).  Each part carries its colour through the 3MF materials
extension, so Bambu Studio opens its colour dialog and maps every part to a
filament (adding any that are missing) without re-painting anything.

Usage:  python3 make_3mf.py [stl_dir] [out.3mf] [thumbnail.png]
"""
import os
import sys
import zipfile

import numpy as np
import trimesh

PARTS = [  # (file stem, part name, colour)
    ("1_cloud", "Cloud", "#8C8AB8"),
    ("2_toga", "Toga", "#1E3C9A"),
    ("3_skin", "Skin", "#D9A07A"),
    ("4_hair", "Hair and beard", "#F2F2F2"),
    ("5_gold", "Gold", "#D4A133"),
]
BED = (256.0, 256.0)
GROUP_ID = 10
ASSEMBLY_ID = len(PARTS) + 1

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Default Extension="config" ContentType="text/xml"/>
</Types>
"""


def rels(thumbnail):
    thumb = ('\n <Relationship Target="/Metadata/thumbnail.png" Id="rel-2" '
             'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>'
             if thumbnail else "")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
            f'{thumb}\n</Relationships>\n')


def mesh_xml(m):
    v = "\n".join(f'     <vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in m.vertices)
    t = "\n".join(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in m.faces)
    return f"   <mesh>\n    <vertices>\n{v}\n    </vertices>\n    <triangles>\n{t}\n    </triangles>\n   </mesh>\n"


def load_parts(stl_dir):
    meshes = []
    for stem, name, _ in PARTS:
        m = trimesh.load(os.path.join(stl_dir, f"{stem}.stl"), process=True)
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        if not m.is_watertight:
            raise SystemExit(f"{stem}.stl is not watertight")
        meshes.append(m)
    return meshes


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    stl_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "stl", "multicolor")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "storm_lord_one_plate.3mf")
    thumbnail = sys.argv[3] if len(sys.argv) > 3 else None

    meshes = load_parts(stl_dir)
    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    hi = np.max([m.bounds[1] for m in meshes], axis=0)
    # centre the assembled figure on the plate, standing on it
    shift = np.array([BED[0] / 2 - (lo[0] + hi[0]) / 2, BED[1] / 2 - (lo[1] + hi[1]) / 2, -lo[2]])

    colors = "\n".join(f'   <m:color color="{c}"/>' for _, _, c in PARTS)
    objects = []
    for i, ((stem, name, _), m) in enumerate(zip(PARTS, meshes)):
        objects.append(f'  <object id="{i + 1}" name="{name}" type="model" pid="{GROUP_ID}" '
                       f'pindex="{i}">\n{mesh_xml(m)}  </object>\n')
    components = "\n".join(f'    <component objectid="{i + 1}"/>' for i in range(len(PARTS)))
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">\n'
        ' <metadata name="Title">Storm Lord</metadata>\n'
        ' <metadata name="Description">Wizard101 Storm Lord fan figure, five colour parts '
        'assembled on one plate.</metadata>\n'
        ' <resources>\n'
        f'  <m:colorgroup id="{GROUP_ID}">\n{colors}\n  </m:colorgroup>\n'
        + "".join(objects) +
        f'  <object id="{ASSEMBLY_ID}" name="Storm Lord" type="model">\n'
        f'   <components>\n{components}\n   </components>\n  </object>\n'
        ' </resources>\n'
        ' <build>\n'
        f'  <item objectid="{ASSEMBLY_ID}" transform="1 0 0 0 1 0 0 0 1 '
        f'{shift[0]:.4f} {shift[1]:.4f} {shift[2]:.4f}"/>\n'
        ' </build>\n'
        '</model>\n')

    # Bambu Studio / Orca part names (read for third-party 3MF files too)
    parts_cfg = "".join(
        f'    <part id="{i + 1}" subtype="normal_part">\n'
        f'      <metadata key="name" value="{name}"/>\n'
        f'    </part>\n' for i, (_, name, _) in enumerate(PARTS))
    settings = ('<?xml version="1.0" encoding="UTF-8"?>\n<config>\n'
                f'  <object id="{ASSEMBLY_ID}">\n'
                '    <metadata key="name" value="Storm Lord"/>\n'
                f'{parts_cfg}  </object>\n</config>\n')

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", rels(thumbnail))
        z.writestr("3D/3dmodel.model", model)
        z.writestr("Metadata/model_settings.config", settings)
        if thumbnail:
            z.write(thumbnail, "Metadata/thumbnail.png")

    size = hi - lo
    print(f"wrote {out}: {os.path.getsize(out) / 1e6:.1f} MB, "
          f"{sum(len(m.faces) for m in meshes)} triangles, "
          f"figure {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm centred on "
          f"{BED[0]:.0f} x {BED[1]:.0f} plate")


if __name__ == "__main__":
    main()
