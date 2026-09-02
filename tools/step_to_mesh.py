"""Tessellate voronoi.step into a triangle mesh for GLMeshItem.

gmsh embeds OpenCASCADE, so it can read the AP214 B-rep (5064 faces, 2835 NURBS
surfaces) and produce a surface triangulation. Only the 2D surface mesh is
generated -- a 3D volume mesh would be far more work for geometry that is only
ever going to be drawn as a shell.

Writes an .npz of float32 vertices and uint32 faces, which is what the widget
loads: no mesh library needed at runtime, and no parsing cost on startup.
"""
import os
import sys
import time

import numpy as np
import gmsh

PROJ = r"C:\Users\Brij Nandan Dogra\Desktop\Ground Station"
STEP = os.path.join(PROJ, "voronoi.step")
OUT = os.path.join(PROJ, "assets", "cansat_mesh.npz")

# Characteristic length as a fraction of the model's bounding-box diagonal.
# Coarser than a CAD tessellation would be: this is a 400x200 px live view, not
# a drawing, and every triangle costs time on a 20 Hz redraw.
LC_FRACTION = float(sys.argv[1]) if len(sys.argv) > 1 else 0.012

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

t0 = time.time()
gmsh.model.add("cansat")
print("reading %s ..." % os.path.basename(STEP))
gmsh.model.occ.importShapes(STEP)
gmsh.model.occ.synchronize()
print("  imported in %.1f s" % (time.time() - t0))

solids = gmsh.model.getEntities(3)
faces = gmsh.model.getEntities(2)
print("  solids: %d   faces: %d" % (len(solids), len(faces)))

xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
print("  bounding box: %.2f x %.2f x %.2f (diagonal %.2f)"
      % (xmax - xmin, ymax - ymin, zmax - zmin, diag))

lc = diag * LC_FRACTION
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.5)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.Algorithm", 6)          # frontal-Delaunay
print("  target edge length: %.3f (%.1f%% of diagonal)" % (lc, LC_FRACTION * 100))

t1 = time.time()
gmsh.model.mesh.generate(2)                          # surface mesh only
print("  meshed in %.1f s" % (time.time() - t1))

node_tags, coords, _ = gmsh.model.mesh.getNodes()
coords = np.array(coords, dtype=np.float64).reshape(-1, 3)
# Node tags are not necessarily contiguous; map tag -> row index.
tag_to_row = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
tag_to_row[np.asarray(node_tags, dtype=np.int64)] = np.arange(len(node_tags))

tri_types, tri_tags, tri_nodes = gmsh.model.mesh.getElements(2)
tris = []
for etype, nodes in zip(tri_types, tri_nodes):
    props = gmsh.model.mesh.getElementProperties(etype)
    if props[3] != 3:            # not a 3-node triangle
        continue
    tris.append(tag_to_row[np.asarray(nodes, dtype=np.int64)].reshape(-1, 3))
faces_arr = np.vstack(tris) if tris else np.zeros((0, 3), dtype=np.int64)

gmsh.finalize()

print()
print("RAW TESSELLATION")
print("  vertices : %d" % len(coords))
print("  triangles: %d" % len(faces_arr))

# Drop vertices no triangle references, so the arrays uploaded to GL are tight.
used = np.unique(faces_arr)
remap = np.full(len(coords), -1, dtype=np.int64)
remap[used] = np.arange(len(used))
verts = coords[used]
faces_arr = remap[faces_arr]

# Normalise to match the rocket's visual treatment exactly, because the two
# share one camera and one ground plane:
#   * same on-screen height (the rocket spans z -2.06 .. +2.00), so neither
#     vehicle looks arbitrarily larger than the other;
#   * centred on the origin, which is what the attitude display rotates about.
#     Offsetting the mesh to stand on the ground plane instead makes it swing
#     around a point outside itself as soon as it rotates.
TARGET_HEIGHT = 4.06
verts -= (verts.min(axis=0) + verts.max(axis=0)) / 2.0
extent = verts.max(axis=0) - verts.min(axis=0)
verts *= TARGET_HEIGHT / extent.max()
verts -= (verts.min(axis=0) + verts.max(axis=0)) / 2.0

print()
print("NORMALISED")
print("  vertices : %d" % len(verts))
print("  triangles: %d" % len(faces_arr))
print("  extent   : %.2f x %.2f x %.2f"
      % tuple(verts.max(axis=0) - verts.min(axis=0)))
print("  z range  : %.2f .. %.2f" % (verts[:, 2].min(), verts[:, 2].max()))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT,
                    vertices=verts.astype(np.float32),
                    faces=faces_arr.astype(np.uint32))
print()
print("wrote %s (%.2f MB)" % (OUT, os.path.getsize(OUT) / 1e6))
