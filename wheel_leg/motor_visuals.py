"""Procedural annular motor shells, embedded directly in MJCF (no CAD files)."""
import math
import xml.etree.ElementTree as ET


def add_ring_mesh(asset, name, radius, half_height, segments=128):
    """Front-half annulus: radius R/2..R, axial Y=0..H/2."""
    vertices, faces = [], []
    for r, y in ((radius, 0), (radius, half_height),
                 (radius/2, 0), (radius/2, half_height)):
        for i in range(segments):
            angle = 2*math.pi*i/segments
            vertices.append((r*math.cos(angle), y, r*math.sin(angle)))
    for i in range(segments):
        j = (i+1)%segments
        # Outer wall, inner wall, front annulus, back annulus.
        quads = ((i,segments+i,segments+j,j),
                 (2*segments+i,2*segments+j,3*segments+j,3*segments+i),
                 (segments+i,3*segments+i,3*segments+j,segments+j),
                 (i,j,2*segments+j,2*segments+i))
        for a,b,c,d in quads:
            faces.extend(((a,b,c),(a,c,d)))
    ET.SubElement(asset, "mesh", {
        "name": name, "vertex": " ".join(f"{x:.12g}" for v in vertices for x in v),
        "face": " ".join(str(x) for f in faces for x in f),
        "smoothnormal": "true",
    })
