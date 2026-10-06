"""Generate a free-floating, gravity/contact-enabled robot with estimated masses."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .build_biped_model import make_biped_xml
from .geometry import Geometry
from .build_model import HIP_STATOR_INNER_FACE_OFFSET_M
from .paths import GENERATED_MODEL_DIR, PHYSICAL_PARAMETERS_PATH, REPORT_DIR


def numbers(values):
    return " ".join(f"{v:.12g}" for v in values)


def box_inertia(mass, size):
    x, y, z = size
    return mass / 12 * np.array([y*y+z*z, x*x+z*z, x*x+y*y])


def annulus(mass, inner, outer, width):
    axial = mass * (inner*inner + outer*outer) / 2
    transverse = mass * (3*(inner*inner+outer*outer)+width*width) / 12
    return np.array([transverse, axial, transverse])


def make_ground_xml(parameters=None):
    p = (json.loads(PHYSICAL_PARAMETERS_PATH.read_text(encoding="utf-8"))
         if parameters is None else parameters)
    numeric = [value for value in p.values() if isinstance(value, (int, float))]
    if not np.isfinite(numeric).all():
        raise ValueError("Physical parameters must be finite")
    rho = p["aluminium_density_kg_m3"]
    rubber = p["rubber_density_kg_m3"]
    dims = np.array([.2, .15, .1])
    inner = dims - 2*p["chassis_wall_thickness_m"]
    if (min(rho, rubber, p["chassis_wall_thickness_m"], p["battery_mass_kg"],
            p["electronics_mass_kg"], p["rubber_friction"]) <= 0
            or np.min(inner) <= 0
            or not 0 < p["rim_inner_radius_m"] < p["tire_inner_radius_m"] < .05):
        raise ValueError("Invalid density, shell thickness, mass, friction or wheel radii")
    g = Geometry()
    root = ET.fromstring(make_biped_xml(g))
    root.set("model", "free_floating_wheel_leg_balance")
    root.find("compiler").set("inertiafromgeom", "auto")
    root.find("default/geom").attrib.pop("mass")
    root.find("default/geom").set("density", "0")
    root.find("option").set("gravity", "0 0 -9.81")
    root.find("option").set("cone", "elliptic")
    body = root.find(".//body[@name='chassis']")
    body.set("pos", f"0 0 {g.nominal_length+g.wheel_radius+.01}")
    ET.SubElement(body, "freejoint", {"name": "base"})
    outer_mass, inner_mass = rho*np.prod(dims), rho*np.prod(inner)
    shell_mass = outer_mass-inner_mass
    shell_I = box_inertia(outer_mass, dims)-box_inertia(inner_mass, inner)
    battery, electronics = p["battery_mass_kg"], p["electronics_mass_kg"]
    z_battery = p["battery_center_z_m"]
    z_electronics = p["electronics_center_z_m"]
    battery_dims = np.asarray(p["battery_dimensions_m"], dtype=float)
    electronics_dims = np.asarray(p["electronics_dimensions_m"], dtype=float)
    central_width = dims[1]-2*HIP_STATOR_INNER_FACE_OFFSET_M
    for size,z in ((battery_dims,z_battery),(electronics_dims,z_electronics)):
        if (size.shape != (3,) or not np.isfinite(size).all() or np.min(size)<=0
                or size[0]>inner[0] or size[1]>=central_width
                or abs(z)+size[2]/2 > inner[2]/2):
            raise ValueError("Payload envelope does not fit between internal hip motors and chassis walls")
    if abs(z_battery-z_electronics) < (battery_dims[2]+electronics_dims[2])/2:
        raise ValueError("Battery and electronics envelopes overlap")
    total = shell_mass+battery+electronics
    com_z = (battery*z_battery+electronics*z_electronics)/total
    inertia = shell_I + box_inertia(battery, battery_dims)
    inertia += box_inertia(electronics, electronics_dims)
    inertia[:2] += (shell_mass*com_z**2 + battery*(z_battery-com_z)**2
                   + electronics*(z_electronics-com_z)**2)
    ET.SubElement(body, "inertial", {
        "pos": numbers([0,0,com_z]), "mass": str(total),
        "diaginertia": numbers(inertia),
    })
    floor = root.find(".//geom[@name='floor']")
    floor.set("size", "5 5 .1")
    floor.set("contype", "1")
    floor.set("conaffinity", "2")
    floor.set("friction", f"{p['rubber_friction']} .001 .00001")
    for site in list(root.findall("worldbody/site")):
        root.find("worldbody").remove(site)
    structural = ("fixed_arm_OB", "rigid_link_BCW", "active_arm_OA",
                  "coupler_AC", "hip_knee_flange", "output_bearing_boss", "wheel_carrier")
    for side in ("left", "right"):
        for name in structural:
            geom = root.find(f".//geom[@name='{name}_{side}']")
            geom.attrib.pop("mass", None)
            geom.set("density", str(rho))
        # MuJoCo computes rod inertias from their equivalent solid geometry.
        tire_mass = rubber*np.pi*(.05**2-p["tire_inner_radius_m"]**2)*.032
        rim_mass = rho*np.pi*(p["tire_inner_radius_m"]**2-p["rim_inner_radius_m"]**2)*.024
        tire = root.find(f".//geom[@name='tire_{side}']")
        tire.set("mass", str(tire_mass))
        tire.set("friction", f"{p['rubber_friction']} .001 .00001")
        tire.set("condim", "3")
        tire.set("solref", ".008 1")
        rim = root.find(f".//geom[@name='wheel_rim_{side}']")
        rim.set("mass", str(rim_mass))
        rim.set("size", f"{p['tire_inner_radius_m']} .012")
        wheel = root.find(f".//body[@name='wheel_{side}']")
        wheel_I = annulus(tire_mass, p["tire_inner_radius_m"], .05, .032)
        wheel_I += annulus(rim_mass, p["rim_inner_radius_m"], p["tire_inner_radius_m"], .024)
        wheel_I += annulus(.18, 0, .034, .0445)
        ET.SubElement(wheel, "inertial", {"pos": "0 0 0",
            "mass": str(tire_mass+rim_mass+.18), "diaginertia": numbers(wheel_I)})
    # Robot-ground collision only; no overlapping visual/structural self contacts.
    for geom in root.findall(".//body/geom"):
        if (geom.get("name") == "chassis_box" or "density" in geom.attrib
                or float(geom.get("mass", "0")) > 0):
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
    root.find("statistic").set("center", "0 0 .15")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def main():
    GENERATED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_MODEL_DIR / "ground_robot.xml"
    path.write_text(make_ground_xml(), encoding="utf-8")
    m = mujoco.MjModel.from_xml_path(str(path))
    report = {
        "status": "engineering_estimates_not_measured",
        "total_mass_kg": float(m.body_subtreemass[m.body('chassis').id]),
        "chassis_including_payload_kg": float(m.body('chassis').mass[0]),
        "bodies": {m.body(i).name: {
            "mass_kg": float(m.body_mass[i]),
            "com_local_m": m.body_ipos[i].tolist(),
            "principal_inertia_kg_m2": m.body_inertia[i].tolist(),
        } for i in range(1, m.nbody)},
    }
    (REPORT_DIR / "MASS_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Written {path}; mass={report['total_mass_kg']:.4f} kg")


if __name__ == "__main__":
    main()
