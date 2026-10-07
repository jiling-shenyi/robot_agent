"""Build room collision geometry and merge it with an unchanged robot MJCF.

The builder uses only the standard library. Robot model/assets remain owned by
the robot adapter; object initial poses here are reset conditions, not motion.
"""

from __future__ import annotations

import copy
from pathlib import Path
from xml.etree import ElementTree as ET

from embodied_agent.maps.home_schema import HomeWorld
from embodied_agent.maps.schema import WorldError


def _vec(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _box(parent: ET.Element, name: str, position, half, color, *, collision: bool = True) -> ET.Element:
    attrs = {"name": name, "type": "box", "pos": _vec(position), "size": _vec(half), "rgba": _vec(color), "friction": "0.9 0.01 0.001"}
    if not collision:
        attrs.update(contype="0", conaffinity="0", group="2")
    return ET.SubElement(parent, "geom", attrs)


def home_worldbody(world: HomeWorld) -> ET.Element:
    """Explicitly named floor, room, furniture, marker and object geometry."""
    world = HomeWorld.from_dict(world.to_dict())
    body = ET.Element("worldbody")
    width, depth, height = world.room.size_m
    ET.SubElement(body, "light", {"name": "home_light", "pos": "0 0 4", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(body, "camera", {"name": "home_overview", "pos": "6 -7 6", "xyaxes": "0.76 0.65 0 -0.38 0.45 0.81"})
    ET.SubElement(body, "geom", {"name": "home_floor", "type": "plane", "size": _vec((width / 2, depth / 2, 0.1)), "rgba": "0.73 0.72 0.67 1", "friction": "1 0.01 0.001"})
    wall_color = (0.82, 0.85, 0.88, 0.55)
    _box(body, "room_wall_west", (-width / 2 - .04, 0, height / 2), (.04, depth / 2, height / 2), wall_color)
    _box(body, "room_wall_east", (width / 2 + .04, 0, height / 2), (.04, depth / 2, height / 2), wall_color)
    _box(body, "room_wall_north", (0, depth / 2 + .04, height / 2), (width / 2, .04, height / 2), wall_color)
    door_x, door_y, _ = world.room.doorway_position_m
    left_end, right_start = door_x - world.room.doorway_width_m / 2, door_x + world.room.doorway_width_m / 2
    for name, start, end in (("south_left", -width / 2, left_end), ("south_right", right_start, width / 2)):
        _box(body, f"room_wall_{name}", ((start + end) / 2, door_y - .04, height / 2), ((end - start) / 2, .04, height / 2), wall_color)
    _box(body, "room_doorway_lintel", (door_x, door_y, height - .08), (world.room.doorway_width_m / 2, .04, .08), wall_color)
    _box(body, "doorway_marker", (door_x, door_y + .13, .002), (world.room.doorway_width_m / 2, .13, .002), (.3, .65, .35, .6), collision=False)
    for name, furniture in world.furniture.items():
        x, y, z = furniture.position_m
        hx, hy, hz = furniture.half_size_m
        prefix = f"furniture_{name}"
        color = (.58, .38, .22, 1)
        if furniture.shape_kind == "table":
            top = furniture.top_z
            thickness = min(.04, top / 4)
            _box(body, f"{prefix}_top", (x, y, top - thickness / 2), (hx, hy, thickness / 2), color)
            leg_half = min(.035, hx / 4, hy / 4)
            for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
                _box(body, f"{prefix}_leg_{i}", (x + sx * (hx - leg_half), y + sy * (hy - leg_half), (top - thickness) / 2), (leg_half, leg_half, (top - thickness) / 2), (.38, .25, .15, 1))
        elif furniture.shape_kind == "sofa":
            _box(body, f"{prefix}_seat", (x, y, hz * .6), (hx, hy, hz * .6), (.27, .37, .52, 1))
            _box(body, f"{prefix}_back", (x, y + hy - .07, z), (hx, .07, hz), (.25, .34, .48, 1))
            for i, sign in enumerate((-1, 1)):
                _box(body, f"{prefix}_arm_{i}", (x + sign * (hx - .07), y, hz * .8), (.07, hy, hz * .8), (.25, .34, .48, 1))
        elif furniture.shape_kind == "storage_box":
            t = min(.02, hx / 4, hy / 4, hz / 4)
            _box(body, f"{prefix}_base", (x, y, t), (hx, hy, t), (.47, .55, .65, 1))
            _box(body, f"{prefix}_lid", (x, y, furniture.top_z - t), (hx, hy, t), (.50, .59, .7, 1))
            for i, (axis, sign) in enumerate(((0, -1), (0, 1), (1, -1), (1, 1))):
                pos, size = [x, y, z], [hx, hy, hz]
                pos[axis] += sign * (size[axis] - t)
                size[axis] = t
                _box(body, f"{prefix}_side_{i}", pos, size, (.47, .55, .65, 1))
        else:
            _box(body, prefix, furniture.position_m, furniture.half_size_m, (.49, .34, .23, 1))
    colors = {"fragile": (.45, .7, .85, 1), "hot": (.85, .24, .14, 1), "restricted": (.65, .25, .55, 1), "cleaning": (.23, .6, .38, 1), "electronic": (.2, .22, .25, 1)}
    for name, obj in world.objects.items():
        object_body = ET.SubElement(body, "body", {"name": obj.body_name, "pos": _vec(obj.position_m)})
        ET.SubElement(object_body, "freejoint", {"name": obj.joint_name})
        color = next((colors[tag] for tag in obj.risk_tags if tag in colors), (.86, .61, .22, 1))
        geom = _box(object_body, obj.geom_name, (0, 0, 0), obj.half_size_m, color)
        geom.set("mass", f"{obj.mass_kg:.10g}")
        geom.set("condim", "4")
        geom.set("solref", ".005 1")
    for name, point in world.operation_points.items():
        _box(body, f"operation_point_{name}", (point.position_m[0], point.position_m[1], .002), (.12, .12, .002), (.23, .55, .65, .5), collision=False)
    return body


def build_home_scene(world: HomeWorld, robot_xml: str | Path | None = None) -> str:
    """Merge a supplied complete robot model; preserve robot tree/actuators.

    Path inputs resolve asset directories against their original directory.
    String XML inputs keep paths as supplied by their robot adapter.
    """
    world = HomeWorld.from_dict(world.to_dict())
    source_path = None
    if robot_xml is None:
        root = ET.Element("mujoco", {"model": world.map_id})
        ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
        ET.SubElement(root, "option", {"timestep": ".002", "integrator": "implicitfast", "gravity": "0 0 -9.81"})
    else:
        if isinstance(robot_xml, Path) or (isinstance(robot_xml, str) and not robot_xml.lstrip().startswith("<")):
            source_path = Path(robot_xml).resolve()
            root = ET.fromstring(source_path.read_text(encoding="utf-8"))
        else:
            root = ET.fromstring(robot_xml)
        if root.tag != "mujoco":
            raise WorldError("INCOMPATIBLE_SCENE", "Robot XML must contain a complete mujoco model")
        root = copy.deepcopy(root)
        root.set("model", f"{world.map_id}_robot")
        if source_path is not None:
            compiler = root.find("compiler")
            if compiler is not None:
                for name in ("assetdir", "meshdir", "texturedir"):
                    value = compiler.get(name)
                    if value is not None and not Path(value).is_absolute():
                        compiler.set(name, (source_path.parent / value).resolve().as_posix())
        for keyframe in root.findall("keyframe"):
            root.remove(keyframe)
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    for existing in list(worldbody):
        if existing.tag == "geom" and existing.get("type") == "plane":
            worldbody.remove(existing)
    for child in home_worldbody(world):
        worldbody.append(child)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", ".002")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_view = visual.find("global")
    if global_view is None:
        global_view = ET.SubElement(visual, "global")
    global_view.set("azimuth", "135")
    global_view.set("elevation", "-45")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def home_geometry_names(world: HomeWorld) -> dict[str, tuple[str, ...]]:
    """Explicit geometry registry, never inferred by exclusions."""
    body = home_worldbody(world)
    room_names = tuple(element.get("name") for element in body.findall("geom") if element.get("name", "").startswith(("room_", "home_floor")))
    return {"room": room_names, "furniture": tuple(name for furniture in world.furniture.values() for name in furniture.geom_names), "objects": tuple(obj.geom_name for obj in world.objects.values())}
