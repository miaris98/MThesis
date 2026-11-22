"""Privileged object tokens for the closed-loop teacher test (2026-11-17, a ceiling diagnostic).

The privileged teacher (train_wor.py --privileged_boxes 20; E:/MThesis_EXP/live_20261108_spatial2/w8_teacher) reads the 20 nearest
logged objects instead of the camera. In closed loop those objects come from the simulator. This module rebuilds PDM-Lite's box dicts
from the live actors, following Bench2Drive's team_code (data_agent.get_bounding_boxes, autopilot's traffic-light and stop-sign lists,
transfuser_utils.get_traffic_light_waypoints), and turns them into tokens with build_box_targets' own box_token and selection, so the
closed-loop input is the training input.

Diagnostic only: it reads true object positions, which the RGB-only thesis agent may never do (no-extra-sensors rule). Switched on by the
checkpoint itself (privileged_boxes > 0 in its run_config), never by an environment flag.
"""
import math
from typing import Dict, List

import numpy as np

from scripts.training.build_box_targets import BOX_CLASSES, BOX_FEATS, box_token, in_image

VARIANTS = ("", "_vis", "_visns")   # train_wor.py --privileged_variant: all objects / only those in the camera image / those without speed

BB_SAVE_RADIUS = 64.0   # PDM-Lite config.bb_save_radius: vehicles, walkers and statics are logged within 64 m
LIGHT_RADIUS = 64.0     # config.light_radius: traffic lights (trigger-area centre) and stop signs (trigger volume) within 64 m
SELECT_RADIUS = 50.0    # build_box_targets.frame_targets drops objects beyond 50 m (2-D distance in the ego frame)
SIGN_EXTENT = (1.5, 1.5, 0.5)  # autopilot: every traffic-light and stop-sign box has this extent


def normalize_angle(x: float) -> float:
    """transfuser_utils.normalize_angle: into [-pi, pi)."""
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return float(x)


def relative_position(ego_matrix: np.ndarray, other_matrix: np.ndarray) -> np.ndarray:
    """transfuser_utils.get_relative_transform: the other actor's position in the ego frame (x forward, y right, z up)."""
    ego_matrix, other_matrix = np.asarray(ego_matrix, dtype=np.float64), np.asarray(other_matrix, dtype=np.float64)
    return ego_matrix[:3, :3].T @ (other_matrix[:3, 3] - ego_matrix[:3, 3])


def forward_speed(transform, velocity) -> float:
    """autopilot._get_forward_speed: velocity projected on the actor's heading (m/s)."""
    pitch, yaw = np.deg2rad(transform.rotation.pitch), np.deg2rad(transform.rotation.yaw)
    heading = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    return float(np.dot(np.array([velocity.x, velocity.y, velocity.z]), heading))


def tokens_from_boxes(boxes: List[Dict], m: int, variant: str = "", crop: float = 0.0) -> np.ndarray:
    """build_box_targets.frame_targets for a frame without recovery-camera offset: the objects whose class is in BOX_CLASSES and whose
    2-D distance is at most 50 m, nearest first (stable sort, so ties keep the log order), the first m as tokens. Padding rows are zero
    (valid flag 0). variant '_vis' keeps only objects whose ground footprint projects into the camera image (cropped by `crop`, the
    checkpoint's crop_bottom_frac); '_visns' also zeroes their speed (2026-11-18, the camera-reachable rungs of the teacher ladder)."""
    if variant not in VARIANTS:
        raise ValueError(f"privileged variant {variant!r} not in {VARIANTS}")
    tok = np.zeros((m, BOX_FEATS), dtype=np.float32)
    if variant:
        from src.config.camera import PDM_LITE_CAMERA
        W, H = float(PDM_LITE_CAMERA["width"]), float(PDM_LITE_CAMERA["height"])
        Hc = H * (1.0 - float(crop))
    objs = []
    for b in boxes:
        if b.get("class") not in BOX_CLASSES:
            continue
        x, y = float(b["position"][0]), float(b["position"][1])
        yaw = float(b.get("yaw", 0.0) or 0.0)
        d = math.hypot(x, y)
        if d > SELECT_RADIUS:
            continue
        if variant:
            bex, bey = (float(b["extent"][0]), float(b["extent"][1])) if b.get("extent") else (0.5, 0.5)
            if not in_image(x, y, yaw, bex, bey, W, Hc):
                continue
        objs.append((d, b, x, y, yaw))
    objs.sort(key=lambda o: o[0])
    for i, (_, b, x, y, yaw) in enumerate(objs[:m]):
        tok[i] = box_token(b, x, y, yaw)
        if variant == "_visns":
            tok[i, 6] = 0.0   # no speed, as in training
    return tok


def traffic_light_waypoints(light, carla_map):
    """transfuser_utils.get_traffic_light_waypoints: (trigger-area centre, stop waypoints of the lanes the light controls)."""
    import carla
    base_transform = light.get_transform()
    base_loc = light.get_location()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(light.trigger_volume.location)
    area_ext = light.trigger_volume.extent
    area = []
    for x in np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0):
        px = math.cos(math.radians(base_rot)) * x   # rotate_point(Vector3D(x, 0, z), yaw): the y component is 0
        py = math.sin(math.radians(base_rot)) * x
        area.append(area_loc + carla.Location(x=px, y=py))
    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
            ini_wps.append(wpx)
    wps = []
    for wpx in ini_wps:
        distance_to_light = base_loc.distance(wpx.transform.location)
        eu_wps = [wpx]
        next_distance_to_light = distance_to_light + 1.0
        while not wpx.is_intersection:
            next_wp = wpx.next(0.5)[0]
            next_distance_to_light = base_loc.distance(next_wp.transform.location)
            if next_wp and not next_wp.is_intersection and next_distance_to_light <= distance_to_light:
                eu_wps.append(next_wp)
                distance_to_light = next_distance_to_light
                wpx = next_wp
            else:
                break
        if not next_distance_to_light <= distance_to_light and len(eu_wps) >= 4:
            wps.append(eu_wps[-4])
        else:
            wps.append(wpx)
    return area_loc, wps


class PrivilegedBoxes:
    """PDM-Lite's box dicts for the ego's current state, from the live simulator. Build once per route (the traffic-light stop
    waypoints are precomputed, as the expert does at route setup), then call tokens(ego, m) every tick."""

    def __init__(self, world, carla_map):
        self.world, self.map = world, carla_map
        self.lights = []
        for actor in world.get_actors():
            if "traffic_light" in actor.type_id:
                center, wps = traffic_light_waypoints(actor, carla_map)
                self.lights.append((actor, center, wps))

    def boxes(self, ego) -> List[Dict]:
        import carla
        ego_transform = ego.get_transform()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
        ego_loc = ego.get_location()
        actors = self.world.get_actors()
        out = []

        def rel(transform):
            return [float(v) for v in relative_position(ego_matrix, np.array(transform.get_matrix()))]

        for v in actors.filter("*vehicle*"):           # data_agent: class car (every vehicle, two-wheelers too)
            if v.id == ego.id or v.get_location().distance(ego_loc) >= BB_SAVE_RADIUS:
                continue
            t, e = v.get_transform(), v.bounding_box.extent
            out.append({"class": "car", "extent": [e.x, e.y, e.z], "position": rel(t),
                        "yaw": normalize_angle(np.deg2rad(t.rotation.yaw) - ego_yaw), "speed": forward_speed(t, v.get_velocity())})
        for w in actors.filter("*walker*"):             # class walker
            if w.get_location().distance(ego_loc) >= BB_SAVE_RADIUS:
                continue
            t, e = w.get_transform(), w.bounding_box.extent
            out.append({"class": "walker", "extent": [e.x, e.y, e.z], "position": rel(t),
                        "yaw": normalize_angle(np.deg2rad(t.rotation.yaw) - ego_yaw), "speed": forward_speed(t, w.get_velocity())})
        for light, center, wps in self.lights:          # autopilot.close_traffic_lights: one box per controlled lane
            if carla.Location(center).distance(ego_loc) > LIGHT_RADIUS:
                continue
            rot = light.get_transform().rotation
            for wp in wps:
                t = carla.Transform(wp.transform.location, carla.Rotation(pitch=rot.pitch, yaw=rot.yaw, roll=rot.roll))
                out.append({"class": "traffic_light", "extent": list(SIGN_EXTENT), "position": rel(t),
                            "yaw": normalize_angle(np.deg2rad(rot.yaw) - ego_yaw), "state": str(light.state)})
        for sign in actors.filter("*traffic.stop*"):    # autopilot.close_stop_signs: the box sits on the trigger volume
            centre = sign.get_transform().transform(sign.trigger_volume.location)
            centre = carla.Location(x=centre.x, y=centre.y, z=centre.z)
            if centre.distance(ego_loc) >= LIGHT_RADIUS:
                continue
            rot = sign.get_transform().rotation
            out.append({"class": "stop_sign", "extent": list(SIGN_EXTENT), "position": rel(carla.Transform(centre, rot)),
                        "yaw": normalize_angle(np.deg2rad(rot.yaw) - ego_yaw)})
        for s in actors.filter("static.*"):             # class static_car (car meshes) or static; extent x/y swapped as PDM-Lite logs it
            if s.get_location().distance(ego_loc) >= BB_SAVE_RADIUS:
                continue
            if s.type_id == "static.prop.mesh":
                if "Car" not in s.attributes.get("mesh_path", ""):
                    continue
                cls = "static_car"
            elif s.type_id == "static.prop.trafficwarning":
                cls = "static_trafficwarning"          # logged under its own class, which the teacher's tokens skip
            else:
                cls = "static"
            t, e = s.get_transform(), s.bounding_box.extent
            out.append({"class": cls, "extent": [e.y, e.x, e.z], "position": rel(t),
                        "yaw": normalize_angle(np.deg2rad(t.rotation.yaw) - ego_yaw)})
        return out

    def tokens(self, ego, m: int, variant: str = "", crop: float = 0.0) -> np.ndarray:
        return tokens_from_boxes(self.boxes(ego), m, variant, crop)
