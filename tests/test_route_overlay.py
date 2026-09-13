"""Tests for the route overlay: the projection has to be geometrically right, not just non-crashing."""
import math
import numpy as np
import pytest

from src.config.camera import PDM_LITE_CAMERA
from src.config.route_overlay import (
    focal_length_px, project_ego_points, draw_route_overlay)


def test_focal_length_matches_fov():
    """f = (W/2)/tan(fov/2). A 90 deg camera has f exactly W/2, which is the easy check."""
    assert focal_length_px(1024, 90.0) == pytest.approx(512.0, abs=1e-6)
    f110 = focal_length_px(1024, 110.0)
    assert f110 == pytest.approx(512.0 / math.tan(math.radians(55.0)), abs=1e-6)


def test_point_straight_ahead_projects_to_horizontal_centre():
    """A route point directly in front (y=0) must land on the image's vertical centre line,
    whatever its distance - otherwise the overlay would systematically bias steering."""
    for dist in (5.0, 20.0, 80.0):
        pts = project_ego_points([(dist, 0.0)], width=1024, height=512)
        assert pts.shape == (1, 2)
        assert pts[0, 0] == pytest.approx(512.0, abs=1e-3)


def test_ground_points_project_below_the_horizon():
    """The camera is 2 m above the road looking level, so ground points are always in the lower
    half of the frame. A bug that put them above the centre would be drawing on the sky."""
    for dist in (5.0, 20.0, 80.0):
        v = project_ego_points([(dist, 0.0)], width=1024, height=512)[0, 1]
        assert v > 256.0, f"ground point at {dist} m projected above the horizon (v={v})"


def test_further_points_approach_the_horizon_monotonically():
    """Perspective sanity: as distance grows, the ground point rises toward the horizon."""
    vs = [project_ego_points([(d, 0.0)], 1024, 512)[0, 1] for d in (5, 10, 20, 40, 80)]
    assert all(vs[i] > vs[i + 1] for i in range(len(vs) - 1)), f"not monotonic: {vs}"
    assert vs[-1] > 256.0


def test_lateral_offset_projects_to_the_correct_side():
    """+y is right in CARLA's ego frame, so a point to the right must land right of centre."""
    right = project_ego_points([(20.0, 3.0)], 1024, 512)[0, 0]
    left = project_ego_points([(20.0, -3.0)], 1024, 512)[0, 0]
    assert right > 512.0 > left
    assert (right - 512.0) == pytest.approx(512.0 - left, abs=1e-3)


def test_camera_mounting_offset_is_applied():
    """The camera sits 1.5 m behind the ego origin, so a point 1.0 m ahead of the ego is 2.5 m
    ahead of the camera - not 1.0. Ignoring the offset would misplace every near point."""
    assert PDM_LITE_CAMERA["x"] == -1.5
    near = project_ego_points([(1.0, 1.0)], 1024, 512)
    # With the offset applied the forward distance is 2.5 m; without it, 1.0 m. The horizontal
    # deflection differs by a factor of 2.5, which is far outside any tolerance.
    f = focal_length_px(1024, 110.0)
    assert near[0, 0] == pytest.approx(512.0 + f * (1.0 / 2.5), abs=1e-2)


def test_points_behind_the_camera_are_dropped():
    """Behind-camera points have negative forward distance; projecting them would mirror them
    into the frame as though they were ahead."""
    pts = project_ego_points([(-10.0, 0.0), (-2.0, 1.0), (20.0, 0.0)], 1024, 512)
    assert pts.shape == (1, 2), "behind-camera points were not dropped"


def test_all_points_behind_returns_empty():
    assert project_ego_points([(-5.0, 0.0)], 1024, 512).shape == (0, 2)


def test_overlay_modifies_image_and_preserves_shape_and_dtype():
    rgb = np.zeros((512, 1024, 3), dtype=np.uint8)
    out = draw_route_overlay(rgb, [(4.0, 0.0), (10.0, 0.0), (25.0, 0.0)])
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert out.sum() > 0, "overlay drew nothing"
    assert rgb.sum() == 0, "overlay mutated its input in place"


def test_overlay_is_a_noop_when_route_is_not_visible():
    """No visible route must yield exactly the original frame, not a blend against black."""
    rgb = np.full((512, 1024, 3), 128, dtype=np.uint8)
    out = draw_route_overlay(rgb, [(-20.0, 0.0), (-5.0, 0.0)])
    assert np.array_equal(out, rgb)


def test_overlay_draws_in_the_lower_half_for_a_forward_route():
    rgb = np.zeros((512, 1024, 3), dtype=np.uint8)
    out = draw_route_overlay(rgb, [(4.0, 0.0), (30.0, 0.0)], thickness=3)
    painted = np.argwhere(out.sum(axis=2) > 0)
    assert painted.size > 0
    assert painted[:, 0].min() > 200, "route drawn into the sky region"


def test_lane_rails_draw_two_separated_lines():
    rgb = np.zeros((512, 1024, 3), dtype=np.uint8)
    single = draw_route_overlay(rgb, [(5.0, 0.0), (30.0, 0.0)], thickness=3)
    rails = draw_route_overlay(rgb, [(5.0, 0.0), (30.0, 0.0)], thickness=3,
                               lane_half_width_m=1.75)
    assert rails.sum() > single.sum(), "two rails should paint more pixels than one line"
    # Somewhere in the frame the two rails must be visibly separate. Scanning rather than
    # assuming a particular row: where they separate depends on the projection, and pinning a
    # row would test the test's arithmetic rather than the rendering.
    def n_runs(img, r):
        row = (img[r].sum(axis=1) > 0).astype(np.int8)
        return int((np.diff(np.concatenate(([0], row, [0]))) == 1).sum())

    assert any(n_runs(rails, r) == 2 for r in range(rails.shape[0])), \
        "rails never appear as two separated lines on any row"
    assert all(n_runs(single, r) <= 1 for r in range(single.shape[0])), \
        "single centre line should never split into two runs"
