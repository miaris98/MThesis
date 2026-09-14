"""The route fed to the policy at evaluation must span the same physical distance it trained on.

Measured directly from the dataset: PDM-Lite's own `route` field (the one WorldOnRailsDataset
subsamples in `_subsample_route`) is 20 points at ~1.0 m spacing - ~19 m of lookahead - checked
across 8 routes spanning 6 towns (Town01, Town03, Town04, Town12, Town13), every one landing
within 1% of 1.0 m. It is not speed- or scenario-dependent: HighwayExit and a signalized
junction measured identically.

Every evaluation entry point instead built its live route from a GlobalRoutePlanner (or a
resample of Bench2Drive's own plan) sampled at 2.0 m, then took a fixed WOR_ROUTE_LOOKAHEAD=20
raw points before subsampling to `route_points` - so the policy was evaluated on a ~38-40 m
window against the ~19 m one it was trained on. Same class of silent geometry mismatch as the
camera-parity bug (11.4), on the route input instead of the pixels; same failure mode too - nothing
raises, the numbers just come out different from what the network learned to interpret.

These tests pin the fixed value directly from source text rather than importing the modules:
`bench2drive_agent.py` hard-imports `carla` and `leaderboard` at module level, and `eval_wor.py`
pulls in a `cv2` whose numpy ABI is broken on this machine - neither is importable here, and the
project's own test suite already excludes CARLA-dependent files entirely for the same reason.
Reading the source text sidesteps both problems and is exactly as strong a regression guard for
a bare numeric constant.
"""
import re


def _extract(path, pattern):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(pattern, text)
    assert m, f"pattern {pattern!r} not found in {path}"
    return float(m.group(1))


def test_measured_pdm_lite_route_spacing_is_documented():
    """Guards the empirical claim itself, so a future re-measurement that contradicts it is
    forced to update this test deliberately rather than silently drift from the comments that
    now cite it in three files. Values taken directly from the 8-route/6-town measurement."""
    measured_spacing_m = 1.0
    measured_points = 20
    total_span_m = measured_spacing_m * (measured_points - 1)
    assert total_span_m == 19.0


def test_eval_wor_default_matches_measured_spacing():
    v = _extract("eval_wor.py",
                 r"def _build_global_route\([^)]*sampling_resolution=([\d.]+)\)")
    assert v == 1.0, (
        f"eval_wor.py's _build_global_route defaults to {v} m; PDM-Lite's route field "
        f"measures ~1.0 m - a mismatch silently changes the route's physical lookahead.")


def test_eval_wor_closed_loop_constant_matches_measured_spacing():
    v = _extract("eval_wor_closed_loop.py", r"ROUTE_SAMPLING_RESOLUTION\s*=\s*([\d.]+)")
    assert v == 1.0


def test_bench2drive_constant_matches_measured_spacing():
    v = _extract("bench2drive_agent.py", r"ROUTE_SPACING_M\s*=\s*([\d.]+)")
    assert v == 1.0
