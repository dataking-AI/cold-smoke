"""Convert a Cold Smoke SkiExport JSON recording to ski_camera H5 format.

The H5 keypoints are the game's procedural skeleton. Positions and velocities
are written in the ski_camera FLU convention: X forward, Y left, Z up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


KEYPOINT_NAMES = [
    "head", "neck", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "hip_centre", "l_hip", "r_hip", "l_knee",
    "r_knee", "l_ankle", "r_ankle",
]


def convert(source: Path, destination: Path, session: str | None = None) -> Path:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the user's ski_rl environment
        raise RuntimeError("h5py is required; run this script in the ski_camera/ski_rl environment") from exc

    with source.open("r", encoding="utf-8") as stream:
        recording = json.load(stream)
    samples = recording.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("the JSON must contain at least two samples")
    if recording.get("format") != "cold-smoke-ski-v1":
        raise ValueError("unsupported export format")

    # The viewer requires strictly increasing timestamps. A paused display
    # frame can repeat game time, so discard repeated timestamps here.
    times = np.asarray([float(s["t"]) for s in samples], dtype=np.float64)
    if not np.isfinite(times).all():
        raise ValueError("sample timestamps t must be finite")
    keep = np.r_[True, np.diff(times) > 0.0]
    samples = [sample for sample, valid in zip(samples, keep) if valid]
    times = times[keep]
    if len(samples) < 2:
        raise ValueError("the export has fewer than two distinct timestamps")

    try:
        keypoints = np.asarray(
            [[sample["keypoints_world"][name] for name in KEYPOINT_NAMES] for sample in samples],
            dtype=np.float32,
        )
        human_pos = np.asarray([sample["human_pos"] for sample in samples], dtype=np.float32)
        velocity = np.asarray([sample["hip_centre_vel"] for sample in samples], dtype=np.float32)
        gravity_body = np.asarray([sample["gravity_body"] for sample in samples], dtype=np.float32)
        ground_z = np.asarray([sample["ground_z"] for sample in samples], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"missing FLU field in sample: {exc}") from exc
    if (keypoints.shape != (len(samples), 15, 3) or human_pos.shape != (len(samples), 3)
            or velocity.shape != (len(samples), 3)):
        raise ValueError("invalid keypoint or human_pos shape")
    if not np.isfinite(keypoints).all() or not np.isfinite(human_pos).all() or not np.isfinite(velocity).all():
        raise ValueError("positions and velocity must be finite")
    if ground_z.shape != (len(samples),) or not np.isfinite(ground_z).all():
        raise ValueError("ground_z must be finite with shape (frames,)")

    # Keep the recorded, potentially irregular cadence.  ``measured_hz`` is
    # only a sampling hint from the browser and must not be used to construct
    # the replay clock.  Round relative ``t`` to integer nanoseconds so the
    # H5 reader can use an exact, monotonic integer time base.
    capture_time_ns = np.rint((times - times[0]) * 1e9).astype(np.int64)
    if np.any(np.diff(capture_time_ns) <= 0):
        raise ValueError("timestamps are not strictly increasing at nanosecond precision")
    measured_hz = float((len(times) - 1) / (times[-1] - times[0]))
    if (gravity_body.shape != (len(samples), 3) or not np.isfinite(gravity_body).all()
            or not np.allclose(np.linalg.norm(gravity_body, axis=1), 1.0, atol=0.01, rtol=0.0)):
        raise ValueError("gravity_body must be finite with shape (frames, 3)")
    velocity[0] = 0.0

    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as output:
        output.create_dataset("capture_time_ns", data=capture_time_ns)
        output.create_dataset("keypoints_world", data=keypoints)
        output.create_dataset("human_pos", data=human_pos)
        output.create_dataset("hip_centre_vel", data=velocity)
        output.create_dataset("gravity_body", data=gravity_body)
        output.create_dataset("ground_z", data=ground_z)
        string_type = h5py.string_dtype(encoding="utf-8")
        output.attrs["keypoint_names"] = np.asarray(KEYPOINT_NAMES, dtype=string_type)
        output.attrs["hip_centre_index"] = 8
        # Report the cadence actually present in t.  Keep the browser's
        # requested rate separately for diagnostics when it was provided.
        output.attrs["measured_hz"] = measured_hz
        requested_hz = recording.get("requested_hz", recording.get("measured_hz"))
        if requested_hz is not None:
            output.attrs["requested_hz"] = float(requested_hz)
        output.attrs["frame_count"] = len(samples)
        output.attrs["keypoint_count"] = 15
        output.attrs["session_name"] = session or source.stem
        output.attrs["source"] = "Cold Smoke procedural skier; not human motion capture"
        output.attrs["coordinate_system"] = "FLU"
        output.attrs["position_unit"] = "m"
        output.attrs["ground_z_semantics"] = "FLU terrain z at recorded human x,y; no lateral terrain measurement"
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="SkiExport JSON file")
    parser.add_argument("destination", type=Path, nargs="?", help="output H5 path")
    parser.add_argument("--session", default=None, help="session_name H5 attribute")
    args = parser.parse_args()
    destination = args.destination or args.source.with_suffix(".h5")
    print(convert(args.source, destination, args.session))


if __name__ == "__main__":
    main()
