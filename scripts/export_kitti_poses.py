"""Convert one demo.py camera directory to KITTI's 12-column c2w trajectory."""

import argparse
from pathlib import Path

import numpy as np


def convert(camera_dir: Path, output_txt: Path, expected_count: int | None = None) -> int:
    files = sorted(camera_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {camera_dir}")
    indices = []
    for path in files:
        if not path.stem.isdigit():
            raise ValueError(f"Unexpected camera filename: {path.name}")
        indices.append(int(path.stem))
    if indices != list(range(len(files))):
        raise ValueError(f"Camera files must be numbered contiguously from 000000 in {camera_dir}")
    if expected_count is not None and len(files) != expected_count:
        raise ValueError(f"Expected {expected_count} camera poses, found {len(files)} in {camera_dir}")
    poses = []
    for path in files:
        with np.load(path) as data:
            pose = data["pose"]
        if pose.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 pose in {path}, got {pose.shape}")
        if not np.isfinite(pose).all():
            raise ValueError(f"Non-finite camera pose in {path}")
        if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError(f"Invalid homogeneous bottom row in {path}")
        poses.append(pose[:3].reshape(12))
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_txt, np.asarray(poses), fmt="%.9f")
    return len(poses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera_dir", type=Path, help="demo output camera/ directory")
    parser.add_argument("output_txt", type=Path, help="KITTI trajectory text file")
    parser.add_argument("--expected-count", type=int, default=None,
                        help="fail if the camera count differs (e.g. from KITTI ground truth)")
    args = parser.parse_args()
    count = convert(args.camera_dir, args.output_txt, args.expected_count)
    print(f"Wrote {count} poses to {args.output_txt}")


if __name__ == "__main__":
    main()
