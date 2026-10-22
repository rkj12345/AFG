"""Prepare aligned long Bonn RGB/depth sequences for bonn_s1_* evaluation."""

import argparse
import shutil
from pathlib import Path

import numpy as np


SEQUENCES = ("balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous")
DEFAULT_LENGTHS = (50, 100, 110, 150, 200, 250, 300, 350, 400, 450, 500)
DEFAULT_OUTPUT = Path("data/long_bonn_s1/rgbd_bonn_dataset")


def prepare(source_root: Path, output_root: Path, lengths,
            start_frame=30, sequences=SEQUENCES, overwrite=False):
    if start_frame < 0 or not lengths or any(length <= 0 for length in lengths):
        raise ValueError("--start-frame must be nonnegative and lengths positive")

    jobs = []
    for seq in sequences:
        source = source_root / f"rgbd_bonn_{seq}"
        rgb_files = sorted((source / "rgb").glob("*.png"))
        depth_files = sorted((source / "depth").glob("*.png"))
        gt_file = source / "groundtruth.txt"
        if not rgb_files or not depth_files or not gt_file.is_file():
            raise FileNotFoundError(f"Missing Bonn RGB, depth, or ground truth in {source}")
        gt = np.loadtxt(gt_file, ndmin=2)
        available = min(len(rgb_files), len(depth_files), len(gt)) - start_frame
        if available <= 0:
            raise ValueError(f"No aligned Bonn frames after index {start_frame} in {source}")
        target = output_root / source.name
        for length in lengths:
            count = min(available, length)
            rgb_out = target / f"rgb_{length}"
            depth_out = target / f"depth_{length}"
            gt_out = target / f"groundtruth_{length}.txt"
            if not overwrite and any(p.exists() for p in (rgb_out, depth_out, gt_out)):
                raise FileExistsError(
                    f"Prepared Bonn output exists under {target}; "
                    "use --overwrite to replace it"
                )
            jobs.append((seq, length, count, rgb_files[start_frame:start_frame + count],
                         depth_files[start_frame:start_frame + count],
                         gt[start_frame:start_frame + count], rgb_out, depth_out, gt_out))

    for seq, length, count, rgbs, depths, gt, rgb_out, depth_out, gt_out in jobs:
        if overwrite:
            for folder in (rgb_out, depth_out):
                if folder.exists():
                    shutil.rmtree(folder)
        rgb_out.mkdir(parents=True)
        depth_out.mkdir(parents=True)
        for rgb in rgbs:
            shutil.copy2(rgb, rgb_out / rgb.name)
        for depth in depths:
            shutil.copy2(depth, depth_out / depth.name)
        np.savetxt(gt_out, gt)
        print(f"Bonn {seq}, length {length}: copied {count} RGB/depth/pose rows")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path,
                        help="original rgbd_bonn_dataset directory")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTHS)
    parser.add_argument("--sequences", nargs="+", choices=SEQUENCES, default=SEQUENCES)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the generated directories for selected sequences and lengths")
    args = parser.parse_args(argv)
    prepare(args.source_root, args.output_root, args.lengths,
            args.start_frame, args.sequences, args.overwrite)


if __name__ == "__main__":
    main()
