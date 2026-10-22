"""Prepare the original KITTI depth sequences used by kitti_s1_* evaluation.

Expected source layout (as in the MonST3R KITTI data preparation):
  <source>/val/<drive>/proj_depth/groundtruth/image_02/<frame>.png
  <source>/<date>/<drive>/image_02/data/<frame>.png

This is the KITTI depth dataset, not the VGGT-Long KITTI Odometry pose data.
"""

import argparse
import shutil
from pathlib import Path


DEFAULT_LENGTHS = (50, 100, 110, 150, 200, 250, 300, 350, 400, 450, 500)
DEFAULT_OUTPUT = Path("data/long_kitti_s1/depth_selection/val_selection_cropped")


def prepare(source_root: Path, output_root: Path, lengths, overwrite=False):
    depth_dirs = sorted((source_root / "val").glob(
        "*/proj_depth/groundtruth/image_02"
    ))
    if not depth_dirs:
        raise FileNotFoundError(
            f"No KITTI depth sequences under {source_root / 'val'}"
        )
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("--lengths must contain positive frame counts")

    jobs = []
    for depth_dir in depth_dirs:
        drive = depth_dir.parents[2].name
        date = "_".join(drive.split("_")[:3])
        rgb_dir = source_root / date / drive / "image_02" / "data"
        depth_files = sorted(depth_dir.glob("*.png"))
        if not depth_files:
            raise ValueError(f"No ground-truth depth frames in {depth_dir}")
        for length in lengths:
            pairs = [(rgb_dir / path.name, path) for path in depth_files[:length]]
            missing = [rgb for rgb, _ in pairs if not rgb.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} matching KITTI RGB frames are missing; first: {missing[0]}"
                )
            seq = f"{drive}_02"
            rgb_out = output_root / f"image_gathered_{length}" / seq
            depth_out = output_root / f"groundtruth_depth_gathered_{length}" / seq
            if not overwrite and (rgb_out.exists() or depth_out.exists()):
                raise FileExistsError(
                    f"Prepared KITTI depth output exists: {rgb_out} or {depth_out}; "
                    "use --overwrite to replace it"
                )
            jobs.append((length, seq, pairs, rgb_out, depth_out))

    for length, seq, pairs, rgb_out, depth_out in jobs:
        if overwrite:
            for folder in (rgb_out, depth_out):
                if folder.exists():
                    shutil.rmtree(folder)
        rgb_out.mkdir(parents=True)
        depth_out.mkdir(parents=True)
        for rgb, depth in pairs:
            shutil.copy2(rgb, rgb_out / rgb.name)
            shutil.copy2(depth, depth_out / depth.name)
        print(f"KITTI depth {seq}, length {length}: copied {len(pairs)} RGB/depth pairs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path,
                        help="original KITTI root containing val/ and date/ directories")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTHS)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the generated directories for selected lengths")
    args = parser.parse_args(argv)
    prepare(args.source_root, args.output_root, args.lengths, args.overwrite)


if __name__ == "__main__":
    main()
