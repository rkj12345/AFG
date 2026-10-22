"""Run and evaluate KITTI Odometry 00-10 in the VGGT-Long data layout.

Each sequence is processed at full frame rate. The demo writes c2w camera
poses, the converter writes KITTI 3x4 rows, and evo reports Sim(3)-aligned ATE.
"""

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from export_kitti_poses import convert


REPO_ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = tuple(f"{i:02d}" for i in range(11))
MODES = ("cut3r", "ttt3r", "ttsa3r", "cut_memix", "ttt_memix",
         "ttsa_memix", "afg_img", "afg_pose")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def pose_count(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    poses = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if poses.shape[1] != 12 or not np.isfinite(poses).all():
        raise ValueError(f"Expected finite KITTI 3x4 rows in {path}, got {poses.shape}")
    return len(poses)


def image_count(kitti_root: Path, sequence: str) -> int:
    image_dir = kitti_root / sequence / "image_2"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    images = [p for p in image_dir.iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise ValueError(f"No KITTI images in {image_dir}")
    return len(images)


def parse_rmse(output: str) -> float:
    matches = re.findall(r"(?m)^\s*rmse\s+([0-9.eE+-]+)\s*$", output)
    if not matches:
        raise ValueError("evo_ape output has no RMSE row")
    value = float(matches[-1])
    if not math.isfinite(value):
        raise ValueError("evo_ape returned a non-finite RMSE")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-root", type=Path,
                        help="VGGT-Long KITTI Odometry root: 00/image_2 ... 10/image_2")
    parser.add_argument("--gt-dir", type=Path,
                        help="KITTI Odometry 00.txt ... 10.txt ground truth")
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "src/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-update-type", choices=MODES, default="afg_pose")
    parser.add_argument("--sequences", nargs="+", choices=SEQUENCES, default=SEQUENCES)
    parser.add_argument("--phase", choices=("all", "infer", "convert", "eval"), default="all")
    parser.add_argument("--python", default=sys.executable, help="Python executable with the AFG dependencies")
    parser.add_argument("--evo-ape", default="evo_ape", help="evo_ape executable")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reset-interval", type=int, default=100)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="allow demo to replace an existing sequence output")
    args = parser.parse_args()

    if args.phase in ("all", "infer") and args.kitti_root is None:
        parser.error("--kitti-root is required for inference")
    if args.phase in ("all", "eval") and args.gt_dir is None:
        parser.error("--gt-dir is required for evaluation")
    if args.reset_interval <= 0 or args.frame_interval <= 0:
        parser.error("frame and reset intervals must be positive")
    if args.phase in ("all", "eval") and args.frame_interval != 1:
        parser.error("ATE evaluation requires --frame-interval 1 to match every GT row")

    args.kitti_root = args.kitti_root.resolve() if args.kitti_root else None
    args.gt_dir = args.gt_dir.resolve() if args.gt_dir else None
    args.weights = args.weights.resolve()
    output_dir = (args.output_dir or REPO_ROOT / "eval_results/kitti_odometry" / args.model_update_type).resolve()
    results = []
    for seq in args.sequences:
        camera_dir = output_dir / seq / "camera"
        trajectory = output_dir / "trajectories" / f"{seq}.txt"
        gt_file = args.gt_dir / f"{seq}.txt" if args.gt_dir else None
        gt_count = pose_count(gt_file) if gt_file else None
        input_count = image_count(args.kitti_root, seq) if args.kitti_root else None
        if gt_count is not None and input_count is not None and gt_count != input_count:
            raise ValueError(f"Sequence {seq}: {input_count} images but {gt_count} GT poses")

        if args.phase in ("all", "infer"):
            if not args.weights.is_file():
                raise FileNotFoundError(args.weights)
            if camera_dir.exists() and any(camera_dir.glob("*.npz")) and not args.force:
                raise FileExistsError(f"{camera_dir} contains predictions; use --force to replace them")
            log_path = output_dir / "logs" / f"infer_{seq}.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [args.python, str(REPO_ROOT / "demo.py"),
                       "--model_path", str(args.weights),
                       "--seq_path", str(args.kitti_root / seq / "image_2"),
                       "--output_dir", str(output_dir / seq),
                       "--model_update_type", args.model_update_type,
                       "--size", "512", "--frame_interval", str(args.frame_interval),
                       "--reset_interval", str(args.reset_interval), "--no_viewer",
                       "--device", args.device]
            print(shlex.join(command), flush=True)
            with log_path.open("w") as log:
                completed = subprocess.run(command, cwd=REPO_ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise RuntimeError(f"Inference failed for {seq}; see {log_path}")

        if args.phase in ("all", "convert"):
            expected = gt_count if gt_count is not None else input_count
            count = convert(camera_dir, trajectory, expected_count=expected)
            print(f"{seq}: converted {count} camera poses to {trajectory}", flush=True)

        if args.phase in ("all", "eval"):
            pred_count = pose_count(trajectory)
            if pred_count != gt_count:
                raise ValueError(f"Sequence {seq}: {pred_count} predictions but {gt_count} GT poses")
            command = [args.evo_ape, "kitti", str(gt_file), str(trajectory), "-r", "full", "-as"]
            print(shlex.join(command), flush=True)
            result = subprocess.run(command, cwd=REPO_ROOT, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            log_path = output_dir / "logs" / f"ape_{seq}.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(result.stdout)
            if result.returncode:
                raise RuntimeError(f"evo_ape failed for {seq}; see {log_path}")
            rmse = parse_rmse(result.stdout)
            results.append((seq, rmse))
            print(f"{seq}: ATE RMSE = {rmse:.6f} m", flush=True)

    if results:
        output_dir.mkdir(parents=True, exist_ok=True)
        mean = sum(value for _, value in results) / len(results)
        csv_path = output_dir / "ate_rmse.csv"
        with csv_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("sequence", "ate_rmse_m"))
            writer.writerows(results)
            writer.writerow(("mean", mean))
        (output_dir / "ate_rmse.json").write_text(json.dumps({
            "protocol": "KITTI Odometry; evo_ape kitti -r full -as",
            "sequences": list(args.sequences),
            "model_update_type": args.model_update_type,
            "per_sequence_ate_rmse_m": dict(results),
            "mean_ate_rmse_m": mean,
        }, indent=2) + "\n")
        print(f"Mean ATE RMSE over {len(results)} sequences: {mean:.6f} m", flush=True)


if __name__ == "__main__":
    main()
