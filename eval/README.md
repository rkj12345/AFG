# Evaluation

Run these commands from the repository root after installing the dependencies and placing the CUT3R checkpoint at `src/cut3r_512_dpt_4_64.pth`. Dataset downloads are governed by each benchmark's own terms. The data layouts below match the evaluation metadata and use relative paths under `data/`.

## Camera pose

`eval/relpose/metadata.py` defines TUM (`data/long_tum_s1`), ScanNet (`data/long_scannet_s3`), and Sintel layouts. Dataset keys include `tum_s1_1000` and `scannet_s3_1000`. For example:

```bash
accelerate launch --num_processes 1 eval/relpose/launch.py \
  --weights src/cut3r_512_dpt_4_64.pth \
  --eval_dataset tum_s1_1000 --size 512 \
  --model_update_type afg_pose \
  --output_dir eval_results/pose/tum_s1_1000/afg_pose
```

The launcher writes predicted trajectories and ATE/RPE summaries. KITTI **pose** uses the KITTI Odometry 00–10 RGB sequences in the [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) layout (`<kitti-root>/00/image_2/*.png` through `10/image_2/*.png`), with the corresponding 12-column odometry ground-truth files (`<gt-dir>/00.txt` through `10.txt`). It does not use the KITTI depth dataset. The paper protocol processes every frame at size 512, resets the recurrent state every 100 frames, converts the saved camera-to-world poses to KITTI 3×4 rows, and computes per-sequence ATE RMSE with `evo_ape kitti -r full -as` (Sim(3) alignment). The table average is the unweighted mean of the 11 sequence RMSE values.

The complete inference → conversion → evaluation command is:

```bash
python scripts/run_kitti_odometry.py \
  --kitti-root /path/to/VGGT_Long_outdoor/kitti_00_10 \
  --gt-dir /path/to/VGGT_Long_outdoor/gt_poses_kitti \
  --weights src/cut3r_512_dpt_4_64.pth \
  --model-update-type afg_pose \
  --output-dir eval_results/kitti_odometry/afg_pose
```

This writes `trajectories/00.txt`–`10.txt`, per-sequence `logs/ape_*.txt`, `ate_rmse.csv`, and `ate_rmse.json`. Run TTT3R or AFG-Img by changing `--model-update-type` and using a separate `--output-dir`. `--phase infer`, `--phase convert`, and `--phase eval` run individual steps on existing files; `--sequences 00 01` selects a subset. The runner checks RGB/GT/prediction frame counts and refuses to overwrite existing inference output unless `--force` is set. For a single sequence, the equivalent manual commands are:

```bash
python demo.py --model_path src/cut3r_512_dpt_4_64.pth \
  --seq_path /path/to/VGGT_Long_outdoor/kitti_00_10/00/image_2 \
  --output_dir eval_results/kitti_odometry/afg_pose/00 \
  --model_update_type afg_pose --size 512 --frame_interval 1 \
  --reset_interval 100 --no_viewer
python scripts/export_kitti_poses.py \
  eval_results/kitti_odometry/afg_pose/00/camera \
  eval_results/kitti_odometry/afg_pose/trajectories/00.txt
evo_ape kitti /path/to/VGGT_Long_outdoor/gt_poses_kitti/00.txt \
  eval_results/kitti_odometry/afg_pose/trajectories/00.txt -r full -as
```

For the inference-time comparisons, replace `--model_update_type afg_pose` with `ttt3r`, `ttsa3r`, or `ttt_memix`. The MeMix repository names its TTT3R combination `ttt_memix`; the other combinations are `cut_memix` and `ttsa_memix`. These modes use the same CUT3R checkpoint. MeMix defaults to its official 708-of-768 patch selector; override with `--memix_topk` / `--memix_num_patches` if required by a different protocol.

The source image and ground-truth paths above are examples; set them to the corresponding VGGT-Long KITTI Odometry locations on your machine.

## Video depth

For Bonn 500-frame data, prepare the original `rgbd_bonn_dataset` first. The script starts at frame index 30 and copies up to 500 paired RGB, depth, and ground-truth pose rows per sequence. The resulting layout includes `data/long_bonn_s1/rgbd_bonn_dataset/rgbd_bonn_balloon2/rgb_500/*.png` and its matching `depth_500/*.png` (and the other Bonn sequences listed in `eval/video_depth/metadata.py`). Depth predictions are written as `frame_*.npy` under each sequence directory.

```bash
python scripts/prepare_bonn.py \
  --source-root /path/to/rgbd_bonn_dataset --lengths 500
```

Then run inference and evaluation:

```bash
accelerate launch --num_processes 1 eval/video_depth/launch.py \
  --weights src/cut3r_512_dpt_4_64.pth \
  --eval_dataset bonn_s1_500 --size 512 \
  --model_update_type afg_pose \
  --output_dir eval_results/depth/bonn_s1_500/afg_pose
python eval/video_depth/eval_depth.py \
  --eval_dataset bonn_s1_500 --align metric \
  --output_dir eval_results/depth/bonn_s1_500/afg_pose
```

`--align scale` and `--align scale&shift` are also supported. The paper's Bonn headline uses metric alignment.

The depth evaluator checks that RGB, ground-truth, and prediction folders have matching sequences and frame counts before computing summary metrics. Bonn requires all five listed sequences.

KITTI **depth** is a separate KITTI depth dataset. Its existing evaluation keys (`kitti_s1_50` through `kitti_s1_500`) expect RGB and ground-truth depth under `data/long_kitti_s1/depth_selection/val_selection_cropped/image_gathered_<length>/<drive>_02/` and `groundtruth_depth_gathered_<length>/<drive>_02/`; see `eval/video_depth/metadata.py`. This is unrelated to the VGGT-Long odometry RGB/pose files above. The preparation script expects the original depth at `<source-root>/val/<drive>/proj_depth/groundtruth/image_02/*.png` and matching RGB at `<source-root>/<date>/<drive>/image_02/data/*.png`, following the original MonST3R KITTI layout. To prepare the 500-frame depth set:

```bash
python scripts/prepare_kitti_depth.py \
  --source-root /path/to/MonST3R/data/kitti --lengths 500
```

Both depth preparation scripts refuse to replace an existing generated directory unless `--overwrite` is passed.

Then run inference and evaluation:

```bash
accelerate launch --num_processes 1 eval/video_depth/launch.py \
  --weights src/cut3r_512_dpt_4_64.pth \
  --eval_dataset kitti_s1_500 --size 512 \
  --kitti_depth_root data/long_kitti_s1/depth_selection/val_selection_cropped \
  --model_update_type afg_pose \
  --output_dir eval_results/depth/kitti_s1_500/afg_pose
python eval/video_depth/eval_depth.py \
  --eval_dataset kitti_s1_500 --align metric \
  --kitti_depth_root data/long_kitti_s1/depth_selection/val_selection_cropped \
  --output_dir eval_results/depth/kitti_s1_500/afg_pose
```

The pose runner never reads KITTI depth data; depth evaluation does not consume the odometry pose files.

## 3D reconstruction

The 7-Scenes and NRGBD launchers accept `--data_root` rather than embedding a machine-specific directory. Both support `--max_frames` and all update rules listed in the root README:

```bash
accelerate launch --num_processes 1 eval/mv_recon/launch.py \
  --weights src/cut3r_512_dpt_4_64.pth \
  --data_root data/7scenes --max_frames 500 \
  --model_update_type afg_img --model_name afg_img \
  --output_dir eval_results/recon/7scenes_500/afg_img
accelerate launch --num_processes 1 eval/mv_recon/launch_nrgbd.py \
  --weights src/cut3r_512_dpt_4_64.pth \
  --data_root data/neural_rgbd --max_frames 500 \
  --model_update_type afg_img --model_name afg_img \
  --output_dir eval_results/recon/nrgbd_500/afg_img
```

These dataset classes use their original 7-Scenes/NRGBD image, depth, and camera structures. Refer to `eval/mv_recon/data.py` for scene discovery and exact filenames. The paper samples one frame every two frames for these benchmarks.
