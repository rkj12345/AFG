# Adaptive Frame Gating (AFG)

Code for **Rethinking the State Update Gate for Long-Sequence Recurrent 3D Reconstruction** ([paper](https://arxiv.org/abs/2605.16981)). AFG improves long-sequence inference with a frame-level gate on the recurrent state of a frozen CUT3R model. It adds no learned parameters or extra model forward pass.

## Method

TTT3R computes a per-state-token update rate β from state-to-image cross-attention logits. AFG multiplies it by a scalar frame rate α:

```text
S_t = S_(t-1) + α_t β_t ⊙ (S_new,t − S_(t-1))
```

Two variants are provided. Their names describe the **feature used to compute α**, regardless of earlier internal experiment names:

| `--model_update_type` | Paper name | Frame gate | Default |
| --- | --- | --- | --- |
| `afg_img` | AFG-Img | `sigmoid(||g_t − g_(t-1)||₂ − τ_img)`; `g` is the spatial mean of encoder image tokens | `τ_img=1.5` |
| `afg_pose` | AFG-Pose | `sigmoid(κ (||p_t − p_(t-1)||₂ − τ_pose))`; `p` is the final decoder pose token | `τ_pose=1.0`, `κ=3.0` |
| `ttt3r` | TTT3R baseline | `α=1`, token gate β | — |
| `cut3r` | CUT3R baseline | full state update | — |
| `ttsa3r` | TTSA3R baseline | temporal × spatial token gate | — |
| `ttt_memix` | TTT3R + MeMix | β × MeMix hard patch mask | `708/768` patches |
| `cut_memix` | CUT3R + MeMix | MeMix hard patch mask | `708/768` patches |
| `ttsa_memix` | TTSA3R + MeMix | TTSA3R gate × MeMix hard patch mask | `708/768` patches |

For CUT3R, TTT3R, TTSA3R, and AFG the first frame initializes the state with a full update; MeMix applies its patch selector from the first frame. AFG gates the recurrent state; the model's separate pose memory retains its original per-frame update. In the long KITTI demo protocol, a chunk reset reinitializes the recurrent state while AFG retains pose memory, matching the paper implementation. The implementation is in [`src/dust3r/model.py`](src/dust3r/model.py) and [`src/dust3r/update_gates.py`](src/dust3r/update_gates.py). Evaluation and the demo call `forward_recurrent_lighter` through [`src/dust3r/inference.py`](src/dust3r/inference.py).

## Installation

Use Python 3.11 with a PyTorch/CUDA build appropriate for your GPU. From the repository root:

```bash
python -m pip install -r requirements.txt
cd src/croco/models/curope
python setup.py build_ext --inplace
cd ../../../..
```

Download the pretrained [`cut3r_512_dpt_4_64.pth` checkpoint](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view) and save it at `src/cut3r_512_dpt_4_64.pth`, or pass another trusted checkpoint path to `--model_path` / `--weights`. The checkpoint contains a Python configuration object, so loading it requires trusting its source. AFG uses this frozen checkpoint; no AFG checkpoint is needed.

## Run on a video or image directory

```bash
python demo.py \
  --model_path src/cut3r_512_dpt_4_64.pth \
  --seq_path /path/to/video.mp4 \
  --output_dir outputs/demo \
  --model_update_type afg_pose \
  --size 512 --port 8080
```

`--seq_path` also accepts a directory of ordered images. Use `afg_img`, `ttt3r`, `cut3r`, `ttsa3r`, or a MeMix combination to compare update rules. `--afg_img_threshold`, `--afg_pose_threshold`, and `--afg_pose_sharpness` expose the AFG constants; `--memix_num_patches` and `--memix_topk` expose the MeMix selector (defaults 768 and 708). The demo saves depth, confidence, color, and camera outputs and starts a Viser viewer.

## Evaluation

The evaluation entry points are [`eval/relpose/launch.py`](eval/relpose/launch.py) for camera pose, [`eval/video_depth/launch.py`](eval/video_depth/launch.py) plus [`eval/video_depth/eval_depth.py`](eval/video_depth/eval_depth.py) for depth, and [`eval/mv_recon/launch.py`](eval/mv_recon/launch.py) / [`launch_nrgbd.py`](eval/mv_recon/launch_nrgbd.py) for reconstruction. Bonn and KITTI depth preparation commands are also in [`eval/README.md`](eval/README.md), alongside dataset layouts and evaluation commands. Pass `--model_update_type afg_pose` or `afg_img` consistently; scripts use the feature-based names above.

The paper evaluates camera pose on TUM-RGBD, ScanNet, and KITTI Odometry (following the VGGT-Long sequence layout); video depth on Bonn; and 3D reconstruction on 7-Scenes and NRGBD. KITTI depth evaluation is a separate optional dataset path. The release includes source and evaluation code. Checkpoints, benchmark datasets, generated results, and private submission materials are excluded.

## License and attribution

This project builds on [CUT3R](https://github.com/CUT3R/CUT3R), [TTT3R](https://github.com/Inception3D/TTT3R), [TTSA3R](https://github.com/anonus2357/ttsa3r), [MeMix](https://github.com/dongjiacheng06/MeMix), [DUSt3R](https://github.com/naver/dust3r), and [CroCo](https://github.com/naver/croco). Please read [`LICENSE`](LICENSE) and [`THIRD_PARTY.md`](THIRD_PARTY.md) before redistribution or use; inherited components include CC BY-NC-SA 4.0 restrictions.

```bibtex
@article{ren2026rethinking,
  title={Rethinking the State Update Gate for Long-Sequence Recurrent 3D Reconstruction},
  author={Ren, Kejun and Jin, Lei and Huang, Tianxin and Xu, Lianming and Wang, Li},
  journal={arXiv preprint arXiv:2605.16981},
  year={2026}
}
```
