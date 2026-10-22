import sys
from pathlib import Path


def add_path_to_dust3r(ckpt):
    """Expose the vendored DUSt3R package regardless of checkpoint location."""
    src_dir = str(Path(__file__).resolve().parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
