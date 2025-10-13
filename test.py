import argparse
import os
import sys

import mrcfile
import numpy as np
import napari


def load_mrc_or_synthetic(path: str | None) -> np.ndarray:
    """Load a 3D volume from an .mrc file, or generate a synthetic volume.

    Returns a 3D numpy array suitable for napari `add_image`.
    """
    if path and os.path.exists(path):
        try:
            with mrcfile.open(path, permissive=True) as f:
                data = np.asarray(f.data)
        except Exception as exc:  # Only catch known file I/O/format issues
            print(f"Failed to read MRC file '{path}': {exc}", file=sys.stderr)
            data = None
        if data is not None and data.ndim >= 3:
            # Ensure we pass a 3D volume (z, y, x); squeeze channels if present
            volume = np.squeeze(data)
            if volume.ndim > 3:
                # Heuristic: take the first channel if extra dims exist
                volume = volume.reshape((-1, *volume.shape[-2:]))
            return volume

    # Fallback: generate a small synthetic volume for smoke testing
    rng = np.random.default_rng(42)
    z, y, x = 64, 128, 128
    volume = (rng.random((z, y, x)) * 255).astype(np.uint8)
    return volume


def main() -> None:
    parser = argparse.ArgumentParser(description="View an .mrc volume in napari.")
    parser.add_argument(
        "mrc_path",
        nargs="?",
        default="test_data/fly_brain_em.mrc",
        help="Path to an .mrc file (optional). If missing, a synthetic volume is shown.",
    )
    args = parser.parse_args()

    volume = load_mrc_or_synthetic(args.mrc_path)
    if volume.ndim != 3:
        print(f"Expected a 3D volume, got shape {volume.shape}", file=sys.stderr)
        sys.exit(1)

    viewer = napari.Viewer()
    viewer.add_image(volume, name="volume", rendering="mip")
    napari.run()


if __name__ == "__main__":
    main()