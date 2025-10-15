import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Callable, Any, Dict

import numpy as np

try:
    # Optional dependency: wrapper must work even if nnU-Net is not installed
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # type: ignore
    NNUNET_AVAILABLE = True
except Exception:
    nnUNetPredictor = None  # type: ignore
    NNUNET_AVAILABLE = False

logger = logging.getLogger(__name__)


class NnUNetNotAvailableError(RuntimeError):
    """Raised when nnU-Net is required but not available in the environment."""


class NnUNetWrapper:
    """Lightweight wrapper around nnU-Net v2 predictor with sliding-window defaults.

    This class hides import details and provides a stable interface for the
    orchestrator/inference pipeline. It supports lazy initialization so the
    host application can boot without nnU-Net installed.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        perform_everything_on_device: bool = True,
        checkpoint_name: str = "checkpoint_best.pth",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        self.perform_everything_on_device = perform_everything_on_device
        self.checkpoint_name = checkpoint_name

        self._predictor = None  # Lazy init

    def is_available(self) -> bool:
        return NNUNET_AVAILABLE

    def ensure_initialized(self) -> None:
        if self._predictor is not None:
            return
        if not NNUNET_AVAILABLE:
            raise NnUNetNotAvailableError(
                "nnU-Net v2 is not installed. Install nnunetv2 to enable inference."
            )
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        logger.info("Initializing nnU-Net predictor from %s", self.model_dir)
        predictor = nnUNetPredictor(
            tile_step_size=self.tile_step_size,
            use_gaussian=self.use_gaussian,
            use_mirroring=self.use_mirroring,
            perform_everything_on_device=self.perform_everything_on_device,
            device=self.device,
            verbose=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(self.model_dir), use_folds="all", checkpoint_name=self.checkpoint_name
        )
        self._predictor = predictor

    def predict_numpy(
        self,
        volume: np.ndarray,
        progress_callback: Optional[Callable[[float], Any]] = None,
        tmp_dir: Optional[str] = None,
    ) -> np.ndarray:
        """Run prediction on an in-memory numpy volume.

        Current implementation uses a temporary NIfTI file round-trip to leverage
        nnU-Net's file-based prediction API. This keeps the wrapper simple and
        isolates dependency details. We can replace this with an in-memory path
        once we add a direct-forward implementation.
        """
        self.ensure_initialized()

        # Basic input validation
        if volume.ndim not in (3, 4):
            raise ValueError("Expected a 3D or 4D volume (C,Z,Y,X or Z,Y,X)")

        # If 3D, add a dummy channel axis expected by many medical models
        if volume.ndim == 3:
            volume = volume[None, ...]

        # Normalize to float32 if needed
        if volume.dtype != np.float32:
            volume = volume.astype(np.float32, copy=False)

        # Write to temp NIfTI files to use predict_from_files
        work_dir_cm = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="cm_nnunet_"))
        input_path = work_dir_cm / "input.nii.gz"
        output_path = work_dir_cm / "output.nii.gz"

        try:
            # Lazy import nibabel to avoid hard dependency at server boot
            import nibabel as nib  # type: ignore

            affine = np.eye(4, dtype=np.float32)
            nib.save(nib.Nifti1Image(volume.squeeze(), affine), str(input_path))

            # Hook progress if provided by sending a couple of coarse-grained updates
            if progress_callback:
                progress_callback(0.05)

            assert self._predictor is not None  # for type checkers
            self._predictor.predict_from_files(
                [str(input_path)],
                [str(output_path)],
                save_probabilities=False,
                overwrite=True,
            )

            if progress_callback:
                progress_callback(0.9)

            result_img = nib.load(str(output_path))
            result = result_img.get_fdata().astype(np.float32)

            if result.ndim == 3:
                # return label volume (Z,Y,X)
                return result
            if result.ndim == 4:
                # If channel axis present, choose argmax across channels as labels
                return np.argmax(result, axis=0).astype(np.float32)
            return result
        finally:
            # Best-effort cleanup
            try:
                if input_path.exists():
                    input_path.unlink()
                if output_path.exists():
                    output_path.unlink()
                # Do not remove the temp directory if provided by caller
                if tmp_dir is None and work_dir_cm.exists():
                    try:
                        os.rmdir(work_dir_cm)
                    except Exception:
                        pass
            except Exception:
                pass

    def warmup(self) -> Dict[str, Any]:
        """Attempt to initialize the model and return environment info."""
        available = self.is_available()
        info: Dict[str, Any] = {
            "nnunet_available": available,
            "model_dir": str(self.model_dir),
        }
        if not available:
            info["status"] = "nnU-Net not installed"
            return info
        try:
            self.ensure_initialized()
            assert self._predictor is not None
            info.update(
                {
                    "device": str(self._predictor.device),
                    "cfg": str(self._predictor.cfg),
                }
            )
            info["status"] = "ready"
        except Exception as e:
            info["status"] = f"error: {e}"
        return info


