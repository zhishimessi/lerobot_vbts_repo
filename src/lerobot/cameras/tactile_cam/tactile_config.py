from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["TactileCameraConfig", "ColorMode", "Cv2Rotation"]


@CameraConfig.register_subclass("tactile")
@dataclass
class TactileCameraConfig(CameraConfig):

    index_or_path: int | Path
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    fourcc: str | None = None
    
    # 曝光设置 (Linux V4L2: 通常 1-10000, 较小值=较暗)
    exposure: int = 1500
    auto_exposure: bool = False
    
    # 白平衡设置 (色温范围: 2000-8000K)
    wb_temperature: int = 4000
    auto_wb: bool = False
    
    # RGB增益 (范围: 0.0 - 3.0)
    r_gain: float = 1.0
    g_gain: float = 1.0
    b_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )

        if self.fourcc is not None and (not isinstance(self.fourcc, str) or len(self.fourcc) != 4):
            raise ValueError(
                f"`fourcc` must be a 4-character string (e.g., 'MJPG', 'YUYV'), but '{self.fourcc}' is provided."
            )