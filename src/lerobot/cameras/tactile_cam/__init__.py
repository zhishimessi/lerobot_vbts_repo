"""
触觉传感器模块

提供触觉相机和图像处理功能：
- TactileCamera: 触觉传感器相机类
- TactileCameraConfig: 相机配置类
- LookupTableProcessor: 基于查找表的图像处理器
- GradientProcessor: 基于梯度的图像处理器
- MLPProcessor: 基于神经网络的图像处理器
- TactileVisualizer: 可视化工具
"""

from .tactile_camera import TactileCamera
from .tactile_config import TactileCameraConfig
from .processors import LookupTableProcessor, GradientProcessor, MLPProcessor, BaseProcessor
from .visualization import (
    TactileVisualizer,
    visualize_depth,
    visualize_normals,
    visualize_gradient,
    visualize_diff
)

__all__ = [
    # Camera
    "TactileCamera",
    "TactileCameraConfig",
    # Processors
    "BaseProcessor",
    "LookupTableProcessor", 
    "GradientProcessor",
    "MLPProcessor",
    # Visualization
    "TactileVisualizer",
    "visualize_depth",
    "visualize_normals",
    "visualize_gradient",
    "visualize_diff",
]