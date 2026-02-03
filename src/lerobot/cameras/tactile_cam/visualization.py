"""
触觉传感器可视化工具模块

提供深度图、法向量、梯度等数据的可视化功能。
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def visualize_depth(depth: np.ndarray, colormap: int = cv2.COLORMAP_VIRIDIS,
                   denoise: bool = True) -> np.ndarray:
    """
    将深度图转换为彩色可视化图像
    
    Args:
        depth: 深度数据 (H, W)
        colormap: OpenCV颜色映射，默认VIRIDIS
        denoise: 是否进行降噪处理
        
    Returns:
        彩色深度图 (H, W, 3)
    """
    if depth is None:
        return None
    
    # 归一化
    depth_normalized = depth - np.min(depth)
    if np.max(depth_normalized) > 0:
        depth_normalized = depth_normalized / np.max(depth_normalized)
    
    # 可选降噪
    if denoise:
        depth_normalized = cv2.bilateralFilter(
            depth_normalized.astype(np.float32), 
            d=9, sigmaColor=75, sigmaSpace=75
        )
    
    # 转换为8位并应用颜色映射
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, colormap)


def visualize_normals(normals: np.ndarray, output_bgr: bool = True) -> np.ndarray:
    """
    将法向量转换为彩色可视化图像
    
    法向量的 (x, y, z) 分量映射到 RGB 颜色通道：
    - x: 左(-1)为暗红，右(+1)为亮红
    - y: 上(-1)为暗绿，下(+1)为亮绿
    - z: 平面为暗蓝，凸起为亮蓝
    
    Args:
        normals: 法向量数据 (H, W, 3)，范围 [-1, 1]
        output_bgr: 是否输出BGR格式（用于OpenCV显示）
        
    Returns:
        彩色法向量图 (H, W, 3)
    """
    if normals is None:
        return None
    
    # 归一化到 [0, 1]
    N_disp = 0.5 * (normals + 1.0)
    N_disp = np.clip(N_disp, 0, 1)
    
    # 转换为8位图像
    result = (N_disp * 255).astype(np.uint8)
    
    if output_bgr:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    return result


def visualize_gradient(grad_x: np.ndarray, grad_y: np.ndarray,
                       colormap: int = cv2.COLORMAP_JET) -> Tuple[np.ndarray, np.ndarray]:
    """
    将梯度数据转换为彩色可视化图像
    
    Args:
        grad_x: X方向梯度 (H, W)
        grad_y: Y方向梯度 (H, W)
        colormap: OpenCV颜色映射
        
    Returns:
        (grad_x_colored, grad_y_colored): 彩色梯度图
    """
    if grad_x is None or grad_y is None:
        return None, None
    
    grad_x_norm = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX)
    grad_y_norm = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX)
    
    grad_x_colored = cv2.applyColorMap(grad_x_norm.astype(np.uint8), colormap)
    grad_y_colored = cv2.applyColorMap(grad_y_norm.astype(np.uint8), colormap)
    
    return grad_x_colored, grad_y_colored


def visualize_diff(current: np.ndarray, reference: np.ndarray,
                   scale: float = 2.0, offset: float = 127) -> np.ndarray:
    """
    可视化两帧之间的差异
    
    Args:
        current: 当前帧
        reference: 参考帧
        scale: 差异放大系数
        offset: 偏移量（使差异居中于灰度）
        
    Returns:
        差分可视化图像
    """
    if current is None or reference is None:
        return None
    
    diff = current.astype(np.float32) - reference.astype(np.float32)
    diff_display = np.clip(diff * scale + offset, 0, 255).astype(np.uint8)
    return diff_display


class TactileVisualizer:
    """
    触觉传感器可视化器
    
    管理多个窗口，显示触觉传感器的各种数据。
    """
    
    DEFAULT_WINDOWS = {
        'original': 'Original Frame',
        'depth': 'Depth Map',
        'normal': 'Normal Vector',
        'grad_x': 'Gradient X',
        'grad_y': 'Gradient Y',
        'diff': 'Difference',
        'marker': 'Marker Motion'
    }
    
    def __init__(self, windows: list = None, window_size: Tuple[int, int] = (640, 480)):
        """
        初始化可视化器
        
        Args:
            windows: 要创建的窗口列表，使用DEFAULT_WINDOWS中的键名
                    如果为None，则创建所有窗口
            window_size: 窗口大小 (width, height)
        """
        self.window_size = window_size
        self.active_windows = {}
        
        if windows is None:
            windows = ['original', 'depth', 'normal']
        
        for key in windows:
            if key in self.DEFAULT_WINDOWS:
                name = self.DEFAULT_WINDOWS[key]
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(name, window_size[0], window_size[1])
                self.active_windows[key] = name
    
    def show(self, key: str, image: np.ndarray):
        """
        在指定窗口显示图像
        
        Args:
            key: 窗口键名
            image: 要显示的图像
        """
        if image is None:
            return
        if key in self.active_windows:
            cv2.imshow(self.active_windows[key], image)
    
    def show_all(self, original: np.ndarray = None, 
                 depth_colored: np.ndarray = None,
                 normal_colored: np.ndarray = None,
                 grad_x: np.ndarray = None,
                 grad_y: np.ndarray = None,
                 diff: np.ndarray = None,
                 marker: np.ndarray = None):
        """
        一次性显示所有可用数据
        
        Args:
            original: 原始帧
            depth_colored: 深度彩色图
            normal_colored: 法向量彩色图
            grad_x: X梯度
            grad_y: Y梯度
            diff: 差分图
            marker: 标记点追踪图
        """
        if original is not None:
            self.show('original', original)
        if depth_colored is not None:
            self.show('depth', depth_colored)
        if normal_colored is not None:
            self.show('normal', normal_colored)
        if grad_x is not None:
            grad_x_vis, grad_y_vis = visualize_gradient(grad_x, grad_y)
            self.show('grad_x', grad_x_vis)
            self.show('grad_y', grad_y_vis)
        if diff is not None:
            self.show('diff', diff)
        if marker is not None:
            self.show('marker', marker)
    
    def wait_key(self, delay: int = 1) -> int:
        """
        等待按键输入
        
        Args:
            delay: 等待时间（毫秒）
            
        Returns:
            按键ASCII码
        """
        return cv2.waitKey(delay) & 0xFF
    
    def cleanup(self):
        """关闭所有窗口"""
        cv2.destroyAllWindows()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
