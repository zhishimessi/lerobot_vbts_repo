"""
触觉传感器图像处理器模块

提供两种方法计算法向量和深度：
1. LookupTableProcessor: 基于查找表的方法（适用于校准过的传感器）
2. GradientProcessor: 基于梯度的方法（通用方法）
"""

import numpy as np
import cv2
import os
import math
import scipy.fftpack

from .fast_poisson import fast_poisson


class BaseProcessor:
    """处理器基类，提供公共功能"""
    
    def __init__(self, pad: int = 20, calib_file: str = None):
        """
        初始化处理器
        
        Args:
            pad: 边缘裁剪像素数
            calib_file: 透视变换矩阵文件路径
        """
        self.pad = pad
        self.con_flag = True  # 是否是第一帧
        
        # 透视变换相关
        self.homography_matrix = None
        self.output_size = None
        
        if calib_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            calib_file = os.path.join(current_dir, "data", "calibration_data", "homography_matrix.npz")
        self.calib_file = calib_file
        
        self._load_homography()
    
    def _load_homography(self):
        """加载透视变换矩阵"""
        try:
            calib_data = np.load(self.calib_file)
            self.homography_matrix = calib_data['homography_matrix']
            self.output_size = tuple(int(x) for x in calib_data['output_size'])
            print(f'[INFO] 成功加载透视变换矩阵，输出尺寸: {self.output_size}')
        except FileNotFoundError:
            print(f'[WARNING] 透视变换矩阵文件不存在: {self.calib_file}')
        except Exception as e:
            print(f'[WARNING] 加载透视变换矩阵失败: {e}')
    
    def warp_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        应用透视变换
        
        Args:
            image: 输入BGR图像
            
        Returns:
            变换后的图像
        """
        if self.homography_matrix is None:
            return image
        return cv2.warpPerspective(
            image, 
            self.homography_matrix, 
            self.output_size if self.output_size else (image.shape[1], image.shape[0]),
            flags=cv2.INTER_NEAREST
        )
    
    def _crop_image(self, img: np.ndarray) -> np.ndarray:
        """裁剪图像边缘"""
        if self.pad > 0:
            return img[self.pad:-self.pad, self.pad:-self.pad]
        return img
    
    def reset(self):
        """重置处理器状态，下一帧将作为新的参考帧"""
        self.con_flag = True
        print("[INFO] 处理器已重置，下一帧将作为参考帧")
    
    def process_frame(self, frame: np.ndarray):
        """
        处理单帧图像（子类需实现）
        
        Args:
            frame: BGR格式图像
            
        Returns:
            depth_colored: 深度图可视化
            normal_colored: 法向量可视化
            raw_depth: 原始深度数据
            raw_normals: 原始法向量数据
        """
        raise NotImplementedError


class LookupTableProcessor(BaseProcessor):
    """
    基于查找表的 GelSight 图像处理器
    
    使用预先校准的查找表将颜色差异映射到梯度值，
    然后通过泊松方程重建深度。
    """
    
    def __init__(self, table_path: str = None, pad: int = 20, calib_file: str = None):
        """
        初始化处理器
        
        Args:
            table_path: 查找表文件路径
            pad: 边缘裁剪像素数
            calib_file: 透视变换矩阵文件路径
        """
        super().__init__(pad=pad, calib_file=calib_file)
        
        # 查找表参数
        self.zeropoint = [-38, -30, -66]
        self.lookscale = [80, 62, 132]
        self.bin_num = 90
        
        # 加载查找表
        if table_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            table_path = os.path.join(current_dir, "load", "table_smooth.npy")
        self.table = np.load(table_path)
        
        # 状态变量
        self.ref_blur = None
        self.blur_inverse = None
        self.red_mask = None
        self.dmask = None
        self.kernel = self._make_kernel(9, 'circle')
        self.kernel2 = self._make_kernel(9, 'circle')
        self.reset_shape = True
    
    def _make_kernel(self, n: int, k_type: str) -> np.ndarray:
        """创建形态学核"""
        if k_type == 'circle':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
        return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
    
    def _defect_mask(self, img: np.ndarray) -> np.ndarray:
        """创建缺陷区域掩膜"""
        pad = 20
        im_mask = np.ones(img.shape)
        im_mask[:pad, :] = 0
        im_mask[-pad:, :] = 0
        im_mask[:, :pad * 2 + 20] = 0
        im_mask[:, -pad:] = 0
        return im_mask.astype(int)
    
    def _marker_detection(self, raw_image: np.ndarray) -> np.ndarray:
        """检测标记点区域"""
        m, n = raw_image.shape[1], raw_image.shape[0]
        raw_image_blur = cv2.GaussianBlur(raw_image.astype(np.float32), (5, 5), 0)
        ref_blur = cv2.GaussianBlur(raw_image.astype(np.float32), (25, 25), 0)
        
        diff = ref_blur - raw_image_blur
        diff *= 16.0
        diff[diff < 0.] = 0.
        diff[diff > 255.] = 255.
        
        mask = ((diff[:, :, 0] > 25) & (diff[:, :, 2] > 25) & (diff[:, :, 1] > 120))
        mask = cv2.resize(mask.astype(np.uint8), (m, n))
        mask = cv2.dilate(mask, self.kernel2, iterations=1)
        return mask
    
    def _find_dots(self, binary_image: np.ndarray):
        """查找标记点"""
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 1
        params.maxThreshold = 12
        params.minDistBetweenBlobs = 9
        params.filterByArea = True
        params.minArea = 5
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minInertiaRatio = 0.5
        detector = cv2.SimpleBlobDetector_create(params)
        return detector.detect(binary_image.astype(np.uint8))
    
    def _make_mask(self, img: np.ndarray, keypoints) -> np.ndarray:
        """根据关键点创建掩膜"""
        img_mask = np.zeros_like(img[:, :, 0])
        for kp in keypoints:
            cv2.ellipse(img_mask, (int(kp.pt[0]), int(kp.pt[1])),
                       (9, 6), 0, 0, 360, (1), -1)
        return img_mask
    
    def _matching_v2(self, test_img: np.ndarray, ref_blur: np.ndarray, 
                     blur_inverse: np.ndarray) -> np.ndarray:
        """使用查找表将颜色差异映射到梯度"""
        diff_temp1 = test_img - ref_blur
        diff_temp2 = diff_temp1 * blur_inverse
        
        # 分通道归一化
        diff_temp3 = np.zeros_like(diff_temp2, dtype=np.float32)
        for ch in range(3):
            diff_temp3[..., ch] = (diff_temp2[..., ch] - self.zeropoint[ch]) / self.lookscale[ch]
        
        diff_temp3 = np.clip(diff_temp3, 0, 0.999)
        diff = (diff_temp3 * self.bin_num).astype(int)
        diff = np.clip(diff, 0, self.bin_num - 1)
        
        # 查表获取梯度
        grad_img = self.table[diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], :]
        return grad_img
    
    def process_frame(self, frame: np.ndarray):
        """
        处理单帧图像
        
        Args:
            frame: BGR格式图像（已透视变换）
            
        Returns:
            depth_colored: 深度图可视化 (H, W, 3)
            normal_colored: 法向量可视化 (H, W, 3)
            raw_depth: 原始深度数据 (H, W) 或 None
            raw_normals: 原始法向量数据 (H, W, 3) 或 None
        """
        raw_image = self._crop_image(frame)
        h, w = raw_image.shape[:2]
        
        if self.con_flag:
            # 第一帧作为参考
            ref_image = raw_image.copy()
            marker = self._marker_detection(ref_image)
            keypoints = self._find_dots((1 - marker) * 255)
            
            if self.reset_shape:
                marker_mask = self._make_mask(ref_image, keypoints)
                ref_image = cv2.inpaint(ref_image, marker_mask, 3, cv2.INPAINT_TELEA)
                
                self.red_mask = (ref_image[:, :, 2] > 12).astype(np.uint8)
                self.dmask = self._defect_mask(ref_image[:, :, 0])
                self.ref_blur = cv2.GaussianBlur(ref_image.astype(np.float32), (5, 5), 0)
                self.blur_inverse = 1 + ((np.mean(self.ref_blur) / (self.ref_blur + 1)) - 1) * 2
                self.reset_shape = False
            
            self.con_flag = False
            
            return (
                np.zeros((h, w, 3), dtype=np.uint8),
                np.zeros((h, w, 3), dtype=np.uint8),
                None, None
            )
        
        # 处理后续帧
        raw_image = cv2.GaussianBlur(raw_image.astype(np.float32), (5, 5), 0)
        marker_mask = self._marker_detection(raw_image)
        
        # 计算梯度
        grad_img = self._matching_v2(raw_image, self.ref_blur, self.blur_inverse)
        grad_x = grad_img[:, :, 0] * (1 - marker_mask)
        grad_y = grad_img[:, :, 1] * (1 - marker_mask)
        
        # 平滑梯度
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), sigmaX=0) * (1 - marker_mask)
        grad_y = cv2.GaussianBlur(grad_y, (5, 5), sigmaX=0) * (1 - marker_mask)
        
        # 计算法向量
        denom = np.sqrt(1.0 + grad_x**2 + grad_y**2)
        normal_x = -grad_x / denom
        normal_y = -grad_y / denom
        normal_z = 1.0 / denom
        raw_normals = np.stack([normal_x, normal_y, normal_z], axis=-1)
        
        # 法向量可视化
        N_disp = 0.5 * (raw_normals + 1.0)
        N_disp = np.clip(N_disp, 0, 1)
        normal_colored = (N_disp * 255).astype(np.uint8)
        normal_colored = cv2.cvtColor(normal_colored, cv2.COLOR_RGB2BGR)
        
        # 计算深度
        raw_depth = fast_poisson(grad_x, grad_y)
        depth_min = np.nanmin(raw_depth[raw_depth != 0]) if np.any(raw_depth != 0) else 0
        raw_depth = raw_depth - depth_min
        raw_depth[raw_depth < 0] = 0
        
        # 深度可视化
        depth_denoised = cv2.bilateralFilter(raw_depth.astype(np.float32), d=9, 
                                             sigmaColor=75, sigmaSpace=75)
        depth_normalized = cv2.normalize(depth_denoised, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        return depth_colored, normal_colored, raw_depth, raw_normals
    
    def reset(self):
        """重置处理器状态"""
        super().reset()
        self.reset_shape = True
        self.ref_blur = None
        self.blur_inverse = None


def _recon_poisson_dst(img, frame0, x_ratio=0.5, y_ratio=0.5, bias=1.0):
    """
    使用DST求解泊松方程重建深度
    
    Args:
        img: 当前帧
        frame0: 参考帧
        x_ratio: x方向比例系数
        y_ratio: y方向比例系数
        bias: 偏置系数
        
    Returns:
        result: 重建的深度图
        dx_display: 梯度X
        dy_display: 梯度Y
    """
    img = np.int32(img)
    frame0 = np.int32(frame0)
    diff = (img - frame0) * bias

    dx1 = diff[:, :, 1] * x_ratio / 255.0
    dy1 = (diff[:, :, 2] * y_ratio - diff[:, :, 0] * (1 - y_ratio)) / 255.0
    
    dx1 = np.clip(dx1, -0.99, 0.99)
    dy1 = np.clip(dy1, -0.99, 0.99)
    
    dx_display = dx1 / np.sqrt(1 - dx1 ** 2)
    dy_display = dy1 / np.sqrt(1 - dy1 ** 2)
    
    dx = dx_display / 32
    dy = dy_display / 32

    gxx = dx[:-1, 1:] - dx[:-1, :-1]
    gyy = dy[1:, :-1] - dy[:-1, :-1]

    f = np.zeros(dx.shape)
    f[:-1, 1:] += gxx
    f[1:, :-1] += gyy

    tt = scipy.fftpack.dst(f, norm='ortho')
    fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

    x, y = np.meshgrid(range(1, f.shape[1]+1), range(1, f.shape[0]+1))
    denom = (2*np.cos(math.pi*x/(f.shape[1]+2))-2) + (2*np.cos(math.pi*y/(f.shape[0]+2)) - 2)

    f = fsin / denom

    tt = scipy.fftpack.idst(f, norm='ortho')
    img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

    result = np.zeros(f.shape)
    result[1:-1, 1:-1] = img_tt[1:-1, 1:-1]

    return result, dx_display, dy_display


class GradientProcessor(BaseProcessor):
    """
    基于梯度的 GelSight 图像处理器
    
    使用图像颜色差异直接计算梯度，不依赖查找表。
    适用于未校准或通用的传感器。
    """
    
    def __init__(self, pad: int = 20, sensor_id: str = "right", calib_file: str = None):
        """
        初始化处理器
        
        Args:
            pad: 边缘裁剪像素数
            sensor_id: 传感器ID，"left" 或 "right"
            calib_file: 透视变换矩阵文件路径
        """
        super().__init__(pad=pad, calib_file=calib_file)
        self.sensor_id = sensor_id
        
        # 状态变量
        self.ref_frame = None
        self.ref_blur = None
        self.frame_count = 0
    
    def img2grad(self, frame0: np.ndarray, frame: np.ndarray, 
                 bias: float = 4.0) -> tuple:
        """
        从图像差异计算梯度（线性方法）
        
        Args:
            frame0: 参考帧
            frame: 当前帧
            bias: 偏置系数
            
        Returns:
            dx, dy: x和y方向的梯度
        """
        diff = (frame.astype(np.float32) - frame0.astype(np.float32)) * bias
        dx = diff[:, :, 1] / 255.0  # Green通道
        dy = (diff[:, :, 0] - diff[:, :, 2]) / 255.0  # B - R
        return dx, dy
    
    def img2depth(self, frame0: np.ndarray, frame: np.ndarray, 
                  bias: float = 1.0, x_ratio: float = 0.5, 
                  y_ratio: float = 0.5) -> tuple:
        """
        从图像计算深度
        
        Args:
            frame0: 参考帧
            frame: 当前帧
            bias: 偏置系数
            x_ratio: x方向比例系数
            y_ratio: y方向比例系数
            
        Returns:
            depth: 深度图
            dx, dy: 梯度
        """
        return _recon_poisson_dst(frame, frame0, x_ratio, y_ratio, bias)
    
    def process_frame(self, frame: np.ndarray, apply_warp: bool = True):
        """
        处理单帧图像
        
        Args:
            frame: BGR格式图像
            apply_warp: 是否应用透视变换
            
        Returns:
            depth_colored: 深度图可视化 (H, W, 3)
            normal_colored: 法向量可视化 (H, W, 3)
            raw_depth: 原始深度数据 (H, W) 或 None
            raw_normals: 原始法向量数据 (H, W, 3) 或 None
            grad_x, grad_y: 梯度数据或 None
            diff_img: 差分图（用于调试）或 None
        """
        if apply_warp:
            warped = self.warp_perspective(frame)
        else:
            warped = frame
        raw_image = self._crop_image(warped)
        h, w = raw_image.shape[:2]
        
        if self.con_flag:
            # 第一帧作为参考
            self.ref_frame = raw_image.copy()
            self.ref_blur = cv2.GaussianBlur(self.ref_frame.astype(np.float32), (13, 13), 0)
            self.con_flag = False
            
            return (
                np.zeros((h, w, 3), dtype=np.uint8),
                np.zeros((h, w, 3), dtype=np.uint8),
                None, None, None, None, None
            )
        
        # 当前帧处理
        frame_blur = cv2.GaussianBlur(raw_image.astype(np.float32), (5, 5), 0)
        
        # 差分图（用于显示）
        diff = frame_blur - self.ref_blur
        diff_display = np.clip(diff * 2 + 127, 0, 255).astype(np.uint8)
        
        # 计算深度和梯度
        raw_depth, grad_x, grad_y = self.img2depth(self.ref_blur, frame_blur)
        
        # 打印统计信息（可选）
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            grad_x_valid = grad_x[np.abs(grad_x) > 0.01]
            if len(grad_x_valid) > 0:
                print(f"[梯度法] grad_x: [{np.min(grad_x):.4f}, {np.max(grad_x):.4f}], "
                      f"grad_y: [{np.min(grad_y):.4f}, {np.max(grad_y):.4f}]")
        
        # 计算法向量
        denom = np.sqrt(1.0 + grad_x**2 + grad_y**2)
        normal_x = -grad_x / denom
        normal_y = -grad_y / denom
        normal_z = 1.0 / denom
        raw_normals = np.stack([normal_x, normal_y, normal_z], axis=-1)
        
        # 法向量可视化
        N_disp = 0.5 * (raw_normals + 1.0)
        N_disp = np.clip(N_disp, 0, 1)
        normal_colored = (N_disp * 255).astype(np.uint8)
        normal_colored = cv2.cvtColor(normal_colored, cv2.COLOR_RGB2BGR)
        
        # 深度可视化
        depth_normalized = raw_depth - np.min(raw_depth)
        if np.max(depth_normalized) > 0:
            depth_normalized = depth_normalized / np.max(depth_normalized) * 255
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        
        return depth_colored, normal_colored, raw_depth, raw_normals, grad_x, grad_y, diff_display
    
    def reset(self):
        """重置处理器状态"""
        super().reset()
        self.ref_frame = None
        self.ref_blur = None
        self.frame_count = 0


# ============================================================================
# MLPProcessor: 基于神经网络的方法
# ============================================================================

class MLPProcessor(BaseProcessor):
    """
    基于MLP神经网络的触觉图像处理器
    
    使用训练好的神经网络将颜色映射到梯度，
    然后通过泊松方程重建深度。
    
    与查找表方法相比，MLP方法可以学习更复杂的映射关系。
    """
    
    def __init__(self, model_path: str = None, pad: int = 20, 
                 calib_file: str = None, device: str = None):
        """
        初始化处理器
        
        Args:
            model_path: MLP模型文件路径
            pad: 边缘裁剪像素数
            calib_file: 透视变换矩阵文件路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        super().__init__(pad=pad, calib_file=calib_file)
        
        # 延迟导入torch（避免不需要时的依赖）
        try:
            import torch
            self.torch = torch
        except ImportError:
            print("[ERROR] 需要安装 PyTorch: pip install torch")
            self.torch = None
            self.model = None
            return
        
        # 设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 加载模型
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "load", "mlp_gradient_model.pt")
        
        self.model_path = model_path
        self.model = None
        self.norm_params = None
        self._load_model()
        
        # 参考帧
        self.ref_frame = None
        self.ref_blur = None
        
        # 标记点掩膜阈值
        self.marker_threshold = 60
    
    def _load_model(self):
        """加载MLP模型"""
        if self.torch is None:
            return
            
        try:
            from .model import MLPGradientEncoder
            
            # 加载归一化参数
            norm_path = self.model_path.replace('.pt', '_norm.npz')
            if os.path.exists(norm_path):
                self.norm_params = dict(np.load(norm_path))
                input_dim = int(self.norm_params.get('input_dim', 5))
                hidden_dim = int(self.norm_params.get('hidden_dim', 32))
            else:
                input_dim = 5
                hidden_dim = 32
            
            # 加载模型
            self.model = MLPGradientEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
            state_dict = self.torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
            
            print(f"[INFO] MLP模型已加载: {self.model_path}")
            print(f"[INFO] 输入维度: {input_dim}, 隐藏层: {hidden_dim}, 设备: {self.device}")
            
        except FileNotFoundError:
            print(f"[WARNING] 模型文件不存在: {self.model_path}")
            self.model = None
        except Exception as e:
            print(f"[WARNING] 加载MLP模型失败: {e}")
            self.model = None
    
    def _get_marker_mask(self, image: np.ndarray) -> np.ndarray:
        """获取标记点掩膜"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray < self.marker_threshold).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask
    
    def _infer_gradient(self, image: np.ndarray, ref_image: np.ndarray) -> tuple:
        """使用MLP推理梯度"""
        if self.model is None or self.torch is None:
            return None, None
        
        h, w = image.shape[:2]
        
        # 获取标记点掩膜
        marker_mask = self._get_marker_mask(image)
        ref_marker_mask = self._get_marker_mask(ref_image)
        combined_mask = cv2.bitwise_or(marker_mask, ref_marker_mask)
        
        # 计算差值图像
        diff = cv2.absdiff(image, ref_image)
        diff_gray = np.max(diff, axis=2)
        contact_mask = (diff_gray > 20).astype(np.uint8)
        
        # 有效区域
        valid_mask = contact_mask & (combined_mask == 0)
        
        y_coords, x_coords = np.where(valid_mask > 0)
        
        if len(x_coords) == 0:
            return np.zeros((h, w)), np.zeros((h, w))
        
        # 准备输入特征
        bgr_values = image[y_coords, x_coords].astype(np.float32) / 255.0
        
        if self.norm_params is not None and 'x_max' in self.norm_params:
            x_norm = x_coords.astype(np.float32) / float(self.norm_params['x_max'])
            y_norm = y_coords.astype(np.float32) / float(self.norm_params['y_max'])
        else:
            x_norm = x_coords.astype(np.float32) / w
            y_norm = y_coords.astype(np.float32) / h
        
        features = np.column_stack([
            bgr_values[:, 0], bgr_values[:, 1], bgr_values[:, 2],
            x_norm, y_norm
        ])
        
        # 推理
        features_tensor = self.torch.tensor(features, dtype=self.torch.float32).to(self.device)
        
        with self.torch.no_grad():
            gradients = self.model(features_tensor)
            gradients = gradients.cpu().numpy()
        
        # 重建梯度图
        gx = np.zeros((h, w), dtype=np.float32)
        gy = np.zeros((h, w), dtype=np.float32)
        gx[y_coords, x_coords] = gradients[:, 0]
        gy[y_coords, x_coords] = gradients[:, 1]
        
        return gx, gy
    
    def _compute_normals(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """从梯度计算法向量"""
        gz = np.ones_like(gx)
        magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        magnitude[magnitude == 0] = 1
        normals = np.stack([gx/magnitude, gy/magnitude, gz/magnitude], axis=-1)
        return normals
    
    def process_frame(self, frame: np.ndarray, apply_warp: bool = False):
        """
        处理单帧图像
        
        Args:
            frame: BGR格式图像
            apply_warp: 是否应用透视变换
            
        Returns:
            depth_colored: 深度图可视化
            normal_colored: 法向量可视化
            raw_depth: 原始深度数据
            raw_normals: 原始法向量数据
        """
        if apply_warp:
            frame = self.warp_perspective(frame)
        
        h, w = frame.shape[:2]
        
        # 第一帧作为参考
        if self.con_flag:
            self.ref_frame = frame.copy()
            self.ref_blur = cv2.GaussianBlur(frame.astype(np.float32), (3, 3), 0)
            self.con_flag = False
            return (np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w)),
                    np.zeros((h, w, 3)))
        
        # 推理梯度
        gx, gy = self._infer_gradient(frame, self.ref_frame)
        
        if gx is None:
            return (np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w)),
                    np.zeros((h, w, 3)))
        
        # 泊松重建
        depth = fast_poisson(gx, gy)
        
        # 计算法向量
        normals = self._compute_normals(gx, gy)
        
        # 可视化
        depth_norm = depth - depth.min()
        if depth_norm.max() > 0:
            depth_norm = depth_norm / depth_norm.max()
        depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        normal_colored = ((normals + 1) / 2 * 255).astype(np.uint8)
        
        return depth_colored, normal_colored, depth, normals
    
    def reset(self):
        """重置处理器状态"""
        super().reset()
        self.ref_frame = None
        self.ref_blur = None
