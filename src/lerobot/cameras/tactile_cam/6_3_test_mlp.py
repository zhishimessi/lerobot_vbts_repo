"""
MLP方法测试脚本

使用训练好的MLP模型实时处理触觉传感器图像，
计算深度图和法向量图。

使用方法:
1. 先运行 6_1_collect_mlp_dataset.py 收集数据
2. 运行 6_2_train_mlp.py 训练模型
3. 运行本脚本测试效果
"""

import cv2
import numpy as np
import os
import torch

from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.tactile_cam.processors import BaseProcessor
from lerobot.cameras.tactile_cam.fast_poisson import fast_poisson
from lerobot.cameras.tactile_cam.model import MLPGradientEncoder
from lerobot.cameras.tactile_cam.visualization import TactileVisualizer
from lerobot.cameras.configs import ColorMode, Cv2Rotation


class MLPProcessor(BaseProcessor):
    """
    基于MLP神经网络的触觉图像处理器
    
    使用训练好的神经网络将颜色差值映射到梯度，
    然后通过泊松方程重建深度。
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
        try:
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
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
            
            print(f"[INFO] MLP模型已加载: {self.model_path}")
            print(f"[INFO] 输入维度: {input_dim}, 隐藏层: {hidden_dim}")
            print(f"[INFO] 使用设备: {self.device}")
            
        except FileNotFoundError:
            print(f"[ERROR] 模型文件不存在: {self.model_path}")
            print("[INFO] 请先运行训练脚本 6_2_train_mlp.py")
            self.model = None
        except Exception as e:
            print(f"[ERROR] 加载模型失败: {e}")
            self.model = None
    
    def _get_marker_mask(self, image: np.ndarray) -> np.ndarray:
        """获取标记点掩膜"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray < self.marker_threshold).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask
    
    def _infer_gradient(self, image: np.ndarray, ref_image: np.ndarray) -> tuple:
        """
        使用MLP推理梯度
        
        Args:
            image: 当前帧 (BGR)
            ref_image: 参考帧 (BGR)
            
        Returns:
            (gx, gy) 梯度图
        """
        if self.model is None:
            return None, None
        
        h, w = image.shape[:2]
        
        # 获取标记点掩膜
        marker_mask = self._get_marker_mask(image)
        ref_marker_mask = self._get_marker_mask(ref_image)
        combined_mask = cv2.bitwise_or(marker_mask, ref_marker_mask)
        
        # 计算差值图像（用于检测接触区域）
        diff = cv2.absdiff(image, ref_image)
        diff_gray = np.max(diff, axis=2)
        contact_mask = (diff_gray > 20).astype(np.uint8)
        
        # 有效区域 = 接触区域 - 标记点
        valid_mask = contact_mask & (combined_mask == 0)
        
        # 获取有效像素坐标
        y_coords, x_coords = np.where(valid_mask > 0)
        
        if len(x_coords) == 0:
            return np.zeros((h, w)), np.zeros((h, w))
        
        # 准备输入特征
        # BGR 值（归一化到 [0, 1]）
        bgr_values = image[y_coords, x_coords].astype(np.float32) / 255.0
        
        # 坐标（归一化）
        if self.norm_params is not None and 'x_max' in self.norm_params:
            x_norm = x_coords.astype(np.float32) / float(self.norm_params['x_max'])
            y_norm = y_coords.astype(np.float32) / float(self.norm_params['y_max'])
        else:
            x_norm = x_coords.astype(np.float32) / w
            y_norm = y_coords.astype(np.float32) / h
        
        # 组合特征 [B, G, R, X, Y]
        features = np.column_stack([
            bgr_values[:, 0],  # B
            bgr_values[:, 1],  # G
            bgr_values[:, 2],  # R
            x_norm,            # X
            y_norm             # Y
        ])
        
        # 转换为张量并推理
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
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
        
        # 归一化
        magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        magnitude[magnitude == 0] = 1
        
        normals = np.stack([gx/magnitude, gy/magnitude, gz/magnitude], axis=-1)
        return normals
    
    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """深度图着色"""
        depth_normalized = depth - depth.min()
        max_val = depth_normalized.max()
        if max_val > 0:
            depth_normalized = depth_normalized / max_val
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    def _colorize_normals(self, normals: np.ndarray) -> np.ndarray:
        """法向量着色"""
        normals_vis = ((normals + 1) / 2 * 255).astype(np.uint8)
        return normals_vis
    
    def reset(self):
        """重置处理器"""
        self.ref_frame = None
        self.ref_blur = None
        self.con_flag = True
        print("[INFO] MLP处理器已重置")
    
    def process_frame(self, frame: np.ndarray, apply_warp: bool = False):
        """
        处理单帧图像
        
        Args:
            frame: BGR格式图像
            apply_warp: 是否应用透视变换（输入已变换时设为False）
            
        Returns:
            depth_colored: 深度图可视化
            normal_colored: 法向量可视化
            raw_depth: 原始深度数据
            raw_normals: 原始法向量数据
        """
        # 透视变换
        if apply_warp:
            frame = self.warp_perspective(frame)
        
        # 第一帧作为参考
        if self.con_flag:
            self.ref_frame = frame.copy()
            self.ref_blur = cv2.GaussianBlur(frame.astype(np.float32), (3, 3), 0)
            self.con_flag = False
            
            h, w = frame.shape[:2]
            return (np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w)),
                    np.zeros((h, w, 3)))
        
        # 推理梯度
        gx, gy = self._infer_gradient(frame, self.ref_frame)
        
        if gx is None:
            h, w = frame.shape[:2]
            return (np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w)),
                    np.zeros((h, w, 3)))
        
        # 泊松重建深度
        depth = fast_poisson(gx, gy)
        
        # 计算法向量
        normals = self._compute_normals(gx, gy)
        
        # 可视化
        depth_colored = self._colorize_depth(depth)
        normal_colored = self._colorize_normals(normals)
        
        return depth_colored, normal_colored, depth, normals


def main():
    """主函数：测试MLP方法"""
    
    # 相机配置
    camera_config = TactileCameraConfig(
        index_or_path="/dev/video2", 
        fps=25,                       
        width=640,                   
        height=480,
        color_mode=ColorMode.RGB,     
        rotation=Cv2Rotation.NO_ROTATION, 
    )
    
    # 数据保存目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "data", "tactile_data_mlp")
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化组件
    camera = TactileCamera(camera_config)
    processor = MLPProcessor()
    visualizer = TactileVisualizer(
        windows=['original', 'depth', 'normal'],
        window_size=(640, 480)
    )
    
    if processor.model is None:
        print("[ERROR] 模型未加载，无法继续")
        return
    
    try:
        camera.connect()
        print("[INFO] 相机已连接")
        print("\n=== 触觉传感器测试（MLP方法）===")
        print("操作说明:")
        print("  r - 重置参考帧")
        print("  s - 保存当前数据")
        print("  q - 退出")
        print("=====================================\n")
        
        # 数据收集
        all_depth_maps = []
        frame_count = 0
        SAVE_EVERY = 10
        
        while True:
            try:
                # 读取并转换图像
                frame = camera.async_read(timeout_ms=200)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 透视变换
                warped_frame = processor.warp_perspective(frame_bgr)
                
                # 处理图像
                depth_colored, normal_colored, raw_depth, raw_normals = \
                    processor.process_frame(warped_frame, apply_warp=False)
                
                # 收集数据
                if not processor.con_flag and raw_depth is not None:
                    frame_count += 1
                    if frame_count % SAVE_EVERY == 0:
                        all_depth_maps.append(raw_depth.copy())
                
                # 显示结果
                visualizer.show('original', warped_frame)
                visualizer.show('depth', depth_colored)
                visualizer.show('normal', normal_colored)
                
                # 处理按键
                key = visualizer.wait_key(1)
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    processor.reset()
                    print("[INFO] 处理器已重置")
                elif key == ord('s') and raw_depth is not None:
                    timestamp = int(cv2.getTickCount())
                    np.save(os.path.join(save_dir, f"depth_{timestamp}.npy"), raw_depth)
                    np.save(os.path.join(save_dir, f"normals_{timestamp}.npy"), raw_normals)
                    print(f"[INFO] 数据已保存: depth_{timestamp}.npy")
                    
            except TimeoutError:
                continue
            except RuntimeError as e:
                print(f"[WARNING] 帧读取错误: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    
    finally:
        # 保存收集的数据
        if len(all_depth_maps) > 0:
            np.save(os.path.join(save_dir, "depth_sequence.npy"), np.array(all_depth_maps))
            print(f"[INFO] 保存了 {len(all_depth_maps)} 帧数据到 {save_dir}")
        
        visualizer.cleanup()
        camera.disconnect()
        print("[INFO] 相机已断开")


if __name__ == "__main__":
    main()
