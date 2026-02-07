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
import sys
import torch
import pyvista as pv

# 确保可以导入 lerobot 模块
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.tactile_cam.processors import BaseProcessor
from lerobot.cameras.tactile_cam.fast_poisson import fast_poisson
from lerobot.cameras.tactile_cam.model import MLPGradientEncoder
from lerobot.cameras.tactile_cam.visualization import TactileVisualizer, visualize_gradient
from lerobot.cameras.configs import ColorMode, Cv2Rotation


class MLPProcessor(BaseProcessor):
    """
    基于MLP神经网络的触觉图像处理器
    
    使用训练好的神经网络将颜色差值映射到梯度，
    然后通过泊松方程重建深度。
    """
    
    def __init__(self, model_path: str = None, pad: int = 20, 
                 calib_file: str = None, device: str = None, has_marker: bool = False):
        """
        初始化处理器
        
        Args:
            model_path: MLP模型文件路径
            pad: 边缘裁剪像素数
            calib_file: 透视变换矩阵文件路径
            device: 计算设备 ('cuda' 或 'cpu')
            has_marker: 是否有标记点，False则不进行marker检测
        """
        super().__init__(pad=pad, calib_file=calib_file)
        
        # 是否有marker
        self.has_marker = has_marker
        
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
        
        # 多帧平均参考帧
        self.ref_frames_buffer = []
        self.ref_avg_count = 10  # 用于平均的帧数
        
        # 标记点掩膜阈值
        self.marker_threshold = 60
        
        # 深度范围平滑（避免闪烁）
        self.depth_min_smooth = None
        self.depth_max_smooth = None
        self.depth_smooth_alpha = 0.1  # 指数平滑因子
    
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
        
        使用颜色差分值作为输入，与训练数据一致
        这样无接触区域的差分值接近0，模型输出也接近0
        
        Args:
            image: 当前帧 (BGR)
            ref_image: 参考帧 (BGR)
            
        Returns:
            (gx, gy) 梯度图
        """
        if self.model is None:
            return None, None
        
        h, w = image.shape[:2]
        
        # 计算颜色差分图像（当前帧 - 参考帧）
        diff_image = image.astype(np.float32) - ref_image.astype(np.float32)
        
        # 计算差分幅度（用于检测接触区域）
        diff_abs = np.abs(diff_image)
        diff_gray = np.max(diff_abs, axis=2)
        
        # 接触检测阈值（必须有足够大的差分才认为是接触）
        # 阈值太小会导致噪声被误认为接触，无接触时深度/法向量不正确
        contact_threshold = 8  # 增大阈值，过滤掉噪声
        contact_mask = (diff_gray > contact_threshold).astype(np.uint8)
        
        # 根据是否有marker决定有效区域
        if self.has_marker:
            marker_mask = self._get_marker_mask(image)
            ref_marker_mask = self._get_marker_mask(ref_image)
            combined_mask = cv2.bitwise_or(marker_mask, ref_marker_mask)
            valid_mask = contact_mask & (combined_mask == 0)
        else:
            valid_mask = contact_mask
        
        # 形态学操作去除噪点
        kernel = np.ones((3, 3), np.uint8)
        valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_OPEN, kernel)
        valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)
        
        # 获取有效像素坐标
        y_coords, x_coords = np.where(valid_mask > 0)
        
        if len(x_coords) == 0:
            return np.zeros((h, w)), np.zeros((h, w))
        
        # 准备输入特征：颜色差分值（归一化到 [-1, 1]）
        diff_values = diff_image[y_coords, x_coords] / 255.0
        
        if self.norm_params is not None and 'x_max' in self.norm_params:
            x_norm = x_coords.astype(np.float32) / float(self.norm_params['x_max'])
            y_norm = y_coords.astype(np.float32) / float(self.norm_params['y_max'])
        else:
            x_norm = x_coords.astype(np.float32) / w
            y_norm = y_coords.astype(np.float32) / h
        
        # 组合特征 [dB, dG, dR, X, Y]
        features = np.column_stack([
            diff_values[:, 0],  # dB (差分)
            diff_values[:, 1],  # dG (差分)
            diff_values[:, 2],  # dR (差分)
            x_norm, y_norm
        ])
        
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            gradients = self.model(features_tensor)
            gradients = gradients.cpu().numpy()
        
        # 重建梯度图（只在接触区域有值）
        gx = np.zeros((h, w), dtype=np.float32)
        gy = np.zeros((h, w), dtype=np.float32)
        
        gx[y_coords, x_coords] = gradients[:, 0]
        gy[y_coords, x_coords] = gradients[:, 1]
        
        return gx, gy
    
    def _compute_normals(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """
        从梯度计算法向量
        
        无接触时 gx=0, gy=0，法向量应该是 (0, 0, 1) 即指向Z轴（纯蓝色）
        """
        gz = np.ones_like(gx)
        
        # 归一化
        magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        magnitude[magnitude == 0] = 1
        
        normals = np.stack([gx/magnitude, gy/magnitude, gz/magnitude], axis=-1)
        
        # 确保无梯度区域（gx=0, gy=0）的法向量是 (0, 0, 1)
        # 这已经是正确的，因为 gx=0, gy=0 时，归一化后就是 (0, 0, 1)
        
        return normals
    
    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        深度图着色
        使用动态归一化
        
        无接触时深度应该是均匀的（接近0），显示统一颜色
        """
        h, w = depth.shape
        
        # 检测是否有真正的接触
        depth_range = depth.max() - depth.min()
        min_depth_threshold = 0.05  # 增大阈值，更好地判断是否有接触
        
        if depth_range < min_depth_threshold:
            # 无接触，返回统一的深紫色（表示平坦表面）
            return np.full((h, w, 3), [128, 0, 68], dtype=np.uint8)
        
        # 动态归一化
        depth_normalized = (depth - depth.min()) / depth_range
        depth_normalized = np.clip(depth_normalized, 0, 1)
        
        # 双边滤波降噪
        depth_normalized = cv2.bilateralFilter(
            depth_normalized.astype(np.float32), 
            d=9, sigmaColor=75, sigmaSpace=75
        )
        
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
    
    def _colorize_normals(self, normals: np.ndarray) -> np.ndarray:
        """
        法向量着色
        
        法向量 [nx, ny, nz] 映射到颜色：
        - nx (X梯度) -> R (红色)
        - ny (Y梯度) -> G (绿色)  
        - nz (Z分量) -> B (蓝色)
        
        平坦表面（无接触）[0, 0, 1] -> 纯蓝色 (R=128, G=128, B=255)
        向右倾斜 [1, 0, 0] -> 红色偏移
        向下倾斜 [0, 1, 0] -> 绿色偏移
        """
        # normals: [H, W, 3] 其中 [:,:,0]=nx, [:,:,1]=ny, [:,:,2]=nz
        # 范围 [-1, 1] -> [0, 255]
        normals_normalized = (normals + 1) / 2
        
        # OpenCV 使用 BGR 格式，所以需要调整通道顺序
        # RGB [nx, ny, nz] -> BGR [nz, ny, nx]
        normals_bgr = np.stack([
            normals_normalized[:, :, 2],  # B <- nz
            normals_normalized[:, :, 1],  # G <- ny
            normals_normalized[:, :, 0],  # R <- nx
        ], axis=-1)
        
        normals_vis = (normals_bgr * 255).astype(np.uint8)
        return normals_vis
    
    def reset(self):
        """重置处理器"""
        self.ref_frame = None
        self.ref_blur = None
        self.ref_frames_buffer = []
        self.con_flag = True
        self.depth_min_smooth = None
        self.depth_max_smooth = None
        print("[INFO] MLP处理器已重置，将采集多帧平均作为参考帧")
    
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
            gx: X方向梯度
            gy: Y方向梯度
            diff_img: 差分图像
        """
        # 透视变换
        if apply_warp:
            frame = self.warp_perspective(frame)
        
        h, w = frame.shape[:2]
        
        # 采集多帧作为参考帧平均
        if self.con_flag:
            self.ref_frames_buffer.append(frame.astype(np.float32))
            
            if len(self.ref_frames_buffer) < self.ref_avg_count:
                # 还在采集中，显示进度
                progress = len(self.ref_frames_buffer)
                progress_img = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(progress_img, f"Collecting ref: {progress}/{self.ref_avg_count}", 
                           (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                return (progress_img, progress_img, np.zeros((h, w)),
                        np.zeros((h, w, 3)), None, None, None)
            else:
                # 采集完成，计算平均
                self.ref_frame = np.mean(self.ref_frames_buffer, axis=0).astype(np.uint8)
                self.ref_blur = cv2.GaussianBlur(self.ref_frame.astype(np.float32), (3, 3), 0)
                self.con_flag = False
                self.ref_frames_buffer = []  # 清空缓冲
                print(f"[INFO] 参考帧已设置（{self.ref_avg_count}帧平均）")
                
                return (np.zeros((h, w, 3), dtype=np.uint8),
                        np.zeros((h, w, 3), dtype=np.uint8),
                        np.zeros((h, w)),
                        np.zeros((h, w, 3)),
                        None, None, None)
        
        # 计算差分图像
        diff = cv2.absdiff(frame, self.ref_frame)
        diff_img = np.max(diff, axis=2)
        diff_img = cv2.applyColorMap((diff_img * 3).clip(0, 255).astype(np.uint8), cv2.COLORMAP_HOT)
        
        # 推理梯度
        gx, gy = self._infer_gradient(frame, self.ref_frame)
        
        if gx is None:
            return (np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w)),
                    np.zeros((h, w, 3)),
                    None, None, diff_img)
        
        # 泊松重建深度
        depth = fast_poisson(gx, gy)
        
        # 计算法向量
        normals = self._compute_normals(gx, gy)
        
        # 可视化
        depth_colored = self._colorize_depth(depth)
        normal_colored = self._colorize_normals(normals)
        
        return depth_colored, normal_colored, depth, normals, gx, gy, diff_img


def main():
    """主函数：测试MLP方法"""
    
    # 相机配置（与 1_quick_roi_calibrator.py 一致）
    camera_config = TactileCameraConfig(
        index_or_path="/dev/video2",  # 替换为你的TactileCamera设备路径
        fps=25,
        width=320,
        height=240,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
        # 曝光设置 (范围: 1-10000, 较小值=较暗)
        exposure=600,
        auto_exposure=False,
        # 白平衡设置 (色温范围: 2000-8000K)
        wb_temperature=4000,
        auto_wb=False,
        # RGB增益 (范围: 0.0 - 3.0)
        r_gain=1.0,
        g_gain=1.0,
        b_gain=1.0,
    )
    
    # 数据保存目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "data", "tactile_data_mlp_grad")
    os.makedirs(save_dir, exist_ok=True)
    
    # 是否有标记点（marker）
    HAS_MARKER = False  # 设置为False表示没有marker
    
    # 初始化组件
    camera = TactileCamera(camera_config)
    processor = MLPProcessor(has_marker=HAS_MARKER)
    visualizer = TactileVisualizer(
        windows=['original', 'depth', 'normal'],
        window_size=(640, 480)
    )
    
    # 初始化3D可视化
    plotter = pv.Plotter(window_size=(800, 600), title="3D Depth")
    plotter.set_background('white')
    plotter.add_axes()
    plotter.show(interactive_update=True, auto_close=False)
    
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
                depth_colored, normal_colored, raw_depth, raw_normals, grad_x, grad_y, diff_img = \
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
                
                # 更新3D可视化
                if raw_depth is not None and not processor.con_flag:
                    h, w = raw_depth.shape
                    # 降采样以提高性能
                    step = 4
                    depth_ds = raw_depth[::step, ::step]
                    hh, ww = depth_ds.shape
                    
                    # 归一化深度到 [0, 1]
                    depth_range = depth_ds.max() - depth_ds.min()
                    if depth_range > 0.01:
                        depth_norm = (depth_ds - depth_ds.min()) / depth_range
                    else:
                        depth_norm = np.zeros_like(depth_ds)
                    
                    # 创建坐标网格
                    x = np.arange(ww)
                    y = np.arange(hh)
                    x, y = np.meshgrid(x, y)
                    
                    # 创建StructuredGrid，归一化深度适度放大
                    grid = pv.StructuredGrid(x, y, depth_norm * 10.0)
                    grid["depth"] = depth_norm.flatten(order="F")
                    
                    # 更新plotter
                    plotter.clear()
                    plotter.add_mesh(grid, scalars="depth", cmap='viridis', show_scalar_bar=True)
                    plotter.update()
                
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
        
        plotter.close()
        visualizer.cleanup()
        camera.disconnect()
        print("[INFO] 相机已断开")


if __name__ == "__main__":
    main()
