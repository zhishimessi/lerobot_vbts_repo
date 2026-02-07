"""
MLP训练数据收集脚本

使用标准球压印采集训练数据：
1. 拍摄参考图和球压印图
2. 手动选择接触圆区域
3. 根据球的几何形状计算真实梯度
4. 保存 [B, G, R, X, Y, gx, gy] 数据对

使用方法:
1. 先运行 1_quick_roi_calibrator.py 标定透视变换
2. 运行本脚本采集数据
3. 按空格拍照，按ESC完成
"""

import cv2
import numpy as np
import os
import sys
from scipy import signal

# 确保可以导入 lerobot 模块
# __file__ -> tactile_cam/6_1_xxx.py
# 向上3层: tactile_cam -> cameras -> lerobot -> src (包含 lerobot 包)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.tactile_cam.processors import BaseProcessor
from lerobot.cameras.configs import ColorMode, Cv2Rotation


class MLPDataCollector(BaseProcessor):
    """MLP训练数据收集器"""
    
    def __init__(self, ball_radius_mm: float = 4.0, mm_per_pixel: float = 0.1316, 
                 pad: int = 20, calib_file: str = None, has_marker: bool = False):
        """
        初始化数据收集器
        
        Args:
            ball_radius_mm: 标定球半径 (mm)
            mm_per_pixel: 像素尺寸 (mm/pixel)，默认值来自 calibration_data/mm_per_pixel.npz
            pad: 边缘裁剪像素数
            calib_file: 透视变换矩阵文件路径
            has_marker: 是否有标记点，False则不进行marker检测
        """
        super().__init__(pad=pad, calib_file=calib_file)
        
        self.ball_radius_mm = ball_radius_mm
        self.mm_per_pixel = mm_per_pixel
        self.ball_radius_pixel = ball_radius_mm / mm_per_pixel
        self.has_marker = has_marker
        
        self.ref_image = None
        self.data_list = []
    
    def set_reference(self, ref_image: np.ndarray):
        """设置参考图像"""
        self.ref_image = ref_image.copy()
        print(f"[INFO] 参考图像已设置，尺寸: {ref_image.shape}")
    
    def get_marker_mask(self, image: np.ndarray, threshold: int = 60) -> np.ndarray:
        """
        获取标记点掩膜
        
        Args:
            image: BGR图像
            threshold: 亮度阈值
            
        Returns:
            标记点掩膜 (0或255)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray < threshold).astype(np.uint8) * 255
        # 膨胀一下扩大掩膜范围
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask
    
    def select_contact_region(self, image: np.ndarray, ref_image: np.ndarray) -> tuple:
        """
        交互式选择接触区域
        
        Args:
            image: 当前帧
            ref_image: 参考帧
            
        Returns:
            (center, radius) 或 None（如果取消）
        """
        # 自动检测接触区域
        diff = cv2.absdiff(image, ref_image)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            print("[WARNING] 未检测到接触区域")
            return None
        
        # 找最大轮廓
        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)
        
        # 限制半径不超过球半径
        radius = min(radius, self.ball_radius_pixel - 1)
        
        # 交互式调整
        print("\n=== 调整接触区域 ===")
        print("  W/S: 上/下移动")
        print("  A/D: 左/右移动")
        print("  M/N: 增大/减小半径")
        print("  ESC: 跳过此图")
        print("  Enter: 确认")
        print("========================\n")
        
        while True:
            display = image.copy()
            cv2.circle(display, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(display, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.imshow('Select Contact Region', display)
            
            key = cv2.waitKey(100)
            if key == 27:  # ESC
                return None
            elif key == 13 or key == 10:  # Enter
                break
            elif key == ord('w'):
                y -= 1
            elif key == ord('s'):
                y += 1
            elif key == ord('a'):
                x -= 1
            elif key == ord('d'):
                x += 1
            elif key == ord('m'):
                radius = min(radius + 1, self.ball_radius_pixel - 1)
            elif key == ord('n'):
                radius = max(radius - 1, 5)
        
        cv2.destroyWindow('Select Contact Region')
        return (int(x), int(y)), int(radius)
    
    def compute_gradient_from_sphere(self, center: tuple, radius: int, 
                                     image_shape: tuple, valid_mask: np.ndarray) -> tuple:
        """
        根据球面几何计算真实梯度
        
        Args:
            center: 接触圆心 (x, y)
            radius: 接触圆半径 (pixel)
            image_shape: 图像形状 (H, W)
            valid_mask: 有效区域掩膜
            
        Returns:
            (gx, gy) 梯度图
        """
        h, w = image_shape[:2]
        x = np.linspace(0, w - 1, w)
        y = np.linspace(0, h - 1, h)
        xv, yv = np.meshgrid(x, y)
        
        # 相对于圆心的坐标
        xv = xv - center[0]
        yv = yv - center[1]
        
        # 球半径（像素）
        R = self.ball_radius_pixel
        
        # 距离图
        rv = np.sqrt(xv**2 + yv**2)
        
        # 创建圆形掩膜
        circle_mask = (rv < radius).astype(np.float32)
        inner_mask = (rv < radius - 1).astype(np.float32)
        
        # 计算高度图 z = sqrt(R^2 - x^2 - y^2)
        temp = (xv * circle_mask)**2 + (yv * circle_mask)**2
        temp_mm = temp * (self.mm_per_pixel ** 2)
        
        height_map = np.sqrt(self.ball_radius_mm**2 - temp_mm) * circle_mask
        height_map -= np.sqrt(self.ball_radius_mm**2 - (radius * self.mm_per_pixel)**2)
        height_map *= circle_mask
        height_map[np.isnan(height_map)] = 0
        
        # 计算梯度 (使用卷积)
        kernel_x = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]])
        kernel_y = kernel_x.T
        
        gx = signal.convolve2d(height_map, kernel_x, boundary='symm', mode='same') * inner_mask / self.mm_per_pixel
        gy = signal.convolve2d(height_map, kernel_y, boundary='symm', mode='same') * inner_mask / self.mm_per_pixel
        
        return gx, gy
    
    def collect_sample(self, image: np.ndarray, center: tuple, radius: int, 
                       add_no_contact_samples: bool = True) -> np.ndarray:
        """
        收集单个样本的训练数据
        
        使用颜色差分值作为输入，而不是原始颜色
        同时收集无接触区域的数据（差分≈0，梯度=0），让模型学会区分
        
        Args:
            image: 当前帧（BGR）
            center: 接触圆心
            radius: 接触圆半径
            add_no_contact_samples: 是否添加无接触区域样本
            
        Returns:
            数据数组 [N, 7]: [dB, dG, dR, X, Y, gx, gy]  # 差分颜色
        """
        if self.ref_image is None:
            raise ValueError("请先设置参考图像")
        
        h, w = image.shape[:2]
        
        # 计算颜色差分图像（当前帧 - 参考帧）
        # 使用有符号差分，范围 [-255, 255]
        diff_image = image.astype(np.float32) - self.ref_image.astype(np.float32)
        
        # 创建接触区域掩膜
        contact_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(contact_mask, center, radius, 255, -1)
        
        # 根据是否有marker决定有效区域
        if self.has_marker:
            # 获取标记点掩膜
            marker_mask = self.get_marker_mask(image)
            ref_marker_mask = self.get_marker_mask(self.ref_image)
            combined_marker_mask = cv2.bitwise_or(marker_mask, ref_marker_mask)
            # 有效区域 = 接触区域 - 标记点
            valid_mask = (contact_mask > 0) & (combined_marker_mask == 0)
            # 无接触有效区域 = 非接触区域 - 标记点 - 边界
            no_contact_valid = (contact_mask == 0) & (combined_marker_mask == 0)
        else:
            # 没有marker时，接触区域即为有效区域
            valid_mask = contact_mask > 0
            no_contact_valid = contact_mask == 0
        
        valid_mask = valid_mask.astype(np.float32)
        
        # 计算真实梯度
        gx, gy = self.compute_gradient_from_sphere(center, radius, image.shape, valid_mask)
        
        # 提取有效像素的数据（接触区域）
        y_coords, x_coords = np.where(valid_mask > 0)
        
        if len(x_coords) == 0:
            return np.array([])
        
        # 颜色差分值（归一化到 [-1, 1]）
        diff_values = diff_image[y_coords, x_coords] / 255.0
        
        # 梯度值
        gx_values = gx[y_coords, x_coords]
        gy_values = gy[y_coords, x_coords]
        
        # 过滤掉NaN值
        valid_idx = ~(np.isnan(gx_values) | np.isnan(gy_values))
        
        # 组合数据 [dB, dG, dR, X, Y, gx, gy]
        contact_data = np.column_stack([
            diff_values[valid_idx, 0],  # dB (差分)
            diff_values[valid_idx, 1],  # dG (差分)
            diff_values[valid_idx, 2],  # dR (差分)
            x_coords[valid_idx],        # X
            y_coords[valid_idx],        # Y
            gx_values[valid_idx],       # gx
            gy_values[valid_idx]        # gy
        ])
        
        # 添加无接触区域的样本（差分小，梯度=0）
        if add_no_contact_samples:
            # 排除边界区域（距离接触圆太近的区域）
            buffer_radius = radius + 10
            buffer_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(buffer_mask, center, buffer_radius, 255, -1)
            no_contact_valid = no_contact_valid & (buffer_mask == 0)
            
            # 排除图像边缘
            edge_margin = 10
            no_contact_valid[:edge_margin, :] = False
            no_contact_valid[-edge_margin:, :] = False
            no_contact_valid[:, :edge_margin] = False
            no_contact_valid[:, -edge_margin:] = False
            
            # 提取无接触像素
            nc_y, nc_x = np.where(no_contact_valid)
            
            if len(nc_x) > 0:
                # 随机采样，数量与接触区域相当
                n_contact = len(contact_data)
                n_sample = min(len(nc_x), n_contact)
                sample_idx = np.random.choice(len(nc_x), n_sample, replace=False)
                
                nc_diff = diff_image[nc_y[sample_idx], nc_x[sample_idx]] / 255.0
                
                # 无接触区域梯度为0
                no_contact_data = np.column_stack([
                    nc_diff[:, 0],  # dB
                    nc_diff[:, 1],  # dG
                    nc_diff[:, 2],  # dR
                    nc_x[sample_idx],  # X
                    nc_y[sample_idx],  # Y
                    np.zeros(n_sample),  # gx = 0
                    np.zeros(n_sample)   # gy = 0
                ])
                
                # 合并接触和无接触数据
                contact_data = np.vstack([contact_data, no_contact_data])
                print(f"[INFO] 添加了 {n_sample} 个无接触样本")
        
        return contact_data


def main():
    """主函数：收集MLP训练数据"""
    
    # 相机配置（与 1_quick_roi_calibrator.py 一致）
    camera_config = TactileCameraConfig(
        index_or_path="/dev/video0",  # 替换为你的TactileCamera设备路径
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
    save_dir = os.path.join(current_dir, "data", "mlp_calibration")
    os.makedirs(save_dir, exist_ok=True)
    
    # 参数设置
    BALL_RADIUS_MM = 4.0  # 标定球半径 (mm)
    
    # 尝试加载 mm_per_pixel
    mm_per_pixel_file = os.path.join(current_dir, "calibration_data", "mm_per_pixel.npz")
    if os.path.exists(mm_per_pixel_file):
        data = np.load(mm_per_pixel_file)
        MM_PER_PIXEL = float(data['mm_per_pixel'])
        print(f"[INFO] 加载 mm_per_pixel: {MM_PER_PIXEL:.4f}")
    else:
        MM_PER_PIXEL = 0.1316  # 默认值
        print(f"[WARNING] 使用默认 mm_per_pixel: {MM_PER_PIXEL:.4f}")
    
    # 初始化
    camera = TactileCamera(camera_config)
    
    # 是否有标记点（marker）
    HAS_MARKER = False  # 设置为False表示没有marker
    
    collector = MLPDataCollector(
        ball_radius_mm=BALL_RADIUS_MM,
        mm_per_pixel=MM_PER_PIXEL,
        pad=20,
        has_marker=HAS_MARKER
    )
    
    print("\n" + "="*60)
    print("MLP训练数据收集")
    print("="*60)
    print(f"球半径: {BALL_RADIUS_MM} mm")
    print(f"像素尺寸: {MM_PER_PIXEL:.4f} mm/pixel")
    print(f"球半径(像素): {BALL_RADIUS_MM/MM_PER_PIXEL:.1f} pixels")
    print(f"是否有marker: {HAS_MARKER}")
    print("="*60)
    
    # 检查是否有已存在的数据集
    dataset_path = os.path.join(save_dir, "mlp_dataset.npy")
    ref_path = os.path.join(save_dir, "ref.jpg")
    all_data = []
    sample_count = 0
    
    if os.path.exists(dataset_path):
        print(f"\n[INFO] 发现已存在的数据集: {dataset_path}")
        existing_data = np.load(dataset_path)
        print(f"[INFO] 已有数据点: {len(existing_data)}")
        
        # 统计已有样本数量
        sample_files = [f for f in os.listdir(save_dir) if f.startswith("sample_") and f.endswith(".jpg")]
        existing_samples = len(sample_files)
        print(f"[INFO] 已有样本: {existing_samples}")
        
        # 询问用户选择
        print("\n选择操作模式:")
        print("  [c] 继续采集 - 在已有数据基础上继续")
        print("  [n] 重新采集 - 清除所有数据重新开始")
        print("  [q] 退出")
        
        while True:
            choice = input("\n请输入选择 [c/n/q]: ").strip().lower()
            if choice == 'c':
                # 继续采集
                all_data = existing_data.tolist()
                sample_count = existing_samples
                print(f"[INFO] 将继续采集，从样本 {sample_count + 1} 开始")
                
                # 加载已有的参考图
                if os.path.exists(ref_path):
                    ref_image = cv2.imread(ref_path)
                    collector.set_reference(ref_image)
                    print("[INFO] 已加载之前的参考图")
                break
            elif choice == 'n':
                # 重新采集
                print("[INFO] 清除已有数据...")
                # 删除所有数据文件
                for f in os.listdir(save_dir):
                    if f.startswith("sample_") or f in ["mlp_dataset.npy", "ref.jpg"]:
                        os.remove(os.path.join(save_dir, f))
                print("[INFO] 已清除，将重新开始采集")
                break
            elif choice == 'q':
                print("[INFO] 用户取消")
                return
            else:
                print("[WARNING] 无效选择，请输入 c/n/q")
    
    print("\n操作说明:")
    print("  1. 首先拍摄参考图（无接触）- 按 'r'")
    print("  2. 用标定球压印传感器 - 按 空格 拍照")
    print("  3. 调整接触区域后按 Enter 确认")
    print("  4. 重复步骤2-3收集更多数据")
    print("  5. 按 'q' 保存并退出")
    print("="*60 + "\n")
    
    try:
        camera.connect()
        print("[INFO] 相机已连接")
        
        while True:
            # 读取当前帧
            frame = camera.async_read(timeout_ms=200)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 透视变换
            warped = collector.warp_perspective(frame_bgr)
            
            # 显示
            display = warped.copy()
            status = f"Samples: {sample_count} | Total points: {len(all_data)}"
            if collector.ref_image is None:
                status += " | Press 'r' to set reference"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('MLP Data Collection', display)
            
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                break
            
            elif key == ord('r'):
                # 设置参考图
                collector.set_reference(warped)
                cv2.imwrite(os.path.join(save_dir, "ref.jpg"), warped)
                print("[INFO] 参考图已保存")
            
            elif key == 32:  # 空格
                if collector.ref_image is None:
                    print("[WARNING] 请先按'r'设置参考图")
                    continue
                
                # 选择接触区域
                result = collector.select_contact_region(warped, collector.ref_image)
                if result is None:
                    print("[INFO] 跳过此样本")
                    continue
                
                center, radius = result
                
                # 收集数据
                data = collector.collect_sample(warped, center, radius)
                
                if len(data) > 0:
                    all_data.extend(data)
                    sample_count += 1
                    
                    # 保存图像
                    cv2.imwrite(os.path.join(save_dir, f"sample_{sample_count}.jpg"), warped)
                    
                    # 保存位置信息
                    with open(os.path.join(save_dir, f"sample_{sample_count}.txt"), 'w') as f:
                        f.write(f"{center[0]} {center[1]} {radius}\n")
                    
                    print(f"[INFO] 样本 {sample_count}: 收集了 {len(data)} 个数据点")
                else:
                    print("[WARNING] 未收集到有效数据")
    
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    
    finally:
        cv2.destroyAllWindows()
        camera.disconnect()
        
        # 保存数据集
        if len(all_data) > 0:
            dataset = np.array(all_data)
            dataset_path = os.path.join(save_dir, "mlp_dataset.npy")
            np.save(dataset_path, dataset)
            print(f"\n[INFO] 数据集已保存: {dataset_path}")
            print(f"[INFO] 总数据点数: {len(dataset)}")
            print(f"[INFO] 数据形状: {dataset.shape}")
        else:
            print("\n[WARNING] 未收集到任何数据")
        
        print("[INFO] 相机已断开")


if __name__ == "__main__":
    main()
