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
from scipy import signal

from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.tactile_cam.processors import BaseProcessor
from lerobot.cameras.configs import ColorMode, Cv2Rotation


class MLPDataCollector(BaseProcessor):
    """MLP训练数据收集器"""
    
    def __init__(self, ball_radius_mm: float = 4.0, mm_per_pixel: float = 0.0595, 
                 pad: int = 20, calib_file: str = None):
        """
        初始化数据收集器
        
        Args:
            ball_radius_mm: 标定球半径 (mm)
            mm_per_pixel: 像素尺寸 (mm/pixel)
            pad: 边缘裁剪像素数
            calib_file: 透视变换矩阵文件路径
        """
        super().__init__(pad=pad, calib_file=calib_file)
        
        self.ball_radius_mm = ball_radius_mm
        self.mm_per_pixel = mm_per_pixel
        self.ball_radius_pixel = ball_radius_mm / mm_per_pixel
        
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
    
    def collect_sample(self, image: np.ndarray, center: tuple, radius: int) -> np.ndarray:
        """
        收集单个样本的训练数据
        
        Args:
            image: 当前帧（BGR）
            center: 接触圆心
            radius: 接触圆半径
            
        Returns:
            数据数组 [N, 7]: [B, G, R, X, Y, gx, gy]
        """
        if self.ref_image is None:
            raise ValueError("请先设置参考图像")
        
        h, w = image.shape[:2]
        
        # 获取标记点掩膜
        marker_mask = self.get_marker_mask(image)
        ref_marker_mask = self.get_marker_mask(self.ref_image)
        combined_marker_mask = cv2.bitwise_or(marker_mask, ref_marker_mask)
        
        # 创建接触区域掩膜
        contact_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(contact_mask, center, radius, 255, -1)
        
        # 有效区域 = 接触区域 - 标记点
        valid_mask = (contact_mask > 0) & (combined_marker_mask == 0)
        valid_mask = valid_mask.astype(np.float32)
        
        # 计算真实梯度
        gx, gy = self.compute_gradient_from_sphere(center, radius, image.shape, valid_mask)
        
        # 提取有效像素的数据
        y_coords, x_coords = np.where(valid_mask > 0)
        
        if len(x_coords) == 0:
            return np.array([])
        
        # BGR值
        bgr_values = image[y_coords, x_coords]
        
        # 梯度值
        gx_values = gx[y_coords, x_coords]
        gy_values = gy[y_coords, x_coords]
        
        # 过滤掉NaN值
        valid_idx = ~(np.isnan(gx_values) | np.isnan(gy_values))
        
        # 组合数据 [B, G, R, X, Y, gx, gy]
        data = np.column_stack([
            bgr_values[valid_idx, 0],  # B
            bgr_values[valid_idx, 1],  # G
            bgr_values[valid_idx, 2],  # R
            x_coords[valid_idx],       # X
            y_coords[valid_idx],       # Y
            gx_values[valid_idx],      # gx
            gy_values[valid_idx]       # gy
        ])
        
        return data


def main():
    """主函数：收集MLP训练数据"""
    
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
    save_dir = os.path.join(current_dir, "data", "mlp_calibration")
    os.makedirs(save_dir, exist_ok=True)
    
    # 参数设置
    BALL_RADIUS_MM = 4.0  # 标定球半径 (mm)
    
    # 尝试加载 mm_per_pixel
    mm_per_pixel_file = os.path.join(current_dir, "data", "calibration_data", "mm_per_pixel.npz")
    if os.path.exists(mm_per_pixel_file):
        data = np.load(mm_per_pixel_file)
        MM_PER_PIXEL = float(data['mm_per_pixel'])
        print(f"[INFO] 加载 mm_per_pixel: {MM_PER_PIXEL:.4f}")
    else:
        MM_PER_PIXEL = 0.0595  # 默认值
        print(f"[WARNING] 使用默认 mm_per_pixel: {MM_PER_PIXEL:.4f}")
    
    # 初始化
    camera = TactileCamera(camera_config)
    collector = MLPDataCollector(
        ball_radius_mm=BALL_RADIUS_MM,
        mm_per_pixel=MM_PER_PIXEL,
        pad=20
    )
    
    print("\n" + "="*60)
    print("MLP训练数据收集")
    print("="*60)
    print(f"球半径: {BALL_RADIUS_MM} mm")
    print(f"像素尺寸: {MM_PER_PIXEL:.4f} mm/pixel")
    print(f"球半径(像素): {BALL_RADIUS_MM/MM_PER_PIXEL:.1f} pixels")
    print("="*60)
    print("\n操作说明:")
    print("  1. 首先拍摄参考图（无接触）- 按 'r'")
    print("  2. 用标定球压印传感器 - 按 空格 拍照")
    print("  3. 调整接触区域后按 Enter 确认")
    print("  4. 重复步骤2-3收集更多数据")
    print("  5. 按 'q' 保存并退出")
    print("="*60 + "\n")
    
    all_data = []
    sample_count = 0
    
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
