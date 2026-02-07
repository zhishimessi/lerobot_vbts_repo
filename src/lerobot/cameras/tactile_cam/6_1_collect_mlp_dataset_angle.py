"""

使用标准球压印采集训练数据：
1. 拍摄参考图和球压印图
2. 手动选择接触圆区域
3. 根据球的几何形状计算真实梯度角度 (gxyangles)
4. 保存 [B, G, R, X, Y] -> [gx_angle, gy_angle] 数据对

与 gs_sdk/calibration 逻辑一致：
- 特征: BGRXY 归一化到 [0,1]
- 标签: 梯度角度 gxyangles = arctan2(dxy, dz)

使用方法:
1. 运行本脚本采集数据（或使用已有的 test_data 目录）
2. 运行 6_2_train_mlp_v2.py 训练模型
3. 运行 6_3_test_mlp_v2.py 测试效果
"""

import cv2
import numpy as np
import os
import sys
import glob

# 确保可以导入 lerobot 模块
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.tactile_cam.processors import BaseProcessor
from lerobot.cameras.configs import ColorMode, Cv2Rotation


def image2bgrxys(image: np.ndarray) -> np.ndarray:
    """
    将BGR图像转换为BGRXY特征 (与gs_sdk一致)
    
    Args:
        image: np.array (H, W, 3); BGR图像
        
    Returns:
        np.array (H, W, 5); BGRXY特征，全部归一化到[0,1]
    """
    h, w = image.shape[:2]
    ys = np.linspace(0, 1, h, endpoint=False, dtype=np.float32)
    xs = np.linspace(0, 1, w, endpoint=False, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    bgrxys = np.concatenate(
        [image.astype(np.float32) / 255.0, xx[..., np.newaxis], yy[..., np.newaxis]],
        axis=2,
    )
    return bgrxys


class MLPDataCollectorV2(BaseProcessor):
    """MLP训练数据收集器 (v2 - 与gs_sdk一致)"""
    
    def __init__(self, ball_radius_mm: float = 4.0, ppmm: float = 7.6,
                 pad: int = 20, calib_file: str = None, has_marker: bool = False):
        """
        初始化数据收集器
        
        Args:
            ball_radius_mm: 标定球半径 (mm)
            ppmm: 像素每毫米 (pixel/mm)，与 gs_sdk 一致
            pad: 边缘裁剪像素数
            calib_file: 透视变换矩阵文件路径
            has_marker: 是否有标记点
        """
        super().__init__(pad=pad, calib_file=calib_file)
        
        self.ball_radius_mm = ball_radius_mm
        self.ppmm = ppmm  # pixel per mm
        self.ball_radius_pixel = ball_radius_mm * ppmm
        self.has_marker = has_marker
        
        self.ref_image = None
        self.data_list = []
        
        print(f"[INFO] 球半径: {ball_radius_mm} mm")
        print(f"[INFO] ppmm: {ppmm} pixel/mm")
        print(f"[INFO] 球半径(像素): {self.ball_radius_pixel:.1f} pixels")
    
    def set_reference(self, ref_image: np.ndarray):
        """设置参考图像"""
        self.ref_image = ref_image.copy()
        print(f"[INFO] 参考图像已设置，尺寸: {ref_image.shape}")
    
    def get_marker_mask(self, image: np.ndarray, threshold: int = 60) -> np.ndarray:
        """获取标记点掩膜"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray < threshold).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask
    
    def contact_detection(self, image: np.ndarray, ref_image: np.ndarray, 
                          marker_mask: np.ndarray = None) -> tuple:
        """
        检测接触区域并交互式调整
        
        Returns:
            (contact_mask, center, radius)
        """
        blur = cv2.GaussianBlur(ref_image.astype(np.float32), (3, 3), 0)
        diff_img = np.max(np.abs(image.astype(np.float32) - blur), axis=2)
        
        if marker_mask is None:
            marker_mask = np.zeros_like(diff_img, dtype=np.uint8)
        
        contact_mask = (diff_img > 30).astype(np.uint8) * (1 - marker_mask)
        
        contours, _ = cv2.findContours(contact_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, None, None
        
        areas = [cv2.contourArea(c) for c in contours]
        cnt = contours[np.argmax(areas)]
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        
        # 交互式调整
        print("\n=== 调整接触区域 ===")
        print("  W/S: 上/下移动")
        print("  A/D: 左/右移动")
        print("  M/N: 增大/减小半径")
        print("  ESC: 跳过此图")
        print("  Enter: 确认")
        
        key = -1
        while key != 27 and key != 13 and key != 10:
            center = (int(x), int(y))
            radius_int = int(radius)
            im2show = cv2.circle(image.copy(), center, radius_int, (0, 255, 0), 2)
            cv2.imshow('contact', im2show)
            key = cv2.waitKey(100)
            
            if key == ord('w'):
                y -= 1
            elif key == ord('s'):
                y += 1
            elif key == ord('a'):
                x -= 1
            elif key == ord('d'):
                x += 1
            elif key == ord('m'):
                radius += 1
            elif key == ord('n'):
                radius = max(radius - 1, 5)
        
        if key == 27:
            return None, None, None
        
        center = (int(x), int(y))
        radius_int = int(radius)
        
        contact_mask = np.zeros_like(diff_img, dtype=np.uint8)
        cv2.circle(contact_mask, center, radius_int, 1, -1)
        if marker_mask is not None:
            contact_mask = contact_mask * (1 - marker_mask)
        
        return contact_mask, center, radius_int
    
    def compute_gxyangles_from_sphere(self, center: tuple, radius: int, 
                                       image_shape: tuple, radius_reduction: float = 4.0) -> tuple:
        """
        根据球面几何计算梯度角度 (与gs_sdk一致)
        
        梯度角度定义:
        gx_angle = arctan2(dx, dz)
        gy_angle = arctan2(dy, dz)
        
        其中 dx, dy 是相对于圆心的偏移，dz 是球面高度
        
        Args:
            center: 接触圆心 (x, y)
            radius: 接触圆半径 (pixel)
            image_shape: 图像形状 (H, W)
            radius_reduction: 半径缩减量，确保标注像素都在接触区域内
            
        Returns:
            (gxyangles, mask): 梯度角度图和有效掩膜
        """
        h, w = image_shape[:2]
        
        # 创建坐标网格
        xs = np.arange(w)
        ys = np.arange(h)
        xv, yv = np.meshgrid(xs, ys, indexing="xy")
        
        # 相对于圆心的坐标
        dxys = np.stack([xv - center[0], yv - center[1]], axis=-1)
        
        # 距离图
        dists = np.linalg.norm(dxys, axis=-1)
        
        # 有效掩膜（在缩减后的半径内）
        effective_radius = radius - radius_reduction
        mask = dists < effective_radius
        
        # 球半径（像素）
        ball_radius_pixel = self.ball_radius_pixel
        
        # 检查是否压得太深
        if ball_radius_pixel < effective_radius:
            print(f"[WARNING] 压得太深！球半径{ball_radius_pixel:.1f}px < 接触半径{effective_radius:.1f}px")
            return None, None
        
        # 计算球面高度 dz = sqrt(R^2 - dist^2)
        dists_masked = dists.copy()
        dists_masked[~mask] = 0
        dzs = np.sqrt(ball_radius_pixel**2 - np.square(dists_masked))
        
        # 计算梯度角度
        gxangles = np.arctan2(dxys[:, :, 0], dzs)
        gyangles = np.arctan2(dxys[:, :, 1], dzs)
        
        # 组合
        gxyangles = np.stack([gxangles, gyangles], axis=-1)
        gxyangles[~mask] = np.array([0.0, 0.0])
        
        return gxyangles, mask
    
    def collect_sample(self, image: np.ndarray, center: tuple, radius: int,
                       radius_reduction: float = 4.0) -> dict:
        """
        收集单个样本的训练数据 (与gs_sdk一致)
        
        Args:
            image: 当前帧（BGR）
            center: 接触圆心
            radius: 接触圆半径
            radius_reduction: 半径缩减量
            
        Returns:
            dict: {'bgrxys': (N, 5), 'gxyangles': (N, 2)}
        """
        # 计算梯度角度
        gxyangles, mask = self.compute_gxyangles_from_sphere(
            center, radius, image.shape, radius_reduction
        )
        
        if gxyangles is None:
            return None
        
        # 计算BGRXY特征
        bgrxys = image2bgrxys(image)
        
        # 提取有效像素
        valid_bgrxys = bgrxys[mask]
        valid_gxyangles = gxyangles[mask]
        
        print(f"[INFO] 收集了 {len(valid_bgrxys)} 个有效像素")
        
        return {
            'bgrxys': valid_bgrxys,
            'gxyangles': valid_gxyangles
        }


def process_existing_data(data_dir: str, ppmm: float = 7.6, ball_radius_mm: float = 4.0,
                          has_marker: bool = False, radius_reduction: float = 4.0,
                          use_labels: bool = True):
    """
    处理已有的采集数据目录
    
    Args:
        data_dir: 数据目录（包含 ref.jpg 和 sample*.jpg）
        ppmm: 像素每毫米
        ball_radius_mm: 球半径
        has_marker: 是否有marker
        radius_reduction: 半径缩减量
        use_labels: 是否使用已有的标注文件 (.txt)
    """
    pad = 20
    
    # 初始化收集器
    collector = MLPDataCollectorV2(
        ball_radius_mm=ball_radius_mm,
        ppmm=ppmm,
        pad=pad,
        has_marker=has_marker
    )
    
    # 加载参考图
    ref_path = os.path.join(data_dir, "ref.jpg")
    if not os.path.exists(ref_path):
        print(f"[ERROR] 参考图不存在: {ref_path}")
        return None
    
    ref_img = cv2.imread(ref_path)
    ref_img = collector._crop_image(ref_img)
    collector.set_reference(ref_img)
    
    # 获取marker掩膜
    if has_marker:
        marker_mask = collector.get_marker_mask(ref_img)
        # 使用inpaint修复参考图
        ref_img = cv2.inpaint(ref_img, marker_mask, 3, cv2.INPAINT_TELEA)
    else:
        marker_mask = np.zeros_like(ref_img[:, :, 0], dtype=np.uint8)
    
    # 处理所有样本图像
    img_list = sorted(glob.glob(os.path.join(data_dir, "sample*.jpg")),
                      key=lambda x: int(os.path.basename(x).replace('sample_', '').replace('.jpg', '')))
    print(f"[INFO] 找到 {len(img_list)} 张样本图像")
    
    all_bgrxys = []
    all_gxyangles = []
    
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        print(f"\n处理: {img_name}")
        img = cv2.imread(img_path)
        img = collector._crop_image(img)
        
        if has_marker:
            curr_marker_mask = collector.get_marker_mask(img)
            combined_mask = cv2.bitwise_or(marker_mask, curr_marker_mask)
        else:
            combined_mask = marker_mask
        
        # 尝试读取已有标注
        label_path = img_path.replace('.jpg', '.txt')
        if use_labels and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                parts = f.read().strip().split()
                if len(parts) >= 3:
                    # 标签坐标是原始图像的，需要减去裁剪偏移量
                    orig_cx, orig_cy = int(parts[0]), int(parts[1])
                    center = (orig_cx - pad, orig_cy - pad)
                    radius = int(parts[2])
                    print(f"  使用已有标注: 原始center=({orig_cx},{orig_cy}), 裁剪后center={center}, radius={radius}")
                else:
                    print(f"  [WARNING] 标注格式错误，跳过")
                    continue
        else:
            # 交互式检测
            contact_mask, center, radius = collector.contact_detection(
                img, ref_img, combined_mask
            )
            if contact_mask is None:
                print("[INFO] 跳过此图")
                continue
        
        # 收集数据
        data = collector.collect_sample(img, center, radius, radius_reduction)
        
        if data is not None:
            all_bgrxys.append(data['bgrxys'])
            all_gxyangles.append(data['gxyangles'])
    
    cv2.destroyAllWindows()
    
    if len(all_bgrxys) == 0:
        print("[ERROR] 没有收集到有效数据")
        return None
    
    # 合并数据
    all_bgrxys = np.concatenate(all_bgrxys, axis=0)
    all_gxyangles = np.concatenate(all_gxyangles, axis=0)
    
    print(f"\n[INFO] 总共收集 {len(all_bgrxys)} 个数据点")
    print(f"[INFO] BGRXY范围: [{all_bgrxys.min():.3f}, {all_bgrxys.max():.3f}]")
    print(f"[INFO] gxyangles范围: [{all_gxyangles.min():.4f}, {all_gxyangles.max():.4f}] rad")
    
    # 添加背景数据（与gs_sdk一致）
    print("\n[INFO] 添加背景数据（无接触，梯度角度=0）...")
    bg_bgrxys = image2bgrxys(ref_img).reshape(-1, 5)
    bg_gxyangles = np.zeros((len(bg_bgrxys), 2), dtype=np.float32)
    
    # 随机采样背景数据，数量与接触数据相当
    n_bg = len(all_bgrxys) // 5  # 使用1/5的背景数据
    perm = np.random.permutation(len(bg_bgrxys))[:n_bg]
    bg_bgrxys = bg_bgrxys[perm]
    bg_gxyangles = bg_gxyangles[perm]
    
    print(f"[INFO] 添加 {len(bg_bgrxys)} 个背景数据点")
    
    # 合并
    all_bgrxys = np.concatenate([all_bgrxys, bg_bgrxys], axis=0)
    all_gxyangles = np.concatenate([all_gxyangles, bg_gxyangles], axis=0)
    
    # 打乱
    perm = np.random.permutation(len(all_bgrxys))
    all_bgrxys = all_bgrxys[perm]
    all_gxyangles = all_gxyangles[perm]
    
    return {
        'bgrxys': all_bgrxys,
        'gxyangles': all_gxyangles,
        'ref_image': ref_img
    }


def main():
    """主函数"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 数据目录 - 使用已有的 mlp_calibration 目录
    data_dir = os.path.join(current_dir, "data", "mlp_calibration")
    save_dir = os.path.join(current_dir, "data", "mlp_calibration_v2")
    os.makedirs(save_dir, exist_ok=True)
    
    # 参数设置
    BALL_RADIUS_MM = 4.0  # 标定球半径 (mm)
    HAS_MARKER = False
    RADIUS_REDUCTION = 4.0  # 半径缩减量（像素）
    
    # 尝试加载 ppmm (从 mm_per_pixel 转换)
    mm_per_pixel_file = os.path.join(current_dir, "calibration_data", "mm_per_pixel.npz")
    if os.path.exists(mm_per_pixel_file):
        data = np.load(mm_per_pixel_file)
        mm_per_pixel = float(data['mm_per_pixel'])
        PPMM = 1.0 / mm_per_pixel  # 转换为 pixel per mm
        print(f"[INFO] 加载 mm_per_pixel: {mm_per_pixel:.4f}, ppmm: {PPMM:.2f}")
    else:
        PPMM = 7.6  # 默认值 (GelSight Mini)
        print(f"[WARNING] 使用默认 ppmm: {PPMM:.2f}")
    
    print("\n" + "=" * 60)
    print("MLP训练数据收集 (v2 - 与gs_sdk一致)")
    print("=" * 60)
    
    # 检查数据目录
    if os.path.exists(data_dir):
        print(f"[INFO] 使用已有数据目录: {data_dir}")
        result = process_existing_data(
            data_dir, 
            ppmm=PPMM, 
            ball_radius_mm=BALL_RADIUS_MM,
            has_marker=HAS_MARKER,
            radius_reduction=RADIUS_REDUCTION,
            use_labels=True  # 使用已有的标注文件
        )
    else:
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        print("[INFO] 请先采集数据或指定正确的数据目录")
        return
    
    if result is None:
        return
    
    # 保存数据
    dataset_path = os.path.join(save_dir, "mlp_dataset_v2.npz")
    np.savez(
        dataset_path,
        bgrxys=result['bgrxys'],
        gxyangles=result['gxyangles']
    )
    print(f"\n[INFO] 数据集已保存: {dataset_path}")
    print(f"[INFO] 数据形状: bgrxys={result['bgrxys'].shape}, gxyangles={result['gxyangles'].shape}")
    
    # 保存参考图
    ref_path = os.path.join(save_dir, "background.png")
    cv2.imwrite(ref_path, result['ref_image'])
    print(f"[INFO] 参考图已保存: {ref_path}")
    
    print("\n" + "=" * 60)
    print("完成！现在可以运行 6_2_train_mlp_v2.py 训练模型")
    print("=" * 60)


if __name__ == "__main__":
    main()
