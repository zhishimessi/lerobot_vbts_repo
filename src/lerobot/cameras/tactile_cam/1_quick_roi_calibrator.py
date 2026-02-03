"""
快速ROI标定工具 - 视触觉传感器（仅核心功能）
仅保留：选4个角点→计算透视变换矩阵→保存矩阵
无内参/去畸变逻辑，仅使用TactileCamera
"""

import numpy as np
import cv2
from pathlib import Path
import time
from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation


class QuickROICalibrator:
    def __init__(self, output_size=None, save_dir=None):
        """
        初始化
        Args:
            output_size: 输出图像固定尺寸 (width, height)
            save_dir: 标定参数保存目录
        """
        self.output_size = output_size
        
        # 保存目录
        self.save_dir = Path(save_dir) if save_dir else Path(__file__).parent / "calibration_data"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 核心变量
        self.roi_points = []
        self.image_size = None
        self.current_frame = None
        self.homography_matrix = None  # 透视变换矩阵
        
        # 仅使用TactileCamera
        self.camera = None
    
    def initialize_camera(self, camera_config):
        """初始化TactileCamera"""
        print("[INFO] 使用TactileCamera...")
        self.camera = TactileCamera(camera_config)
        self.camera.connect()
        print("[INFO] TactileCamera初始化成功")
    
    def read_frame(self):
        frame = self.camera.async_read(timeout_ms=200)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.roi_points) < 4:
            self.roi_points.append((x, y))
            print(f"[INFO] 选择点 {len(self.roi_points)}: ({x}, {y})")
            
            if len(self.roi_points) == 4:
                self._compute_homography()
                print("[INFO] 4个点已选择完成！按's'保存矩阵，按'r'重新选择")
    
    def _compute_homography(self):
        """计算透视变换矩阵（核心逻辑）"""
        # 源点：手动选择的4个角点
        src_points = np.array(self.roi_points, dtype=np.float32)
        # 目标点：输出尺寸的四个角
        w, h = self.output_size
        dst_points = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        
        self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        print(f"[INFO] 透视变换矩阵计算完成：\n{self.homography_matrix}")
    
    def apply_correction(self, frame):
        """应用透视变换（预览矫正效果）"""
        if self.homography_matrix is None:
            return frame
        return cv2.warpPerspective(frame, self.homography_matrix, self.output_size)
    
    def save_calibration(self, filename="homography_matrix.npz"):
        """保存变换矩阵（仅保留核心参数）"""
        if self.homography_matrix is None:
            print("[ERROR] 请先选择4个角点计算变换矩阵")
            return
        
        save_path = self.save_dir / filename
        params = {
            'homography_matrix': self.homography_matrix,  
            'roi_points': np.array(self.roi_points, dtype=np.float32),
            'output_size': np.array(self.output_size)
        }
        np.savez(save_path, **params)
        
        print(f"\n[INFO] 变换矩阵已保存到: {save_path}")
        print(f"[INFO] 核心透视变换矩阵：\n{self.homography_matrix}")
        print(f"[INFO] 选中的ROI角点：{self.roi_points}")
        print(f"[INFO] 输出图像尺寸：{self.output_size}")
    
    def run(self):
        window_name = "ROI Calibrator"
        preview_name = "Corrected Preview"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        # 操作提示
        print("\n=== 极简ROI标定工具 ===")
        print("操作说明：")
        print("  1. 按顺序点击4个角点（左上→右上→右下→左下）")
        print("  2. 按'r' → 重新选点")
        print("  3. 按's' → 保存变换矩阵")
        print("  4. 按'q' → 退出")
        print("=========================\n")
        
        while True:
            try:
                frame_raw = self.read_frame()

                if self.image_size is None:
                    self.image_size = (frame_raw.shape[1], frame_raw.shape[0])
                
                self.current_frame = frame_raw.copy()
                display = frame_raw.copy()
                
                for i, pt in enumerate(self.roi_points):
                    color = (0, 255, 0) if len(self.roi_points) == 4 else (0, 0, 255)
                    cv2.circle(display, pt, 8, color, -1)
                    cv2.putText(display, ['TL','TR','BR','BL'][i], (pt[0]+10, pt[1]+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if len(self.roi_points) >= 2:
                    for i in range(len(self.roi_points)-1):
                        cv2.line(display, self.roi_points[i], self.roi_points[i+1], (0,255,255), 2)
                    if len(self.roi_points) == 4:
                        cv2.line(display, self.roi_points[3], self.roi_points[0], (0,255,255), 2)
                
                # 显示选点进度
                cv2.putText(display, f"Points: {len(self.roi_points)}/4", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.imshow(window_name, display)
                
                # 预览矫正后的图像
                if self.homography_matrix is not None:
                    corrected = self.apply_correction(frame_raw)
                    cv2.putText(corrected, f"Fixed Size: {corrected.shape[1]}x{corrected.shape[0]}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.imshow(preview_name, corrected)
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    self.roi_points = []
                    self.homography_matrix = None
                    cv2.destroyWindow(preview_name)
                    print("[INFO] 已重置选点")
                elif key == ord('s'):
                    self.save_calibration()
                elif key == ord('q'):
                    break
                    
            except Exception as e:
                print(f"[ERROR] 帧处理失败: {e}")
                time.sleep(0.1)
                continue
        
        self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self.camera is not None:
            self.camera.disconnect()
        cv2.destroyAllWindows()


def main():
    """主函数：仅初始化TactileCamera并运行"""
    # 配置TactileCamera（根据实际设备修改路径）
    camera_config = TactileCameraConfig(
        index_or_path="/dev/video2",  # 替换为你的TactileCamera设备路径
        fps=25,
        width=640,
        height=480,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
    )
    
    calibrator = QuickROICalibrator(output_size=(640, 480))
    
    try:
        calibrator.initialize_camera(camera_config)
        calibrator.run()
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断程序")
    except Exception as e:
        print(f"[ERROR] 程序异常: {e}")
        raise
    finally:
        calibrator.cleanup()


if __name__ == "__main__":
    main()
