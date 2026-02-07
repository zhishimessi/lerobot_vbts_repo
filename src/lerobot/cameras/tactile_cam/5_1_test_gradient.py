"""
触觉传感器测试脚本 - 梯度方法

使用 GradientProcessor 处理触觉传感器图像，
计算法向量和深度，并实时可视化。

与查找表方法相比，梯度方法不需要预先校准的查找表，
直接从图像颜色差异计算梯度。
"""

import cv2
import numpy as np
import os
import sys

# 确保可以导入 lerobot 模块
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.tactile_cam.processors import GradientProcessor
from lerobot.cameras.tactile_cam.visualization import TactileVisualizer, visualize_gradient
from lerobot.cameras.configs import ColorMode, Cv2Rotation


def main():
    """主函数：使用梯度方法测试触觉传感器"""
    
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
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(save_dir, "data", "tactile_data_grad")
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化组件
    camera = TactileCamera(camera_config)
    processor = GradientProcessor(pad=10, sensor_id="right")
    visualizer = TactileVisualizer(
        windows=['original', 'depth', 'normal', 'grad_x', 'grad_y', 'diff'],
        window_size=(640, 480)
    )
    
    try:
        camera.connect()
        print("[INFO] 相机已连接")
        print("\n=== 触觉传感器测试（梯度方法）===")
        print("操作说明:")
        print("  r - 重置参考帧")
        print("  s - 保存当前数据")
        print("  q - 退出")
        print("===================================\n")
        
        # 数据收集
        all_depth = []
        all_normals = []
        all_gradients = []
        frame_count = 0
        SAVE_EVERY = 10
        
        while True:
            try:
                # 读取并转换图像
                frame = camera.async_read(timeout_ms=200)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 处理帧
                result = processor.process_frame(frame_bgr)
                depth_colored, normal_colored, raw_depth, raw_normals, grad_x, grad_y, diff_img = result
                
                # 透视变换后的原始图像
                warped = processor.warp_perspective(frame_bgr)
                
                # 显示结果
                visualizer.show('original', warped)
                visualizer.show('depth', depth_colored)
                visualizer.show('normal', normal_colored)
                
                if diff_img is not None:
                    visualizer.show('diff', diff_img)
                
                if grad_x is not None:
                    grad_x_vis, grad_y_vis = visualize_gradient(grad_x, grad_y)
                    visualizer.show('grad_x', grad_x_vis)
                    visualizer.show('grad_y', grad_y_vis)
                    
                    # 收集数据
                    frame_count += 1
                    if frame_count % SAVE_EVERY == 0 and raw_depth is not None:
                        all_depth.append(raw_depth.copy())
                        all_normals.append(raw_normals.copy())
                        all_gradients.append(np.stack([grad_x, grad_y], axis=-1))
                
                # 处理按键
                key = visualizer.wait_key(1)
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    processor.reset()
                elif key == ord('s') and raw_depth is not None:
                    timestamp = int(cv2.getTickCount())
                    np.save(os.path.join(save_dir, f"depth_grad_{timestamp}.npy"), raw_depth)
                    np.save(os.path.join(save_dir, f"normal_grad_{timestamp}.npy"), raw_normals)
                    np.save(os.path.join(save_dir, f"gradient_{timestamp}.npy"), 
                           np.stack([grad_x, grad_y], axis=-1))
                    print(f"[INFO] 数据已保存: depth_grad_{timestamp}.npy")
                    
            except TimeoutError:
                continue
            except RuntimeError as e:
                print(f"[WARNING] 帧读取错误: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    
    finally:
        # 保存收集的数据
        if len(all_depth) > 0:
            np.save(os.path.join(save_dir, "depth_gradient.npy"), np.array(all_depth))
            np.save(os.path.join(save_dir, "normals_gradient.npy"), np.array(all_normals))
            np.save(os.path.join(save_dir, "gradients.npy"), np.array(all_gradients))
            print(f"[INFO] 保存了 {len(all_depth)} 帧数据到 {save_dir}")
        
        visualizer.cleanup()
        camera.disconnect()
        print("[INFO] 相机已断开")


if __name__ == "__main__":
    main()
