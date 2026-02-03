"""
触觉传感器测试脚本 - 查找表方法

使用 LookupTableProcessor 处理触觉传感器图像，
计算法向量和深度，并实时可视化。
"""

import cv2
import numpy as np
import os

from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.tactile_cam.processors import LookupTableProcessor
from lerobot.cameras.tactile_cam.visualization import TactileVisualizer
from lerobot.cameras.tactile_cam.gelsight_marker_tracker import GelSightMarkerTracker
from lerobot.cameras.configs import ColorMode, Cv2Rotation


def main():
    """主函数：使用查找表方法测试触觉传感器"""
    
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
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(save_dir, "data", "tactile_data_lut")
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化组件
    camera = TactileCamera(camera_config)
    processor = LookupTableProcessor()
    marker_tracker = GelSightMarkerTracker()
    visualizer = TactileVisualizer(
        windows=['original', 'depth', 'normal', 'marker'],
        window_size=(640, 480)
    )
    
    try:
        camera.connect()
        print("[INFO] 相机已连接")
        print("\n=== 触觉传感器测试（查找表方法）===")
        print("操作说明:")
        print("  r - 重置参考帧")
        print("  s - 保存当前数据")
        print("  q - 退出")
        print("=====================================\n")
        
        # 数据收集
        all_normal_maps = []
        all_depth_maps = []
        all_displacements = []
        frame_count = 0
        marker_initialized = False
        SAVE_EVERY = 10
        
        while True:
            try:
                # 读取并转换图像
                frame = camera.async_read(timeout_ms=200)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 透视变换
                warped_frame = processor.warp_perspective(frame_bgr)
                
                # 初始化标记点追踪器（跳过前几帧）
                if not marker_initialized and frame_count > 5:
                    marker_tracker.reinit(warped_frame)
                    marker_tracker.start_display_markerIm()
                    marker_initialized = True
                    print("[INFO] 标记点追踪器已初始化")
                
                # 处理图像
                depth_colored, normal_colored, raw_depth, raw_normals = \
                    processor.process_frame(warped_frame)
                
                # 更新标记点追踪
                if marker_initialized:
                    marker_tracker.update_markerMotion(warped_frame)
                displacements = marker_tracker.get_marker_displacements()
                
                # 收集数据
                if not processor.con_flag and raw_depth is not None:
                    frame_count += 1
                    if frame_count % SAVE_EVERY == 0:
                        all_normal_maps.append(raw_normals.copy())
                        all_depth_maps.append(raw_depth.copy())
                        if displacements is not None:
                            all_displacements.append(displacements.copy())
                
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
                    marker_initialized = False
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
        if len(all_normal_maps) > 0:
            np.save(os.path.join(save_dir, "normals.npy"), np.array(all_normal_maps))
            np.save(os.path.join(save_dir, "depth.npy"), np.array(all_depth_maps))
            np.save(os.path.join(save_dir, "displacements.npy"), 
                   all_displacements, allow_pickle=True)
            print(f"[INFO] 保存了 {len(all_normal_maps)} 帧数据到 {save_dir}")
        
        visualizer.cleanup()
        marker_tracker.cleanup()
        camera.disconnect()
        print("[INFO] 相机已断开")


if __name__ == "__main__":
    main()


