import numpy as np
import time
import cv2
import os
from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation

class CalibrationImageCapture:
    """捕获并矫正标定图像的类（仅保存矫正后图像）"""
    
    def __init__(self, camera_config):
        """
        初始化标定图像捕获器
        
        Args:
            camera_config: 相机配置
        """
        self.camera_config = camera_config
        self.save_dir1 = "/home/donghy/lerobot/src/lerobot/cameras/tactile_cam/data/test_data"
        self.save_dir2 = "/home/donghy/lerobot/src/lerobot/cameras/tactile_cam/data/calibration_data"
        self.camera = None
        self.homography_matrix = None
        self.output_size = None     
        self.calib_file = "/home/donghy/lerobot/src/lerobot/cameras/tactile_cam/data/calibration_data/homography_matrix.npz"
    
    def initialize_camera(self):
        print("[INFO] 初始化相机...")
        self.camera = TactileCamera(self.camera_config)
        self.camera.connect()
        print("[INFO] 相机连接成功")

    def process_and_save_image1(self, image, save_name):
        """
        基于预加载的透视矩阵矫正图像并保存
        Args:
            image: 输入BGR图像
            save_name: 保存的基础文件名
        """
        if self.homography_matrix is None:
            print(f'[WARNING] {save_name}: 透视变换矩阵未加载，跳过保存')
            return
        
        warped_img = cv2.warpPerspective(
            image, 
            self.homography_matrix, 
            self.output_size,  # 使用标定时的输出尺寸
            flags=cv2.INTER_NEAREST  # 最近邻，不改变像素值
        )

        save_path = os.path.join(self.save_dir1, f"{save_name}.jpg")
        cv2.imwrite(save_path, warped_img)
        print(f'[INFO] 已保存矫正后图像: {save_path}')

    def process_and_save_image2(self, image, save_name):
        """
        基于预加载的透视矩阵矫正图像并保存
        Args:
            image: 输入BGR图像
            save_name: 保存的基础文件名
        """
        if self.homography_matrix is None:
            print(f'[WARNING] {save_name}: 透视变换矩阵未加载，跳过保存')
            return
        
        warped_img = cv2.warpPerspective(
            image, 
            self.homography_matrix, 
            self.output_size,  # 使用标定时的输出尺寸
            flags=cv2.INTER_NEAREST  # 最近邻，不改变像素值
        )

        save_path = os.path.join(self.save_dir2, f"{save_name}.jpg")
        cv2.imwrite(save_path, warped_img)
        print(f'[INFO] 已保存矫正后图像: {save_path}')
    
    def capture_images(self, num_images):
        """
        捕获、矫正并保存标定图像（基于预计算的透视变换矩阵）
        
        Args:
            num_images: 需要捕获的标定图像数量
        """
        if self.camera is None:
            raise RuntimeError("相机未初始化，请先调用 initialize_camera()")
        
        try:
            calib_data = np.load(self.calib_file)
            self.homography_matrix = calib_data['homography_matrix']
            self.output_size = tuple(int(x) for x in calib_data['output_size'])
            print(f'[INFO] 成功加载透视变换矩阵')
            print(f'[INFO] 输出尺寸：{self.output_size}')
        except FileNotFoundError:
            raise FileNotFoundError(f"标定文件未找到，请检查路径：{self.calib_file}")
        except KeyError:
            raise KeyError("标定文件中未找到 'homography_matrix' 字段")
        except Exception as e:
            raise RuntimeError(f"加载标定矩阵失败：{e}")
        
        print(f'[INFO] 开始捕获 {num_images} 张标定图像...')
        print('[INFO] 第一张图像为参考图像 (请勿按压传感器)...')
        print('[INFO] 后续图像将在按压标准球后捕获...')
        time.sleep(2)
        
        count = 0
        
        while True:
            try:
                frame = self.camera.async_read(timeout_ms=200)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                warped_img = cv2.warpPerspective(
                    frame_bgr, 
                    self.homography_matrix, 
                    self.output_size,  # 使用标定时的输出尺寸
                    flags=cv2.INTER_NEAREST  # 最近邻，不改变像素值
                )
                
                cv2.imshow('Calibration Image Capture', warped_img)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):
                    if count == 0:
                        self.process_and_save_image1(frame_bgr, 'ref')
                        count += 1
                    elif count <= num_images:
                        self.process_and_save_image1(frame_bgr, f'sample_{count}')
                        count += 1
                    if count > num_images:
                        print(f'[INFO] 已完成 {num_images} 张标定图像的捕获与矫正')
                        break
                    
                # elif key == ord('c'):
                #     if count <= num_images:
                #         self.process_and_save_image2(frame_bgr, f'calibration_{count}')
                #         count += 1
                #     if count > num_images:
                #         print(f'[INFO] 已完成 {num_images} 张标定图像的捕获与矫正')
                #         break

                elif key == ord('q'):
                    print('[INFO] 用户取消捕获')
                    break
                    
            except TimeoutError as e:
                print(f'[WARNING] 帧读取超时: {e}')
                continue
            except RuntimeError as e:
                print(f'[WARNING] 帧读取错误: {e}')
                continue
        
        cv2.destroyAllWindows()
    
    def cleanup(self):
        """清理资源"""
        if self.camera is not None:
            self.camera.disconnect()
            print("[INFO] 相机已断开连接")

def main():
    ov5647_config = TactileCameraConfig(
        index_or_path="/dev/video2", 
        fps=25,                       
        width=640,                   
        height=480,
        color_mode=ColorMode.RGB,     
        rotation=Cv2Rotation.NO_ROTATION, 
        exposure=600,        # 曝光值
        wb_temperature=4500, # 白平衡色温
        r_gain=1.0,          # RGB增益
        g_gain=1.0,
        b_gain=1.0
    )

    capture = CalibrationImageCapture(ov5647_config)
    
    try:
        capture.initialize_camera()
        capture.capture_images(80) 
        
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断捕获")
    except Exception as e:
        print(f'[ERROR] 捕获/矫正过程中发生错误: {e}')
        raise
    finally:
        capture.cleanup()

if __name__ == "__main__":
    main()