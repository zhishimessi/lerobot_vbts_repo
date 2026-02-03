import logging
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
from numpy.typing import NDArray  

if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2  

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_backend, get_cv2_rotation
from .tactile_config import ColorMode, TactileCameraConfig

MAX_OPENCV_INDEX = 60

logger = logging.getLogger(__name__)


class TactileCamera(Camera):

    def __init__(self, config: TactileCameraConfig):

        super().__init__(config)

        self.config = config
        self.index_or_path = config.index_or_path

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s

        self.videocapture: cv2.VideoCapture | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.backend: int = get_cv2_backend()

        # 曝光设置 (Linux V4L2: 通常 1-10000)
        self.exposure_value: int = config.exposure
        self.auto_exposure: bool = config.auto_exposure
        
        # 白平衡设置
        self.wb_temperature: int = config.wb_temperature
        self.auto_wb: bool = config.auto_wb
        
        # RGB增益设置 (范围: 0.0 - 3.0)
        self.r_gain: float = config.r_gain
        self.g_gain: float = config.g_gain
        self.b_gain: float = config.b_gain

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected and opened."""
        return isinstance(self.videocapture, cv2.VideoCapture) and self.videocapture.isOpened()

    def connect(self, warmup: bool = True) -> None:
        """
        Connects to the OpenCV camera specified in the configuration.

        Initializes the OpenCV VideoCapture object, sets desired camera properties
        (FPS, width, height), and performs initial checks.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ConnectionError: If the specified camera index/path is not found or the camera is found but fails to open.
            RuntimeError: If the camera opens but fails to apply requested FPS/resolution settings.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Use 1 thread for OpenCV operations to avoid potential conflicts or
        # blocking in multi-threaded applications, especially during data collection.
        cv2.setNumThreads(1)

        self.videocapture = cv2.VideoCapture(self.index_or_path, self.backend)

        if not self.videocapture.isOpened():
            self.videocapture.release()
            self.videocapture = None
            raise ConnectionError(
                f"Failed to open {self}.Run `lerobot-find-cameras opencv` to find available cameras."
            )

        self._configure_capture_settings()

        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    def _configure_capture_settings(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Cannot configure settings for {self} as it is not connected.")

        # Set FOURCC first (if specified) as it can affect available FPS/resolution options
        if self.config.fourcc is not None:
            self._validate_fourcc()
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        default_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        default_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if self.width is None or self.height is None:
            self.width, self.height = default_width, default_height
            self.capture_width, self.capture_height = default_width, default_height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = default_height, default_width
                self.capture_width, self.capture_height = default_width, default_height
        else:
            self._validate_width_and_height()

        if self.fps is None:
            self.fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        else:
            self._validate_fps()
        
        # 自动配置曝光、白平衡等相机控制参数
        self._configure_camera_controls()

    def _configure_camera_controls(self) -> None:
        """配置相机控制参数：曝光、白平衡等 (Linux V4L2)."""
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")
        
        self._configure_v4l2_controls()
        
        print(f"{self} 相机参数配置完成: 曝光={self.exposure_value}, 白平衡={self.wb_temperature}K, "
                    f"RGB增益=({self.r_gain:.2f}, {self.g_gain:.2f}, {self.b_gain:.2f})")
    
    def _configure_exposure(self) -> None:
        """配置手动曝光模式和曝光值 (兼容旧接口)."""
        self._configure_camera_controls()
    
    def _configure_v4l2_controls(self) -> None:
        """使用 v4l2-ctl 设置曝光和白平衡."""
        device_path = self.index_or_path
        if isinstance(device_path, int):
            device_path = f"{device_path}"
        
        try:
            # 1. 设置曝光
            if self.auto_exposure:
                subprocess.run(
                    ["v4l2-ctl", "-d", str(device_path), "-c", "auto_exposure=3"],
                    check=False, capture_output=True, timeout=5
                )
                logger.debug(f"{self} 自动曝光已启用")
            else:
                # 禁用自动曝光 (auto_exposure: 1=手动, 3=自动)
                subprocess.run(
                    ["v4l2-ctl", "-d", str(device_path), "-c", "auto_exposure=1"],
                    check=False, capture_output=True, timeout=5
                )
                # 设置曝光值
                subprocess.run(
                    ["v4l2-ctl", "-d", str(device_path), "-c", f"exposure_time_absolute={int(self.exposure_value)}"],
                    check=False, capture_output=True, timeout=5
                )
                logger.debug(f"{self} 手动曝光: {self.exposure_value}")
            
            # 2. 设置白平衡
            if self.auto_wb:
                subprocess.run(
                    ["v4l2-ctl", "-d", str(device_path), "-c", "white_balance_temperature_auto=1"],
                    check=False, capture_output=True, timeout=5
                )
                logger.debug(f"{self} 自动白平衡已启用")
            else:
                subprocess.run(
                    ["v4l2-ctl", "-d", str(device_path), "-c", "white_balance_temperature_auto=0"],
                    check=False, capture_output=True, timeout=5
                )
                subprocess.run(
                    ["v4l2-ctl", "-d", str(device_path), "-c", f"white_balance_temperature={self.wb_temperature}"],
                    check=False, capture_output=True, timeout=5
                )
                logger.debug(f"{self} 手动白平衡: {self.wb_temperature}K")
                
        except FileNotFoundError:
            logger.warning(f"{self} v4l2-ctl 未安装，请运行: sudo apt install v4l-utils")
        except subprocess.TimeoutExpired:
            logger.warning(f"{self} v4l2-ctl 超时")
        except Exception as e:
            logger.warning(f"{self} V4L2 相机参数设置失败: {e}")

    def _validate_fps(self) -> None:
        """Validates and sets the camera's FPS (兼容驱动返回False)."""
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        if self.fps is None:
            raise ValueError(f"{self} FPS is not set")

        # 尝试设置FPS
        self.videocapture.set(cv2.CAP_PROP_FPS, float(self.fps))
        actual_fps = self.videocapture.get(cv2.CAP_PROP_FPS)

        # 浮点兼容比较，仅警告不抛异常
        if not math.isclose(self.fps, actual_fps, rel_tol=0.1):  # 放宽容差到10%
            logger.warning(
                f"{self} FPS mismatch: requested {self.fps}, actual {actual_fps:.1f}. Using actual FPS."
            )
            self.fps = actual_fps

    def _validate_fourcc(self) -> None:
        """Validates and sets the camera's FOURCC code (兼容Linux驱动)."""
        fourcc_code = cv2.VideoWriter_fourcc(*self.config.fourcc)

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        # 先尝试OpenCV原生方式设置
        success = self.videocapture.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        actual_fourcc_code = self.videocapture.get(cv2.CAP_PROP_FOURCC)

        # 转换实际FOURCC为字符串（兼容空值）
        actual_fourcc = ""
        if actual_fourcc_code != 0:
            actual_fourcc_code_int = int(actual_fourcc_code)
            actual_fourcc = "".join([chr((actual_fourcc_code_int >> 8 * i) & 0xFF) for i in range(4)])

        # 仅警告，不中断流程（Linux驱动常不支持CAP_PROP_FOURCC直接设置）
        if not success or actual_fourcc != self.config.fourcc:
            logger.warning(
                f"{self} failed to set fourcc={self.config.fourcc} (actual={actual_fourcc}, success={success}). "
                f"Continuing with hardware default format."
            )

    def _validate_width_and_height(self) -> None:
        """Validates and sets the camera's frame capture width and height (兼容驱动返回False)."""
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        if self.capture_width is None or self.capture_height is None:
            raise ValueError(f"{self} capture_width or capture_height is not set")

        # 尝试设置分辨率（驱动可能返回False，但实际已生效）
        self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
        self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))

        # 以实际读取的分辨率为准，而非set的返回值
        actual_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # 仅当实际分辨率与目标差距过大时警告，不抛异常
        if actual_width != self.capture_width or actual_height != self.capture_height:
            logger.warning(
                f"{self} resolution mismatch: requested ({self.capture_width}x{self.capture_height}), actual ({actual_width}x{actual_height}). Using actual resolution."
            )
            self.capture_width = actual_width
            self.capture_height = actual_height

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available OpenCV cameras connected to the system.

        On Linux, it scans '/dev/video*' paths. On other systems (like macOS, Windows),
        it checks indices from 0 up to `MAX_OPENCV_INDEX`.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'id' (port index or path),
            and the default profile properties (width, height, fps, format).
        """
        found_cameras_info = []

        targets_to_scan: list[str | int]
        if platform.system() == "Linux":
            possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            targets_to_scan = [str(p) for p in possible_paths]
        else:
            targets_to_scan = [int(i) for i in range(MAX_OPENCV_INDEX)]

        for target in targets_to_scan:
            camera = cv2.VideoCapture(target)
            if camera.isOpened():
                default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                default_fps = camera.get(cv2.CAP_PROP_FPS)
                default_format = camera.get(cv2.CAP_PROP_FORMAT)

                # Get FOURCC code and convert to string
                default_fourcc_code = camera.get(cv2.CAP_PROP_FOURCC)
                default_fourcc_code_int = int(default_fourcc_code)
                default_fourcc = "".join([chr((default_fourcc_code_int >> 8 * i) & 0xFF) for i in range(4)])

                camera_info = {
                    "name": f"OpenCV Camera @ {target}",
                    "type": "OpenCV",
                    "id": target,
                    "backend_api": camera.getBackendName(),
                    "default_stream_profile": {
                        "format": default_format,
                        "fourcc": default_fourcc,
                        "width": default_width,
                        "height": default_height,
                        "fps": default_fps,
                    },
                }

                found_cameras_info.append(camera_info)
                camera.release()

        return found_cameras_info

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        Reads a single frame synchronously from the camera.

        This is a blocking call. It waits for the next available frame from the
        camera hardware via OpenCV.

        Args:
            color_mode (Optional[ColorMode]): If specified, overrides the default
                color mode (`self.color_mode`) for this read operation (e.g.,
                request RGB even if default is BGR).

        Returns:
            np.ndarray: The captured frame as a NumPy array in the format
                       (height, width, channels), using the specified or default
                       color mode and applying any configured rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading the frame from the camera fails or if the
                          received frame dimensions don't match expectations before rotation.
            ValueError: If an invalid `color_mode` is requested.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")
        
        ret, frame = self.videocapture.read()

        if not ret or frame is None:
            raise RuntimeError(f"{self} read failed (status={ret}).")

        processed_frame = self._postprocess_image(frame, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return processed_frame

    def _postprocess_image(self, image: NDArray[Any], color_mode: ColorMode | None = None) -> NDArray[Any]:

        requested_color_mode = self.color_mode if color_mode is None else color_mode

        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        processed_image = image
        
        # 应用RGB增益
        if self.r_gain != 1.0 or self.g_gain != 1.0 or self.b_gain != 1.0:
            processed_image = self._apply_rgb_gain(processed_image)
        
        if requested_color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image
    
    def _apply_rgb_gain(self, image: NDArray[Any]) -> NDArray[Any]:
        """
        应用RGB增益到图像.
        
        Args:
            image: 输入BGR图像
        
        Returns:
            调整后的图像
        """
        # 分离通道 (OpenCV是BGR顺序)
        b, g, r = cv2.split(image.astype(np.float32))
        
        # 应用增益
        r = np.clip(r * self.r_gain, 0, 255)
        g = np.clip(g * self.g_gain, 0, 255)
        b = np.clip(b * self.b_gain, 0, 255)
        
        # 合并通道
        result = cv2.merge([b, g, r]).astype(np.uint8)
        return result
    
    # ========== 动态调整方法 ==========
    
    def set_exposure(self, value: int) -> None:
        """
        动态设置曝光值 (Linux V4L2).
        
        Args:
            value: 曝光值 (通常 1-10000, 较小值=较暗)
        """
        self.exposure_value = value
        self.auto_exposure = False
        
        if self.is_connected:
            device_path = self.index_or_path
            if isinstance(device_path, int):
                device_path = f"/dev/video{device_path}"
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", str(device_path), "-c", "auto_exposure=1"],
                    check=False, capture_output=True, timeout=5
                )
                subprocess.run(
                    ["v4l2-ctl", "-d", str(device_path), "-c", f"exposure_time_absolute={int(value)}"],
                    check=False, capture_output=True, timeout=5
                )
            except Exception as e:
                logger.warning(f"设置曝光失败: {e}")
        
        logger.debug(f"{self} 曝光设置为: {value}")
    
    def set_wb_temperature(self, value: int) -> None:
        """
        动态设置白平衡色温 (Linux V4L2).
        
        Args:
            value: 色温值 (范围: 2000-8000K)
        """
        self.wb_temperature = max(2000, min(8000, value))
        self.auto_wb = False
        
        if self.is_connected:
            device_path = self.index_or_path
            if isinstance(device_path, int):
                device_path = f"/dev/video{device_path}"
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", str(device_path), "-c", "white_balance_temperature_auto=0"],
                    check=False, capture_output=True, timeout=5
                )
                subprocess.run(
                    ["v4l2-ctl", "-d", str(device_path), "-c", f"white_balance_temperature={self.wb_temperature}"],
                    check=False, capture_output=True, timeout=5
                )
            except Exception as e:
                logger.warning(f"设置白平衡失败: {e}")
        
        logger.debug(f"{self} 白平衡设置为: {self.wb_temperature}K")
    
    def set_rgb_gain(self, r: float = 1.0, g: float = 1.0, b: float = 1.0) -> None:
        """
        设置RGB通道增益.
        
        Args:
            r: 红色通道增益 (0.0 - 3.0)
            g: 绿色通道增益 (0.0 - 3.0)
            b: 蓝色通道增益 (0.0 - 3.0)
        """
        self.r_gain = max(0.0, min(3.0, r))
        self.g_gain = max(0.0, min(3.0, g))
        self.b_gain = max(0.0, min(3.0, b))
        logger.debug(f"{self} RGB增益设置为: R={self.r_gain:.2f}, G={self.g_gain:.2f}, B={self.b_gain:.2f}")
    
    def reset_rgb_gain(self) -> None:
        """重置RGB增益为默认值 1.0."""
        self.r_gain = self.g_gain = self.b_gain = 1.0
        logger.debug(f"{self} RGB增益已重置")
    
    def get_camera_settings(self) -> dict:
        """
        获取当前相机设置.
        
        Returns:
            包含所有相机设置的字典
        """
        return {
            "exposure": self.exposure_value,
            "auto_exposure": self.auto_exposure,
            "wb_temperature": self.wb_temperature,
            "auto_wb": self.auto_wb,
            "r_gain": self.r_gain,
            "g_gain": self.g_gain,
            "b_gain": self.b_gain,
        }

    def _read_loop(self) -> None:
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a color frame
        2. Stores result in latest_frame (thread-safe)
        3. Sets new_frame_event to notify listeners

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        while not self.stop_event.is_set():
            try:
                color_image = self.read()

                with self.frame_lock:
                    self.latest_frame = color_image
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def disconnect(self) -> None:
        """
        Disconnects from the camera and cleans up resources.

        Stops the background read thread (if running) and releases the OpenCV
        VideoCapture object.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.videocapture is not None:
            self.videocapture.release()
            self.videocapture = None

        logger.info(f"{self} disconnected.")