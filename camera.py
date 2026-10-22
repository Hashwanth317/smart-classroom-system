import os
import cv2
import time
import threading
import numpy as np

class CameraManager:
    """
    Thread-safe camera manager for MacBook built-in webcam.
    Continuously captures frames in a background thread to prevent streaming latency.
    Includes graceful handling for permission denial, hardware disconnection, or testing environments.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CameraManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, camera_index=0):
        if self._initialized:
            return
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.thread = None
        self.lock = threading.Lock()
        self.has_hardware_camera = False
        self.last_access_time = time.time()
        self._initialized = True

    def start(self):
        """Start camera streaming thread."""
        with self.lock:
            if self.is_running:
                return True
            
            # Try initializing hardware webcam
            try:
                # On macOS, AVFoundation is standard backend
                self.cap = cv2.VideoCapture(self.camera_index)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        self.has_hardware_camera = True
                        self.current_frame = test_frame
                    else:
                        self.has_hardware_camera = False
                else:
                    self.has_hardware_camera = False
            except Exception as e:
                print(f"[CameraManager] Camera init error: {e}")
                self.has_hardware_camera = False

            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return True

    def stop(self):
        """Stop camera streaming and release hardware resource."""
        with self.lock:
            self.is_running = False
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.has_hardware_camera = False

    def is_online(self):
        """Check if camera capture is currently active."""
        return self.is_running

    def _generate_placeholder_frame(self, text="Webcam Offline / Permission Needed"):
        """Generates a professional standby frame when webcam hardware is not accessible."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Gradient background
        for y in range(480):
            val = int(24 + (y / 480.0) * 20)
            frame[y, :] = (val, val + 5, val + 15)

        # Draw decorative camera icon circle
        cv2.circle(frame, (320, 200), 55, (59, 130, 246), -1)
        cv2.circle(frame, (320, 200), 25, (255, 255, 255), 3)
        cv2.circle(frame, (320, 200), 10, (255, 255, 255), -1)

        # Status text
        cv2.putText(frame, text, (110, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (241, 245, 249), 2, cv2.LINE_AA)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"System Time: {timestamp}", (190, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(frame, "Click 'Start Camera' or grant camera permission in System Settings", (70, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 116, 139), 1, cv2.LINE_AA)

        return frame

    def _capture_loop(self):
        """Background thread continuously pulling frames."""
        frame_interval = 1.0 / 30.0 # ~30 FPS
        while self.is_running:
            start_t = time.time()
            if self.has_hardware_camera and self.cap is not None and self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        # Flip horizontally for natural mirror feel on webcam
                        flipped = cv2.flip(frame, 1)
                        with self.lock:
                            self.current_frame = flipped
                    else:
                        with self.lock:
                            self.current_frame = self._generate_placeholder_frame("Webcam Frame Read Failed")
                except Exception:
                    with self.lock:
                        self.current_frame = self._generate_placeholder_frame("Camera Exception")
            else:
                # Standby / synthetic preview frame
                with self.lock:
                    self.current_frame = self._generate_placeholder_frame("Live Camera Ready - Standby Mode")

            elapsed = time.time() - start_t
            sleep_time = max(0.005, frame_interval - elapsed)
            time.sleep(sleep_time)

    def get_raw_frame(self):
        """Return the latest raw frame copy."""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return self._generate_placeholder_frame("Camera Initializing...")

# Global singleton
camera_manager = CameraManager()
