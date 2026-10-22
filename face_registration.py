import os
import re
import cv2
import json
import time
import numpy as np

# Ensure DeepFace uses local offline weights directory BEFORE importing DeepFace
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["DEEPFACE_HOME"] = os.path.join(BASE_DIR, ".deepface")
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from deepface import DeepFace
import database

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
CASCADES_DIR = os.path.join(BASE_DIR, "cascades")

FACE_CASCADE_PATH = os.path.join(CASCADES_DIR, "haarcascade_frontalface_default.xml")
if not os.path.exists(FACE_CASCADE_PATH):
    FACE_CASCADE_PATH = os.path.join(BASE_DIR, ".deepface", ".deepface", "weights", "haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH) if os.path.exists(FACE_CASCADE_PATH) else None

POSE_GUIDELINES = [
    {"pose": "Front", "instruction": "Look directly at the camera (Front)", "range": (1, 5)},
    {"pose": "Slight Left", "instruction": "Turn your head slightly to the LEFT", "range": (6, 10)},
    {"pose": "Slight Right", "instruction": "Turn your head slightly to the RIGHT", "range": (11, 15)},
    {"pose": "Slight Up", "instruction": "Tilt your head slightly UP", "range": (16, 20)},
    {"pose": "Slight Down", "instruction": "Tilt your head slightly DOWN", "range": (21, 25)},
]

def sanitize_filename(name):
    """Sanitize string for file/directory paths."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name.strip())

class RegistrationSession:
    """Manages an interactive registration session for capturing and processing student face samples."""
    def __init__(self, roll_no, name, target_samples=25):
        self.roll_no = roll_no.strip().upper()
        self.name = name.strip()
        self.target_samples = target_samples
        self.samples = []
        self.last_capture_time = 0
        self.min_capture_interval = 0.28 # Minimum 280ms between samples to prevent identical consecutive frames
        self.last_face_box = None
        self.status = "in_progress" # in_progress, completed, failed
        self.message = "Please face the camera to begin registration."
        self.current_pose = POSE_GUIDELINES[0]["instruction"]
        self.student_dir = os.path.join(DATASET_DIR, f"{sanitize_filename(self.roll_no)}_{sanitize_filename(self.name)}")
        os.makedirs(self.student_dir, exist_ok=True)
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    def get_current_guidance(self):
        count = len(self.samples) + 1
        for guide in POSE_GUIDELINES:
            if guide["range"][0] <= count <= guide["range"][1]:
                return guide["instruction"]
        return "Holding pose for final verification..."

    def process_frame(self, frame):
        """
        Processes a camera frame for registration:
        - Detects single face.
        - Validates face quality.
        - Captures frame if conditions are met.
        Returns:
            dict: progress info
        """
        if self.status == "completed":
            return {
                "status": "completed",
                "progress": f"{len(self.samples)} / {self.target_samples} images captured",
                "count": len(self.samples),
                "total": self.target_samples,
                "message": "Registration Completed Successfully",
                "current_pose": "Done",
                "error": None
            }

        if frame is None:
            return {"status": "error", "message": "No frame received", "error": "Camera offline"}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = []
        if face_cascade and not face_cascade.empty():
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(60, 60)
            )
            # If not detected, try with looser parameters for dim lighting or tilted angles
            if len(faces) == 0:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(50, 50)
                )

        # Rule 16: Check number of faces detected
        if len(faces) == 0:
            self.message = "No face detected. Please face the camera."
            return {
                "status": "in_progress",
                "progress": f"{len(self.samples)} / {self.target_samples} images captured",
                "count": len(self.samples),
                "total": self.target_samples,
                "message": self.message,
                "current_pose": self.get_current_guidance(),
                "face_box": None,
                "error": None
            }
        
        if len(faces) > 1:
            self.message = "Only one student should be visible during registration."
            return {
                "status": "in_progress",
                "progress": f"{len(self.samples)} / {self.target_samples} images captured",
                "count": len(self.samples),
                "total": self.target_samples,
                "message": self.message,
                "current_pose": self.get_current_guidance(),
                "face_box": None,
                "error": "Multiple faces detected"
            }

        x, y, w, h = faces[0]
        face_roi = frame[max(0, y):min(frame.shape[0], y+h), max(0, x):min(frame.shape[1], x+w)]

        # Quality check: Reject excessively blurry frames (tolerant threshold for webcams)
        lap_var = cv2.Laplacian(gray[max(0, y):min(frame.shape[0], y+h), max(0, x):min(frame.shape[1], x+w)], cv2.CV_64F).var()
        if lap_var < 8.0:
            self.message = "Frame too blurry. Please stay steady."
            return {
                "status": "in_progress",
                "progress": f"{len(self.samples)} / {self.target_samples} images captured",
                "count": len(self.samples),
                "total": self.target_samples,
                "message": self.message,
                "current_pose": self.get_current_guidance(),
                "face_box": [int(x), int(y), int(w), int(h)],
                "error": None
            }

        now = time.time()
        # Do not save identical consecutive frames (check interval & motion)
        should_capture = False
        if now - self.last_capture_time >= self.min_capture_interval:
            should_capture = True

        if should_capture and len(self.samples) < self.target_samples:
            sample_idx = len(self.samples) + 1
            filename = f"img_{sample_idx:02d}.jpg"
            filepath = os.path.join(self.student_dir, filename)
            
            # Save the face crop with margin
            pad_y = int(h * 0.15)
            pad_x = int(w * 0.15)
            y1 = max(0, y - pad_y)
            y2 = min(frame.shape[0], y + h + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(frame.shape[1], x + w + pad_x)
            cropped_face = frame[y1:y2, x1:x2]

            cv2.imwrite(filepath, cropped_face)
            self.samples.append(filepath)
            self.last_capture_time = now
            self.last_face_box = (x, y, w, h)
            self.message = f"{len(self.samples)} / {self.target_samples} images captured"

            # Check if target reached
            if len(self.samples) >= self.target_samples:
                self.status = "processing_embeddings"
                self.message = "Generating representative face embeddings..."

        return {
            "status": self.status,
            "progress": f"{len(self.samples)} / {self.target_samples} images captured",
            "count": len(self.samples),
            "total": self.target_samples,
            "message": self.message,
            "current_pose": self.get_current_guidance(),
            "face_box": [int(x), int(y), int(w), int(h)],
            "error": None
        }

    def finalize_registration(self):
        """
        Generates FaceNet embeddings for captured samples, averages them,
        normalizes the representative embedding, and saves to database.
        """
        if len(self.samples) < 5:
            self.status = "failed"
            self.message = "Insufficient face samples captured."
            return False, self.message

        valid_embeddings = []
        for sample_path in self.samples:
            try:
                # DeepFace represent with FaceNet model
                representations = DeepFace.represent(
                    img_path=sample_path,
                    model_name="Facenet",
                    enforce_detection=False
                )
                if representations and len(representations) > 0:
                    emb = np.array(representations[0]["embedding"], dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        valid_embeddings.append(emb / norm)
            except Exception as e:
                # Reject bad frame if embedding extraction fails
                print(f"[Registration] Skipping bad frame {sample_path}: {e}")

        if len(valid_embeddings) == 0:
            self.status = "failed"
            self.message = "Failed to extract valid face embeddings from samples."
            return False, self.message

        # Compute averaged representative embedding
        mean_embedding = np.mean(valid_embeddings, axis=0)
        # Normalize the averaged embedding
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        embedding_list = mean_embedding.tolist()

        # Save to SQLite database
        student_id = database.add_student(self.roll_no, self.name, embedding_list)

        # Also store backup embedding file
        emb_file = os.path.join(EMBEDDINGS_DIR, f"{sanitize_filename(self.roll_no)}.json")
        with open(emb_file, "w") as f:
            json.dump({
                "student_id": student_id,
                "roll_no": self.roll_no,
                "name": self.name,
                "samples_count": len(valid_embeddings),
                "embedding": embedding_list
            }, f, indent=2)

        self.status = "completed"
        self.message = "Registration Completed Successfully"
        return True, self.message
