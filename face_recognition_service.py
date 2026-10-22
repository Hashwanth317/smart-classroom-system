import os
import cv2
import time
import json
import threading
import numpy as np

# Ensure DeepFace uses local offline weights directory BEFORE importing DeepFace
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["DEEPFACE_HOME"] = os.path.join(BASE_DIR, ".deepface")
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from deepface import DeepFace
import database
from liveness import LivenessDetector

# ==================================================
# CONFIGURATION SECTION
# ==================================================
MATCH_THRESHOLD = 0.40         # Cosine distance threshold (lower = stricter match)
COSINE_SIM_THRESHOLD = 0.60    # Equivalent cosine similarity threshold (1 - distance)
COOLDOWN_SECONDS = 25          # Minimum seconds between attendance actions for same student
SPOOF_COOLDOWN_SECONDS = 15    # Cooldown for logging spoof attempts
UNKNOWN_COOLDOWN_SECONDS = 15  # Cooldown for logging unknown face sightings
# ==================================================

CASCADES_DIR = os.path.join(BASE_DIR, "cascades")
FACE_CASCADE_PATH = os.path.join(CASCADES_DIR, "haarcascade_frontalface_default.xml")
if not os.path.exists(FACE_CASCADE_PATH):
    FACE_CASCADE_PATH = os.path.join(BASE_DIR, ".deepface", ".deepface", "weights", "haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH) if os.path.exists(FACE_CASCADE_PATH) else None

def cosine_distance(vec_a, vec_b):
    """Compute cosine distance between two 1D normalized vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = np.dot(a, b) / (norm_a * norm_b)
    return max(0.0, float(1.0 - similarity))

class FaceRecognitionService:
    """
    Core Face Recognition and Attendance Engine:
    - Runs face detection on incoming webcam frames
    - Verifies anti-spoofing / liveness
    - Extracts FaceNet embeddings and performs cosine distance matching
    - Updates SQLite database for ENTRY and EXIT modes with student cooldowns
    """

    def __init__(self):
        self.mode = "ENTRY" # "ENTRY" or "EXIT"
        self.lock = threading.Lock()
        self.registered_students = []
        self.student_cooldowns = {} # student_id -> last_processed_timestamp
        self.last_spoof_log_time = 0
        self.last_unknown_log_time = 0
        self.liveness_detector = LivenessDetector()
        self.reload_students()

    def set_mode(self, mode_str):
        """Toggle between ENTRY and EXIT mode."""
        with self.lock:
            mode_upper = mode_str.strip().upper()
            if mode_upper in ["ENTRY", "EXIT"]:
                self.mode = mode_upper
                database.log_activity("mode", f"Switched attendance system to {self.mode} MODE")
                return True, f"Mode set to {self.mode}"
            return False, "Invalid mode (must be ENTRY or EXIT)"

    def get_mode(self):
        with self.lock:
            return self.mode

    def reload_students(self):
        """Reload registered student embeddings from database."""
        with self.lock:
            try:
                self.registered_students = database.get_all_students_with_embeddings()
                print(f"[FaceRecognitionService] Loaded {len(self.registered_students)} student embeddings.")
            except Exception as e:
                print(f"[FaceRecognitionService] Error loading students: {e}")
                self.registered_students = []

    def _find_best_match(self, face_emb):
        """
        Compare face_emb against all registered students.
        Returns: (best_student, min_dist)
        """
        best_student = None
        min_distance = 999.0

        for student in self.registered_students:
            registered_emb = student.get("embedding")
            if not registered_emb or len(registered_emb) != len(face_emb):
                continue
            dist = cosine_distance(face_emb, registered_emb)
            if dist < min_distance:
                min_distance = dist
                best_student = student

        if min_distance <= MATCH_THRESHOLD:
            return best_student, min_distance
        return None, min_distance

    def process_and_annotate_frame(self, frame):
        """
        Processes a single camera frame:
        - Detects faces
        - Verifies liveness
        - Runs face recognition
        - Marks attendance according to ENTRY or EXIT mode
        - Annotates bounding boxes, student information, and system status
        """
        if frame is None:
            return None

        annotated = frame.copy()
        h, w = annotated.shape[:2]
        now = time.time()

        # 1. Draw top status bar
        mode = self.get_mode()
        bar_color = (199, 90, 37) if mode == "ENTRY" else (147, 51, 234) # BGR colors
        cv2.rectangle(annotated, (0, 0), (w, 42), bar_color, -1)
        
        mode_text = f"ATTENDANCE CAMERA  |  {mode} MODE ACTIVE"
        cv2.putText(annotated, mode_text, (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        time_str = time.strftime("%I:%M:%S %p")
        cv2.putText(annotated, time_str, (w - 150, 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, (241, 245, 249), 1, cv2.LINE_AA)

        # 2. Face Detection
        gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = []
        if face_cascade and not face_cascade.empty():
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(60, 60)
            )
            if len(faces) == 0:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(50, 50)
                )

        # 3. Process Each Detected Face
        for (x, y, w_box, h_box) in faces:
            # Expand box slightly for better visualization
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + w_box)
            y2 = min(h, y + h_box)

            face_roi = annotated[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            # STEP 1: Liveness / Anti-spoofing verification
            liveness_res = self.liveness_detector.verify_liveness(annotated, (x1, y1, w_box, h_box))
            is_live = liveness_res.get("is_live", True)

            if not is_live:
                # SPOOF DETECTED: Reject & Display Warning
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 225), 3) # Red border
                
                # Pill background for text
                pill_y1 = max(45, y1 - 32)
                cv2.rectangle(annotated, (x1, pill_y1), (x1 + 240, pill_y1 + 28), (0, 0, 200), -1)
                cv2.putText(annotated, "Possible Spoof Attempt", (x1 + 8, pill_y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                # Log spoof attempt once per cooldown
                if now - self.last_spoof_log_time >= SPOOF_COOLDOWN_SECONDS:
                    database.log_activity("spoof", "Possible spoof attempt detected by anti-spoofing filter")
                    self.last_spoof_log_time = now

                continue # Do not mark attendance!

            # STEP 2: Extract FaceNet Embedding
            recognized_student = None
            min_dist = 1.0
            try:
                # Calculate embedding for detected face ROI
                res = DeepFace.represent(
                    face_roi,
                    model_name="Facenet",
                    enforce_detection=False
                )
                if res and len(res) > 0:
                    current_emb = np.array(res[0]["embedding"], dtype=np.float32)
                    norm = np.linalg.norm(current_emb)
                    if norm > 1e-6:
                        current_emb = current_emb / norm
                        recognized_student, min_dist = self._find_best_match(current_emb)
            except Exception as e:
                # Fallback on embedding extraction failure
                recognized_student = None

            # STEP 3: Handle Match Result
            if recognized_student is not None:
                # RECOGNIZED STUDENT
                student_id = recognized_student["id"]
                roll_no = recognized_student["roll_no"]
                name = recognized_student["name"]

                # Draw vibrant green border
                box_color = (34, 197, 94) # BGR Green
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

                # Determine attendance action & cooldown
                last_time = self.student_cooldowns.get(student_id, 0)
                in_cooldown = (now - last_time < COOLDOWN_SECONDS)
                action_status = "RECOGNIZED"

                if not in_cooldown:
                    if mode == "ENTRY":
                        success, msg, rec = database.record_entry(student_id)
                        if success:
                            action_status = "ENTRY RECORDED"
                            self.student_cooldowns[student_id] = now
                        else:
                            action_status = "PRESENT"
                    elif mode == "EXIT":
                        success, msg, rec = database.record_exit(student_id)
                        if success:
                            action_status = "EXIT RECORDED"
                            self.student_cooldowns[student_id] = now
                        else:
                            if "not entered" in msg.lower():
                                action_status = "NOT ENTERED YET"
                            else:
                                action_status = "ALREADY LEFT"
                else:
                    action_status = "RECOGNIZED"

                # Info Pill Overlay above/below face
                pill_height = 68
                pill_y1 = y2 + 6 if y2 + pill_height < h else max(46, y1 - pill_height - 6)
                pill_width = max(230, int(w_box * 1.1))

                # Background card
                cv2.rectangle(annotated, (x1, pill_y1), (x1 + pill_width, pill_y1 + pill_height), (15, 23, 42), -1)
                cv2.rectangle(annotated, (x1, pill_y1), (x1 + pill_width, pill_y1 + pill_height), box_color, 1)

                # Line 1: Student Name
                cv2.putText(annotated, name, (x1 + 10, pill_y1 + 22),
                            cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
                # Line 2: Roll Number
                cv2.putText(annotated, roll_no, (x1 + 10, pill_y1 + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)
                # Line 3: Recognition & Attendance Status
                status_color = (74, 222, 128) if "RECORDED" in action_status or action_status == "RECOGNIZED" else (248, 113, 113)
                cv2.putText(annotated, action_status, (x1 + 10, pill_y1 + 60),
                            cv2.FONT_HERSHEY_DUPLEX, 0.48, status_color, 1, cv2.LINE_AA)

            else:
                # UNKNOWN PERSON
                box_color = (14, 165, 233) # BGR Amber/Yellow
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                pill_y1 = max(45, y1 - 30)
                cv2.rectangle(annotated, (x1, pill_y1), (x1 + 120, pill_y1 + 26), (15, 23, 42), -1)
                cv2.rectangle(annotated, (x1, pill_y1), (x1 + 120, pill_y1 + 26), box_color, 1)
                cv2.putText(annotated, "UNKNOWN", (x1 + 8, pill_y1 + 18),
                            cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

                # Log unknown sighting with cooldown
                if now - self.last_unknown_log_time >= UNKNOWN_COOLDOWN_SECONDS:
                    database.log_activity("unknown", "Unknown face detected by camera")
                    self.last_unknown_log_time = now

        return annotated

# Global service instance
recognition_service = FaceRecognitionService()
