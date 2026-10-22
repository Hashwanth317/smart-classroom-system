import os
import json
import unittest
import numpy as np

# Ensure environment variables are set before imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["DEEPFACE_HOME"] = os.path.join(BASE_DIR, ".deepface")
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import database
from app import app
from liveness import LivenessDetector
from face_recognition_service import recognition_service, cosine_distance, MATCH_THRESHOLD

class SmartClassroomTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        database.init_db()

    def test_01_database_init(self):
        """Verify database tables and initial status."""
        status = database.get_classroom_status()
        self.assertIsNotNone(status)
        self.assertIn("occupancy", status)
        self.assertIn("light_status", status)
        self.assertIn("fan_status", status)

    def test_02_student_registration_and_duplicate_protection(self):
        """Test student registration and duplicate roll number protection."""
        test_roll = "TEST999"
        test_name = "Alex Morgan"
        
        # Clean up any leftover test data
        database.delete_student(test_roll)

        # 1. Register student
        res = self.app.post("/api/students/register", json={"roll_no": test_roll, "name": test_name})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])

        # 2. Add fake embedding directly to database to simulate completed registration
        dummy_emb = [0.05] * 128
        database.add_student(test_roll, test_name, dummy_emb)
        recognition_service.reload_students()

        # 3. Try to register with the exact same roll number -> should be rejected
        res_dup = self.app.post("/api/students/register", json={"roll_no": test_roll, "name": "Duplicate Student"})
        self.assertEqual(res_dup.status_code, 409)
        data_dup = res_dup.get_json()
        self.assertFalse(data_dup["success"])
        self.assertIn("already registered", data_dup["error"].lower())

    def test_03_attendance_entry_and_exit_logic(self):
        """Test entry and exit logic, duplicate entry prevention, and exit without entry rejection."""
        test_roll = "ATT001"
        test_name = "Test Attendee"
        database.delete_student(test_roll)
        
        # Register student
        student_id = database.add_student(test_roll, test_name, [0.01]*128)
        
        # Test exit BEFORE entry -> Must fail
        res_exit_early = self.app.post("/api/attendance/exit", json={"roll_no": test_roll})
        self.assertEqual(res_exit_early.status_code, 200)
        data_exit_early = res_exit_early.get_json()
        self.assertFalse(data_exit_early["success"])
        self.assertIn("not entered", data_exit_early["message"].lower())

        # Test valid entry
        res_entry = self.app.post("/api/attendance/entry", json={"roll_no": test_roll})
        self.assertEqual(res_entry.status_code, 200)
        data_entry = res_entry.get_json()
        self.assertTrue(data_entry["success"])
        self.assertEqual(data_entry["record"]["status"], "Present")

        # Test duplicate entry on the same day -> Must not duplicate
        res_entry_dup = self.app.post("/api/attendance/entry", json={"roll_no": test_roll})
        self.assertEqual(res_entry_dup.status_code, 200)
        data_entry_dup = res_entry_dup.get_json()
        self.assertFalse(data_entry_dup["success"])
        self.assertIn("already recorded entry", data_entry_dup["message"].lower())

        # Test valid exit
        res_exit = self.app.post("/api/attendance/exit", json={"roll_no": test_roll})
        self.assertEqual(res_exit.status_code, 200)
        data_exit = res_exit.get_json()
        self.assertTrue(data_exit["success"])
        self.assertEqual(data_exit["record"]["status"], "Left")

        # Cleanup
        database.delete_student(test_roll)

    def test_04_hardware_update_endpoint(self):
        """Test POST /api/hardware/update endpoint with ESP32 inCount, outCount, and roomCount."""
        payload = {
            "inCount": 10,
            "outCount": 2,
            "roomCount": 8,
            "lightStatus": True,
            "fanStatus": True
        }
        res = self.app.post("/api/hardware/update", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["roomCount"], 8)
        self.assertEqual(data["data"]["inCount"], 10)
        self.assertEqual(data["data"]["outCount"], 2)
        self.assertEqual(data["data"]["classroom_status"], "OCCUPIED")
        self.assertEqual(data["data"]["light_status"], "ON")
        self.assertEqual(data["data"]["fan_status"], "ON")

    def test_05_dashboard_and_activity_api(self):
        """Test GET /api/dashboard and GET /api/activity."""
        res_dash = self.app.get("/api/dashboard")
        self.assertEqual(res_dash.status_code, 200)
        dash_data = res_dash.get_json()
        self.assertTrue(dash_data["success"])
        self.assertIn("metrics", dash_data)
        self.assertIn("attendance", dash_data)
        self.assertIn("activity", dash_data)

        res_act = self.app.get("/api/activity")
        self.assertEqual(res_act.status_code, 200)
        act_data = res_act.get_json()
        self.assertTrue(act_data["success"])
        self.assertIsInstance(act_data["activity"], list)

    def test_06_attendance_mode_switching(self):
        """Test toggling ENTRY and EXIT modes."""
        res_exit = self.app.post("/api/attendance/mode", json={"mode": "EXIT"})
        self.assertEqual(res_exit.status_code, 200)
        self.assertEqual(res_exit.get_json()["current_mode"], "EXIT")

        res_entry = self.app.post("/api/attendance/mode", json={"mode": "ENTRY"})
        self.assertEqual(res_entry.status_code, 200)
        self.assertEqual(res_entry.get_json()["current_mode"], "ENTRY")

    def test_07_liveness_detector(self):
        """Test anti-spoofing and liveness detector behavior."""
        det = LivenessDetector()
        # Test on synthetic frame
        synthetic_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        res = det.verify_liveness(synthetic_frame, (100, 100, 120, 120))
        self.assertIn("is_live", res)
        self.assertIn("confidence", res)
        self.assertIn("reason", res)

    def test_08_cosine_matching(self):
        """Verify vector cosine distance calculation."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        
        # Identical vectors should have distance 0.0
        self.assertAlmostEqual(cosine_distance(v1, v2), 0.0, places=4)
        # Orthogonal vectors should have distance 1.0
        self.assertAlmostEqual(cosine_distance(v1, v3), 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
