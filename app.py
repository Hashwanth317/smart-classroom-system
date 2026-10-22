import os
import cv2
import time
import json
import base64
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

# Ensure offline DeepFace directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["DEEPFACE_HOME"] = os.path.join(BASE_DIR, ".deepface")
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import database
from camera import camera_manager
from face_recognition_service import recognition_service
from face_registration import RegistrationSession, sanitize_filename

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Auto-initialize database on startup
database.init_db()

# Active registration session holder
active_reg_session = None

# ==================================================
# WEB PAGES
# ==================================================

@app.route('/')
def index():
    """Render the main Smart Classroom Dashboard."""
    return render_template('index.html')

@app.route('/register')
def register_page():
    """Render the Student Face Registration Page."""
    return render_template('register.html')

# ==================================================
# VIDEO STREAMING ENDPOINTS
# ==================================================

def generate_attendance_stream():
    """Continuously processes raw camera frames through face recognition & anti-spoofing."""
    while True:
        if not camera_manager.is_online():
            placeholder = camera_manager._generate_placeholder_frame("Camera Offline - Click 'Start Camera'")
            ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.3)
            continue

        raw_frame = camera_manager.get_raw_frame()
        if raw_frame is None:
            time.sleep(0.04)
            continue

        annotated_frame = recognition_service.process_and_annotate_frame(raw_frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            time.sleep(0.04)
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04) # ~25 FPS

@app.route('/video_feed')
def video_feed():
    """MJPEG streaming route for live attendance camera."""
    return Response(generate_attendance_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_registration_stream():
    """Video stream with overlay for interactive student registration."""
    global active_reg_session
    while True:
        if not camera_manager.is_online():
            placeholder = camera_manager._generate_placeholder_frame("Camera Offline - Start Camera to Register")
            ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.3)
            continue

        raw_frame = camera_manager.get_raw_frame()
        if raw_frame is None:
            time.sleep(0.04)
            continue

        annotated = raw_frame.copy()
        h, w = annotated.shape[:2]

        # Overlay registration guidance
        if active_reg_session and active_reg_session.status == "in_progress":
            reg_info = active_reg_session.process_frame(raw_frame)
            
            # Draw top banner
            cv2.rectangle(annotated, (0, 0), (w, 42), (37, 99, 235), -1)
            cv2.putText(annotated, f"REGISTERING: {active_reg_session.name} ({active_reg_session.roll_no})",
                        (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw face bounding box if detected
            box = reg_info.get("face_box")
            if box:
                bx, by, bw, bh = box
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (34, 197, 94), 2)

            # Draw bottom guidance banner
            cv2.rectangle(annotated, (0, h - 50), (w, h), (15, 23, 42), -1)
            guidance = reg_info.get("current_pose", "")
            prog = reg_info.get("progress", "")
            cv2.putText(annotated, f"Pose: {guidance}", (16, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, prog, (w - 220, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (74, 222, 128), 2, cv2.LINE_AA)

            if reg_info.get("status") == "processing_embeddings":
                # Automatically trigger finalization
                active_reg_session.finalize_registration()
                recognition_service.reload_students()
        else:
            # Standby preview
            cv2.rectangle(annotated, (0, 0), (w, 36), (71, 85, 105), -1)
            cv2.putText(annotated, "REGISTRATION PREVIEW - Enter Roll No & Name to Begin",
                        (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)

@app.route('/registration_feed')
def registration_feed():
    """MJPEG streaming route for registration page preview."""
    return Response(generate_registration_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==================================================
# API ENDPOINTS
# ==================================================

@app.route('/api/students', methods=['GET'])
def get_students():
    """List all registered students."""
    students = database.get_all_students()
    return jsonify({"success": True, "count": len(students), "students": students})

@app.route('/api/students/register', methods=['POST'])
def start_registration():
    """Start face capture registration session for a new student."""
    global active_reg_session
    data = request.get_json(silent=True) or {}
    roll_no = data.get("roll_no", "").strip()
    name = data.get("name", "").strip()

    if not roll_no or not name:
        return jsonify({"success": False, "error": "Both Roll Number and Student Name are required."}), 400

    # Rule 1: Duplicate roll number protection
    if database.student_exists(roll_no):
        return jsonify({
            "success": False,
            "error": f"Roll number '{roll_no.upper()}' is already registered. Please check or use admin delete."
        }), 409

    # Ensure camera is running
    camera_manager.start()

    # Create new active registration session
    active_reg_session = RegistrationSession(roll_no=roll_no, name=name, target_samples=25)
    return jsonify({
        "success": True,
        "message": f"Face registration started for {name} ({roll_no.upper()}). Please face the camera.",
        "roll_no": roll_no.upper(),
        "name": name,
        "target_samples": 25
    })

@app.route('/api/students/registration_status', methods=['GET'])
def get_registration_status():
    """Poll progress for the active registration session."""
    global active_reg_session
    if not active_reg_session:
        return jsonify({
            "active": False,
            "status": "idle",
            "message": "No registration currently active."
        })

    is_completed = (active_reg_session.status == "completed")
    res = {
        "active": True,
        "roll_no": active_reg_session.roll_no,
        "name": active_reg_session.name,
        "status": active_reg_session.status,
        "count": len(active_reg_session.samples),
        "total": active_reg_session.target_samples,
        "progress": f"{len(active_reg_session.samples)} / {active_reg_session.target_samples} images captured",
        "message": active_reg_session.message,
        "current_pose": active_reg_session.get_current_guidance()
    }

    if is_completed:
        # Reset session after completion
        active_reg_session = None

    return jsonify(res)

@app.route('/api/students/process_registration_frame', methods=['POST'])
def process_registration_frame():
    """Process webcam frame sent directly from browser for student registration."""
    global active_reg_session
    if not active_reg_session or active_reg_session.status != "in_progress":
        return jsonify({"success": False, "status": "idle", "message": "No active registration session."})

    data = request.get_json(silent=True) or {}
    img_data = data.get("image", "")
    if not img_data:
        return jsonify({"success": False, "error": "No image payload"}), 400

    try:
        if "," in img_data:
            img_data = img_data.split(",")[1]
        raw_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "error": "Image decode failed"}), 400

        # Process through active registration session
        reg_info = active_reg_session.process_frame(frame)

        if reg_info.get("status") == "processing_embeddings":
            active_reg_session.finalize_registration()
            recognition_service.reload_students()
            reg_info["status"] = "completed"
            reg_info["message"] = "Registration Completed Successfully"

        reg_info["success"] = True
        return jsonify(reg_info)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/attendance/process_frame', methods=['POST'])
def process_attendance_frame():
    """Process live camera frame sent directly from browser for attendance and anti-spoofing."""
    data = request.get_json(silent=True) or {}
    img_data = data.get("image", "")
    if not img_data:
        return jsonify({"success": False, "error": "No image payload"}), 400

    try:
        if "," in img_data:
            img_data = img_data.split(",")[1]
        raw_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "error": "Image decode failed"}), 400

        annotated = recognition_service.process_and_annotate_frame(frame)
        ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_out = ""
        if ret:
            b64_out = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

        return jsonify({
            "success": True,
            "mode": recognition_service.get_mode(),
            "annotated_frame": b64_out
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/students/<roll_no>', methods=['DELETE'])
def remove_student(roll_no):
    """Admin endpoint to delete a registered student."""
    deleted = database.delete_student(roll_no)
    if deleted:
        recognition_service.reload_students()
        return jsonify({"success": True, "message": f"Student {roll_no.upper()} removed successfully."})
    return jsonify({"success": False, "error": f"Student {roll_no.upper()} not found."}), 404

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    """Retrieve today's attendance table, supports ?search= query parameter."""
    search_q = request.args.get('search', '').strip()
    attendance_data = database.get_today_attendance(search_query=search_q if search_q else None)
    return jsonify({"success": True, "count": len(attendance_data), "attendance": attendance_data})

@app.route('/api/attendance/entry', methods=['POST'])
def manual_entry():
    """Record student entry manually or via API."""
    data = request.get_json(silent=True) or {}
    roll_no = data.get("roll_no", "").strip()
    student = database.get_student_by_roll(roll_no)
    if not student:
        return jsonify({"success": False, "error": "Student not found"}), 404

    success, msg, rec = database.record_entry(student["id"])
    return jsonify({"success": success, "message": msg, "record": rec})

@app.route('/api/attendance/exit', methods=['POST'])
def manual_exit():
    """Record student exit manually or via API."""
    data = request.get_json(silent=True) or {}
    roll_no = data.get("roll_no", "").strip()
    student = database.get_student_by_roll(roll_no)
    if not student:
        return jsonify({"success": False, "error": "Student not found"}), 404

    success, msg, rec = database.record_exit(student["id"])
    return jsonify({"success": success, "message": msg, "record": rec})

@app.route('/api/attendance/mode', methods=['POST'])
def set_attendance_mode():
    """Toggle between ENTRY and EXIT mode."""
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "ENTRY")
    success, msg = recognition_service.set_mode(mode)
    return jsonify({"success": success, "message": msg, "current_mode": recognition_service.get_mode()})

@app.route('/api/activity', methods=['GET'])
def get_activity():
    """Retrieve the 10 most recent activity logs."""
    limit = int(request.args.get('limit', 10))
    activities = database.get_recent_activities(limit=limit)
    return jsonify({"success": True, "activity": activities})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Current system status."""
    hw_status = database.get_classroom_status()
    students_present = database.get_present_students_count()
    total_registered = len(database.get_all_students())
    
    return jsonify({
        "success": True,
        "students_present": students_present,
        "total_registered": total_registered,
        "occupancy": hw_status.get("occupancy", 0),
        "classroom_status": "OCCUPIED" if hw_status.get("occupancy", 0) > 0 else "VACANT",
        "light_status": "ON" if hw_status.get("light_status") == 1 else "OFF",
        "fan_status": "ON" if hw_status.get("fan_status") == 1 else "OFF",
        "pir_status": "ACTIVE" if hw_status.get("pir_status") == 1 else "INACTIVE",
        "camera_status": "ONLINE" if camera_manager.is_online() else "OFFLINE",
        "face_recognition_status": "ACTIVE" if camera_manager.is_online() else "STANDBY",
        "attendance_mode": recognition_service.get_mode(),
        "hardware_updated_at": hw_status.get("updated_at", "")
    })

@app.route('/api/hardware/update', methods=['POST'])
def update_hardware():
    """
    ESP32 Hardware Telemetry API:
    Receives JSON from ESP32 with occupancy count and relay states.
    Example payload:
    {
        "occupancy": 4,
        "lightStatus": true,
        "fanStatus": true,
        "pirStatus": true
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid or missing JSON payload"}), 400

    # Accommodate both camelCase and snake_case keys
    occupancy = data.get("occupancy")
    light_status = data.get("lightStatus", data.get("light_status"))
    fan_status = data.get("fanStatus", data.get("fan_status"))
    pir_status = data.get("pirStatus", data.get("pir_status"))

    updated = database.update_classroom_status(
        occupancy=occupancy,
        light_status=light_status,
        fan_status=fan_status,
        pir_status=pir_status
    )

    return jsonify({
        "success": True,
        "message": "ESP32 classroom hardware status updated successfully",
        "data": {
            "occupancy": updated["occupancy"],
            "classroom_status": "OCCUPIED" if updated["occupancy"] > 0 else "VACANT",
            "light_status": "ON" if updated["light_status"] == 1 else "OFF",
            "fan_status": "ON" if updated["fan_status"] == 1 else "OFF",
            "pir_status": "ACTIVE" if updated["pir_status"] == 1 else "INACTIVE",
            "updated_at": updated["updated_at"]
        }
    })

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start laptop webcam."""
    camera_manager.start()
    return jsonify({"success": True, "camera_status": "ONLINE"})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop laptop webcam."""
    camera_manager.stop()
    return jsonify({"success": True, "camera_status": "OFFLINE"})

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """Consolidated endpoint for dashboard real-time polling."""
    hw_status = database.get_classroom_status()
    students_present = database.get_present_students_count()
    attendance_list = database.get_today_attendance()
    recent_activity = database.get_recent_activities(limit=10)

    return jsonify({
        "success": True,
        "metrics": {
            "students_present": students_present,
            "occupancy": hw_status.get("occupancy", 0),
            "classroom_status": "OCCUPIED" if hw_status.get("occupancy", 0) > 0 else "VACANT",
            "light_status": "ON" if hw_status.get("light_status") == 1 else "OFF",
            "fan_status": "ON" if hw_status.get("fan_status") == 1 else "OFF",
            "pir_status": "ACTIVE" if hw_status.get("pir_status") == 1 else "INACTIVE",
            "camera_status": "ONLINE" if camera_manager.is_online() else "OFFLINE",
            "face_recognition_status": "ACTIVE" if camera_manager.is_online() else "STANDBY",
            "attendance_mode": recognition_service.get_mode(),
            "hardware_updated_at": hw_status.get("updated_at", "")
        },
        "attendance": attendance_list,
        "activity": recent_activity
    })

def find_available_port(preferred_port=5001):
    import socket
    for p in [preferred_port, 5000, 5050, 8080]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('0.0.0.0', p))
            s.close()
            return p
        except OSError:
            continue
    return preferred_port

if __name__ == '__main__':
    # Initialize camera at startup
    camera_manager.start()
    port = int(os.environ.get("PORT", find_available_port(5000)))
    print(f"\n=======================================================")
    print(f" [Smart Classroom] Application Live:")
    print(f" http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
