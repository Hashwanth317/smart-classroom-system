# Smart Classroom Attendance & Occupancy Management System

**Automated Attendance and Energy Management System**  
*Developed by Team Tech Nova*

A complete, locally hosted Python/Flask application running on macOS. The system integrates laptop webcam face recognition (using pre-trained FaceNet via DeepFace and multi-factor anti-spoofing) with an automated SQLite attendance pipeline, coupled with an API bridge ready for ESP32 hardware occupancy, light, and fan relay control.

---

## 🏗️ Architecture Overview

```text
Laptop Webcam
      │
      ▼
Face Detection (OpenCV)
      │
      ▼
Anti-Spoofing & Liveness Check (Blink & Micro-motion Verification)
      │
      ▼
FaceNet Embeddings (128-D Cosine Matching)
      │
      ▼
Student Identified & Cooldown Verified
      │
      ▼
SQLite Attendance Record (ENTRY / EXIT Modes)
      │
      ▼
Flask Real-time API
      │
      ▼
HTML/CSS/JS College Dashboard

─────────────────────────────────────────────────────────────────

IR Sensors ──► ESP32 Logic ──► POST /api/hardware/update ──► SQLite ──► Dashboard
(Occupancy Counting & Relays)                                          (Energy Cards)
```

---

## 📋 Features

1. **Student Face Registration (`/register`)**:
   - Collects 25 multi-pose face samples with on-screen guidance (Front, Slight Left, Slight Right, Slight Up, Slight Down).
   - Rejects duplicate roll numbers, multiple faces, and blurry frames.
   - Extracts normalized 128-D FaceNet embeddings, averages them, and saves them to SQLite.
2. **Attendance Camera (`/video_feed`)**:
   - Real-time video stream with face bounding boxes and status labels (`RECOGNIZED`, `UNKNOWN`, `Possible Spoof Attempt`).
3. **Anti-Spoofing & Liveness Verification**:
   - Analyzes eye blinking, natural physiological micro-movement, and texture variance.
   - Rejects static printed photos and phone screen images.
   - Graceful fallback mode ensures zero crashes.
4. **Attendance Entry / Exit Logic**:
   - **Entry Mode**: Marks `Present` with exact entry timestamp.
   - **Exit Mode**: Marks `Left` with exact exit timestamp (rejects exit if the student hasn't entered).
   - 25-second cooldown prevents repeated attendance entries every frame.
5. **ESP32 Energy Management API Bridge (`POST /api/hardware/update`)**:
   - Ingests occupancy count, light status, fan status, and PIR motion state.
   - Dynamically updates dashboard energy cards.
   - Built-in UI simulator panel allows testing hardware telemetry right from the dashboard.
6. **Live Dashboard (`/`)**:
   - Metric cards: Students Present, Current Occupancy, Classroom Status, Light, Fan, Camera, Face Recognition.
   - Searchable today's attendance table (search by roll number or student name).
   - Real-time recent activity log (last 10 events).

---

## 💻 Tech Stack

- **Backend**: Python 3.11+, Flask
- **Database**: SQLite3 (`database/classroom.db`)
- **Computer Vision**: OpenCV (`cv2`), DeepFace (FaceNet ONNX 128-D)
- **Anti-Spoofing**: Haar Cascades & Multi-frame Motion / Texture Analysis
- **Frontend**: HTML5, CSS3, Vanilla JavaScript (No React, no external cloud dependencies)

---

## 🚀 Setup & Execution on macOS

### 1. Clone or Open Project Directory

```bash
cd "/Users/hashwanth/Downloads/smart clasroom"
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

### 5. Open in Web Browser

Navigate to:
```
http://127.0.0.1:5001
```
*(or `http://127.0.0.1:5000` if AirPlay Receiver is disabled in macOS System Settings)*

---

## 📖 User Workflow Guide

### 1. Register Your First Student
1. Open the dashboard at `http://127.0.0.1:5001` and click **Student Face Registration** (or visit `http://127.0.0.1:5001/register`).
2. Enter the student's **Roll Number** (e.g., `23CSR001`) and **Student Name** (e.g., `Arun Kumar`).
3. Click **Start Face Registration**.
4. Position your face in front of your laptop webcam.
5. Follow the live pose prompts on screen:
   - Samples 1–5: Look straight ahead (Front)
   - Samples 6–10: Turn head slightly Left
   - Samples 11–15: Turn head slightly Right
   - Samples 16–20: Tilt head slightly Up
   - Samples 21–25: Tilt head slightly Down
6. Once 25 valid samples are gathered, the system averages the FaceNet embeddings and saves the student to SQLite.
7. You will see **"Registration Completed Successfully"**, and the student appears in the Registered Students Directory below.

### 2. Operating Entry Mode
1. On the main dashboard (`http://127.0.0.1:5000`), click the **Entry Mode** button (active by default).
2. The webcam stream displays `[ ATTENDANCE CAMERA | ENTRY MODE ACTIVE ]`.
3. When the registered student steps in front of the camera:
   - Anti-spoofing verifies live presence.
   - FaceNet identifies the student.
   - A green bounding box displays `Arun Kumar`, `23CSR001`, and `ENTRY RECORDED`.
   - In the attendance table, the status becomes **Present** and the entry time is logged (e.g., `09:04:32 AM`).
   - If the student remains in view, a 25-second cooldown prevents repeated duplicate logs.

### 3. Operating Exit Mode
1. Click the **Exit Mode** button on the dashboard.
2. The top bar switches to purple: `[ ATTENDANCE CAMERA | EXIT MODE ACTIVE ]`.
3. When a student who entered earlier faces the camera:
   - Green bounding box displays `Arun Kumar`, `23CSR001`, and `EXIT RECORDED`.
   - The attendance status updates to **Left** with the recorded exit time.
   - *Note*: If a student who has not entered attempts to exit, the system displays `NOT ENTERED YET` and rejects the exit record.

---

## 🔌 ESP32 Hardware Integration API

Your ESP32 hardware (with IR sensors and light/fan relays) can send HTTP POST requests directly over Wi-Fi to your laptop's IP address on port 5000:

### Endpoint:
```http
POST /api/hardware/update
Content-Type: application/json
```

### JSON Payload:
```json
{
  "occupancy": 4,
  "lightStatus": true,
  "fanStatus": true,
  "pirStatus": true
}
```

### Expected JSON Response:
```json
{
  "success": true,
  "message": "ESP32 classroom hardware status updated successfully",
  "data": {
    "occupancy": 4,
    "classroom_status": "OCCUPIED",
    "light_status": "ON",
    "fan_status": "ON",
    "pir_status": "ACTIVE",
    "updated_at": "2026-10-22 09:30:00"
  }
}
```

*(You can also test this endpoint directly using the **ESP32 Energy Management API Bridge** console at the bottom of the web dashboard).*

---

## 🧪 Testing

Run the automated verification suite anytime:

```bash
python test_app.py
```

This verifies:
- Database auto-creation and schema integrity
- Student registration and duplicate roll number rejection
- Attendance entry and exit rules
- Anti-spoofing and liveness evaluation
- ESP32 hardware telemetry endpoint
- Cosine distance matching thresholds
