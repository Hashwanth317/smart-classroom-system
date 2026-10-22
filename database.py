import os
import sqlite3
import json
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(__file__), "database")
DB_PATH = os.path.join(DB_DIR, "classroom.db")

def get_db_connection():
    """Create and return a thread-safe SQLite connection with foreign keys enabled."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initialize database tables and seed default status if not already present."""
    os.makedirs(DB_DIR, exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Students table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                embedding TEXT NOT NULL,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. Attendance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                entry_time TEXT,
                exit_time TEXT,
                status TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
                UNIQUE(student_id, date)
            );
        """)

        # 3. Activity log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 4. Classroom status table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS classroom_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occupancy INTEGER DEFAULT 0,
                light_status INTEGER DEFAULT 0,
                fan_status INTEGER DEFAULT 0,
                pir_status INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Seed initial classroom status record if empty
        cursor.execute("SELECT COUNT(*) FROM classroom_status WHERE id = 1;")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO classroom_status (id, occupancy, light_status, fan_status, pir_status, updated_at)
                VALUES (1, 0, 0, 0, 0, datetime('now', 'localtime'));
            """)

        conn.commit()

# --- STUDENT HELPERS ---

def student_exists(roll_no):
    """Check if a student roll number is already registered."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM students WHERE UPPER(roll_no) = UPPER(?);", (roll_no.strip(),))
        return cursor.fetchone() is not None

def add_student(roll_no, name, embedding_list):
    """Add a new student with serialized embedding."""
    roll_no_clean = roll_no.strip().upper()
    name_clean = name.strip()
    embedding_json = json.dumps(embedding_list)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO students (roll_no, name, embedding, registered_at)
            VALUES (?, ?, ?, datetime('now', 'localtime'));
        """, (roll_no_clean, name_clean, embedding_json))
        student_id = cursor.lastrowid
        conn.commit()
    
    log_activity("registration", f"Student registered: {name_clean} ({roll_no_clean})")
    return student_id

def get_all_students():
    """Retrieve list of all registered students."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, roll_no, name, registered_at FROM students ORDER BY roll_no ASC;")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_all_students_with_embeddings():
    """Retrieve all students along with their parsed embedding vectors."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, roll_no, name, embedding FROM students;")
        rows = cursor.fetchall()
        students = []
        for r in rows:
            try:
                emb = json.loads(r["embedding"])
            except Exception:
                emb = []
            students.append({
                "id": r["id"],
                "roll_no": r["roll_no"],
                "name": r["name"],
                "embedding": emb
            })
        return students

def get_student_by_roll(roll_no):
    """Retrieve student details by roll number."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, roll_no, name, registered_at FROM students WHERE UPPER(roll_no) = UPPER(?);", (roll_no.strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None

def delete_student(roll_no):
    """Delete a student and related attendance records."""
    roll_no_clean = roll_no.strip().upper()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM students WHERE UPPER(roll_no) = ?;", (roll_no_clean,))
        row = cursor.fetchone()
        student_name = row["name"] if row else roll_no_clean

        cursor.execute("DELETE FROM students WHERE UPPER(roll_no) = ?;", (roll_no_clean,))
        deleted = cursor.rowcount > 0
        conn.commit()

    if deleted:
        log_activity("registration", f"Student removed: {student_name} ({roll_no_clean})")
    return deleted

# --- ATTENDANCE HELPERS ---

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

def get_current_time():
    return datetime.now().strftime("%I:%M:%S %p")

def record_entry(student_id):
    """Record student entry for today. Returns (success, message, record)."""
    today = get_current_date()
    now_time = get_current_time()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance WHERE student_id = ? AND date = ?;", (student_id, today))
        record = cursor.fetchone()

        if record:
            # Already marked entry
            return False, f"Already recorded entry at {record['entry_time']}", dict(record)

        cursor.execute("""
            INSERT INTO attendance (student_id, date, entry_time, exit_time, status)
            VALUES (?, ?, ?, NULL, 'Present');
        """, (student_id, today, now_time))
        conn.commit()

        cursor.execute("SELECT roll_no, name FROM students WHERE id = ?;", (student_id,))
        student = cursor.fetchone()
        name = student["name"] if student else "Student"
        roll = student["roll_no"] if student else ""

    log_activity("entry", f"{name} ({roll}) entered classroom at {now_time}")
    return True, f"Entry recorded at {now_time}", {"student_id": student_id, "date": today, "entry_time": now_time, "status": "Present"}

def record_exit(student_id):
    """Record student exit for today. Returns (success, message, record)."""
    today = get_current_date()
    now_time = get_current_time()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance WHERE student_id = ? AND date = ?;", (student_id, today))
        record = cursor.fetchone()

        if not record or not record["entry_time"]:
            return False, "Cannot record exit - student has not entered yet today.", None

        if record["status"] == "Left":
            return False, f"Already recorded exit at {record['exit_time']}", dict(record)

        cursor.execute("""
            UPDATE attendance
            SET exit_time = ?, status = 'Left'
            WHERE student_id = ? AND date = ?;
        """, (now_time, student_id, today))
        conn.commit()

        cursor.execute("SELECT roll_no, name FROM students WHERE id = ?;", (student_id,))
        student = cursor.fetchone()
        name = student["name"] if student else "Student"
        roll = student["roll_no"] if student else ""

    log_activity("exit", f"{name} ({roll}) exited classroom at {now_time}")
    return True, f"Exit recorded at {now_time}", {"student_id": student_id, "date": today, "exit_time": now_time, "status": "Left"}

def get_today_attendance(search_query=None):
    """Retrieve full attendance list for today with student metadata and optional search filter."""
    today = get_current_date()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Left join students with today's attendance so all registered students are listed
        query = """
            SELECT 
                s.id as student_id,
                s.roll_no,
                s.name,
                a.id as attendance_id,
                COALESCE(a.date, ?) as date,
                a.entry_time,
                a.exit_time,
                COALESCE(a.status, 'Not Entered') as status
            FROM students s
            LEFT JOIN attendance a ON s.id = a.student_id AND a.date = ?
        """
        params = [today, today]

        if search_query:
            query += " WHERE UPPER(s.roll_no) LIKE ? OR UPPER(s.name) LIKE ?"
            like_val = f"%{search_query.strip().upper()}%"
            params.extend([like_val, like_val])

        query += " ORDER BY CASE WHEN a.status = 'Present' THEN 1 WHEN a.status = 'Left' THEN 2 ELSE 3 END, s.roll_no ASC;"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_present_students_count():
    """Count of students currently present in class today (entered and not yet left)."""
    today = get_current_date()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM attendance 
            WHERE date = ? AND status = 'Present';
        """, (today,))
        return cursor.fetchone()[0]

# --- ACTIVITY LOG HELPERS ---

def log_activity(event_type, message):
    """Record an event into the activity log."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO activity (event_type, message, timestamp)
            VALUES (?, ?, datetime('now', 'localtime'));
        """, (event_type, message))
        conn.commit()

def get_recent_activities(limit=10):
    """Retrieve the most recent activity log entries."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, event_type, message, timestamp
            FROM activity
            ORDER BY id DESC
            LIMIT ?;
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

# --- CLASSROOM / HARDWARE STATUS HELPERS ---

def get_classroom_status():
    """Retrieve the current classroom hardware telemetry state."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM classroom_status WHERE id = 1;")
        row = cursor.fetchone()
        if not row:
            return {"occupancy": 0, "light_status": 0, "fan_status": 0, "pir_status": 0, "updated_at": ""}
        return dict(row)

def update_classroom_status(occupancy=None, light_status=None, fan_status=None, pir_status=None):
    """Update hardware status received from ESP32."""
    current = get_classroom_status()
    new_occ = int(occupancy) if occupancy is not None else current["occupancy"]
    new_light = int(1 if light_status else 0) if light_status is not None else current["light_status"]
    new_fan = int(1 if fan_status else 0) if fan_status is not None else current["fan_status"]
    new_pir = int(1 if pir_status else 0) if pir_status is not None else current["pir_status"]

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE classroom_status
            SET occupancy = ?, light_status = ?, fan_status = ?, pir_status = ?, updated_at = datetime('now', 'localtime')
            WHERE id = 1;
        """, (new_occ, new_light, new_fan, new_pir))
        conn.commit()

    # Log significant status changes
    status_summary = f"Hardware update: Occupancy={new_occ}, Light={'ON' if new_light else 'OFF'}, Fan={'ON' if new_fan else 'OFF'}"
    log_activity("hardware", status_summary)

    return get_classroom_status()
