from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import os
import shutil
import sqlite3
import cv2
import subprocess
import sys

from app.database import create_tables
from app.auth import verify_password, create_access_token, SECRET_KEY, ALGORITHM

app = FastAPI()
security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        username = payload.get("sub")

        if not username:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )

        return username

    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables
create_tables()

# Face images folder
FACES_FOLDER = "faces"
os.makedirs(FACES_FOLDER, exist_ok=True)

DATABASE = "fras.db"


# Database connection
def get_connection():
    return sqlite3.connect(DATABASE)

def save_attendance(student_id):

    from datetime import datetime

    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Check today's attendance
    cursor.execute(
        """
        SELECT id FROM attendance
        WHERE student_id = ? AND date = ?
        """,
        (student_id, date)
    )

    existing = cursor.fetchone()

    if existing:
        conn.close()
        return False, date, time

    cursor.execute(
        """
        INSERT INTO attendance
        (student_id, date, time, status)
        VALUES (?, ?, ?, ?)
        """,
        (student_id, date, time, "Present")
    )

    conn.commit()
    conn.close()

    return True, date, time


# Student registration data
class Student(BaseModel):
    student_id: str
    name: str
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(login_data: LoginRequest):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT username, password
        FROM admin_users
        WHERE username = ?
        """,
        (login_data.username,)
    )

    user = cursor.fetchone()
    conn.close()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    username = user[0]
    hashed_password = user[1]

    if not verify_password(login_data.password, hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    token = create_access_token(
        data={"sub": username}
    )

    return {
        "message": "Login successful!",
        "access_token": token,
        "token_type": "bearer"
    }

# Home
@app.get("/")
def home():
    return {
        "message": "Face Recognition Attendance System is running!"
    }


# Register student
@app.post("/register")
def register_student(
    student: Student,
    current_user: str = Depends(get_current_user)
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO students (student_id, name)
        VALUES (?, ?)
        """,
        (student.student_id, student.name)
    )

    conn.commit()
    conn.close()

    return {
        "message": "Student registered successfully!",
        "student_id": student.student_id,
        "name": student.name
    }


# Upload student's face image
@app.post("/register-face")
async def register_face(
    student_id: str = Form(...),
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):

    save_dir = os.path.join(FACES_FOLDER, student_id)

    os.makedirs(save_dir, exist_ok=True)

    # Count existing samples
    existing_files = [
        f for f in os.listdir(save_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    sample_number = len(existing_files) + 1

    filename = f"{student_id}_{sample_number}.jpg"

    file_path = os.path.join(
        save_dir,
        filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Train only after 30 samples
    total_samples = sample_number

    if total_samples >= 30:

        try:

            result = subprocess.run(
                [
                    sys.executable,
                    "app/train_model.py"
                ],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:

                print("Model training failed:")
                print(result.stderr)

                return {
                    "message": "Face samples saved, but model training failed.",
                    "student_id": student_id,
                    "sample": sample_number
                }

            print("Model retrained successfully.")
            print(result.stdout)

            return {
                "message": "30 face samples completed and model trained successfully!",
                "student_id": student_id,
                "sample": sample_number,
                "training": True
            }

        except Exception as e:

            print("Training error:", e)

            return {
                "message": "Face samples saved, but automatic training failed.",
                "student_id": student_id,
                "sample": sample_number
            }

    return {
        "message": f"Face sample {sample_number}/30 saved successfully!",
        "student_id": student_id,
        "sample": sample_number,
        "training": False
    }


# Mark attendance
@app.post("/attendance")
def mark_attendance(
    student_id: str,
    date: str,
    time: str,
    current_user: str = Depends(get_current_user)
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO attendance
        (student_id, date, time, status)
        VALUES (?, ?, ?, ?)
        """,
        (student_id, date, time, "Present")
    )

    conn.commit()
    conn.close()

    return {
        "message": "Attendance marked successfully!",
        "student_id": student_id,
        "date": date,
        "time": time,
        "status": "Present"
    }


# Get all students
@app.get("/students")
def get_students(
    current_user: str = Depends(get_current_user)
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT student_id, name FROM students"
    )

    rows = cursor.fetchall()

    conn.close()

    return [
        {
            "student_id": row[0],
            "name": row[1]
        }
        for row in rows
    ]


# Get all attendance
@app.get("/attendance")
def get_attendance(
    current_user: str = Depends(get_current_user)
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT attendance.student_id,
               students.name,
               attendance.date,
               attendance.time,
               attendance.status
        FROM attendance
        LEFT JOIN students
        ON attendance.student_id = students.student_id
        ORDER BY attendance.id DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    return [
        {
            "student_id": row[0],
            "name": row[1],
            "date": row[2],
            "time": row[3],
            "status": row[4]
        }
        for row in rows
    ]
# Delete student
@app.delete("/students/{student_id}")
def delete_student(
    student_id: str,
    current_user: str = Depends(get_current_user)
    ):

    conn = get_connection()
    cursor = conn.cursor()

    # Check student exists
    cursor.execute(
        "SELECT student_id FROM students WHERE student_id = ?",
        (student_id,)
    )

    student = cursor.fetchone()

    if not student:
        conn.close()
        return {
            "message": "Student not found!"
        }

    # Delete attendance records
    cursor.execute(
        "DELETE FROM attendance WHERE student_id = ?",
        (student_id,)
    )

    # Delete student
    cursor.execute(
        "DELETE FROM students WHERE student_id = ?",
        (student_id,)
    )

    conn.commit()
    conn.close()

    # Delete face samples folder
    face_folder = os.path.join(FACES_FOLDER, student_id)

    if os.path.exists(face_folder):
        shutil.rmtree(face_folder)

    return {
        "message": "Student deleted successfully!",
        "student_id": student_id
    }
    
    # Face recognition endpoint
@app.post("/recognize")
async def recognize_face(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
    ):

    import numpy as np

    # Read uploaded image
    contents = await file.read()
    image_array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {"message": "Invalid image!"}

    # Load trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("models/face_model.yml")

    # Load student mapping
    student_map = {}

    with open("models/student_map.txt", "r") as f:
        for line in f:
            label, student_id = line.strip().split(",")
            student_map[int(label)] = student_id

    # Face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )
    if len(faces) == 0:
        return {
           "recognized": False,
           "student_id": None,
           "message": "Face not detected"
    }
    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        if confidence < 90 and label in student_map:

            student_id = student_map[label]

            marked, date, time = save_attendance(student_id)

            return {
                "recognized": True,
                "student_id": student_id,
                "confidence": round(confidence, 1),
                "attendance_marked": marked,
                "date": date,
                "time": time
          
            }

        return {
        "recognized": False,
        "student_id": None,
        "message": "Unknown face"
        }