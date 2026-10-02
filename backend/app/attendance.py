import cv2
import sqlite3
from datetime import datetime

# Database
DATABASE = "fras.db"

# Face model
MODEL_PATH = "models/face_model.yml"

# Haar Cascade
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Load face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# Load face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Student ID mapping
students = {
    1: "MCA001"
}

# Camera
camera = cv2.VideoCapture(0)

print("Automatic Attendance System started...")
print("Camera ke saamne face lao.")
print("Q press karke band karo.")


def already_marked_today(student_id, date):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id FROM attendance
        WHERE student_id = ?
        AND date = ?
        """,
        (student_id, date)
    )

    result = cursor.fetchone()

    conn.close()

    return result is not None


attendance_marked = False

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera open nahi ho raha!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        if confidence < 80 and label in students:

            student_id = students[label]

            now = datetime.now()

            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            # Check duplicate attendance
            if not attendance_marked:

                if already_marked_today(student_id, date):

                    print("Attendance already marked today!")
                    print("Student:", student_id)

                else:

                    conn = sqlite3.connect(DATABASE)
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

                    print("Attendance marked successfully!")
                    print("Student:", student_id)
                    print("Date:", date)
                    print("Time:", time)

                attendance_marked = True

            text = f"{student_id} - Present"

        else:

            text = "Unknown"

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Automatic Attendance System",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print("Attendance system stopped.")