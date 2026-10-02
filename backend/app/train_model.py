import cv2
import os
import numpy as np

# Training data folder
data_path = "faces"

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Training data
faces = []
labels = []

# Student ID mapping
student_map = {}

print("Loading face samples...")

# Get all student folders
student_folders = sorted(os.listdir(data_path))

label_id = 1

for student_id in student_folders:

    student_path = os.path.join(
        data_path,
        student_id
    )

    # Only process folders
    if not os.path.isdir(student_path):
        continue

    student_map[label_id] = student_id

    print(f"Loading samples for: {student_id}")

    for filename in os.listdir(student_path):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            student_path,
            filename
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        # Detect face
        detected_faces = face_cascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(detected_faces) == 0:
            print(
                f"Face not detected: {filename}"
            )
            continue

        # Use the largest detected face
        x, y, w, h = max(
            detected_faces,
            key=lambda rect: rect[2] * rect[3]
        )

        face_crop = image[
            y:y+h,
            x:x+w
        ]

        # Resize face for consistent training
        face_crop = cv2.resize(
            face_crop,
            (200, 200)
        )

        faces.append(face_crop)
        labels.append(label_id)

    label_id += 1


print(
    f"Loaded {len(faces)} detected face samples."
)

if len(faces) == 0:
    print("No face samples found!")
    exit()


print("Student mapping:")
print(student_map)

print("Training model...")

# LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

recognizer.train(
    faces,
    np.array(labels)
)

# Create models folder
os.makedirs(
    "models",
    exist_ok=True
)

# Save model
recognizer.save(
    "models/face_model.yml"
)

# Save student mapping
with open(
    "models/student_map.txt",
    "w"
) as file:

    for label, student_id in student_map.items():

        file.write(
            f"{label},{student_id}\n"
        )


print(
    "Training completed successfully!"
)

print(
    "Model saved at: models/face_model.yml"
)

print(
    "Student mapping saved at: models/student_map.txt"
)