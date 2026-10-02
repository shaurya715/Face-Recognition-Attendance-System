import cv2

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/face_model.yml")


# Load student mapping
student_map = {}

with open("models/student_map.txt", "r") as file:
    for line in file:
        label, student_id = line.strip().split(",")
        student_map[int(label)] = student_id


# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# Camera
cap = cv2.VideoCapture(0)

print("Face recognition started...")
print("Camera ke saamne face lao.")
print("Q press karke band karo.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera open nahi hua.")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        if confidence < 80 and label in student_map:

            student_id = student_map[label]

            text = f"{student_id} - {confidence:.1f}"

        else:

            text = "Unknown"


        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )


    cv2.imshow(
        "FRAS - Face Recognition",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()