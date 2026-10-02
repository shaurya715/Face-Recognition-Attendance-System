import cv2
import os
import sys

# Student ID command line se lena
if len(sys.argv) < 2:
    print("Student ID provide karo.")
    print("Example: python app\\collect_faces.py MCA002")
    sys.exit()

student_id = sys.argv[1]

# Folder
save_dir = os.path.join("faces", student_id)
os.makedirs(save_dir, exist_ok=True)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Camera
cap = cv2.VideoCapture(0)

count = 0

print("Camera starting...")
print("Student ID:", student_id)
print("Face camera ke saamne rakho.")
print("Q press karke stop kar sakte ho.")

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

        count += 1

        # Face crop
        face = gray[y:y+h, x:x+w]

        # Save sample
        filename = os.path.join(
            save_dir,
            f"{student_id}_{count}.jpg"
        )

        cv2.imwrite(
            filename,
            face
        )

        # Rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {count}/30",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "Collect Face Samples",
        frame
    )

    # 30 samples complete
    if count >= 30:

        print(
            "30 face samples collected successfully!"
        )

        break

    # Q press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print(
    f"Saved {count} samples in: {save_dir}"
)