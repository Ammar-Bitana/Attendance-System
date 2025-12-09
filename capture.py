import cv2
import os
import time

# Create dataset folder if not exists
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize webcam
cap = cv2.VideoCapture(0)

print("📸 Starting face capture... Look into the camera.")
count = 0
faces_data = []
capture_limit = 50  # Number of images per person

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y+h, x:x+w]
        faces_data.append(face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Capturing {count}/{capture_limit}", (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Capture Faces", frame)

    # Stop automatically after limit
    if count >= capture_limit:
        break

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop early
        break

cap.release()
cv2.destroyAllWindows()

# After capture, ask for name
person_name = input("Enter the person's name: ").strip()
person_folder = os.path.join(dataset_path, person_name)
os.makedirs(person_folder, exist_ok=True)

# Save the captured faces
for i, face in enumerate(faces_data):
    cv2.imwrite(f"{person_folder}/{person_name}_{i+1}.jpg", face)

print(f"✅ Saved {len(faces_data)} images for {person_name} in {person_folder}")