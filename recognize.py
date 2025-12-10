import sys
import os
import cv2
import torch
import numpy as np
from datetime import datetime
import pytz
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import csv
import pickle
import hashlib

# Check CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
if device.type == 'cpu':
    print("⚠️  CUDA not available, using CPU (will be slower)")

# Initialize models
try:
    # Use keep_all=False for single face detection
    mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device, keep_all=False)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

dataset_path = 'dataset'
cache_file = 'face_encodings_cache.pkl'

def get_file_hash(filepath):
    """Get hash of file to detect changes"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_cache():
    """Load cached embeddings if available"""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return {}
    return {}

def save_cache(cache_data):
    """Save embeddings to cache"""
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

print("🔍 Loading face encodings...")

# Load existing cache
cache = load_cache()
embeddings = []
names = []

# Track current dataset files
current_files = {}
for person in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person)
    if not os.path.isdir(person_dir):
        continue
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        if os.path.isfile(img_path):
            current_files[img_path] = (person, get_file_hash(img_path))

# Separate new and cached files
new_files = []
for img_path, (person, file_hash) in current_files.items():
    cache_key = f"{img_path}_{file_hash}"
    if cache_key in cache:
        # Use cached embedding
        embeddings.append(cache[cache_key])
        names.append(person)
    else:
        new_files.append((img_path, person, file_hash))

if new_files:
    print(f"📦 Found {len(current_files) - len(new_files)} cached encodings")
    print(f"🆕 Encoding {len(new_files)} new/changed images...")
    
    processed = 0
    for img_path, person, file_hash in new_files:
        img = cv2.imread(img_path)
        if img is None:
            processed += 1
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            face = mtcnn(img_rgb)
            if face is not None:
                face_embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                embeddings.append(face_embedding)
                names.append(person)
                # Cache the new embedding
                cache_key = f"{img_path}_{file_hash}"
                cache[cache_key] = face_embedding
        except Exception as e:
            print(f"⚠️  Error processing {os.path.basename(img_path)}: {e}")
        
        processed += 1
        if processed % 5 == 0:
            print(f"Progress: {processed}/{len(new_files)} new images processed...")
    
    # Save updated cache
    save_cache(cache)
    print("💾 Cache updated")
else:
    print(f"✅ Using {len(embeddings)} cached encodings (no new images to process)")

if len(embeddings) == 0:
    print("❌ No faces found in dataset! Please check your dataset folder.")
    exit(1)

embeddings = np.vstack(embeddings)
print(f"✅ Ready with {len(names)} faces from {len(set(names))} people.")

# Initialize webcam
print("📹 Initializing webcam...")
cap = cv2.VideoCapture(0)

today = datetime.now().strftime("%Y-%m-%d")
attendance_file = f"attendance_{today}.csv"

# Load existing attendance records
attendance_records = {}  # Format: {name: {'In-Time': time or None, 'Out-Time': time or None}}

if os.path.exists(attendance_file):
    import pandas as pd
    try:
        df = pd.read_csv(attendance_file, skiprows=2)
        for _, row in df.iterrows():
            name = row['Name']
            intime = None if pd.isna(row['In-Time']) or row['In-Time'] == 'NA' else row['In-Time']
            outtime = None if pd.isna(row['Out-Time']) or row['Out-Time'] == 'NA' else row['Out-Time']
            attendance_records[name] = {'In-Time': intime, 'Out-Time': outtime}
    except (pd.errors.EmptyDataError, KeyError):
        pass  # Skip if file is empty or columns don't exist
else:
    # Create new CSV file with headers
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Morning", "Evening"])

def save_attendance_to_csv():
    """Save all attendance records to CSV with date at top"""
    ist = pytz.timezone('Asia/Kolkata')
    today = datetime.now(ist).strftime("%Y-%m-%d")
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"Date: {today}"])
        writer.writerow([])  # Empty row for spacing
        writer.writerow(["Name", "In-Time", "Out-Time"])
        for name, sessions in attendance_records.items():
            intime = sessions['In-Time'] if sessions['In-Time'] else 'NA'
            outtime = sessions['Out-Time'] if sessions['Out-Time'] else 'NA'
            writer.writerow([name, intime, outtime])

print("📹 Starting real-time recognition... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(img_rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face_img = img_rgb[y1:y2, x1:x2]
            
            try:
                face_tensor = mtcnn(face_img)

                if face_tensor is not None:
                    embedding = resnet(face_tensor.unsqueeze(0).to(device)).detach().cpu().numpy()
                    sims = cosine_similarity(embedding, embeddings)
                    idx = np.argmax(sims)
                    sim_score = sims[0][idx]

                    if sim_score > 0.6:  # threshold for recognition
                        name = names[idx]
                    else:
                        name = "Unknown"

                    # Draw on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{name} ({sim_score:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                    # Mark attendance
                    if name != "Unknown":
                        current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
                        time_now = current_time.strftime("%H:%M:%S")
                        hour = current_time.hour
                        
                        # Determine session: In-Time (before 2 PM) or Out-Time (2 PM onwards)
                        session = "In-Time" if hour < 14 else "Out-Time"
                        
                        # Initialize record if person not seen today
                        if name not in attendance_records:
                            attendance_records[name] = {'In-Time': None, 'Out-Time': None}
                        
                        # Mark attendance only if this session hasn't been marked yet
                        if not attendance_records[name][session]:
                            attendance_records[name][session] = time_now
                            save_attendance_to_csv()
                            print(f"🟢 Marked {session} attendance for {name} at {time_now}")
                        else:
                            # Show on screen that attendance already marked for this session
                            cv2.putText(frame, f"{session} attendance already marked", (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
            except Exception as e:
                # Skip faces that can't be processed
                pass

    cv2.imshow("Face Recognition (CUDA)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()