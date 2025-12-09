import streamlit as st
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
import os
import pandas as pd
from PIL import Image
import time
import streamlit.components.v1 as components
import base64
import zipfile
import io

# Page config
st.set_page_config(
    page_title="Attendance System",
    layout="wide"
)

def create_auto_capture_html(person_name, num_photos=50):
    """Create HTML/JS for automatic photo capture"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial; text-align: center; padding: 20px; background: #f5f5f5; }}
            #video {{ border: 3px solid #4CAF50; border-radius: 10px; }}
            .status {{ font-size: 24px; margin: 20px; font-weight: bold; }}
            canvas {{ display: none; }}
        </style>
    </head>
    <body>
        <h2>📸 Auto-Capturing for {person_name}</h2>
        <video id="video" width="640" height="480" autoplay></video>
        <canvas id="canvas" width="640" height="480"></canvas>
        <div class="status">Photos: <span id="count">0</span>/{num_photos}</div>
        <div id="message" style="font-size: 18px; color: #2196F3;">Starting...</div>
        <div id="download"></div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            let count = 0;
            const maxPhotos = {num_photos};
            const photos = [];
            
            navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: 'user' }} }})
                .then(stream => {{
                    video.srcObject = stream;
                    document.getElementById('message').textContent = 'Starting in 2 seconds...';
                    setTimeout(startCapture, 2000);
                }});
            
            function startCapture() {{
                document.getElementById('message').textContent = 'Capturing...';
                const interval = setInterval(() => {{
                    if (count >= maxPhotos) {{
                        clearInterval(interval);
                        finish();
                        return;
                    }}
                    ctx.drawImage(video, 0, 0, 640, 480);
                    canvas.toBlob(blob => photos.push(blob), 'image/jpeg', 0.85);
                    count++;
                    document.getElementById('count').textContent = count;
                }}, 150);
            }}
            
            async function finish() {{
                video.srcObject.getTracks()[0].stop();
                document.getElementById('message').textContent = 'Creating ZIP...';
                
                const zip = new JSZip();
                photos.forEach((blob, i) => {{
                    zip.file('{person_name}_' + (i+1) + '.jpg', blob);
                }});
                
                const zipBlob = await zip.generateAsync({{type: 'blob'}});
                const url = URL.createObjectURL(zipBlob);
                
                document.getElementById('download').innerHTML = 
                    '<a href="' + url + '" download="{person_name}_photos.zip" style="background:#4CAF50;color:white;padding:15px 30px;text-decoration:none;border-radius:5px;font-size:18px;">📥 Download ZIP</a>';
                document.getElementById('message').textContent = '✅ Ready! Click to download';
            }}
        </script>
    </body>
    </html>
    """

# Initialize session state
if 'recognition_active' not in st.session_state:
    st.session_state.recognition_active = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'names' not in st.session_state:
    st.session_state.names = None

# Page config
st.set_page_config(
    page_title="Attendance System",
    page_icon="📸",
    layout="wide"
)

# Initialize session state
dataset_path = 'dataset'
cache_file = 'face_encodings_cache.pkl'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Functions
@st.cache_resource
def load_models():
    """Load face detection and recognition models"""
    mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device, keep_all=False)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet

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

def load_face_encodings(mtcnn, resnet):
    """Load or create face encodings"""
    # Create dataset directory if it doesn't exist
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        return None, None
    
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
            embeddings.append(cache[cache_key])
            names.append(person)
        else:
            new_files.append((img_path, person, file_hash))
    
    if new_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (img_path, person, file_hash) in enumerate(new_files):
            status_text.text(f"Encoding {idx+1}/{len(new_files)}: {os.path.basename(img_path)}")
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            try:
                face = mtcnn(img_rgb)
                if face is not None:
                    face_embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                    embeddings.append(face_embedding)
                    names.append(person)
                    cache_key = f"{img_path}_{file_hash}"
                    cache[cache_key] = face_embedding
            except Exception as e:
                st.warning(f"Error processing {os.path.basename(img_path)}: {e}")
            
            progress_bar.progress((idx + 1) / len(new_files))
        
        save_cache(cache)
        progress_bar.empty()
        status_text.empty()
    
    if len(embeddings) == 0:
        return None, None
    
    return np.vstack(embeddings), names

def get_attendance_records(attendance_file):
    """Load existing attendance records"""
    attendance_records = {}
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        for _, row in df.iterrows():
            name = row['Name']
            morning_time = None if pd.isna(row['Morning']) or row['Morning'] == 'NA' else row['Morning']
            evening_time = None if pd.isna(row['Evening']) or row['Evening'] == 'NA' else row['Evening']
            attendance_records[name] = {'Morning': morning_time, 'Evening': evening_time}
    return attendance_records

def save_attendance_to_csv(attendance_file, attendance_records):
    """Save all attendance records to CSV"""
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Morning", "Evening"])
        for name, sessions in attendance_records.items():
            morning = sessions['Morning'] if sessions['Morning'] else 'NA'
            evening = sessions['Evening'] if sessions['Evening'] else 'NA'
            writer.writerow([name, morning, evening])

def mark_attendance(name, attendance_file, attendance_records):
    """Mark attendance for a person"""
    # Use Indian timezone (IST - UTC+5:30)
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    time_now = current_time.strftime("%H:%M:%S")
    hour = current_time.hour
    
    session = "Morning" if hour < 14 else "Evening"
    
    if name not in attendance_records:
        attendance_records[name] = {'Morning': None, 'Evening': None}
    
    if not attendance_records[name][session]:
        attendance_records[name][session] = time_now
        save_attendance_to_csv(attendance_file, attendance_records)
        return True, session, time_now
    return False, session, time_now

# UI
st.title("📸 Attendance System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Statistics")
    
    if os.path.exists(dataset_path):
        total_people = len([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        st.metric("Total People", total_people)
    
    # Use IST timezone
    ist = pytz.timezone('Asia/Kolkata')
    today = datetime.now(ist).strftime("%Y-%m-%d")
    attendance_file = f"attendance_{today}.csv"
    
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        st.metric("Today's Attendance", len(df))

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📹 Live Recognition", "📋 Attendance Records", "➕ Add New Person", "🗑️ Remove Person"])

# Tab 1: Live Recognition
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📸 Face Recognition")
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, help="Reload face encodings"):
            st.session_state.embeddings = None
            st.rerun()
    
    # Load models if not loaded
    if not st.session_state.models_loaded:
        with st.spinner("Loading models..."):
            mtcnn, resnet = load_models()
            st.session_state.mtcnn = mtcnn
            st.session_state.resnet = resnet
            st.session_state.models_loaded = True
    
    # Load encodings if not loaded
    if st.session_state.embeddings is None:
        with st.spinner("Loading face encodings..."):
            embeddings, names = load_face_encodings(st.session_state.mtcnn, st.session_state.resnet)
            if embeddings is not None:
                st.session_state.embeddings = embeddings
                st.session_state.names = names
                st.success(f"✅ Loaded {len(names)} faces from {len(set(names))} people")
            else:
                st.warning("⚠️ No faces found in dataset. Please add people using the 'Add New Person' tab first.")
                st.session_state.embeddings = []
                st.session_state.names = []
    
    if st.session_state.embeddings is not None and len(st.session_state.embeddings) > 0:
        # Camera selection
        camera_option = st.radio("📷 Camera:", ["Front Camera (Selfie)", "Back Camera"], horizontal=True)
        camera_mode = "user" if "Front" in camera_option else "environment"
        
        # Use Streamlit's camera_input for mobile compatibility
        html_code = f"""
        <script>
        const video = parent.document.querySelector('video');
        if (video) {{
            const constraints = {{ video: {{ facingMode: '{camera_mode}' }} }};
            navigator.mediaDevices.getUserMedia(constraints).then(stream => {{
                video.srcObject = stream;
            }});
        }}
        </script>
        """
        components.html(html_code, height=0)
        
        img_file = st.camera_input("Take a photo for attendance", key=f"recognition_cam_{camera_mode}")
        
        if img_file is not None:
            # Read the image
            image = Image.open(img_file)
            img_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Detect and recognize face
            with st.spinner("Recognizing face..."):
                boxes, probs = st.session_state.mtcnn.detect(img_rgb)
                
                if boxes is not None and len(boxes) > 0:
                    recognized = False
                    for box in boxes:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        face_img = img_rgb[y1:y2, x1:x2]
                        
                        try:
                            face_tensor = st.session_state.mtcnn(face_img)
                            
                            if face_tensor is not None:
                                embedding = st.session_state.resnet(face_tensor.unsqueeze(0).to(device)).detach().cpu().numpy()
                                sims = cosine_similarity(embedding, st.session_state.embeddings)
                                idx = np.argmax(sims)
                                sim_score = sims[0][idx]
                                
                                recognition_threshold = 0.6
                                
                                if sim_score > recognition_threshold:
                                    name = st.session_state.names[idx]
                                    
                                    # Draw rectangle on image
                                    cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(img_array, f"{name} ({sim_score:.2f})", (x1, y1 - 10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                    
                                    # Initialize attendance file
                                    if not os.path.exists(attendance_file):
                                        with open(attendance_file, "w", newline="") as f:
                                            writer = csv.writer(f)
                                            writer.writerow(["Name", "Morning", "Evening"])
                                    
                                    attendance_records = get_attendance_records(attendance_file)
                                    
                                    # Try to mark attendance
                                    marked, session, time_mark = mark_attendance(name, attendance_file, attendance_records)
                                    
                                    if marked:
                                        st.success(f"✅ **{name}** - {session} attendance marked at {time_mark}")
                                        recognized = True
                                    else:
                                        st.info(f"ℹ️ **{name}** - {session} attendance already marked today")
                                        recognized = True
                                else:
                                    st.error("❌ Face not recognized. Please try again or contact admin.")
                        except Exception as e:
                            st.error(f"⚠️ Error processing face: {e}")
                    
                    # Display image with detection box
                    st.image(img_array, caption="Recognition Result", use_container_width=True)
                    
                    if not recognized:
                        st.warning("⚠️ No recognized face found. Please ensure you are registered in the system.")
                else:
                    st.error("❌ No face detected in the image. Please retake with your face clearly visible.")
        
        # Show recent attendance
        st.markdown("---")
        st.subheader("📋 Today's Attendance")
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
            if not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No attendance marked yet today")
        else:
            st.info("No attendance records yet")
    else:
        st.warning("⚠️ No people registered in the system. Please add people first in the 'Add New Person' tab.")

# Tab 2: Attendance Records
with tab2:
    st.subheader("📋 Attendance Records")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        date_filter = st.date_input("Select Date", datetime.now(pytz.timezone('Asia/Kolkata')))
    
    selected_file = f"attendance_{date_filter.strftime('%Y-%m-%d')}.csv"
    
    if os.path.exists(selected_file):
        df = pd.read_csv(selected_file)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entries", len(df))
        with col2:
            st.metric("Unique People", df['Name'].nunique())
        with col3:
            morning_present = df['Morning'].ne('NA').sum()
            evening_present = df['Evening'].ne('NA').sum()
            st.metric("Morning / Evening", f"{morning_present} / {evening_present}")
        
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=selected_file,
            mime="text/csv"
        )
    else:
        st.info(f"No attendance records for {date_filter.strftime('%Y-%m-%d')}")

# Tab 3: Add New Person
with tab3:
    st.subheader("➕ Add New Person to Dataset")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        person_name = st.text_input("Person Name")
        
        # Check if running locally (has webcam access)
        is_local = not os.path.exists('/mount/src')
        
        if is_local:
            st.info("🖥️ **Desktop Mode:** Will automatically capture 50 photos")
            
            if st.button("📸 Capture Photos", use_container_width=True):
                if person_name:
                    person_dir = os.path.join(dataset_path, person_name)
                    os.makedirs(person_dir, exist_ok=True)
                    
                    cap = cv2.VideoCapture(0)
                    st.info("Capturing 50 photos... Please look at the camera!")
                    
                    progress_bar = st.progress(0)
                    img_placeholder = st.empty()
                    
                    for i in range(50):
                        ret, frame = cap.read()
                        if ret:
                            img_path = os.path.join(person_dir, f"{person_name}_{i+1}.jpg")
                            cv2.imwrite(img_path, frame)
                            img_placeholder.image(frame, channels="BGR", width=400)
                            progress_bar.progress((i + 1) / 50)
                            time.sleep(0.1)
                    
                    cap.release()
                    progress_bar.empty()
                    img_placeholder.empty()
                    
                    st.success(f"✅ Successfully captured 50 photos for {person_name}!")
                    st.info("The system will automatically refresh encodings.")
                    st.session_state.embeddings = None
                else:
                    st.error("Please enter a person name")
        else:
            capture_method = st.radio(
                "Select Method:",
                ["🚀 Auto-Capture (Recommended)", "👆 Manual Capture"],
                horizontal=True
            )
            
            if capture_method == "🚀 Auto-Capture (Recommended)":
                if person_name:
                    if st.button("📸 Start Auto-Capture", use_container_width=True, type="primary"):
                        html_content = create_auto_capture_html(person_name, num_photos=50)
                        components.html(html_content, height=700, scrolling=False)
                    
                    st.divider()
                    st.write("**Upload the ZIP file after download**")
                    uploaded_zip = st.file_uploader("Upload the photos ZIP", type=['zip'])
                    
                    if uploaded_zip:
                        with st.spinner("Processing photos..."):
                            person_dir = os.path.join(dataset_path, person_name)
                            os.makedirs(person_dir, exist_ok=True)
                            
                            try:
                                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                                    zip_ref.extractall(person_dir)
                                
                                st.success(f"✅ Successfully added {person_name}!")
                                st.balloons()
                                st.session_state.embeddings = None  # Refresh encodings
                            except Exception as e:
                                st.error(f"Error extracting ZIP: {str(e)}")
                else:
                    st.warning("Please enter a person name first")
            
            else:  # Manual Capture
                # Simplified capture mode
                if person_name and st.button("📸 Start Manual Capture (50 photos)", use_container_width=True, type="primary"):
                    person_dir = os.path.join(dataset_path, person_name)
                    os.makedirs(person_dir, exist_ok=True)
                    st.session_state.capture_mode = True
                    st.session_state.capture_count = 0
                    st.session_state.person_name = person_name
                    st.rerun()
                
                if st.session_state.get('capture_mode', False):
                    capture_count = st.session_state.get('capture_count', 0)
                    person_name = st.session_state.get('person_name', '')
                    person_dir = os.path.join(dataset_path, person_name)
                    
                    # Camera selection for manual capture
                    if capture_count == 0:
                        camera_choice = st.radio("📷 Select Camera:", ["Front (Selfie)", "Back"], horizontal=True, key="manual_cam_choice")
                        st.session_state.camera_facing = "user" if "Front" in camera_choice else "environment"
                    
                    st.progress(capture_count / 50, text=f"Progress: {capture_count}/50 photos")
                    
                    if capture_count < 50:
                        # Auto-submit hack: use a unique key that changes each time
                        img_file = st.camera_input(
                            f"📸 Photo {capture_count + 1}/50 - Just keep clicking!", 
                            key=f"cam_{capture_count}_{person_name}_{st.session_state.get('camera_facing', 'user')}"
                        )
                        
                        if img_file is not None:
                            # Save immediately
                            image = Image.open(img_file)
                            img_path = os.path.join(person_dir, f"{person_name}_{capture_count + 1}.jpg")
                            image.save(img_path, quality=85, optimize=True)
                            
                            # Update counter and rerun immediately
                            st.session_state.capture_count += 1
                            
                            # Show success briefly
                            if capture_count % 10 == 0:
                                st.success(f"✅ {capture_count + 1} photos saved!")
                            
                            st.rerun()
                    else:
                        st.success(f"🎉 All 50 photos captured for {person_name}!")
                        st.balloons()
                        
                        # Reset and refresh
                        st.session_state.embeddings = None
                        st.session_state.capture_mode = False
                        st.session_state.capture_count = 0
                        
                        st.info("✅ Face encodings will be refreshed. Go to 'Live Recognition' tab!")
                        
                        if st.button("🔄 Start Fresh Recognition", type="primary"):
                            st.rerun()
    
    with col2:
        if not st.session_state.get('capture_mode', False):
            # Show current people
            if os.path.exists(dataset_path):
                people_list = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
                if people_list:
                    st.markdown("### Registered People")
                    for person in sorted(people_list):
                        st.text(f"• {person}")

# Tab 4: Remove Person
with tab4:
    st.subheader("🗑️ Remove Person from Dataset")
    
    # Get list of people in dataset
    people_list = []
    if os.path.exists(dataset_path):
        people_list = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        if people_list:
            person_to_remove = st.selectbox("Select person to remove", sorted(people_list))
            
            if person_to_remove:
                person_dir = os.path.join(dataset_path, person_to_remove)
                
                st.warning(f"⚠️ This will permanently delete **{person_to_remove}** and all attendance records.")
                
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    confirm = st.checkbox("I understand this action cannot be undone")
                
                with col_btn2:
                    if st.button("🗑️ Delete Person", type="primary", use_container_width=True, disabled=not confirm):
                        try:
                            import shutil
                            import glob
                            
                            # Remove person directory
                            shutil.rmtree(person_dir)
                            
                            # Clear cache entries for this person
                            if os.path.exists(cache_file):
                                cache = load_cache()
                                keys_to_remove = [k for k in cache.keys() if person_to_remove in k]
                                for key in keys_to_remove:
                                    del cache[key]
                                save_cache(cache)
                            
                            # Remove from all attendance CSV files
                            attendance_files = glob.glob("attendance_*.csv")
                            removed_count = 0
                            for file in attendance_files:
                                if os.path.exists(file):
                                    df = pd.read_csv(file)
                                    if person_to_remove in df['Name'].values:
                                        df = df[df['Name'] != person_to_remove]
                                        df.to_csv(file, index=False)
                                        removed_count += 1
                            
                            # Clear session state to force reload
                            st.session_state.embeddings = None
                            st.session_state.names = None
                            
                            st.success(f"✅ Successfully removed {person_to_remove} from the dataset!")
                            if removed_count > 0:
                                st.success(f"📋 Removed attendance records from {removed_count} file(s)")
                            st.info("The encodings will be refreshed automatically.")
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error removing person: {e}")
    else:
        st.info("No people found in the dataset.")
    
    if people_list:
        st.markdown("### Current People in Dataset")
        for person in sorted(people_list):
            st.text(f"• {person}")