import streamlit as st
import cv2
import torch
import numpy as np
from datetime import datetime
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import csv
import pickle
import hashlib
import os
import pandas as pd
from PIL import Image
import time

# Page config
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="📸",
    layout="wide"
)

# Initialize session state
if 'recognition_active' not in st.session_state:
    st.session_state.recognition_active = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'names' not in st.session_state:
    st.session_state.names = None

# Constants
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
    current_time = datetime.now()
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
st.title("📸 Face Recognition Attendance System")

# Check if running on Streamlit Cloud
if not os.path.exists('/mount/src'):
    st.markdown("---")
else:
    st.info("ℹ️ **Cloud Deployment Note:** Camera features are not available on Streamlit Cloud. Use the local version for face recognition. This cloud version is for viewing attendance records.")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    device_info = "🚀 GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
    st.info(f"**Device:** {device_info}")
    
    recognition_threshold = st.slider("Recognition Threshold", 0.0, 1.0, 0.6, 0.05)
    
    st.markdown("---")
    st.header("📊 Statistics")
    
    if os.path.exists(dataset_path):
        total_people = len([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        st.metric("Total People", total_people)
    
    today = datetime.now().strftime("%Y-%m-%d")
    attendance_file = f"attendance_{today}.csv"
    
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        st.metric("Today's Attendance", len(df))

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📹 Live Recognition", "📋 Attendance Records", "➕ Add New Person", "🗑️ Remove Person"])

# Tab 1: Live Recognition
with tab1:
    st.subheader("📸 Face Recognition")
    
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
        st.info("📱 **Mobile Instructions:** Tap 'Take Photo' below, allow camera access, capture your face, then tap 'Use Photo'")
        
        # Use Streamlit's camera_input for mobile compatibility
        img_file = st.camera_input("Take a photo for attendance")
        
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
                    st.image(img_array, caption="Recognition Result", use_column_width=True)
                    
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
                st.dataframe(df, use_column_width=True)
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
        date_filter = st.date_input("Select Date", datetime.now())
    
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
        
        st.info("📱 **Mobile Mode:** Take 10 photos from different angles")
        
        if person_name and st.button("📸 Start Photo Capture", use_container_width=True):
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
            
            st.progress(capture_count / 10)
            st.write(f"Photo {capture_count}/10 captured")
            
            if capture_count < 10:
                img_file = st.camera_input(f"Take photo #{capture_count + 1}")
                
                if img_file is not None:
                    image = Image.open(img_file)
                    img_path = os.path.join(person_dir, f"{person_name}_{capture_count + 1}.jpg")
                    image.save(img_path)
                    st.session_state.capture_count += 1
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.success(f"✅ Successfully captured 10 photos for {person_name}!")
                st.balloons()
                if st.button("🔄 Refresh Encodings Now"):
                    st.session_state.embeddings = None
                    st.session_state.capture_mode = False
                    st.session_state.capture_count = 0
                    st.rerun()
    
    with col2:
        st.info("""
        **Instructions:**
        1. Enter the person's name
        2. Click 'Start Photo Capture'
        3. Take 10 photos from different angles:
           - Front face
           - Slight left turn
           - Slight right turn
           - Different expressions
           - With/without glasses (if applicable)
        4. Click 'Refresh Encodings' when done
        
        **Tips:**
        - Ensure good lighting
        - Keep face clearly visible
        - Avoid shadows on face
        - Stand at arm's length from camera
        """)
        
        if not st.session_state.get('capture_mode', False):
            # Show current people
            if os.path.exists(dataset_path):
                people_list = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
                if people_list:
                    st.markdown("### Registered People")
                    for person in sorted(people_list):
                        person_dir = os.path.join(dataset_path, person)
                        num_images = len([f for f in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, f))])
                        st.text(f"• {person} ({num_images} photos)")

# Tab 4: Remove Person
with tab4:
    st.subheader("🗑️ Remove Person from Dataset")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Get list of people in dataset
        people_list = []
        if os.path.exists(dataset_path):
            people_list = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        if people_list:
            person_to_remove = st.selectbox("Select person to remove", sorted(people_list))
            
            if person_to_remove:
                person_dir = os.path.join(dataset_path, person_to_remove)
                num_images = len([f for f in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, f))])
                
                st.warning(f"⚠️ This will permanently delete **{person_to_remove}** and all {num_images} associated images.")
                
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    confirm = st.checkbox("I understand this action cannot be undone")
                
                with col_btn2:
                    if st.button("🗑️ Delete Person", type="primary", use_container_width=True, disabled=not confirm):
                        try:
                            import shutil
                            shutil.rmtree(person_dir)
                            
                            # Clear cache entries for this person
                            if os.path.exists(cache_file):
                                cache = load_cache()
                                keys_to_remove = [k for k in cache.keys() if person_to_remove in k]
                                for key in keys_to_remove:
                                    del cache[key]
                                save_cache(cache)
                            
                            # Clear session state to force reload
                            st.session_state.embeddings = None
                            st.session_state.names = None
                            
                            st.success(f"✅ Successfully removed {person_to_remove} from the dataset!")
                            st.info("The encodings will be refreshed automatically.")
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error removing person: {e}")
        else:
            st.info("No people found in the dataset.")
    
    with col2:
        st.info("""
        **Warning:**
        - This action is **permanent** and cannot be undone
        - All photos of the selected person will be deleted
        - Their face encodings will be removed from the cache
        - They will no longer be recognized by the system
        
        **Note:**
        - This does NOT delete their attendance history
        - Past attendance records remain intact
        - Only the recognition data is removed
        """)
        
        if people_list:
            st.markdown("### Current People in Dataset")
            for person in sorted(people_list):
                person_dir = os.path.join(dataset_path, person)
                num_images = len([f for f in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, f))])
                st.text(f"• {person} ({num_images} images)")
