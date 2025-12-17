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
import subprocess
import shutil
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

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
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: Arial, sans-serif; 
                text-align: center; 
                padding: 10px; 
                background: #f5f5f5;
                overflow-x: hidden;
            }}
            h2 {{ 
                font-size: 20px; 
                margin-bottom: 15px; 
                color: #333;
            }}
            #video {{ 
                max-width: 100%; 
                width: 100%;
                height: auto;
                border: 3px solid #4CAF50; 
                border-radius: 10px;
                display: block;
                margin: 0 auto;
            }}
            .status {{ 
                font-size: 22px; 
                margin: 15px 0; 
                font-weight: bold; 
                color: #333;
            }}
            #message {{ 
                font-size: 16px; 
                color: #2196F3; 
                margin: 10px 0;
                min-height: 25px;
            }}
            #download {{ 
                margin-top: 15px;
                margin-bottom: 20px;
            }}
            #download a {{ 
                background: #4CAF50;
                color: white;
                padding: 12px 25px;
                text-decoration: none;
                border-radius: 5px;
                font-size: 16px;
                display: inline-block;
            }}
            canvas {{ display: none; }}
            @media (max-width: 600px) {{
                h2 {{ font-size: 18px; }}
                .status {{ font-size: 20px; }}
                #message {{ font-size: 14px; }}
                #download a {{ padding: 10px 20px; font-size: 14px; }}
            }}
        </style>
    </head>
    <body>
        <h2>📸 Auto-Capturing for {person_name}</h2>
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas"></canvas>
        <div class="status">Photos: <span id="count">0</span>/{num_photos}</div>
        <div id="message">Starting...</div>
        <div id="download"></div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            let count = 0;
            const maxPhotos = {num_photos};
            const photos = [];
            
            // Set canvas size based on video
            video.addEventListener('loadedmetadata', () => {{
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            }});
            
            navigator.mediaDevices.getUserMedia({{ 
                video: {{ 
                    facingMode: 'user',
                    width: {{ ideal: 640 }},
                    height: {{ ideal: 480 }}
                }} 
            }})
            .then(stream => {{
                video.srcObject = stream;
                document.getElementById('message').textContent = 'Starting in 2 seconds...';
                setTimeout(startCapture, 2000);
            }})
            .catch(err => {{
                document.getElementById('message').textContent = 'Error: ' + err.message;
            }});
            
            function startCapture() {{
                document.getElementById('message').textContent = 'Capturing...';
                const interval = setInterval(() => {{
                    if (count >= maxPhotos) {{
                        clearInterval(interval);
                        finish();
                        return;
                    }}
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
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
                    '<a href="' + url + '" download="{person_name}_photos.zip">📥 Download ZIP</a>';
                document.getElementById('message').textContent = '✅ Ready! Click to download';
                document.getElementById('message').style.color = '#4CAF50';
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
if 'recognized_today' not in st.session_state:
    st.session_state.recognized_today = set()  # Track who's been recognized in this session

# Page config
st.set_page_config(
    page_title="Attendance System",
    layout="wide"
)

# Initialize session state
dataset_path = 'dataset'
cache_file = 'face_encodings_cache.pkl'
roles_file = 'person_roles.json'

# Role management functions
def load_person_roles():
    """Load person roles from JSON file"""
    if os.path.exists(roles_file):
        try:
            with open(roles_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_person_role(name, role):
    """Save a person's role"""
    roles = load_person_roles()
    roles[name] = role
    with open(roles_file, 'w') as f:
        json.dump(roles, f, indent=2)

def get_person_role(name):
    """Get a person's role"""
    roles = load_person_roles()
    return roles.get(name, "Unknown")

# Git sync functions for persistent storage
def git_sync_enabled():
    """Check if we're running on Streamlit Cloud"""
    return os.path.exists('/mount/src')

def git_pull():
    """Pull latest data from GitHub"""
    if git_sync_enabled():
        try:
            subprocess.run(['git', 'pull', 'origin', 'main'], 
                         capture_output=True, check=False, cwd='/mount/src/attendance-system')
        except:
            pass

def git_push_changes(message="Auto-sync data"):
    """Push changes to GitHub"""
    if git_sync_enabled():
        try:
            repo_path = '/mount/src/attendance-system'
            
            # Get GitHub token from secrets
            github_token = st.secrets.get("GITHUB_TOKEN", "")
            if not github_token:
                return  # Skip if no token configured
            
            # Configure git
            subprocess.run(['git', 'config', 'user.email', 'streamlit-app@auto-sync.com'], 
                         cwd=repo_path, capture_output=True)
            subprocess.run(['git', 'config', 'user.name', 'Streamlit App'], 
                         cwd=repo_path, capture_output=True)
            
            # Add files
            subprocess.run(['git', 'add', 'dataset/', '*.csv', '*.pkl', '*.json'], 
                         cwd=repo_path, capture_output=True, check=False)
            
            # Commit
            result = subprocess.run(['git', 'commit', '-m', message], 
                         cwd=repo_path, capture_output=True, check=False)
            
            # Only push if there are changes
            if result.returncode == 0:
                # Use token for authentication
                remote_url = f"https://{github_token}@github.com/Samarth-143/Attendance-System.git"
                subprocess.run(['git', 'push', remote_url, 'main'], 
                             cwd=repo_path, capture_output=True, check=False, timeout=10)
        except Exception as e:
            pass  # Silently fail to not interrupt user experience

# Pull latest data on startup
if git_sync_enabled() and 'initial_sync' not in st.session_state:
    git_pull()
    st.session_state.initial_sync = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Functions
@st.cache_resource
def load_models():
    """Load face detection and recognition models"""
    mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device, keep_all=False)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet

def get_monthly_attendance_summary(year, month):
    """Get attendance summary for a specific month"""
    from calendar import monthrange
    import glob
    
    # Get all days in the month
    num_days = monthrange(year, month)[1]
    monthly_data = {}
    all_people = set()
    
    # First pass: collect all people who appear in any attendance file and track present days
    for day in range(1, num_days + 1):
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        
        # Check both staff and worker files
        for file_pattern in [f"attendance_staff_{date_str}.csv", f"attendance_worker_{date_str}.csv"]:
            if os.path.exists(file_pattern):
                try:
                    df = pd.read_csv(file_pattern, skiprows=2)  # Skip date row and empty row
                    for _, row in df.iterrows():
                        name = row['Name']
                        all_people.add(name)
                        if name not in monthly_data:
                            monthly_data[name] = {
                                'Days Present': 0,
                                'Present Dates': [],
                                'Absent Dates': []
                            }
                        monthly_data[name]['Days Present'] += 1
                        monthly_data[name]['Present Dates'].append(date_str)
                except Exception:
                    pass  # Skip files that can't be read
    
    # Second pass: identify absent dates for each person
    for day in range(1, num_days + 1):
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        for person in all_people:
            if person not in monthly_data:
                monthly_data[person] = {
                    'Days Present': 0,
                    'Present Dates': [],
                    'Absent Dates': []
                }
            if date_str not in monthly_data[person]['Present Dates']:
                monthly_data[person]['Absent Dates'].append(date_str)
    
    # Convert to DataFrame for better display
    if monthly_data:
        summary_list = []
        for name, data in monthly_data.items():
            absent_str = ', '.join(data['Absent Dates']) if data['Absent Dates'] else 'No absences'
            summary_list.append({
                'Name': name,
                'Days Present': data['Days Present'],
                'Absent Dates': absent_str
            })
        return pd.DataFrame(summary_list).sort_values('Days Present', ascending=False)
    return pd.DataFrame()

# Email configuration and functions
def get_email_config():
    """Get email configuration from Streamlit secrets or environment variables"""
    try:
        config = {
            'sender_email': st.secrets.get("SENDER_EMAIL", os.getenv("SENDER_EMAIL", "")),
            'sender_password': st.secrets.get("SENDER_PASSWORD", os.getenv("SENDER_PASSWORD", "")),
            'recipient_email': st.secrets.get("RECIPIENT_EMAIL", os.getenv("RECIPIENT_EMAIL", "")),
            'smtp_server': st.secrets.get("SMTP_SERVER", os.getenv("SMTP_SERVER", "smtp.gmail.com")),
            'smtp_port': int(st.secrets.get("SMTP_PORT", os.getenv("SMTP_PORT", "587")))
        }
        return config
    except Exception as e:
        st.error(f"Error loading email configuration: {e}")
        return None

def send_attendance_email(recipient_email, subject, csv_data, csv_filename, sender_email, sender_password, smtp_server, smtp_port):
    """Send attendance CSV via email"""
    try:
        # Create message
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = recipient_email
        message['Subject'] = subject
        
        # Email body
        body = f"""
Hi,

Please find attached the attendance report.

Report Details:
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- File: {csv_filename}

Best regards,
Attendance System
        """
        
        message.attach(MIMEText(body, 'plain'))
        
        # Attach CSV file
        attachment = MIMEBase('application', 'octet-stream')
        attachment.set_payload(csv_data.encode())
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', f'attachment; filename= {csv_filename}')
        message.attach(attachment)
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
        
        return True, "Email sent successfully!"
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Email authentication failed. Check your credentials."
    except smtplib.SMTPException as e:
        return False, f"❌ SMTP error: {str(e)}"
    except Exception as e:
        return False, f"❌ Error sending email: {str(e)}"

def test_email_config():
    """Test email configuration"""
    config = get_email_config()
    if not config or not all(config.values()):
        return False, "❌ Email configuration incomplete. Please set all required environment variables."
    
    try:
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['sender_email'], config['sender_password'])
        return True, "✅ Email configuration is valid!"
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Email authentication failed. Check your credentials."
    except Exception as e:
        return False, f"❌ Connection error: {str(e)}"

def send_daily_email_auto():
    """Automatically send daily attendance email anytime after 4 PM"""
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    current_hour = current_time.hour
    
    # Send email anytime after 4 PM (16:00) every day
    if current_hour < 21:
        return  # Too early, before 4 PM
    
    # Check if we already sent email today
    email_log_file = f"email_sent_{current_time.strftime('%Y-%m-%d')}.txt"
    if os.path.exists(email_log_file):
        return  # Already sent today
    
    # Get today's attendance file
    today = current_time.strftime('%Y-%m-%d')
    attendance_file = f"attendance_{today}.csv"
    
    if not os.path.exists(attendance_file):
        return  # No attendance data yet
    
    try:
        config = get_email_config()
        if not config or not all(config.values()):
            return  # Email not configured
        
        # Read and send today's attendance
        df = pd.read_csv(attendance_file)
        if df.empty:
            return
        
        csv_data = df.to_csv(index=False)
        success, message = send_attendance_email(
            recipient_email=config['recipient_email'],
            subject=f"Daily Attendance Report - {today}",
            csv_data=csv_data,
            csv_filename=attendance_file,
            sender_email=config['sender_email'],
            sender_password=config['sender_password'],
            smtp_server=config['smtp_server'],
            smtp_port=config['smtp_port']
        )
        
        if success:
            # Mark that we sent email today
            with open(email_log_file, 'w') as f:
                f.write(f"Email sent at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    except Exception as e:
        pass  # Silently fail to not interrupt user experience

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

def get_attendance_records(date_str):
    """Load existing attendance records from both staff and worker files"""
    attendance_records = {}
    
    # Load from staff file
    staff_file = f"attendance_staff_{date_str}.csv"
    if os.path.exists(staff_file):
        try:
            if os.path.getsize(staff_file) > 0:
                df = pd.read_csv(staff_file, skiprows=2)  # Skip date row and empty row
                if not df.empty:
                    for _, row in df.iterrows():
                        name = row['Name']
                        intime = None if pd.isna(row['In-Time']) or row['In-Time'] == 'NA' else row['In-Time']
                        outtime = None if pd.isna(row['Out-Time']) or row['Out-Time'] == 'NA' else row['Out-Time']
                        attendance_records[name] = {'In-Time': intime, 'Out-Time': outtime}
        except Exception:
            pass
    
    # Load from worker file
    worker_file = f"attendance_worker_{date_str}.csv"
    if os.path.exists(worker_file):
        try:
            if os.path.getsize(worker_file) > 0:
                df = pd.read_csv(worker_file, skiprows=2)  # Skip date row and empty row
                if not df.empty:
                    for _, row in df.iterrows():
                        name = row['Name']
                        intime = None if pd.isna(row['In-Time']) or row['In-Time'] == 'NA' else row['In-Time']
                        outtime = None if pd.isna(row['Out-Time']) or row['Out-Time'] == 'NA' else row['Out-Time']
                        attendance_records[name] = {'In-Time': intime, 'Out-Time': outtime}
        except Exception:
            pass
    
    return attendance_records

def save_attendance_to_csv(date_str, attendance_records):
    """Save all attendance records to separate CSV files by role"""
    # Separate records by role
    staff_records = {}
    worker_records = {}
    
    for name, sessions in attendance_records.items():
        role = get_person_role(name)
        if role == "Staff":
            staff_records[name] = sessions
        elif role == "Worker":
            worker_records[name] = sessions
    
    # Save staff file
    if staff_records:
        staff_file = f"attendance_staff_{date_str}.csv"
        with open(staff_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"Attendance Date: {date_str}"])
            writer.writerow([])  # Empty row for spacing
            writer.writerow(["Name", "Role", "In-Time", "Out-Time"])
            for name, sessions in staff_records.items():
                intime = sessions['In-Time'] if sessions['In-Time'] else 'NA'
                outtime = sessions['Out-Time'] if sessions['Out-Time'] else 'NA'
                writer.writerow([name, "Staff", intime, outtime])
    
    # Save worker file
    if worker_records:
        worker_file = f"attendance_worker_{date_str}.csv"
        with open(worker_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"Attendance Date: {date_str}"])
            writer.writerow([])  # Empty row for spacing
            writer.writerow(["Name", "Role", "In-Time", "Out-Time"])
            for name, sessions in worker_records.items():
                intime = sessions['In-Time'] if sessions['In-Time'] else 'NA'
                outtime = sessions['Out-Time'] if sessions['Out-Time'] else 'NA'
                writer.writerow([name, "Worker", intime, outtime])
    
    # Sync to GitHub after saving
    git_push_changes(f"Update attendance: {date_str}")

def mark_attendance(name, date_str, attendance_records):
    """Mark attendance for a person"""
    # Use Indian timezone (IST - UTC+5:30)
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    time_now = current_time.strftime("%H:%M:%S")
    hour = current_time.hour
    
    session = "In-Time" if hour < 16 else "Out-Time"
    
    if name not in attendance_records:
        attendance_records[name] = {'In-Time': None, 'Out-Time': None}
    
    if not attendance_records[name][session]:
        attendance_records[name][session] = time_now
        save_attendance_to_csv(date_str, attendance_records)
        return True, session, time_now
    return False, session, time_now

# UI
# Check if it's time to send daily email (at 5 PM)
send_daily_email_auto()

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
                                    
                                    # Check if already recognized in this session
                                    ist = pytz.timezone('Asia/Kolkata')
                                    current_hour = datetime.now(ist).hour
                                    session_key = f"{name}_{current_hour < 16}"  # True for morning, False for evening
                                    
                                    if session_key not in st.session_state.recognized_today:
                                        # Not yet marked in this session, mark now
                                        today = datetime.now(ist).strftime("%Y-%m-%d")
                                        
                                        # Load existing attendance records
                                        attendance_records = get_attendance_records(today)
                                        
                                        # Try to mark attendance
                                        marked, session, time_mark = mark_attendance(name, today, attendance_records)
                                        
                                        if marked:
                                            st.session_state.recognized_today.add(session_key)  # Add to recognized set
                                            st.success(f"✅ **{name}** - {session} attendance marked at {time_mark}")
                                        else:
                                            st.session_state.recognized_today.add(session_key)  # Already marked in CSV, add to set
                                            st.info(f"ℹ️ **{name}** - {session} attendance already marked today")
                                    
                                    recognized = True
                                else:
                                    st.error("❌ Face not recognized. Please try again or contact admin.")
                        except Exception as e:
                            # Silently skip faces that can't be processed
                            pass
                    
                    # Display image with detection box
                    st.image(img_array, caption="Recognition Result", use_container_width=True)
                    
                    if not recognized:
                        st.warning("⚠️ No recognized face found. Please ensure you are registered in the system.")
                else:
                    st.error("❌ No face detected in the image. Please retake with your face clearly visible.")
        
        # Show recent attendance
        st.markdown("---")
        st.subheader("📋 Today's Attendance")
        
        # Check for both staff and worker files
        ist = pytz.timezone('Asia/Kolkata')
        today = datetime.now(ist).strftime("%Y-%m-%d")
        staff_file_today = f"attendance_staff_{today}.csv"
        worker_file_today = f"attendance_worker_{today}.csv"
        
        has_data = False
        
        # Display staff attendance
        if os.path.exists(staff_file_today):
            try:
                if os.path.getsize(staff_file_today) > 0:
                    df_staff = pd.read_csv(staff_file_today, skiprows=2)
                    if not df_staff.empty:
                        st.markdown("**Staff:**")
                        st.dataframe(df_staff, use_container_width=True)
                        has_data = True
            except Exception:
                pass
        
        # Display worker attendance
        if os.path.exists(worker_file_today):
            try:
                if os.path.getsize(worker_file_today) > 0:
                    df_worker = pd.read_csv(worker_file_today, skiprows=2)
                    if not df_worker.empty:
                        st.markdown("**Worker:**")
                        st.dataframe(df_worker, use_container_width=True)
                        has_data = True
            except Exception:
                pass
        
        if not has_data:
            st.info("No attendance marked yet today")
        else:
            st.info("No attendance records yet")
    else:
        st.warning("⚠️ No people registered in the system. Please add people first in the 'Add New Person' tab.")

# Tab 2: Attendance Records
with tab2:
    st.subheader("📋 Attendance Records")
    
    # Toggle between daily and monthly view
    view_type = st.radio("📊 View:", ["Daily Attendance", "Monthly Attendance"], horizontal=True)
    
    if view_type == "Daily Attendance":
        st.markdown("### 📅 Daily Attendance")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            date_filter = st.date_input("Select Date", datetime.now(pytz.timezone('Asia/Kolkata')), key="daily_date")
        with col2:
            role_filter = st.selectbox("View", ["All", "Staff Only", "Worker Only"])
        
        date_str = date_filter.strftime('%Y-%m-%d')
        staff_file = f"attendance_staff_{date_str}.csv"
        worker_file = f"attendance_worker_{date_str}.csv"
        
        # Determine which files to show
        files_to_show = []
        if role_filter in ["All", "Staff Only"] and os.path.exists(staff_file):
            files_to_show.append(("Staff", staff_file))
        if role_filter in ["All", "Worker Only"] and os.path.exists(worker_file):
            files_to_show.append(("Worker", worker_file))
        
        if files_to_show:
            for role_name, file_path in files_to_show:
                st.markdown(f"#### {role_name} Attendance")
                df = pd.read_csv(file_path, skiprows=2)  # Skip date row and empty row
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Entries", len(df))
                with col2:
                    st.metric("Unique People", df['Name'].nunique())
                with col3:
                    intime_present = df['In-Time'].ne('NA').sum()
                    outtime_present = df['Out-Time'].ne('NA').sum()
                    st.metric("In-Time / Out-Time", f"{intime_present} / {outtime_present}")
                
                st.dataframe(df, use_container_width=True)
                
                # Download and Email buttons for this role
                col1, col2 = st.columns(2)
                with col1:
                    # Read the original file with date header for download
                    with open(file_path, 'r') as f:
                        csv_data = f.read()
                    st.download_button(
                        label=f"📥 Download {role_name} CSV",
                        data=csv_data,
                        file_name=file_path,
                        mime="text/csv",
                        key=f"download_{role_name}"
                    )
                
                with col2:
                    if st.button(f"📧 Send {role_name} to Email", key=f"send_{role_name}_email"):
                        config = get_email_config()
                        if config and all(config.values()):
                            # Read the original file with date header for email
                            with open(file_path, 'r') as f:
                                csv_data = f.read()
                            success, message = send_attendance_email(
                                recipient_email=config['recipient_email'],
                                subject=f"{role_name} Attendance Report - {date_str}",
                                csv_data=csv_data,
                                csv_filename=file_path,
                                sender_email=config['sender_email'],
                                sender_password=config['sender_password'],
                                smtp_server=config['smtp_server'],
                                smtp_port=config['smtp_port']
                            )
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                        else:
                            st.error("❌ Email not configured.")
                
                st.markdown("---")  # Separator between staff and worker sections
        else:
            st.info(f"No attendance records for {date_str}")
    
    else:  # Monthly Attendance
        st.markdown("### 📊 Monthly Attendance Summary")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            selected_month = st.date_input("Select Month", datetime.now(pytz.timezone('Asia/Kolkata')), key="monthly_date")
        
        monthly_data_raw = {}
        from calendar import monthrange
        
        # Get all days in the month
        num_days = monthrange(selected_month.year, selected_month.month)[1]
        all_people = set()
        
        # Collect attendance data from both staff and worker files
        for day in range(1, num_days + 1):
            date_str = f"{selected_month.year:04d}-{selected_month.month:02d}-{day:02d}"
            
            # Check both staff and worker files
            for file_pattern in [f"attendance_staff_{date_str}.csv", f"attendance_worker_{date_str}.csv"]:
                if os.path.exists(file_pattern):
                    try:
                        df_temp = pd.read_csv(file_pattern, skiprows=2)  # Skip date row and empty row
                        for _, row in df_temp.iterrows():
                            name = row['Name']
                            all_people.add(name)
                            if name not in monthly_data_raw:
                                monthly_data_raw[name] = {'present_dates': [], 'absent_dates': []}
                            monthly_data_raw[name]['present_dates'].append(date_str)
                    except Exception:
                        pass  # Skip files that can't be read
        
        # Calculate absent dates
        for day in range(1, num_days + 1):
            date_str = f"{selected_month.year:04d}-{selected_month.month:02d}-{day:02d}"
            for person in all_people:
                if person not in monthly_data_raw:
                    monthly_data_raw[person] = {'present_dates': [], 'absent_dates': []}
                if date_str not in monthly_data_raw[person]['present_dates']:
                    monthly_data_raw[person]['absent_dates'].append(date_str)
        
        if monthly_data_raw:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("People in Dataset", len(monthly_data_raw))
            with col2:
                avg_days = sum(len(data['present_dates']) for data in monthly_data_raw.values()) / len(monthly_data_raw)
                st.metric("Avg Days Present", f"{avg_days:.1f}")
            with col3:
                total_days_present = sum(len(data['present_dates']) for data in monthly_data_raw.values())
                st.metric("Total Person-Days", total_days_present)
            
            st.markdown("---")
            
            # Display each person with expandable absent dates
            for name in sorted(monthly_data_raw.keys()):
                data = monthly_data_raw[name]
                days_present = len(data['present_dates'])
                absent_dates = data['absent_dates']
                
                # Create header with days present
                header = f"**{name}** - {days_present} days present"
                
                if absent_dates:
                    with st.expander(header + f" ({len(absent_dates)} absences)"):
                        st.markdown("**Absent Dates:**")
                        for date in sorted(absent_dates):
                            st.markdown(f"• {date}")
                else:
                    st.success(header + " ✅ (Perfect attendance)")
            
            st.markdown("---")
            
            # Create separate DataFrames for staff and workers
            staff_summary = []
            worker_summary = []
            
            for name in sorted(monthly_data_raw.keys()):
                data = monthly_data_raw[name]
                role = get_person_role(name)
                absent_str = ', '.join(sorted(data['absent_dates'])) if data['absent_dates'] else 'No absences'
                summary_item = {
                    'Name': name,
                    'Days Present': len(data['present_dates']),
                    'Absent Dates': absent_str
                }
                if role == "Staff":
                    staff_summary.append(summary_item)
                elif role == "Worker":
                    worker_summary.append(summary_item)
            
            staff_df = pd.DataFrame(staff_summary) if staff_summary else pd.DataFrame()
            worker_df = pd.DataFrame(worker_summary) if worker_summary else pd.DataFrame()
            
            # Staff Download and Email
            if not staff_df.empty:
                st.markdown("#### Staff Monthly Summary")
                col1, col2 = st.columns(2)
                with col1:
                    # Create CSV with date header
                    csv_lines = [f"Staff Monthly Attendance Report: {selected_month.strftime('%B %Y')}\\n"]
                    csv_lines.append("\\n")  # Empty line
                    csv_lines.append(staff_df.to_csv(index=False))
                    csv_data = "".join(csv_lines)
                    
                    st.download_button(
                        label="📥 Download Staff Monthly Summary",
                        data=csv_data,
                        file_name=f"monthly_staff_{selected_month.strftime('%Y-%m')}.csv",
                        mime="text/csv",
                        key="download_staff_monthly"
                    )
                
                with col2:
                    if st.button("📧 Send Staff to Email", key="send_staff_monthly_email"):
                        config = get_email_config()
                        if config and all(config.values()):
                            # Create CSV with date header
                            csv_lines = [f"Staff Monthly Attendance Report: {selected_month.strftime('%B %Y')}\\n"]
                            csv_lines.append("\\n")  # Empty line
                            csv_lines.append(staff_df.to_csv(index=False))
                            csv_data = "".join(csv_lines)
                            
                            success, message = send_attendance_email(
                                recipient_email=config['recipient_email'],
                                subject=f"Staff Monthly Attendance Report - {selected_month.strftime('%B %Y')}",
                                csv_data=csv_data,
                                csv_filename=f"monthly_staff_{selected_month.strftime('%Y-%m')}.csv",
                                sender_email=config['sender_email'],
                                sender_password=config['sender_password'],
                                smtp_server=config['smtp_server'],
                                smtp_port=config['smtp_port']
                            )
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                        else:
                            st.error("❌ Email not configured.")
            
            # Worker Download and Email
            if not worker_df.empty:
                st.markdown("#### Worker Monthly Summary")
                col1, col2 = st.columns(2)
                with col1:
                    # Create CSV with date header
                    csv_lines = [f"Worker Monthly Attendance Report: {selected_month.strftime('%B %Y')}\\n"]
                    csv_lines.append("\\n")  # Empty line
                    csv_lines.append(worker_df.to_csv(index=False))
                    csv_data = "".join(csv_lines)
                    
                    st.download_button(
                        label="📥 Download Worker Monthly Summary",
                        data=csv_data,
                        file_name=f"monthly_worker_{selected_month.strftime('%Y-%m')}.csv",
                        mime="text/csv",
                        key="download_worker_monthly"
                    )
                
                with col2:
                    if st.button("📧 Send Worker to Email", key="send_worker_monthly_email"):
                        config = get_email_config()
                        if config and all(config.values()):
                            # Create CSV with date header
                            csv_lines = [f"Worker Monthly Attendance Report: {selected_month.strftime('%B %Y')}\n"]
                            csv_lines.append("\n")  # Empty line
                            csv_lines.append(worker_df.to_csv(index=False))
                            csv_data = "".join(csv_lines)
                            
                            success, message = send_attendance_email(
                                recipient_email=config['recipient_email'],
                                subject=f"Worker Monthly Attendance Report - {selected_month.strftime('%B %Y')}",
                                csv_data=csv_data,
                                csv_filename=f"monthly_worker_{selected_month.strftime('%Y-%m')}.csv",
                                sender_email=config['sender_email'],
                                sender_password=config['sender_password'],
                                smtp_server=config['smtp_server'],
                                smtp_port=config['smtp_port']
                                )
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                        else:
                            st.error("❌ Email not configured.")
        else:
            st.info(f"No attendance records for {selected_month.strftime('%B %Y')}")

# Tab 3: Add New Person
with tab3:
    st.subheader("➕ Add New Person to Dataset")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        person_name = st.text_input("Person Name")
        person_role = st.selectbox("Role", ["Staff", "Worker"])
        
        # Check if running locally (has webcam access)
        is_local = not os.path.exists('/mount/src')
        
        if is_local:
            st.info("🖥️ **Desktop Mode:** Will automatically capture 50 photos")
            
            if st.button("📸 Capture Photos", use_container_width=True):
                if person_name:
                    person_dir = os.path.join(dataset_path, person_name)
                    os.makedirs(person_dir, exist_ok=True)
                    
                    # Save role
                    save_person_role(person_name, person_role)
                    
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
                    
                    # Sync to GitHub
                    git_push_changes(f"Add person: {person_name}")
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
                        components.html(html_content, height=800, scrolling=True)
                    
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
                                
                                # Save role
                                save_person_role(person_name, person_role)
                                
                                st.success(f"✅ Successfully added {person_name}!")
                                st.session_state.embeddings = None  # Refresh encodings
                                
                                # Sync to GitHub
                                git_push_changes(f"Add person: {person_name}")
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
                        # Save role
                        save_person_role(person_name, person_role)
                        
                        st.success(f"🎉 All 50 photos captured for {person_name}!")
                        
                        # Sync to GitHub
                        git_push_changes(f"Add person: {person_name}")
                        
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
                    st.markdown("### Current People in Dataset")
                    
                    # Separate by role
                    staff_list = []
                    worker_list = []
                    for person in people_list:
                        role = get_person_role(person)
                        if role == "Staff":
                            staff_list.append(person)
                        elif role == "Worker":
                            worker_list.append(person)
                    
                    if staff_list:
                        st.markdown("**Staff:**")
                        for person in sorted(staff_list):
                            st.text(f"• {person}")
                    
                    if worker_list:
                        st.markdown("**Worker:**")
                        for person in sorted(worker_list):
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
                            
                            # Sync to GitHub
                            git_push_changes(f"Remove person: {person_to_remove}")
                            
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error removing person: {e}")
    else:
        st.info("No people found in the dataset.")
    
    if people_list:
        st.markdown("### Current People in Dataset")
        
        # Separate by role
        staff_list = []
        worker_list = []
        for person in people_list:
            role = get_person_role(person)
            if role == "Staff":
                staff_list.append(person)
            elif role == "Worker":
                worker_list.append(person)
        
        if staff_list:
            st.markdown("**Staff:**")
            for person in sorted(staff_list):
                st.text(f"• {person}")
        
        if worker_list:
            st.markdown("**Worker:**")
            for person in sorted(worker_list):
                st.text(f"• {person}")