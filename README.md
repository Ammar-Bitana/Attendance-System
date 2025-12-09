# 📸 Face Recognition Attendance System

A real-time face recognition attendance system built with Python, OpenCV, and Streamlit. Mark attendance twice a day (morning and evening) with automatic facial recognition.

## 🌟 Features

- **Real-time Face Recognition**: Uses MTCNN and FaceNet for accurate face detection
- **Dual Attendance**: Supports morning and evening attendance marking
- **Smart Caching**: Efficient face encoding caching system for faster startup
- **Web Interface**: Beautiful Streamlit UI accessible from any device
- **Easy Management**: Add/remove people directly from the app
- **Attendance Reports**: View and download attendance records as CSV
- **Mobile Friendly**: Access from your phone or tablet

## 🚀 Quick Start

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/attendance-system.git
cd attendance-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## 📱 Usage

### Desktop Mode
- **Add New Person**: Click "Capture Photos" - automatically takes 50 photos using webcam
- **Live Recognition**: Uses local webcam for real-time recognition

### Mobile/Cloud Mode (Recommended)
- **Auto-Capture (New!)**: 
  1. Enter person name
  2. Click "Start Auto-Capture"
  3. Allow camera access
  4. Wait ~8 seconds (50 photos captured automatically!)
  5. Download the ZIP file
  6. Upload it back to complete registration
- **Manual Capture**: Take 50 photos manually (legacy method)
- **Live Recognition**: Use phone camera for attendance marking

### General Features
1. **View Records**: Check attendance history and download CSV reports
2. **Remove Person**: Delete a person from the system if needed

## 🛠️ Technology Stack

- **Python 3.10**
- **Streamlit**: Web interface
- **OpenCV**: Image processing
- **PyTorch**: Deep learning framework
- **FaceNet-PyTorch**: Face recognition model
- **Pandas**: Data management

## 📊 CSV Format

Attendance is stored in a clean format:
```csv
Name,Morning,Evening
John Doe,09:30:15,NA
Jane Smith,10:15:30,18:45:22
```

## 🔒 Privacy

- All face data is stored locally
- No data is sent to external servers
- Attendance records remain on your system

## 📝 License

MIT License

## 👨‍💻 Author

Built with ❤️ for efficient attendance management
