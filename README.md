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

1. **Add New Person**: Capture 50 photos of a person to register them
2. **Live Recognition**: Start the camera to recognize and mark attendance
3. **View Records**: Check attendance history and download reports
4. **Remove Person**: Delete a person from the system if needed

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
