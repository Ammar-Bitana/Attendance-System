import streamlit as st
import streamlit.components.v1 as components

def auto_camera_capture(num_photos=50):
    """Create an auto-capture camera component"""
    
    html_code = f"""
    <div id="camera-container">
        <video id="video" width="640" height="480" autoplay style="border: 2px solid #4CAF50; border-radius: 10px;"></video>
        <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
        <div id="status" style="margin-top: 10px; font-size: 18px; font-weight: bold;">
            Photos captured: <span id="count">0</span>/{num_photos}
        </div>
        <div id="message" style="margin-top: 10px; color: #4CAF50; font-size: 16px;"></div>
    </div>

    <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    const countSpan = document.getElementById('count');
    const message = document.getElementById('message');
    
    let captureCount = 0;
    const maxPhotos = {num_photos};
    let capturedImages = [];
    
    // Access camera
    navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: 'user' }} }})
        .then(stream => {{
            video.srcObject = stream;
            message.textContent = '✓ Camera ready! Auto-capturing will start in 2 seconds...';
            message.style.color = '#4CAF50';
            
            // Start auto-capture after 2 seconds
            setTimeout(() => {{
                startAutoCapture();
            }}, 2000);
        }})
        .catch(err => {{
            message.textContent = '✗ Error accessing camera: ' + err.message;
            message.style.color = '#f44336';
        }});
    
    function startAutoCapture() {{
        const interval = setInterval(() => {{
            if (captureCount >= maxPhotos) {{
                clearInterval(interval);
                message.textContent = '🎉 All photos captured! Sending data...';
                message.style.color = '#4CAF50';
                
                // Send images back to Streamlit
                setTimeout(() => {{
                    window.parent.postMessage({{
                        type: 'streamlit:setComponentValue',
                        data: {{
                            images: capturedImages,
                            completed: true
                        }}
                    }}, '*');
                }}, 500);
                
                return;
            }
            
            // Capture photo
            context.drawImage(video, 0, 0, 640, 480);
            const imageData = canvas.toDataURL('image/jpeg', 0.9);
            capturedImages.push(imageData);
            
            captureCount++;
            countSpan.textContent = captureCount;
            message.textContent = 'Capturing... Vary your expression slightly!';
            message.style.color = '#2196F3';
        }}, 200); // Capture every 200ms
    }}
    </script>
    
    <style>
    #camera-container {{
        text-align: center;
        padding: 20px;
        background: #f5f5f5;
        border-radius: 15px;
    }}
    </style>
    """
    
    result = components.html(html_code, height=600)
    return result
