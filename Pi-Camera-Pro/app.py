from flask import Flask, Response, render_template_string, jsonify, request, send_file, send_from_directory
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import threading
import os
import glob
from datetime import datetime
import csv
import json
import requests
import psutil

### ULTRA ENHANCED Camera Server for Raspberry Pi ###
### All Features Edition ###

app = Flask(__name__)
auth = HTTPBasicAuth()

# Authentication (change these credentials!)
users = {
    "admin": generate_password_hash("picam2025")
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

# Directories
BASE_DIR = os.path.expanduser("~")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")
TIMELAPSE_DIR = os.path.join(BASE_DIR, "timelapse")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

for directory in [RECORDINGS_DIR, SNAPSHOTS_DIR, TIMELAPSE_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Camera configuration
camera = Picamera2()
camera_config = camera.create_preview_configuration(
    main={"format": 'XRGB8888', "size": (640, 480)}
)
camera.configure(camera_config)
camera.start()

# Global variables
current_resolution = (640, 480)
fps_counter = {"count": 0, "fps": 0, "last_time": time.time()}

# Enhanced camera settings
camera_settings = {
    "brightness": 0.0,
    "contrast": 1.0,
    "quality": 85,
    "flip_horizontal": False,
    "flip_vertical": False,
    "filter": "none",
    "show_histogram": False,
    "motion_detection": False,
    "face_detection": False,
    "edge_detection": "none",
    "text_overlay": "",
    "show_timestamp": False,
    "timelapse_active": False,
    "timelapse_interval": 5,
    "webhook_url": ""
}

# Video recording state
recording_state = {
    "active": False,
    "writer": None,
    "filename": None,
    "start_time": None,
    "frame_count": 0
}

# Motion detection state
motion_state = {
    "previous_frame": None,
    "motion_detected": False,
    "motion_threshold": 25,
    "last_motion_time": None
}

# Timelapse state
timelapse_state = {
    "frames": [],
    "last_capture": 0
}

# Face cascade for detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except:
    face_cascade = None

# Statistics logger
def log_statistics(event, details=""):
    """Log events to CSV"""
    log_file = os.path.join(LOGS_DIR, "camera_stats.csv")
    file_exists = os.path.exists(log_file)
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'event', 'details', 'fps', 'resolution'])
        writer.writerow([
            datetime.now().isoformat(),
            event,
            details,
            fps_counter["fps"],
            f"{current_resolution[0]}x{current_resolution[1]}"
        ])

def send_webhook(event, data):
    """Send webhook notification"""
    if camera_settings["webhook_url"]:
        try:
            payload = {
                "event": event,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            requests.post(camera_settings["webhook_url"], json=payload, timeout=5)
        except:
            pass

def get_system_stats():
    """Get system statistics"""
    stats = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "cpu_temp": "N/A"
    }
    
    # Try to get CPU temperature (Raspberry Pi)
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000.0
            stats["cpu_temp"] = f"{temp:.1f}°C"
    except:
        pass
    
    return stats

def apply_advanced_filter(frame, filter_type):
    """Apply advanced artistic filters"""
    if filter_type == "cartoon":
        # Cartoonify effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    
    elif filter_type == "emboss":
        kernel = np.array([[0,-1,-1],
                          [1,0,-1],
                          [1,1,0]])
        embossed = cv2.filter2D(frame, -1, kernel)
        embossed = cv2.cvtColor(embossed, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == "sharpen":
        kernel = np.array([[-1,-1,-1],
                          [-1,9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(frame, -1, kernel)
    
    elif filter_type == "vintage":
        # Sepia + vignette
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        sepia = cv2.transform(frame, kernel)
        # Add vignette
        rows, cols = frame.shape[:2]
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols/2)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows/2)
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel / kernel.max()
        mask = np.dstack([mask]*3)
        return (sepia * mask).astype(np.uint8)
    
    elif filter_type == "pencil_sketch":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0, 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    return frame

def apply_filter(frame, filter_type):
    """Apply color filter to frame"""
    if filter_type == "grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == "sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        return cv2.transform(frame, kernel)
    
    elif filter_type == "invert":
        return cv2.bitwise_not(frame)
    
    elif filter_type == "cool":
        cool = frame.copy().astype(np.float32)
        cool[:, :, 0] = np.clip(cool[:, :, 0] * 1.3, 0, 255)
        cool[:, :, 1] = np.clip(cool[:, :, 1] * 1.1, 0, 255)
        return cool.astype(np.uint8)
    
    elif filter_type == "warm":
        warm = frame.copy().astype(np.float32)
        warm[:, :, 2] = np.clip(warm[:, :, 2] * 1.3, 0, 255)
        warm[:, :, 1] = np.clip(warm[:, :, 1] * 1.1, 0, 255)
        return warm.astype(np.uint8)
    
    elif filter_type == "sketch":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv_gray = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == "blur":
        return cv2.GaussianBlur(frame, (15, 15), 0)
    
    # Advanced filters
    elif filter_type in ["cartoon", "emboss", "sharpen", "vintage", "pencil_sketch"]:
        return apply_advanced_filter(frame, filter_type)
    
    return frame

def apply_edge_detection(frame, edge_type):
    """Apply edge detection"""
    if edge_type == "canny":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    elif edge_type == "sobel":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel / sobel.max() * 255)
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    
    elif edge_type == "laplacian":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    
    return frame

def detect_motion(frame):
    """Detect motion in frame"""
    global motion_state
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if motion_state["previous_frame"] is None:
        motion_state["previous_frame"] = gray
        return False
    
    frame_delta = cv2.absdiff(motion_state["previous_frame"], gray)
    thresh = cv2.threshold(frame_delta, motion_state["motion_threshold"], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    motion_state["previous_frame"] = gray
    
    if motion_detected and not motion_state["motion_detected"]:
        motion_state["last_motion_time"] = time.time()
        log_statistics("motion_detected")
        send_webhook("motion_detected", {"timestamp": datetime.now().isoformat()})
    
    motion_state["motion_detected"] = motion_detected
    
    if motion_detected:
        cv2.putText(frame, "MOTION DETECTED!", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return motion_detected

def detect_faces(frame):
    """Detect faces in frame"""
    if face_cascade is None:
        return frame
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    
    if len(faces) > 0:
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    return frame

def add_histogram(frame):
    """Add histogram overlay"""
    hist_height = 100
    hist_width = 256
    
    # Calculate histograms
    colors = ('b', 'g', 'r')
    hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        hist = hist / hist.max() * hist_height
        
        for x in range(256):
            cv2.line(hist_img, (x, hist_height), (x, hist_height - int(hist[x])), 
                    (255 if color == 'b' else 0, 
                     255 if color == 'g' else 0, 
                     255 if color == 'r' else 0), 1)
    
    # Overlay on frame
    frame[10:10+hist_height, 10:10+hist_width] = cv2.addWeighted(
        frame[10:10+hist_height, 10:10+hist_width], 0.5, hist_img, 0.5, 0)
    
    return frame

def add_text_overlay(frame):
    """Add text overlay to frame"""
    if camera_settings["text_overlay"]:
        cv2.putText(frame, camera_settings["text_overlay"], (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, camera_settings["text_overlay"], (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    if camera_settings["show_timestamp"]:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return frame

def apply_adjustments(frame):
    """Apply all adjustments to frame"""
    # Apply flip/mirror
    if camera_settings["flip_horizontal"]:
        frame = cv2.flip(frame, 1)
    if camera_settings["flip_vertical"]:
        frame = cv2.flip(frame, 0)
    
    # Apply edge detection (overrides filters)
    if camera_settings["edge_detection"] != "none":
        frame = apply_edge_detection(frame, camera_settings["edge_detection"])
    # Apply filter
    elif camera_settings["filter"] != "none":
        frame = apply_filter(frame, camera_settings["filter"])
    
    # Apply brightness and contrast
    brightness = camera_settings["brightness"]
    contrast = camera_settings["contrast"]
    
    if brightness != 0 or contrast != 1:
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness * 100)
    
    # Motion detection
    if camera_settings["motion_detection"]:
        detect_motion(frame)
    
    # Face detection
    if camera_settings["face_detection"]:
        frame = detect_faces(frame)
    
    # Histogram
    if camera_settings["show_histogram"]:
        frame = add_histogram(frame)
    
    # Text overlay
    frame = add_text_overlay(frame)
    
    return frame

def generate_frames():
    """Generate video frames with all features"""
    global fps_counter, recording_state, timelapse_state
    
    while True:
        frame = camera.capture_array()
        
        # Apply adjustments
        frame = apply_adjustments(frame)
        
        # Recording
        if recording_state["active"] and recording_state["writer"] is not None:
            recording_state["writer"].write(frame)
            recording_state["frame_count"] += 1
            
            # Add recording indicator
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            duration = int(time.time() - recording_state["start_time"])
            cv2.putText(frame, f"REC {duration}s", (frame.shape[1] - 120, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Timelapse
        if camera_settings["timelapse_active"]:
            current_time = time.time()
            if current_time - timelapse_state["last_capture"] >= camera_settings["timelapse_interval"]:
                timelapse_state["frames"].append(frame.copy())
                timelapse_state["last_capture"] = current_time
                cv2.putText(frame, f"Timelapse: {len(timelapse_state['frames'])} frames", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Encode with quality setting
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), camera_settings["quality"]]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = buffer.tobytes()
        
        # Update FPS counter
        fps_counter["count"] += 1
        current_time = time.time()
        if current_time - fps_counter["last_time"] >= 1.0:
            fps_counter["fps"] = fps_counter["count"]
            fps_counter["count"] = 0
            fps_counter["last_time"] = current_time
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
@auth.login_required
def index():
    return render_template_string(HTML_TEMPLATE, 
                                  current_res=f"{current_resolution[0]}x{current_resolution[1]}")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fps')
def get_fps():
    return jsonify({"fps": fps_counter["fps"]})

@app.route('/system_stats')
def system_stats():
    return jsonify(get_system_stats())

@app.route('/snapshot')
def snapshot():
    """Capture a single snapshot"""
    frame = camera.capture_array()
    frame = apply_adjustments(frame)
    
    filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(SNAPSHOTS_DIR, filename)
    cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    log_statistics("snapshot_taken", filename)
    
    return send_file(filepath, mimetype='image/jpeg',
                    as_attachment=True, download_name=filename)

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start video recording"""
    global recording_state
    
    if recording_state["active"]:
        return jsonify({"status": "error", "message": "Already recording"}), 400
    
    filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = os.path.join(RECORDINGS_DIR, filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    recording_state["writer"] = cv2.VideoWriter(
        filepath, fourcc, 20.0, current_resolution)
    recording_state["filename"] = filename
    recording_state["active"] = True
    recording_state["start_time"] = time.time()
    recording_state["frame_count"] = 0
    
    log_statistics("recording_started", filename)
    send_webhook("recording_started", {"filename": filename})
    
    return jsonify({"status": "success", "filename": filename})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Stop video recording"""
    global recording_state
    
    if not recording_state["active"]:
        return jsonify({"status": "error", "message": "Not recording"}), 400
    
    recording_state["active"] = False
    if recording_state["writer"]:
        recording_state["writer"].release()
    
    duration = int(time.time() - recording_state["start_time"])
    filename = recording_state["filename"]
    
    log_statistics("recording_stopped", f"{filename} ({duration}s)")
    send_webhook("recording_stopped", {
        "filename": filename, 
        "duration": duration,
        "frames": recording_state["frame_count"]
    })
    
    recording_state["writer"] = None
    recording_state["filename"] = None
    
    return jsonify({"status": "success", "duration": duration})

@app.route('/start_timelapse', methods=['POST'])
def start_timelapse():
    """Start timelapse capture"""
    global timelapse_state
    
    camera_settings["timelapse_active"] = True
    timelapse_state["frames"] = []
    timelapse_state["last_capture"] = 0
    
    log_statistics("timelapse_started")
    return jsonify({"status": "success"})

@app.route('/stop_timelapse', methods=['POST'])
def stop_timelapse():
    """Stop timelapse and create video"""
    global timelapse_state
    
    camera_settings["timelapse_active"] = False
    
    if len(timelapse_state["frames"]) == 0:
        return jsonify({"status": "error", "message": "No frames captured"}), 400
    
    filename = f"timelapse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = os.path.join(TIMELAPSE_DIR, filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, 10.0, 
                         (timelapse_state["frames"][0].shape[1], 
                          timelapse_state["frames"][0].shape[0]))
    
    for frame in timelapse_state["frames"]:
        out.write(frame)
    
    out.release()
    
    frame_count = len(timelapse_state["frames"])
    timelapse_state["frames"] = []
    
    log_statistics("timelapse_created", f"{filename} ({frame_count} frames)")
    
    return jsonify({"status": "success", "filename": filename, "frames": frame_count})

@app.route('/gallery')
@auth.login_required
def gallery():
    """View snapshot gallery"""
    snapshots = sorted(glob.glob(os.path.join(SNAPSHOTS_DIR, "*.jpg")), reverse=True)
    snapshot_list = [os.path.basename(s) for s in snapshots]
    
    return render_template_string(GALLERY_TEMPLATE, snapshots=snapshot_list)

@app.route('/snapshots/<filename>')
def get_snapshot(filename):
    """Serve snapshot file"""
    return send_from_directory(SNAPSHOTS_DIR, filename)

@app.route('/delete_snapshot/<filename>', methods=['POST'])
def delete_snapshot(filename):
    """Delete a snapshot"""
    filepath = os.path.join(SNAPSHOTS_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 404

@app.route('/recordings_list')
def recordings_list():
    """List all recordings"""
    recordings = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "*.mp4")), reverse=True)
    recording_list = []
    for r in recordings:
        stat = os.stat(r)
        recording_list.append({
            "filename": os.path.basename(r),
            "size": stat.st_size,
            "date": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    return jsonify(recording_list)

@app.route('/change_resolution', methods=['POST'])
def change_resolution():
    """Change camera resolution"""
    global current_resolution
    
    data = request.json
    res_str = data.get('resolution', '640,480')
    width, height = map(int, res_str.split(','))
    current_resolution = (width, height)
    
    # Reconfigure camera
    camera.stop()
    camera.configure(camera.create_preview_configuration(
        main={"format": 'XRGB8888', "size": current_resolution}
    ))
    camera.start()
    
    log_statistics("resolution_changed", f"{width}x{height}")
    
    return jsonify({"status": "success", "resolution": current_resolution})

@app.route('/update_setting', methods=['POST'])
def update_setting():
    """Update camera settings"""
    data = request.json
    setting = data.get('setting')
    value = data.get('value')
    
    if setting in camera_settings:
        camera_settings[setting] = value
        log_statistics("setting_changed", f"{setting}={value}")
        return jsonify({"status": "success", "setting": setting, "value": value})
    
    return jsonify({"status": "error", "message": "Invalid setting"}), 400

# HTML Template (will be created separately due to size)
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>🚀 ULTRA Pi Camera Pro</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            animation: fadeInDown 0.8s ease;
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .header h1 {
            font-size: 3em;
            font-weight: 800;
            margin-bottom: 5px;
            text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.95;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .tab {
            padding: 12px 24px;
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 10px;
            color: white;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .tab:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        
        .tab.active {
            background: white;
            color: #667eea;
            border-color: white;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 1024px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            animation: fadeInUp 0.8s ease;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .card-title {
            font-size: 1.4em;
            font-weight: 700;
            margin-bottom: 20px;
            color: #667eea;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .video-container {
            position: relative;
            width: 100%;
            border-radius: 15px;
            overflow: hidden;
            background: #000;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }
        
        .video-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .video-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        
        .badge {
            background: rgba(0,0,0,0.8);
            backdrop-filter: blur(10px);
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85em;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .fps-badge {
            color: #0f0;
            font-family: 'Courier New', monospace;
        }
        
        .resolution-badge {
            color: #00d4ff;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
            gap: 12px;
            margin: 20px 0;
        }
        
        .stat-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            color: white;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .stat-value {
            font-size: 1.6em;
            font-weight: 800;
            margin-bottom: 3px;
        }
        
        .stat-label {
            font-size: 0.8em;
            opacity: 0.9;
        }
        
        .control-group {
            margin-bottom: 20px;
        }
        
        .control-group label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #444;
            font-size: 0.9em;
        }
        
        .control-group select,
        .control-group input[type="text"],
        .control-group input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        
        .control-group select {
            cursor: pointer;
        }
        
        .control-group select:focus,
        .control-group input:focus {
            border-color: #667eea;
            outline: none;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .control-group input[type="range"] {
            width: 100%;
            height: 8px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 5px;
            outline: none;
            -webkit-appearance: none;
        }
        
        .control-group input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: white;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            border: 3px solid #667eea;
        }
        
        .slider-value {
            display: inline-block;
            min-width: 50px;
            text-align: right;
            font-weight: 700;
            color: #667eea;
        }
        
        .toggle-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .toggle-btn {
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            background: white;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            text-align: center;
            font-size: 0.85em;
        }
        
        .toggle-btn:hover {
            border-color: #667eea;
        }
        
        .toggle-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 5px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            text-transform: uppercase;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.3);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-success {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
            color: white;
        }
        
        .btn-warning {
            background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
            color: white;
        }
        
        .btn-full {
            width: 100%;
            margin: 8px 0;
        }
        
        .button-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        
        .section-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
            margin: 20px 0;
        }
        
        .info-box {
            background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%);
            padding: 15px;
            border-radius: 12px;
            border-left: 4px solid #667eea;
            margin-top: 15px;
            font-size: 0.9em;
        }
        
        .recording-status {
            background: rgba(239, 68, 68, 0.2);
            border: 2px solid #ef4444;
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 15px;
            text-align: center;
            font-weight: 700;
            color: #991b1b;
            display: none;
        }
        
        .recording-status.active {
            display: block;
            animation: pulse 2s ease infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        /* Mobile optimizations */
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .tabs {
                gap: 5px;
            }
            
            .tab {
                padding: 8px 16px;
                font-size: 0.85em;
            }
            
            .card {
                padding: 15px;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .button-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ULTRA Pi Camera Pro</h1>
            <p>All Features Edition - Professional Camera Control System</p>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('main')">📹 Live Feed</div>
            <div class="tab" onclick="switchTab('effects')">🎨 Effects</div>
            <div class="tab" onclick="switchTab('detection')">🔍 Detection</div>
            <div class="tab" onclick="switchTab('recording')">📼 Recording</div>
            <div class="tab" onclick="switchTab('system')">🌡️ System</div>
            <div class="tab" onclick="window.location='/gallery'">🖼️ Gallery</div>
        </div>
        
        <!-- Main Tab -->
        <div id="main" class="tab-content active">
            <div class="main-grid">
                <div class="card">
                    <div class="card-title">📹 Live Video Feed</div>
                    <div class="video-container">
                        <img src="/video_feed" alt="Video Stream" id="videoStream">
                        <div class="video-overlay">
                            <div class="badge fps-badge" id="fps">FPS: --</div>
                            <div class="badge resolution-badge" id="resDisplay">{{ current_res }}</div>
                        </div>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-value" id="statFps">--</div>
                            <div class="stat-label">FPS</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="statQuality">85%</div>
                            <div class="stat-label">Quality</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="statRes">VGA</div>
                            <div class="stat-label">Mode</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-title">⚙️ Basic Controls</div>
                    
                    <div class="control-group">
                        <label>Resolution</label>
                        <select id="resolution" onchange="changeResolution()">
                            <option value="320,240">320x240 - Fast</option>
                            <option value="640,480" selected>640x480 - Default</option>
                            <option value="800,600">800x600 - Good</option>
                            <option value="1280,720">1280x720 - HD</option>
                            <option value="1920,1080">1920x1080 - Full HD</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <label>
                            JPEG Quality: <span class="slider-value" id="qualityValue">85</span>%
                        </label>
                        <input type="range" id="quality" min="1" max="100" value="85" 
                               oninput="updateQuality(this.value)">
                    </div>
                    
                    <div class="control-group">
                        <label>
                            Brightness: <span class="slider-value" id="brightnessValue">0.0</span>
                        </label>
                        <input type="range" id="brightness" min="-1" max="1" step="0.1" value="0" 
                               oninput="updateBrightness(this.value)">
                    </div>
                    
                    <div class="control-group">
                        <label>
                            Contrast: <span class="slider-value" id="contrastValue">1.0</span>
                        </label>
                        <input type="range" id="contrast" min="0.5" max="2" step="0.1" value="1" 
                               oninput="updateContrast(this.value)">
                    </div>
                    
                    <div class="section-divider"></div>
                    
                    <div class="control-group">
                        <label>🔄 Mirror / Flip</label>
                        <div class="toggle-group">
                            <div class="toggle-btn" id="flipH" onclick="toggleFlipH()">
                                ↔️ Horizontal
                            </div>
                            <div class="toggle-btn" id="flipV" onclick="toggleFlipV()">
                                ↕️ Vertical
                            </div>
                        </div>
                    </div>
                    
                    <div class="button-grid">
                        <button class="btn btn-success" onclick="takeSnapshot()">
                            📸 Snapshot
                        </button>
                        <button class="btn btn-primary" onclick="resetSettings()">
                            🔄 Reset
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Effects Tab -->
        <div id="effects" class="tab-content">
            <div class="card">
                <div class="card-title">🎨 Visual Effects & Filters</div>
                
                <div class="control-group">
                    <label>Color Filter</label>
                    <select id="filter" onchange="updateFilter()">
                        <option value="none">None (Original)</option>
                        <option value="grayscale">Grayscale</option>
                        <option value="sepia">Sepia (Vintage)</option>
                        <option value="invert">Inverted</option>
                        <option value="cool">Cool (Blue Tint)</option>
                        <option value="warm">Warm (Orange Tint)</option>
                        <option value="blur">Blur</option>
                        <option value="cartoon">Cartoon Effect</option>
                        <option value="emboss">Emboss</option>
                        <option value="sharpen">Sharpen</option>
                        <option value="vintage">Vintage</option>
                        <option value="pencil_sketch">Pencil Sketch</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Edge Detection</label>
                    <select id="edge_detection" onchange="updateEdgeDetection()">
                        <option value="none">None</option>
                        <option value="canny">Canny Edge</option>
                        <option value="sobel">Sobel Edge</option>
                        <option value="laplacian">Laplacian Edge</option>
                    </select>
                </div>
                
                <div class="section-divider"></div>
                
                <div class="control-group">
                    <label>📊 Overlay Options</label>
                    <div class="toggle-group">
                        <div class="toggle-btn" id="histogramToggle" onclick="toggleHistogram()">
                            📊 Histogram
                        </div>
                        <div class="toggle-btn" id="timestampToggle" onclick="toggleTimestamp()">
                            🕐 Timestamp
                        </div>
                    </div>
                </div>
                
                <div class="control-group">
                    <label>Custom Text Overlay</label>
                    <input type="text" id="textOverlay" placeholder="Enter text..." 
                           onchange="updateTextOverlay()">
                </div>
                
                <div class="info-box">
                    💡 <strong>Tip:</strong> Edge detection overrides color filters. Histogram shows real-time RGB distribution.
                </div>
            </div>
        </div>
        
        <!-- Detection Tab -->
        <div id="detection" class="tab-content">
            <div class="card">
                <div class="card-title">🔍 Detection Features</div>
                
                <div class="control-group">
                    <label>Detection Modes</label>
                    <div class="toggle-group">
                        <div class="toggle-btn" id="motionToggle" onclick="toggleMotion()">
                            🎯 Motion
                        </div>
                        <div class="toggle-btn" id="faceToggle" onclick="toggleFace()">
                            👤 Face
                        </div>
                    </div>
                </div>
                
                <div class="section-divider"></div>
                
                <div class="control-group">
                    <label>
                        Motion Sensitivity: <span class="slider-value" id="motionThresholdValue">25</span>
                    </label>
                    <input type="range" id="motionThreshold" min="10" max="100" value="25" 
                           oninput="updateMotionThreshold(this.value)">
                </div>
                
                <div class="info-box">
                    <strong>🎯 Motion Detection:</strong> Automatically detects movement and draws bounding boxes.<br>
                    <strong>👤 Face Detection:</strong> Uses OpenCV Haar Cascade to detect and track faces in real-time.<br>
                    <strong>🔔 Webhooks:</strong> Configure webhooks in System tab to receive notifications.
                </div>
            </div>
        </div>
        
        <!-- Recording Tab -->
        <div id="recording" class="tab-content">
            <div class="card">
                <div class="card-title">📼 Recording & Timelapse</div>
                
                <div class="recording-status" id="recordingStatus">
                    🔴 RECORDING IN PROGRESS
                </div>
                
                <div class="control-group">
                    <label>Video Recording</label>
                    <div class="button-grid">
                        <button class="btn btn-danger btn-full" id="recordBtn" onclick="toggleRecording()">
                            ⏺️ Start Recording
                        </button>
                    </div>
                </div>
                
                <div class="section-divider"></div>
                
                <div class="control-group">
                    <label>Timelapse</label>
                    <div class="control-group">
                        <label>
                            Interval: <span class="slider-value" id="timelapseIntervalValue">5</span>s
                        </label>
                        <input type="range" id="timelapseInterval" min="1" max="60" value="5" 
                               oninput="updateTimelapseInterval(this.value)">
                    </div>
                    <div class="button-grid">
                        <button class="btn btn-warning btn-full" id="timelapseBtn" onclick="toggleTimelapse()">
                            ⏱️ Start Timelapse
                        </button>
                    </div>
                </div>
                
                <div class="info-box">
                    <strong>📹 Recording:</strong> Saves MP4 video to ~/recordings/<br>
                    <strong>⏱️ Timelapse:</strong> Captures frames at set intervals, creates video on stop<br>
                    Files are automatically timestamped for easy organization
                </div>
            </div>
        </div>
        
        <!-- System Tab -->
        <div id="system" class="tab-content">
            <div class="card">
                <div class="card-title">🌡️ System Statistics</div>
                
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value" id="cpuPercent">--</div>
                        <div class="stat-label">CPU %</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="memPercent">--</div>
                        <div class="stat-label">Memory %</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="diskPercent">--</div>
                        <div class="stat-label">Disk %</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="cpuTemp">--</div>
                        <div class="stat-label">CPU Temp</div>
                    </div>
                </div>
                
                <div class="section-divider"></div>
                
                <div class="control-group">
                    <label>🔔 Webhook URL (for notifications)</label>
                    <input type="text" id="webhookUrl" placeholder="https://your-webhook-url.com" 
                           onchange="updateWebhook()">
                    <small style="color: #666;">Receives JSON for motion/recording events</small>
                </div>
                
                <div class="info-box">
                    <strong>📊 Statistics:</strong> Real-time system monitoring<br>
                    <strong>📝 Logging:</strong> All events logged to ~/logs/camera_stats.csv<br>
                    <strong>🔔 Webhooks:</strong> Get notified of motion, recording start/stop
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentResolution = "{{ current_res }}";
        let flipHActive = false;
        let flipVActive = false;
        let isRecording = false;
        let isTimelapsing = false;
        
        // Tab switching
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        
        // Update FPS and system stats
        setInterval(() => {
            fetch('/fps')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('fps').textContent = `FPS: ${data.fps}`;
                    document.getElementById('statFps').textContent = data.fps;
                });
            
            fetch('/system_stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('cpuPercent').textContent = data.cpu_percent.toFixed(1);
                    document.getElementById('memPercent').textContent = data.memory_percent.toFixed(1);
                    document.getElementById('diskPercent').textContent = data.disk_percent.toFixed(1);
                    document.getElementById('cpuTemp').textContent = data.cpu_temp;
                });
        }, 1000);
        
        function changeResolution() {
            const res = document.getElementById('resolution').value;
            fetch('/change_resolution', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({resolution: res})
            }).then(() => setTimeout(() => location.reload(), 1000));
        }
        
        function updateQuality(value) {
            document.getElementById('qualityValue').textContent = value;
            document.getElementById('statQuality').textContent = value + '%';
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'quality', value: parseInt(value)})
            });
        }
        
        function updateBrightness(value) {
            document.getElementById('brightnessValue').textContent = parseFloat(value).toFixed(1);
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'brightness', value: parseFloat(value)})
            });
        }
        
        function updateContrast(value) {
            document.getElementById('contrastValue').textContent = parseFloat(value).toFixed(1);
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'contrast', value: parseFloat(value)})
            });
        }
        
        function updateFilter() {
            const filter = document.getElementById('filter').value;
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'filter', value: filter})
            });
        }
        
        function updateEdgeDetection() {
            const edge = document.getElementById('edge_detection').value;
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'edge_detection', value: edge})
            });
        }
        
        function toggleFlipH() {
            flipHActive = !flipHActive;
            document.getElementById('flipH').classList.toggle('active');
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'flip_horizontal', value: flipHActive})
            });
        }
        
        function toggleFlipV() {
            flipVActive = !flipVActive;
            document.getElementById('flipV').classList.toggle('active');
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'flip_vertical', value: flipVActive})
            });
        }
        
        function toggleHistogram() {
            const btn = document.getElementById('histogramToggle');
            btn.classList.toggle('active');
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'show_histogram', value: btn.classList.contains('active')})
            });
        }
        
        function toggleTimestamp() {
            const btn = document.getElementById('timestampToggle');
            btn.classList.toggle('active');
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'show_timestamp', value: btn.classList.contains('active')})
            });
        }
        
        function updateTextOverlay() {
            const text = document.getElementById('textOverlay').value;
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'text_overlay', value: text})
            });
        }
        
        function toggleMotion() {
            const btn = document.getElementById('motionToggle');
            btn.classList.toggle('active');
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'motion_detection', value: btn.classList.contains('active')})
            });
        }
        
        function toggleFace() {
            const btn = document.getElementById('faceToggle');
            btn.classList.toggle('active');
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'face_detection', value: btn.classList.contains('active')})
            });
        }
        
        function updateMotionThreshold(value) {
            document.getElementById('motionThresholdValue').textContent = value;
            // You can add endpoint for this if needed
        }
        
        function toggleRecording() {
            const btn = document.getElementById('recordBtn');
            const status = document.getElementById('recordingStatus');
            
            if (!isRecording) {
                fetch('/start_recording', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => {
                        if (data.status === 'success') {
                            isRecording = true;
                            btn.textContent = '⏹️ Stop Recording';
                            btn.classList.remove('btn-danger');
                            btn.classList.add('btn-success');
                            status.classList.add('active');
                        }
                    });
            } else {
                fetch('/stop_recording', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => {
                        isRecording = false;
                        btn.textContent = '⏺️ Start Recording';
                        btn.classList.remove('btn-success');
                        btn.classList.add('btn-danger');
                        status.classList.remove('active');
                        alert(`Recording saved! Duration: ${data.duration}s`);
                    });
            }
        }
        
        function updateTimelapseInterval(value) {
            document.getElementById('timelapseIntervalValue').textContent = value;
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'timelapse_interval', value: parseInt(value)})
            });
        }
        
        function toggleTimelapse() {
            const btn = document.getElementById('timelapseBtn');
            
            if (!isTimelapsing) {
                fetch('/start_timelapse', {method: 'POST'})
                    .then(() => {
                        isTimelapsing = true;
                        btn.textContent = '⏹️ Stop Timelapse';
                        btn.classList.remove('btn-warning');
                        btn.classList.add('btn-success');
                    });
            } else {
                fetch('/stop_timelapse', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => {
                        isTimelapsing = false;
                        btn.textContent = '⏱️ Start Timelapse';
                        btn.classList.remove('btn-success');
                        btn.classList.add('btn-warning');
                        alert(`Timelapse created! ${data.frames} frames saved as ${data.filename}`);
                    });
            }
        }
        
        function updateWebhook() {
            const url = document.getElementById('webhookUrl').value;
            fetch('/update_setting', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({setting: 'webhook_url', value: url})
            });
        }
        
        function takeSnapshot() {
            const link = document.createElement('a');
            link.href = '/snapshot';
            link.download = `snapshot_${Date.now()}.jpg`;
            link.click();
        }
        
        function resetSettings() {
            document.getElementById('brightness').value = 0;
            document.getElementById('contrast').value = 1;
            document.getElementById('quality').value = 85;
            document.getElementById('filter').value = 'none';
            document.getElementById('edge_detection').value = 'none';
            document.getElementById('textOverlay').value = '';
            
            if (flipHActive) toggleFlipH();
            if (flipVActive) toggleFlipV();
            
            updateBrightness(0);
            updateContrast(1);
            updateQuality(85);
            updateFilter();
            updateEdgeDetection();
            updateTextOverlay();
        }
    </script>
</body>
</html>
'''

GALLERY_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>📸 Snapshot Gallery</title>
    <style>
        body { font-family: Arial; background: #1a1a1a; color: white; padding: 20px; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }
        .item { background: #2a2a2a; border-radius: 10px; overflow: hidden; }
        .item img { width: 100%; height: 200px; object-fit: cover; }
        .item .info { padding: 10px; }
        .btn { padding: 8px 16px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .btn:hover { background: #5568d3; }
        .back { margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="back">
        <a href="/" class="btn">← Back to Camera</a>
    </div>
    <h1>📸 Snapshot Gallery</h1>
    <div class="gallery">
        {% for snap in snapshots %}
        <div class="item">
            <img src="/snapshots/{{ snap }}" alt="{{ snap }}">
            <div class="info">
                <p>{{ snap }}</p>
                <button class="btn" onclick="downloadSnapshot('{{ snap }}')">Download</button>
                <button class="btn" style="background: #e53e3e;" onclick="deleteSnapshot('{{ snap }}')">Delete</button>
            </div>
        </div>
        {% endfor %}
    </div>
    <script>
        function downloadSnapshot(filename) {
            window.location.href = '/snapshots/' + filename;
        }
        function deleteSnapshot(filename) {
            if (confirm('Delete ' + filename + '?')) {
                fetch('/delete_snapshot/' + filename, {method: 'POST'})
                    .then(() => location.reload());
            }
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("=" * 70)
    print("🚀  ULTRA Pi Camera Pro - ALL FEATURES EDITION")
    print("=" * 70)
    print(f"📡  Access: http://localhost:5000")
    print(f"🔐  Username: admin  |  Password: picam2025")
    print("=" * 70)
    print("✨  FEATURES:")
    print("   • 📹 Video Recording (MP4)")
    print("   • 🎯 Motion Detection with Alerts")
    print("   • 📊 Real-time Histogram Overlay")
    print("   • 🔍 Face Detection (OpenCV)")
    print("   • 📝 Custom Text & Timestamp Overlay")
    print("   • ⏱️  Timelapse Mode")
    print("   • 🎨 15+ Filters & Edge Detection")
    print("   • 📈 Statistics Logging")
    print("   • 🔐 Password Protection")
    print("   • 🌡️  System Stats (CPU, Memory, Temp)")
    print("   • 🖼️  Snapshot Gallery Manager")
    print("   • 🔔 Webhook Notifications")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, threaded=True)
