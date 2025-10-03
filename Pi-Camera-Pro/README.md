# 🚀 ULTRA Pi Camera Pro - All Features Edition

The most complete Raspberry Pi camera streaming server ever! This enhanced version includes **EVERYTHING** - 14+ advanced features for professional camera control and monitoring.

## 🎯 Features

### 📹 **Core Streaming**
- High-quality MJPEG video streaming
- Real-time FPS monitoring
- Dynamic resolution switching (320x240 up to 1920x1080)
- Adjustable JPEG quality (1-100%)
- Brightness and contrast controls

### 🎨 **Visual Effects** (12 filters!)
- **Basic Filters:** Grayscale, Sepia, Invert, Blur
- **Color Tints:** Cool (Blue), Warm (Orange)
- **Artistic Filters:** Cartoon, Emboss, Sharpen, Vintage, Pencil Sketch
- **Edge Detection:** Canny, Sobel, Laplacian

### 🔍 **Detection & Tracking**
- **Motion Detection** with bounding boxes and alerts
- **Face Detection** using OpenCV Haar Cascades
- Adjustable motion sensitivity
- Real-time detection overlays

### 📼 **Recording**
- **Video Recording** - Save MP4 videos with one click
- **Timelapse Mode** - Interval-based frame capture (1-60s)
- Automatic timestamped filenames
- On-screen recording indicator

### 📊 **Overlays & Info**
- **Real-time RGB Histogram** visualization
- **Custom Text Overlay** - Add your own text
- **Timestamp Overlay** - Date/time on video
- **System Stats** - CPU, Memory, Disk, Temperature

### 🖼️ **Media Management**
- **Snapshot Gallery** - View, download, delete images
- Organized file structure (recordings/, snapshots/, timelapse/, logs/)
- High-quality snapshot capture (95% JPEG quality)

### 🔐 **Security & Monitoring**
- **HTTP Basic Authentication** (username/password)
- **Statistics Logging** - CSV logs of all events
- **Webhook Notifications** - Motion/recording events
- Configurable webhook URLs

### 🎛️ **User Interface**
- **Tabbed Interface** - 5 organized sections
- **Mobile Responsive** - Touch-optimized controls
- **Beautiful Gradients** - Animated purple/pink theme
- Real-time stats dashboard

## 📦 Installation

### Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-opencv libcamera-dev
```

### Install Python Dependencies
```bash
pip3 install -r requirements.txt
```

Or manually:
```bash
pip3 install flask flask-httpauth picamera2 opencv-python numpy psutil requests werkzeug
```

## 🚀 Quick Start

1. **Run the server:**
```bash
python3 camserver.py
```

2. **Access the interface:**
   - Open browser: `http://YOUR_PI_IP:5000`
   - Login credentials:
     - **Username:** `admin`
     - **Password:** `picam2025`

3. **Change the password** (edit `camserver.py`):
```python
users = {
    "admin": generate_password_hash("YOUR_NEW_PASSWORD")
}
```

## 🎮 Usage Guide

### **📹 Live Feed Tab**
- Adjust resolution, quality, brightness, contrast
- Flip video horizontally/vertically
- Take snapshots instantly
- Reset all settings to defaults

### **🎨 Effects Tab**
- Choose from 12+ filters
- Apply edge detection (Canny, Sobel, Laplacian)
- Toggle histogram overlay
- Add custom text and timestamp

### **🔍 Detection Tab**
- Enable motion detection
- Enable face detection
- Adjust motion sensitivity
- Configure webhook notifications

### **📼 Recording Tab**
- Start/stop video recording
- Create timelapse videos
- Adjust timelapse interval (1-60s)
- View recording status

### **🌡️ System Tab**
- Monitor CPU, memory, disk usage
- View CPU temperature (Raspberry Pi)
- Configure webhook URL
- Check system health

### **🖼️ Gallery**
- Browse all snapshots
- Download images
- Delete unwanted photos

## 📂 File Structure

```
~/
├── camserver.py          # Main server application
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── recordings/          # Video recordings (MP4)
├── snapshots/           # Captured images (JPG)
├── timelapse/           # Timelapse videos (MP4)
└── logs/                # Event logs (CSV)
    └── camera_stats.csv # Statistics and events
```

## 🔔 Webhook Integration

Configure a webhook URL in the System tab to receive JSON notifications:

```json
{
  "event": "motion_detected",
  "timestamp": "2025-10-02T21:30:45",
  "data": {
    "timestamp": "2025-10-02T21:30:45"
  }
}
```

**Events sent:**
- `motion_detected` - When motion is detected
- `recording_started` - Video recording begins
- `recording_stopped` - Video recording ends (includes duration)

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main interface |
| `/video_feed` | GET | MJPEG video stream |
| `/fps` | GET | Current FPS |
| `/system_stats` | GET | System statistics |
| `/snapshot` | GET | Capture snapshot |
| `/start_recording` | POST | Start video recording |
| `/stop_recording` | POST | Stop recording |
| `/start_timelapse` | POST | Start timelapse |
| `/stop_timelapse` | POST | Stop & save timelapse |
| `/gallery` | GET | View snapshots |
| `/update_setting` | POST | Update camera settings |
| `/change_resolution` | POST | Change resolution |

## 🐛 Troubleshooting

### Camera not detected
```bash
# Check if camera is enabled
sudo raspi-config
# Enable camera in Interface Options

# Test camera
libcamera-hello
```

### Permission denied
```bash
# Add user to video group
sudo usermod -aG video $USER
# Reboot
sudo reboot
```

### Port already in use
```bash
# Kill process on port 5000
sudo lsof -t -i:5000 | xargs kill -9
```

### Face detection not working
Face detection requires OpenCV to be properly installed with the Haar Cascade files. If it's not working, the feature will gracefully disable.

## ⚡ Performance Tips

1. **Lower resolution** for better FPS (320x240 or 640x480)
2. **Reduce quality** for faster streaming (60-70%)
3. **Disable effects** when not needed (filters add overhead)
4. **Motion detection** works best at lower resolutions
5. **Timelapse** uses memory - don't run for hours

## 🔧 Advanced Configuration

### Run on boot (systemd service)
```bash
sudo nano /etc/systemd/system/picamera.service
```

```ini
[Unit]
Description=Pi Camera Pro
After=network.target

[Service]
Type=simple
User=billy
WorkingDirectory=/home/billy
ExecStart=/usr/bin/python3 /home/billy/camserver.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable picamera
sudo systemctl start picamera
```

### Access from internet (use ngrok or similar)
```bash
# Install ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm.tgz
tar xvf ngrok-v3-stable-linux-arm.tgz
sudo mv ngrok /usr/local/bin/

# Run
ngrok http 5000
```

## 📊 Statistics Logging

All events are logged to `~/logs/camera_stats.csv`:
- Timestamp
- Event type (snapshot, recording, motion, etc.)
- Details
- Current FPS
- Current resolution

Perfect for analyzing usage patterns!

## 🎨 Customization

### Change color theme
Edit the CSS gradients in the HTML template (lines 12-15 in template).

### Add more filters
Add your filter function to `apply_filter()` or `apply_advanced_filter()` in `camserver.py`.

### Custom resolutions
Add options to the resolution dropdown in the HTML template.

## 📝 License

Free to use! Original donation link from base version: https://www.buymeacoffee.com/mmshilleh

## 🙏 Credits

Enhanced with ALL features by AI Assistant. Includes contributions from the Flask, OpenCV, and Picamera2 communities.

## 💡 Feature Ideas (Future)

- [ ] Object detection (YOLO integration)
- [ ] Audio streaming support
- [ ] Multi-camera support
- [ ] Cloud storage integration
- [ ] AI-powered scene detection
- [ ] PTZ camera control
- [ ] Mobile app

---

**Enjoy your ULTRA-powered Pi Camera! 🚀📹**
