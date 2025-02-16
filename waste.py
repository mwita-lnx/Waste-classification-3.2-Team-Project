from flask import Flask, render_template, Response, jsonify, request, send_from_directory, send_file
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import time
import base64
from picamera2 import Picamera2
import signal
import sys
import serial
import serial.tools.list_ports
import os
import json
from datetime import datetime

app = Flask(__name__)

# Constants
MODEL_PATH = 'waste_model.h5'
MODEL_VERSION = '1.0.0'
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
STREAM_WIDTH = 640
STREAM_HEIGHT = 360
SERIAL_BAUDRATE = 9600
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Enhanced system metrics
system_metrics = {
    'start_time': time.time(),
    'camera_start_time': None,
    'items_processed': 0,
    'processing_times': [],  # List to calculate average processing time
    'predictions': {
        'recyclable': 0,
        'organic': 0
    }
}

# Global variables
picam2 = None
model = None
last_prediction = None
frame_lock = threading.Lock()
current_frame = None
serial_port = None
system_status = {
    'camera': False,
    'arduino': False,
    'model': False,
    'model_version': MODEL_VERSION,
    'arduino_port': None,
    'performance': {
        'processing_time': 0,
        'accuracy': 98.5,  # Example fixed value, could be calculated from validation
        'fps': 30
    }
}

def get_system_uptime():
    """Get system uptime in minutes"""
    return int((time.time() - system_metrics['start_time']) / 60)

def get_camera_uptime():
    """Get camera uptime in minutes"""
    if system_metrics['camera_start_time']:
        return int((time.time() - system_metrics['camera_start_time']) / 60)
    return 0

def calculate_avg_processing_time():
    """Calculate average processing time from last 100 predictions"""
    if not system_metrics['processing_times']:
        return 0
    return sum(system_metrics['processing_times'][-100:]) / len(system_metrics['processing_times'][-100:])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_arduino_port():
    """Find the Arduino's serial port"""
    ports = list(serial.tools.list_ports.comports())
    
    for port in ports:
        if 'Arduino' in port.description or 'CH340' in port.description or 'ACM' in port.description:
            return port.device
    return None

def initialize_serial():
    """Initialize serial connection with Arduino"""
    global serial_port, system_status
    try:
        arduino_port = find_arduino_port()
        if arduino_port is None:
            print("Arduino not found!")
            system_status['arduino'] = False
            system_status['arduino_port'] = None
            return False
        
        serial_port = serial.Serial(arduino_port, SERIAL_BAUDRATE, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        system_status['arduino'] = True
        system_status['arduino_port'] = arduino_port
        return True
    except Exception as e:
        print(f"Failed to initialize serial connection: {e}")
        system_status['arduino'] = False
        system_status['arduino_port'] = None
        return False

def send_to_arduino(prediction):
    """Send prediction to Arduino"""
    global serial_port
    try:
        if serial_port and serial_port.is_open:
            # Send 'R' for Recyclable, 'O' for Organic
            command = 'R' if prediction == 'Recyclable Waste' else 'O'
            serial_port.write(command.encode())
            serial_port.flush()
    except Exception as e:
        print(f"Error sending to Arduino: {e}")
        system_status['arduino'] = False

def initialize_camera():
    """Initialize the camera"""
    global picam2, system_status, system_metrics
    try:
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            lores={"size": (STREAM_WIDTH, STREAM_HEIGHT), "format": "YUV420"})
        picam2.configure(camera_config)
        picam2.start()
        system_status['camera'] = True
        system_metrics['camera_start_time'] = time.time()
        return True
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        system_status['camera'] = False
        system_metrics['camera_start_time'] = None
        return False

def load_ml_model():
    """Load the ML model"""
    global model, system_status
    try:
        model = load_model(MODEL_PATH)
        system_status['model'] = True
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        system_status['model'] = False
        return False

def classify_frame(frame):
    """Classify a frame using the ML model"""
    global model, last_prediction, system_metrics, system_status
    try:
        start_time = time.time()
        
        frame_resized = cv2.resize(frame, (64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(frame_resized)
        img_array = img_array.reshape((1, 64, 64, 3))
        
        result = model.predict(img_array)
        prediction = 'Recyclable Waste' if result[0][0] > 0.5 else 'Organic Waste'
        
        # Update metrics
        process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        system_metrics['processing_times'].append(process_time)
        system_metrics['items_processed'] += 1
        if prediction == 'Recyclable Waste':
            system_metrics['predictions']['recyclable'] += 1
        else:
            system_metrics['predictions']['organic'] += 1
            
        # Update system status with latest performance metrics
        system_status['performance']['processing_time'] = int(calculate_avg_processing_time())
        
        if prediction != last_prediction:
            send_to_arduino(prediction)
            last_prediction = prediction
            
        return prediction
    except Exception as e:
        print(f"Error in classification: {e}")
        system_status['model'] = False
        return None

def generate_frames():
    """Generate video frames with predictions"""
    global current_frame, last_prediction
    
    while True:
        try:
            if not system_status['camera']:
                time.sleep(0.1)
                continue

            # Capture frame
            frame = picam2.capture_array("main")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Make prediction every 0.5 seconds
            if time.time() % 0.5 < 0.1:
                prediction = classify_frame(frame_bgr)
                if prediction:
                    last_prediction = prediction
            
            # Add prediction text to frame
            if last_prediction:
                cv2.putText(frame_bgr, f"Prediction: {last_prediction}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 255, 0), 2)
            
            # Resize frame for streaming
            frame_resized = cv2.resize(frame_bgr, (STREAM_WIDTH, STREAM_HEIGHT))
            
            # Encode frame
            with frame_lock:
                _, buffer = cv2.imencode('.jpg', frame_resized)
                current_frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Error generating frame: {e}")
            system_status['camera'] = False
            time.sleep(0.1)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/livefeed')
def livefeed():
    return render_template('livefeed.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get the current status of all system components and metrics"""
    global system_status, system_metrics
    
    return jsonify({
        **system_status,
        'metrics': {
            'system_uptime': get_system_uptime(),
            'camera_uptime': get_camera_uptime(),
            'items_processed': system_metrics['items_processed'],
            'predictions': system_metrics['predictions']
        }
    })

@app.route('/api/metrics')
def get_metrics():
    """Get detailed system metrics"""
    return jsonify({
        'processing_time': system_status['performance']['processing_time'],
        'accuracy': system_status['performance']['accuracy'],
        'items_processed': system_metrics['items_processed'],
        'uptime': get_system_uptime(),
        'camera_uptime': get_camera_uptime(),
        'predictions': system_metrics['predictions']
    })

@app.route('/api/model')
def get_model_info():
    """Get model information"""
    return jsonify({
        'version': MODEL_VERSION,
        'status': system_status['model'],
        'accuracy': system_status['performance']['accuracy'],
        'processing_time': system_status['performance']['processing_time']
    })

@app.route('/uploadimage', methods=['POST'])
def upload_image():
    """Handle image upload and classification"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
        
    try:
        # Process the image
        img = Image.open(file.stream)
        img_resized = img.resize((64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = img_array.reshape((1, 64, 64, 3))
        
        # Make prediction
        start_time = time.time()
        result = model.predict(img_array)
        prediction = 'Recyclable Waste' if result[0][0] > 0.5 else 'Organic Waste'
        
        # Update metrics
        process_time = (time.time() - start_time) * 1000
        system_metrics['processing_times'].append(process_time)
        system_metrics['items_processed'] += 1
        
        if prediction == 'Recyclable Waste':
            system_metrics['predictions']['recyclable'] += 1
        else:
            system_metrics['predictions']['organic'] += 1
        
        # Send to Arduino
        send_to_arduino(prediction)
        
        return jsonify({
            'prediction': prediction,
            'confidence': float(result[0][0]) if prediction == 'Recyclable Waste' else float(1 - result[0][0]),
            'processing_time': int(process_time)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_prediction')
def get_prediction():
    """Get the latest prediction"""
    return jsonify({
        'prediction': last_prediction if last_prediction else 'No prediction available'
    })

def cleanup():
    """Cleanup function to properly close resources"""
    global picam2, serial_port
    if picam2:
        try:
            picam2.stop()
            picam2 = None
            system_status['camera'] = False
            system_metrics['camera_start_time'] = None
        except:
            pass
    
    if serial_port:
        try:
            serial_port.close()
            serial_port = None
            system_status['arduino'] = False
            system_status['arduino_port'] = None
        except:
            pass

def signal_handler(sig, frame):
    """Handle system signals for graceful shutdown"""
    print('Cleaning up...')
    cleanup()
    sys.exit(0)

if __name__ == '__main__':
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize components
        if not initialize_camera():
            print("Warning: Failed to initialize camera")
        
        if not load_ml_model():
            print("Warning: Failed to load ML model")
            
        if not initialize_serial():
            print("Warning: Failed to initialize serial connection")
        
        # Run the Flask app
        app.run(debug=False, host='0.0.0.0', threaded=True)
    except Exception as e:
        print(f"Error running application: {e}")
    finally:
        cleanup()