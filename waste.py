from flask import Flask, render_template, Response, jsonify, request
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

app = Flask(__name__)

# Constants
MODEL_PATH = 'waste_model.h5'
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
STREAM_WIDTH = 640
STREAM_HEIGHT = 360
SERIAL_BAUDRATE = 9600

# Global variables
picam2 = None
model = None
last_prediction = None
frame_lock = threading.Lock()
current_frame = None
serial_port = None

def find_arduino_port():
    """Find the Arduino's serial port"""
    ports = list(serial.tools.list_ports.comports())

    for port in ports:
        print(port.description)
        if 'Arduino' in port.description or 'CH340' in port.description or 'ACM' in port.description:
            return port.device
    return None

def initialize_serial():
    """Initialize serial connection with Arduino"""
    global serial_port
    try:
        arduino_port = find_arduino_port()
        if arduino_port is None:
            print("Arduino not found!")
            return False
        
        serial_port = serial.Serial(arduino_port, SERIAL_BAUDRATE, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        return True
    except Exception as e:
        print(f"Failed to initialize serial connection: {e}")
        return False

def send_to_arduino(prediction):
    """Send prediction to Arduino"""
    global serial_port
    try:
        if serial_port and serial_port.is_open:
            # Send 'R' for Recyclable, 'N' for Non-recyclable/Organic
            command = 'N' if prediction == 'Recyclable Waste' else 'R'
            serial_port.write(command.encode())
            serial_port.flush()
    except Exception as e:
        print(f"Error sending to Arduino: {e}")

def initialize_camera():
    global picam2
    try:
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            lores={"size": (STREAM_WIDTH, STREAM_HEIGHT), "format": "YUV420"})
        picam2.configure(camera_config)
        picam2.start()
        return True
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return False

def load_ml_model():
    global model
    try:
        model = load_model(MODEL_PATH)
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def classify_frame(frame):
    global model, last_prediction
    try:
        frame_resized = cv2.resize(frame, (64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(frame_resized)
        img_array = img_array.reshape((1, 64, 64, 3))
        
        result = model.predict(img_array)
        prediction = 'Recyclable Waste' if result[0][0] == 1 else 'Organic Waste'
        
        # Send prediction to Arduino if it changed
        if prediction != last_prediction:
            send_to_arduino(prediction)
            last_prediction = prediction
            
        return prediction
    except Exception as e:
        print(f"Error in classification: {e}")
        return None

def generate_frames():
    global current_frame, last_prediction
    
    while True:
        try:
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
            time.sleep(0.1)

@app.route('/livefeed')
def livefeed():
    return render_template('livefeed.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def uploadpage():
    return render_template('upload.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/uploadimage', methods=['POST'])
def upload():
    # Load the model
    model = load_model(MODEL_PATH)
    
    # Get the image from the request
    file = request.files['file']
    
    # Process the image
    img = Image.open(file.stream)
    img_resized = img.resize((64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = img_array.reshape((1, 64, 64, 3))
    
    # Make prediction
    result = model.predict(img_array)
    prediction = 'Recyclable Waste' if result[0][0] == 1 else 'Organic Waste'
    
    return jsonify({'prediction': prediction})

@app.route('/get_prediction')
def get_prediction():
    return jsonify({'prediction': last_prediction if last_prediction else 'No prediction available'})

def cleanup():
    global picam2, serial_port
    if picam2:
        try:
            picam2.stop()
            picam2 = None
        except:
            pass
    
    if serial_port:
        try:
            serial_port.close()
            serial_port = None
        except:
            pass

def signal_handler(sig, frame):
    print('Cleaning up...')
    cleanup()
    sys.exit(0)

if __name__ == '__main__':
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize camera, model, and serial connection
        if not initialize_camera():
            print("Failed to initialize camera. Exiting.")
            sys.exit(1)
            
        if not load_ml_model():
            print("Failed to load ML model. Exiting.")
            cleanup()
            sys.exit(1)
            
        if not initialize_serial():
            print("Failed to initialize serial connection. Exiting.")
            cleanup()
            sys.exit(1)
            
        # Run the Flask app
        app.run(debug=False, host='0.0.0.0', threaded=True)
    except Exception as e:
        print(f"Error running application: {e}")
    finally:
        cleanup()