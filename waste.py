from flask import Flask, render_template, Response , jsonify,request
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import time
import base64
FRAME_INTERVAL = 1  # Interval in seconds between processing frames

app = Flask(__name__)

# Constants
MODEL_PATH = 'waste_model.h5'
VIDEO_FEED_URL = 'http://192.168.43.78:8080/video'

# Load the model
model = load_model(MODEL_PATH)
classification = ''


def get_frame():
    cap = cv2.VideoCapture(VIDEO_FEED_URL)
    
    ret, frame = cap.read()
    if ret:
        # Convert frame to PIL image
        # frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Convert PIL image to bytes
        # img_bytes = BytesIO()
        # frame_pil.save(img_bytes, format='JPEG')
        # img_bytes.seek(0)

        # Release the video capture
        cap.release()

        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cap.release()  # Release the video capture if no frame was read
    return None  # Return None if no frame was read


@app.route('/livefeed')
def livefeed():
    return render_template('livefeed.html')

@app.route('/upload')
def uploadpage():
    return render_template('upload.html')

@app.route('/')
def index():
    return render_template('index.html')



last_frame_time = 0
last_prediction_time = 0
last_prediction = None


def classify_frame(frame):
    global model
    # Preprocess the frame
    frame_resized = cv2.resize(frame, (64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(frame_resized)
    img_array = img_array.reshape((1, 64, 64, 3))

    # Perform prediction
    result = model.predict(img_array)
    prediction = 'Recyclable Waste' if result[0][0] == 1 else 'Organic Waste'

    return prediction


def capture_and_process_frames():
        global last_prediction, last_frame_encoded

        frame = get_frame()
        prediction = classify_frame(frame)
        last_prediction = prediction

        # Encode the frame as base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = base64.b64encode(buffer).decode('utf-8')
        last_frame_encoded = frame_encoded

@app.route('/predict')
def predict():
    capture_and_process_frames()
    global last_prediction, last_prediction_time, last_frame_encoded

    if last_prediction:
        return jsonify({'prediction': last_prediction, 'frame': last_frame_encoded})
    else:
        return jsonify({'prediction': 'No prediction available', 'frame': None})
    


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


if __name__ == '__main__':
    # Load the model
    model = load_model(MODEL_PATH)


    # Run the Flask app
    app.run(debug=True)