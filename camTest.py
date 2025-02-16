from picamera2 import Picamera2
import time

# Initialize the camera
picam2 = Picamera2()

# Configure the camera
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)

# Start the camera
picam2.start()

# Wait for the camera to warm up
time.sleep(2)

# Capture an image
picam2.capture_file("test_image.jpg")

# Stop the camera
picam2.stop()