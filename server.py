import cv2
from flask import Flask, request, Response, render_template
import numpy as np
import os
from PPE_check import process_frame, save_intervals
from config import VideoProcessingConfig
app = Flask(__name__)

config = VideoProcessingConfig()

def generate_frames():
    print(os.getcwd())

    video = cv2.VideoCapture(config.video_path)  # Use your video file
    frame_idx = 0
    while True:
        success, frame = video.read()
        if not success:
            break

        processed_frame, safety_intervals = process_frame(frame, frame_idx)
        print(safety_intervals)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        # frame_dimensions = f"Frame Dimensions: {frame.shape[1]}x{frame.shape[0]}"
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               # b'Content-Type: text/plain\r\n\r\n' + frame_dimensions.encode() + b'\r\n')
        frame_idx += 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_result')
def save_result():
    return save_intervals()

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
