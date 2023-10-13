import cv2
from flask import Flask, request, Response, render_template
import numpy as np
import os
from PPE_check import process_frame, save_intervals

app = Flask(__name__)



def generate_frames():
    print(os.getcwd())

    video = cv2.VideoCapture('../../Thingy-Detector/hats1.mp4')  # Use your video file
    frame_idx = 0
    while True:
        success, frame = video.read()
        if not success:
            break

        processed_frame = process_frame(frame, frame_idx)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        frame_idx += 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_result')
def save_result():
    save_intervals()

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
