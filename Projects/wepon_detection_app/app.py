from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Load the custom-trained YOLO model
model = YOLO("custom_yolov8_weapon_detection (1).pt")

# Create a folder to store uploaded files
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        results = model(file_path, conf = 0.4, iou = 0.2)
        output_path = os.path.join(UPLOAD_FOLDER, 'result_' + file.filename)
        for result in results:
            result.save(output_path)
        return redirect(url_for('display_image', filename='result_' + file.filename))

@app.route('/display_image/<filename>')
def display_image(filename):
    return render_template('display_image.html', filename=filename)

@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return redirect(request.url)

    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return redirect(url_for('video_feed', filename=file.filename))

@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(generate_video_feed(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video_feed(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    cap = cv2.VideoCapture(file_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=0.4, iou = 0.8)
        frame_ = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', frame_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/detect_live', methods=['POST'])
def detect_live():
    return redirect(url_for('live_feed'))

@app.route('/live_feed')
def live_feed():
    return Response(generate_live_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_live_feed():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame_ = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', frame_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
