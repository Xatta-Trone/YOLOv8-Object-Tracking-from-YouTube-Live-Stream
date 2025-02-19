from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from collections import defaultdict
import yt_dlp
import os

app = Flask(__name__)

# YOLO model
model = YOLO('yolov8n.pt')  # Replace with your YOLOv8 model path
class_list = model.names

# Define ROI and other parameters
roi_points = [(250, 80), (450, 80), (500, 340), (50, 295)]
roi_polygon = Polygon(roi_points)
line_start = (0, 160)
line_end = (640, 160)
crossed_ids = set()
class_counts = defaultdict(int)

# Fetch YouTube live stream URL
youtube_live_id = "sstbJ2XhmyU"  # Replace with your YouTube Live ID
youtube_link = f"https://www.youtube.com/watch?v={youtube_live_id}"


def get_live_stream_url(youtube_url):
    ydl_opts = {'format': 'best[height<=360]',
                'quiet': True, 'noplaylist': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info['url']
    except Exception as e:
        print(f"Error fetching live stream URL: {e}")
        return None


stream_url = get_live_stream_url(youtube_link)
if not stream_url:
    raise RuntimeError(
        "Failed to fetch the live stream URL. Ensure the YouTube link is correct.")

cap = cv2.VideoCapture(stream_url)


def generate_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        results = model.track(
            frame, classes=[0, 1, 2, 3, 5, 6, 7], persist=True, conf=0.25, iou=0.4)

        # Draw ROI
        cv2.polylines(frame, [np.array(roi_points, np.int32)],
                      isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.line(frame, line_start, line_end, (255, 0, 255), 2)

        if results[0].boxes.data is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist(
            ) if results[0].boxes.id is not None else list(range(len(class_indices)))

            for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = class_list[class_idx]
                point = Point(cx, cy)

                if roi_polygon.contains(point):
                    if (cy < line_start[1] or cy < line_end[1]) and track_id not in crossed_ids:
                        crossed_ids.add(track_id)
                        class_counts[class_name] += 1

                # Draw bounding boxes and labels
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display class counts
        y_offset = 30
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/counts')
def get_counts():
    return jsonify(class_counts)


@app.route('/')
def index():
    return render_template('index.html', youtube_link=youtube_link)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
