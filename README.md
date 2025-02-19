# YOLOv8 Object Tracking from YouTube Live Stream

This repository provides a Flask web application that streams a live YouTube video feed, processes it using YOLOv8 for object detection and tracking, and displays the processed feed with bounding boxes and class counts. The application also provides an API endpoint for retrieving object counts.

## Features
- Fetches a YouTube live stream and processes the video in real-time
- Utilizes YOLOv8 for object detection and tracking
- Defines a region of interest (ROI) for monitoring object movement
- Counts objects that cross a predefined line
- Provides a web-based interface to view the processed video stream
- Offers an API to retrieve real-time object counts

## Requirements
Ensure you have the following dependencies installed before running the application:

```bash
pip install flask opencv-python numpy ultralytics shapely yt-dlp
```

## Installation and Usage
1. Clone this repository:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask application:
```bash
python app.py
```

4. Open a web browser and go to:
```
http://localhost:8080/
```

## Configuration
- Update the `youtube_live_id` variable in `app.py` with the desired YouTube live stream ID.
- Adjust the YOLO model path if using a custom model.
- Modify `roi_points` and `line_start/line_end` to customize the region of interest and crossing line.

## API Endpoints
- `GET /video_feed` - Streams the processed video with object tracking.
- `GET /counts` - Returns a JSON response with the count of detected objects.

## Deployment
To deploy the application, use a cloud server or containerization.

```bash
export PORT=8080
python app.py
```

For Docker deployment:
```bash
docker build -t yolo-stream .
docker run -p 8080:8080 yolo-stream
```

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Preview

---
Feel free to contribute and improve this repository by submitting a pull request!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Xatta-Trone/YOLOv8-Object-Tracking-from-YouTube-Live-Stream&type=Date)](https://star-history.com/#Xatta-Trone/YOLOv8-Object-Tracking-from-YouTube-Live-Stream&Date)