# VisionAI-WebSite

VisionAI-WebSite is a Flask-based application that integrates Computer Vision technologies using OpenCV and the YOLO object detection model. This application provides real-time video processing functionalities such as raw video streaming, masked video visualization, stretched video, and object detection with color-based enhancements. Users can interact with the application to customize video processing parameters such as field of view (FOV) and HSV values for color segmentation.

---

## Features

- **Raw Video Streaming**: Displays unprocessed live video feed.
- **Masked Video**: Shows video feed with applied masks based on user-defined HSV values.
- **Stretched Video**: Visualizes video with a stretched perspective.
- **Object Detection with Color Enhancement**: Utilizes YOLO and color segmentation to highlight detected objects.
- **Customizable Field of View (FOV)**: Allows users to set specific regions for video processing.
- **HSV Parameter Adjustment**: Real-time updates to HSV thresholds for color segmentation.

---

## Prerequisites

To run this application, you need the following installed:

- Python 3.8 or above
- Required Python libraries:
  - OpenCV
  - Flask
  - NumPy
  - `ultralytics` (for YOLO)

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/VisionAI-WebSite.git
   cd VisionAI-WebSite
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
1. Run the application
   ```bash
   python app.py
2. Access the web application in your browser at:
   ```auduino
   http://127.0.0.1:5000/
3. Interact with the following routes:
- `/RawVideo:` Stream raw video feed.
- `/MaskedVideo:` Stream video with applied masks.
- `/StrechedVideo:` Stream stretched video feed.
- `/ObjectDetectionWithColor:` Stream object detection feed with color enhancement.
- `/SetFieldOfView:` Update the field of view by sending a POST request with coordinates.
- `/ResetFieldOfView:` Reset the field of view to default.
- `/UpdateHSV:` Adjust HSV parameters for video processing.

## Folder Structure

```plaintext
VisionAI-WebSite/
├── app.py                      # Main Flask application
├── ComputerVision/
│   └── generate_frames.py      # Core video processing logic
├── static/
│   ├── css/                    # CSS files for styling
│   ├── js/                     # JavaScript files
│   └── media/                  # Images and videos
├── templates/
│   └── Index.html              # Main HTML file for the web app
├── README.md                   # Documentation
└── requirements.txt            # List of dependencies
```

## Screenshots and Media

*Home Page:*

*Raw Video Streaming:*

*Future Enhancements*

## Contribution
Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request.

## License
This project is licensed under the MIT License.
