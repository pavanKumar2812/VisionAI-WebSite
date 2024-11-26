import cv2
import json
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
from ComputerVision.generate_frames import GenerateFrames

app = Flask(__name__)
generate_frames = GenerateFrames()

@app.route("/")
def render_root():
    return render_template("Index.html")

@app.route("/raw_video")
def raw_video():
    return Response(generate_frames.generate_frame("RawVideo"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/streched_video")
def masked_video():
    return Response(generate_frames.generate_frame("StrechedVideo"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/mask_video")
def streched_video():
    return Response(generate_frames.generate_frame("MaskedVideo"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/ObjectDetectionWithColor")
def object_detection_with_color():
    return Response(generate_frames.generate_frame("ObjectDetectionWithColor"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/DetectFaces")
def detect_faces():
    return Response(generate_frames.generate_frame("DetectFaces"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/HandDetection")
def hand_detection():
    return Response(generate_frames.generate_frame("HandDetection"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/HandGestureRecognition")
def hand_gesture_recognition():
    return Response(generate_frames.generate_frame("HandGestureRecognition"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/UpdateHSV", methods=["GET", "POST"])
def UpdateHSV():
    global generate_frames

    try:
        Data = request.get_json()
        generate_frames.lower_h = int(Data["LowerH"])
        generate_frames.lower_s = int(Data["LowerS"])
        generate_frames.lower_v = int(Data["LowerV"])
        generate_frames.upper_h = int(Data["UpperH"])
        generate_frames.upper_s = int(Data["UpperS"])
        generate_frames.upper_v = int(Data["UpperV"])
    except:
        print("Expection")
    return jsonify({"Success": True})

@app.route("/SetFieldOfView", methods=["GET", "POST"])
def set_field_of_view():
    global generate_frames
    try:
        data = request.get_json()
        generate_frames.fov_coordinates[0][0] = int(data["X0"])
        generate_frames.fov_coordinates[0][1] = int(data["Y0"])
        generate_frames.fov_coordinates[1][0] = int(data["X1"])
        generate_frames.fov_coordinates[1][1] = int(data["Y1"])
        generate_frames.fov_coordinates[2][0] = int(data["X2"])
        generate_frames.fov_coordinates[2][1] = int(data["Y2"])
        generate_frames.fov_coordinates[3][0] = int(data["X3"])
        generate_frames.fov_coordinates[3][1] = int(data["Y3"])
        print(f"SetFieldOfView: {generate_frames.fov_coordinates}")
    except:
        print("Exception")
    return jsonify({"Success": True})

@app.route("/ResetFieldOfView", methods=["GET", "POST"])
def ResetFieldOfView():
    global generate_frames
    
    # generate_frames.fov_coordinates = [[0, 0], [generate_frames.frame_width, 0], [0, generate_frames.frame_height], [generate_frames.frame_width, generate_frames.frame_height]]
    generate_frames.fov_coordinates = [[0, 0], [640, 0], [0, 480], [640, 480]]
    print(f"ResetFieldOfView: {generate_frames.fov_coordinates}")
    return jsonify({"Success": True})

# main driver function
if __name__ == "__main__":
    app.run()