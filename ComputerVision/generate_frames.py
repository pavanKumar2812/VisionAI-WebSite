import cv2
import numpy as np
import mediapipe as mp
from .mediapipe import MediapipeMyHands, MediapipePose, MediaPipeFace
from ultralytics import YOLO


class GenerateFrames:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        success, self.next_frame = self.camera.read()
        if not success:
            self.camera.release()
            self.camera = cv2.VideoCapture(1)

        self.frame_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # self.fov_coordinates = [[0, 0], [self.frame_width, 0], [0, self.frame_height], [self.frame_width, self.frame_height]]
        self.fov_coordinates = [[0, 0], [640, 0], [0, 480], [640, 480]]

        self.lower_h = 0
        self.lower_s = 0
        self.lower_v = 206
        self.upper_h = 179
        self.upper_s = 255
        self.upper_v = 255

        self.FindHands = MediapipeMyHands(2)
        self.FindFace = MediaPipeFace()
        self.FindPose = MediapipePose()

    def _raw_video_generator(self):
        while True:
            ret_value, next_frame = self.camera.read()
            next_frame = cv2.resize(next_frame, (640, 480), interpolation=cv2.INTER_AREA)
    
            ret_value, frame_as_jpeg = cv2.imencode(".jpg", next_frame)
            next_frame = frame_as_jpeg.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + next_frame + b"\r\n\r\n")

    def _stretched_video_generator(self):
        while True:
            ret_value, next_frame = self.camera.read()
            next_frame = cv2.resize(next_frame, (640, 480), interpolation=cv2.INTER_AREA)
            selected_corners = np.float32([self.fov_coordinates[0], self.fov_coordinates[1], self.fov_coordinates[2], self.fov_coordinates[3]])
            selected_image_width = int(np.linalg.norm(selected_corners[0] - selected_corners[1]))
            selected_image_height = int(np.linalg.norm(selected_corners[0] - selected_corners[2]))

            mapped_corners = np.float32([[0, 0], [selected_image_width, 0], [0, selected_image_height], [selected_image_width, selected_image_height]])
            transform_matrix = cv2.getPerspectiveTransform(selected_corners, mapped_corners)
            transformed_matrix = cv2.warpPerspective(next_frame, transform_matrix, (selected_image_width, selected_image_height))

            stretched_image = cv2.resize(transformed_matrix, (int(self.frame_width), int(self.frame_height)), interpolation=cv2.INTER_AREA)
            # stretched_image = cv2.resize(transformed_matrix, (640, 480), interpolation=cv2.INTER_AREA)

            ret_value, frame_as_jpeg = cv2.imencode(".jpg", stretched_image)
            next_frame = frame_as_jpeg.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + next_frame + b"\r\n\r\n")

    def _masked_video_generator(self):
        print(f"Mask Video Function")
        while True:
            ret_value, next_frame = self.camera.read()
            next_frame = cv2.resize(next_frame, (640, 480), interpolation=cv2.INTER_AREA)
            selected_corners = np.float32([self.fov_coordinates[0], self.fov_coordinates[1], self.fov_coordinates[2], self.fov_coordinates[3]])
            selected_image_width = int(np.linalg.norm(selected_corners[0] - selected_corners[1]))
            selected_image_height = int(np.linalg.norm(selected_corners[0] - selected_corners[2]))

            mapped_corners = np.float32([[0, 0], [selected_image_width, 0], [0, selected_image_height], [selected_image_width, selected_image_height]])
            transform_matrix = cv2.getPerspectiveTransform(selected_corners, mapped_corners)
            transformed_matrix = cv2.warpPerspective(next_frame, transform_matrix, (selected_image_width, selected_image_height))

            stretched_image = cv2.resize(transformed_matrix, (int(self.frame_width), int(self.frame_height)), interpolation=cv2.INTER_AREA)

            hsv_frame = cv2.cvtColor(stretched_image, cv2.COLOR_BGR2HSV)
            lower_values = np.array([self.lower_h, self.lower_s, self.lower_v])
            upper_values = np.array([self.upper_h, self.upper_s, self.upper_v])
            masked_frame = cv2.inRange(hsv_frame, lower_values, upper_values)

            ret_value, frame_as_jpeg = cv2.imencode(".jpg", masked_frame)
            next_frame = frame_as_jpeg.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + next_frame + b"\r\n\r\n")

    def _object_detection_with_color(self):
        while True:
            ret_value, next_frame = self.camera.read()
            next_frame = cv2.resize(next_frame, (640, 480), interpolation=cv2.INTER_AREA)
            selected_corners = np.float32([self.fov_coordinates[0], self.fov_coordinates[1], self.fov_coordinates[2], self.fov_coordinates[3]])
            selected_image_width = int(np.linalg.norm(selected_corners[0] - selected_corners[1]))
            selected_image_height = int(np.linalg.norm(selected_corners[0] - selected_corners[2]))

            mapped_corners = np.float32([[0, 0], [selected_image_width, 0], [0, selected_image_height], [selected_image_width, selected_image_height]])
            transform_matrix = cv2.getPerspectiveTransform(selected_corners, mapped_corners)
            transformed_matrix = cv2.warpPerspective(next_frame, transform_matrix, (selected_image_width, selected_image_height))

            stretched_image = cv2.resize(transformed_matrix, (int(self.frame_width), int(self.frame_height)), interpolation=cv2.INTER_AREA)

            lower_values = np.array([self.lower_h, self.lower_s, self.lower_v])
            upper_values = np.array([self.upper_h, self.upper_s, self.upper_v])

            HSV_frame = cv2.cvtColor(stretched_image, cv2.COLOR_BGR2HSV)
            masked_frame = cv2.inRange(HSV_frame, lower_values, upper_values)
            _, masked_frame_threshold = cv2.threshold(masked_frame, 254, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(masked_frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            detections = []

            for contour in contours:
                area = cv2.contourArea(contour)

                if area < 500:
                    continue

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Calculate the larger bounding box
                center = rect[0]
                size = rect[1]
                angle = rect[2]
                new_size = (size[0] * 1.2, size[1] * 1.2)
                larger_rect = (center, new_size, angle)
                play_new_box = cv2.boxPoints(larger_rect)
                play_new_box = np.int0(play_new_box)

                cv2.drawContours(stretched_image, [play_new_box], 0, (0, 255, 0), 2)

            ret_value, frame_as_jpeg = cv2.imencode(".jpg", stretched_image)
            next_frame = frame_as_jpeg.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + next_frame + b"\r\n\r\n")

    def _detect_faces(self):
        while True:
            ret_value, next_frame = self.camera.read()
            next_frame = cv2.resize(next_frame, (640, 480), interpolation=cv2.INTER_AREA)
            selected_corners = np.float32([self.fov_coordinates[0], self.fov_coordinates[1], self.fov_coordinates[2], self.fov_coordinates[3]])
            selected_image_width = int(np.linalg.norm(selected_corners[0] - selected_corners[1]))
            selected_image_height = int(np.linalg.norm(selected_corners[0] - selected_corners[2]))

            mapped_corners = np.float32([[0, 0], [selected_image_width, 0], [0, selected_image_height], [selected_image_width, selected_image_height]])
            transform_matrix = cv2.getPerspectiveTransform(selected_corners, mapped_corners)
            transformed_matrix = cv2.warpPerspective(next_frame, transform_matrix, (selected_image_width, selected_image_height))

            stretched_image = cv2.resize(transformed_matrix, (int(self.frame_width), int(self.frame_height)), interpolation=cv2.INTER_AREA)

            FaceLocation = self.FindFace.Marks(stretched_image, int(self.frame_width), int(self.frame_height))
            for Face in FaceLocation:
                cv2.rectangle(stretched_image, Face[0], Face[1], (255, 0, 0), 3)

            ret_value, frame_as_jpeg = cv2.imencode(".jpg", stretched_image)
            next_frame = frame_as_jpeg.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + next_frame + b"\r\n\r\n")
    
    def _hand_detection(self):
        while True:
            ret_value, next_frame = self.camera.read()
            next_frame = cv2.resize(next_frame, (640, 480), interpolation=cv2.INTER_AREA)
            selected_corners = np.float32([self.fov_coordinates[0], self.fov_coordinates[1], self.fov_coordinates[2], self.fov_coordinates[3]])
            selected_image_width = int(np.linalg.norm(selected_corners[0] - selected_corners[1]))
            selected_image_height = int(np.linalg.norm(selected_corners[0] - selected_corners[2]))

            mapped_corners = np.float32([[0, 0], [selected_image_width, 0], [0, selected_image_height], [selected_image_width, selected_image_height]])
            transform_matrix = cv2.getPerspectiveTransform(selected_corners, mapped_corners)
            transformed_matrix = cv2.warpPerspective(next_frame, transform_matrix, (selected_image_width, selected_image_height))

            stretched_image = cv2.resize(transformed_matrix, (int(self.frame_width), int(self.frame_height)), interpolation=cv2.INTER_AREA)

            Font = cv2.FONT_HERSHEY_COMPLEX
            FontColor = (255, 0, 0)

            FaceLocation = self.FindFace.Marks(stretched_image, int(self.frame_width), int(self.frame_height))
            for Face in FaceLocation:
                cv2.rectangle(stretched_image, Face[0], Face[1], (255, 0, 0), 3)

            HandLandMarks, HandsTypeData = self.FindHands.Marks(stretched_image, int(self.frame_width), int(self.frame_height))
            for hand,handType in zip(HandLandMarks,HandsTypeData):
                if handType == "Right":
                    Label = "Right"
                if handType == "Left":
                    Label = "Left"
                cv2.putText(stretched_image, Label, hand[8], Font, 2, FontColor, 2)
            
            PoseLandmarks = self.FindPose.Marks(stretched_image, int(self.frame_width), int(self.frame_height))
            if PoseLandmarks is not None and len(PoseLandmarks) >= max([13, 14, 15, 16]) + 1:
                for index in [13, 14, 15, 16]:
                    cv2.circle(stretched_image, PoseLandmarks[index], 20, (0, 255, 0), -1)

            ret_value, frame_as_jpeg = cv2.imencode(".jpg", stretched_image)
            next_frame = frame_as_jpeg.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + next_frame + b"\r\n\r\n")

    def _hand_gesture_recognition(self):
        while True:
            ret_value, next_frame = self.camera.read()
            next_frame = cv2.resize(next_frame, (640, 480), interpolation=cv2.INTER_AREA)
            selected_corners = np.float32([self.fov_coordinates[0], self.fov_coordinates[1], self.fov_coordinates[2], self.fov_coordinates[3]])
            selected_image_width = int(np.linalg.norm(selected_corners[0] - selected_corners[1]))
            selected_image_height = int(np.linalg.norm(selected_corners[0] - selected_corners[2]))

            mapped_corners = np.float32([[0, 0], [selected_image_width, 0], [0, selected_image_height], [selected_image_width, selected_image_height]])
            transform_matrix = cv2.getPerspectiveTransform(selected_corners, mapped_corners)
            transformed_matrix = cv2.warpPerspective(next_frame, transform_matrix, (selected_image_width, selected_image_height))

            stretched_image = cv2.resize(transformed_matrix, (int(self.frame_width), int(self.frame_height)), interpolation=cv2.INTER_AREA)
            
            myHands = []
            Hands = mp.solutions.hands.Hands(static_image_mode=False, 
                                max_num_hands=2, 
                                min_detection_confidence=0.5, 
                                min_tracking_confidence=0.5)

            mpDraw = mp.solutions.drawing_utils

            RGBFrames = cv2.cvtColor(stretched_image, cv2.COLOR_BGR2RGB)
            Results = Hands.process(RGBFrames)
            # print(Results)
            if Results.multi_hand_landmarks is not None:
                for HandLandMarks in Results.multi_hand_landmarks:
                    myHand = []
                    # print(HandLandMarks)
                    mpDraw.draw_landmarks(stretched_image, HandLandMarks, mp.solutions.hands.HAND_CONNECTIONS)
                    for LandMark in HandLandMarks.landmark:
                        # print(LandMark.x, LandMark.y, LandMark.z)
                        myHand.append((int(LandMark.x * int(self.frame_width)), int(LandMark.y * int(self.frame_height))))
                    print("")
                    myHands.append(myHand)
                    print(myHands)

            ret_value, frame_as_jpeg = cv2.imencode(".jpg", stretched_image)
            next_frame = frame_as_jpeg.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + next_frame + b"\r\n\r\n")

    def generate_frame(self, video_type):
        if video_type == "RawVideo":
            return self._raw_video_generator()
        elif video_type == "StrechedVideo":
            return self._stretched_video_generator()
        elif video_type == "MaskedVideo":
            return self._masked_video_generator()
        elif video_type == "ObjectDetectionWithColor":
            return self._object_detection_with_color()
        elif video_type == "DetectFaces":
            return self._detect_faces()
        elif video_type == "HandDetection":
            return self._hand_detection()
        elif video_type == "HandGestureRecognition":
            return self._hand_gesture_recognition()