import cv2

Font = cv2.FONT_HERSHEY_COMPLEX
FontColor = (255, 0, 0)

class MediapipeMyHands():
    import mediapipe as mp
    def __init__(self, maxHands = 2, tol1 = .5, tol2 = .5):
        self.hands = self.mp.solutions.hands.Hands(
                                                    static_image_mode=False, 
                                                    max_num_hands=maxHands, 
                                                    min_detection_confidence=tol1, 
                                                    min_tracking_confidence=tol2
                                                )
        
    def Marks(self, frame, width, height):
        MyHands = []
        HandsType=[]
        RGBFrames = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Results = self.hands.process(RGBFrames)
        if Results.multi_hand_landmarks is not None:
            print(Results.multi_hand_landmarks)
            for Hand in Results.multi_handedness:
                print(Hand)
                #print(Hand.classification)
                #print(Hand.classification[0])
                HandType=Hand.classification[0].label
                HandsType.append(HandType)
            for HandLandMarks in Results.multi_hand_landmarks:
                myHand=[]
                for LandMark in HandLandMarks.landmark:
                    myHand.append((int(LandMark.x*width),int(LandMark.y*height)))
                MyHands.append(myHand)
        return MyHands, HandsType


class MediapipePose:
    import mediapipe as mp

    def __init__(self, still=False, upperBody=False, smoothData=True, tol1=0.5, tol2=0.5):
        self.myPose = self.mp.solutions.pose.Pose(
            static_image_mode=still,
            smooth_landmarks=smoothData,
            min_detection_confidence=tol1,
            min_tracking_confidence=tol2
        )

        if not still:
            self.myPose._upper_body_only = upperBody

    def Marks(self,frame, width, height):
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.myPose.process(frameRGB)
        poseLandmarks=[]
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark: 
                poseLandmarks.append((int(lm.x*width),int(lm.y*height)))
        return poseLandmarks

class MediaPipeFace:
    import mediapipe as mp
    
    def __init__(self):
        self.MyFace = self.mp.solutions.face_detection.FaceDetection()
    
    def Marks(self, frame, width, height):
        RGBFrames = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Results = self.MyFace.process(RGBFrames)
        FaceBoundingBoxs = []
        if Results.detections is not None:
            for Face in Results.detections:
                BoundingBox = Face.location_data.relative_bounding_box
                TopLeft = (int(BoundingBox.xmin * width), int(BoundingBox.ymin * height))
                BottomRight = (int((BoundingBox.xmin + BoundingBox.width) * width), int((BoundingBox.ymin + BoundingBox.height) * height))
                FaceBoundingBoxs.append((TopLeft, BottomRight))
        return FaceBoundingBoxs