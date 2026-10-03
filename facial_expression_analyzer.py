import sys
import os
import math
import urllib.request
import cv2
import pandas as pd
from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

class FacialExpressionAnalyzer:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, "face_landmarker.task")
        
        if not os.path.exists(self.model_path):
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            
        with open(self.model_path, "rb") as f:
            model_buffer = f.read()
            
        base_options = python.BaseOptions(model_asset_buffer=model_buffer)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(options)
        
        self.data_records = []
        self.frame_counter = 0
        self.current_emotion = "Calculating..."

    def calculate_distance(self, face_landmarks):
        nose = face_landmarks[168]
        eyebrow = face_landmarks[336]
        raw_distance = math.hypot(eyebrow.x - nose.x, eyebrow.y - nose.y)
        
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        eye_distance = math.hypot(right_eye.x - left_eye.x, right_eye.y - left_eye.y)
        
        if eye_distance > 0:
            return raw_distance / eye_distance
        return raw_distance

    def run(self):
        if not self.camera.isOpened():
            print("Error: Could not open the camera.")
            return

        print("Program started. Press 'q' to quit.")
        try:
            while True:
                success, frame = self.camera.read()
                if not success:
                    print("Error: Could not read frame from camera.")
                    break

                self.frame_counter += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                detection_result = self.face_detector.detect(mp_image)
                distance = None
                
                if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                    face_landmarks = detection_result.face_landmarks[0]
                    distance = self.calculate_distance(face_landmarks)

                if self.frame_counter % 30 == 0:
                    try:
                        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                        if analysis and isinstance(analysis, list):
                            self.current_emotion = analysis[0]['dominant_emotion']
                    except Exception:
                        pass

                if distance is not None:
                    self.data_records.append({
                        "Frame": self.frame_counter,
                        "Micro_Expression_Distance": round(distance, 4),
                        "Dominant_Emotion": self.current_emotion
                    })

                cv2.putText(frame, f"Emotion: {self.current_emotion}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                if distance is not None:
                    cv2.putText(frame, f"Expression Ratio: {distance:.3f}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

                cv2.imshow("Facial Emotion and Expression Analysis", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    break

        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            
            if self.data_records:
                df = pd.DataFrame(self.data_records)
                df.to_csv("results.csv", index=False)
                print(f"Saved {len(self.data_records)} records to 'results.csv'.")
            else:
                print("No face data to save.")


if __name__ == "__main__":
    app = FacialExpressionAnalyzer()
    app.run()
