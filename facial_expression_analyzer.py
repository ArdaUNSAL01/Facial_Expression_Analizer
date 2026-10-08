import sys
import os
import math
import urllib.request
import threading
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
            print("MediaPipe modeli indiriliyor, lütfen bekleyin...")
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
        self.is_analyzing = False

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

    def calculate_metrics(self, face_landmarks):
        def dist(p1, p2):
            return math.hypot(p2.x - p1.x, p2.y - p1.y)

        left_eye_outer = face_landmarks[33]
        right_eye_outer = face_landmarks[263]
        eye_distance = dist(left_eye_outer, right_eye_outer)
        if eye_distance <= 0:
            eye_distance = 1.0

        nose = face_landmarks[168]
        right_eyebrow = face_landmarks[336]
        left_eyebrow = face_landmarks[107]
        right_brow_dist = dist(nose, right_eyebrow) / eye_distance
        left_brow_dist = dist(nose, left_eyebrow) / eye_distance

        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]
        mouth_open_ratio = dist(upper_lip, lower_lip) / eye_distance

        mouth_left = face_landmarks[61]
        mouth_right = face_landmarks[291]
        smile_width_ratio = dist(mouth_left, mouth_right) / eye_distance

        left_eye_h = dist(face_landmarks[33], face_landmarks[133])
        left_eye_v = dist(face_landmarks[159], face_landmarks[145])
        left_ear = (left_eye_v / left_eye_h) if left_eye_h > 0 else 0.0

        right_eye_h = dist(face_landmarks[362], face_landmarks[263])
        right_eye_v = dist(face_landmarks[386], face_landmarks[374])
        right_ear = (right_eye_v / right_eye_h) if right_eye_h > 0 else 0.0

        avg_ear = (left_ear + right_ear) / 2.0

        return {
            "right_brow_distance": right_brow_dist,
            "left_brow_distance": left_brow_dist,
            "mouth_open_ratio": mouth_open_ratio,
            "smile_width_ratio": smile_width_ratio,
            "eye_aspect_ratio": avg_ear
        }

    def draw_face_visuals(self, frame, face_landmarks):
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min = max(0, min(xs) - 15)
        x_max = min(w, max(xs) + 15)
        y_min = max(0, min(ys) - 15)
        y_max = min(h, max(ys) + 15)
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 220, 255), 2)
        cv2.putText(frame, "Face Tracked", (x_min, max(20, y_min - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

        for pt in points:
            cv2.circle(frame, pt, 1, (0, 230, 115), -1)

        cv2.line(frame, points[168], points[336], (0, 255, 255), 2)
        cv2.line(frame, points[168], points[107], (0, 255, 255), 2)
        cv2.circle(frame, points[168], 4, (0, 255, 0), -1)
        cv2.circle(frame, points[336], 4, (0, 255, 255), -1)
        cv2.circle(frame, points[107], 4, (0, 255, 255), -1)

        cv2.line(frame, points[33], points[263], (255, 255, 0), 1)
        cv2.circle(frame, points[33], 3, (255, 255, 0), -1)
        cv2.circle(frame, points[263], 3, (255, 255, 0), -1)

        cv2.line(frame, points[61], points[291], (255, 0, 255), 2)
        cv2.line(frame, points[13], points[14], (0, 165, 255), 2)
        cv2.circle(frame, points[61], 4, (255, 0, 255), -1)
        cv2.circle(frame, points[291], 4, (255, 0, 255), -1)
        cv2.circle(frame, points[13], 3, (0, 165, 255), -1)
        cv2.circle(frame, points[14], 3, (0, 165, 255), -1)

        cv2.circle(frame, points[152], 4, (200, 200, 255), -1)
        cv2.circle(frame, points[10], 4, (200, 200, 255), -1)

    def draw_hud(self, frame, metrics):
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (370, 215), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, f"Emotion: {self.current_emotion}", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        if metrics is not None:
            cv2.putText(frame, f"Expression Ratio: {metrics['right_brow_distance']:.3f}", (25, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
            cv2.putText(frame, f"Left Brow Ratio: {metrics['left_brow_distance']:.3f}", (25, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.putText(frame, f"Mouth Open Ratio: {metrics['mouth_open_ratio']:.3f}", (25, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
            cv2.putText(frame, f"Smile Width Ratio: {metrics['smile_width_ratio']:.3f}", (25, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
            cv2.putText(frame, f"Eye Aspect Ratio: {metrics['eye_aspect_ratio']:.3f}", (25, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
        else:
            cv2.putText(frame, "Face: Searching...", (25, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    def _async_emotion_analysis(self, face_image):
        try:
            try:
                analysis = DeepFace.analyze(
                    face_image,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip'
                )
            except Exception:
                analysis = DeepFace.analyze(
                    face_image,
                    actions=['emotion'],
                    enforce_detection=False
                )

            if analysis and isinstance(analysis, list):
                self.current_emotion = analysis[0]['dominant_emotion']
        except Exception:
            pass
        finally:
            self.is_analyzing = False

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
                metrics = None
                face_landmarks = None
                face_roi = None
                
                if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                    face_landmarks = detection_result.face_landmarks[0]
                    metrics = self.calculate_metrics(face_landmarks)

                    h, w, _ = frame.shape
                    xs = [int(lm.x * w) for lm in face_landmarks]
                    ys = [int(lm.y * h) for lm in face_landmarks]
                    pad = 25
                    x1, x2 = max(0, min(xs) - pad), min(w, max(xs) + pad)
                    y1, y2 = max(0, min(ys) - pad), min(h, max(ys) + pad)
                    if x2 > x1 and y2 > y1:
                        face_roi = frame[y1:y2, x1:x2].copy()

                if self.frame_counter % 10 == 0 and not self.is_analyzing and face_roi is not None:
                    self.is_analyzing = True
                    threading.Thread(
                        target=self._async_emotion_analysis,
                        args=(face_roi,),
                        daemon=True
                    ).start()

                if metrics is not None:
                    self.data_records.append({
                        "Frame": self.frame_counter,
                        "Micro_Expression_Distance": round(metrics["right_brow_distance"], 4),
                        "Right_Brow_Distance": round(metrics["right_brow_distance"], 4),
                        "Left_Brow_Distance": round(metrics["left_brow_distance"], 4),
                        "Mouth_Open_Ratio": round(metrics["mouth_open_ratio"], 4),
                        "Smile_Width_Ratio": round(metrics["smile_width_ratio"], 4),
                        "Eye_Aspect_Ratio": round(metrics["eye_aspect_ratio"], 4),
                        "Dominant_Emotion": self.current_emotion
                    })

                if face_landmarks is not None:
                    self.draw_face_visuals(frame, face_landmarks)

                self.draw_hud(frame, metrics)

                cv2.imshow("Facial Emotion and Expression Analysis", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    break

        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            
            import datetime
                df = pd.DataFrame(self.data_records)
                csv_path = os.path.join(self.current_dir, "results.csv")
                df.to_csv(csv_path, index=False, sep=';', encoding="utf-8-sig")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                history_csv_path = os.path.join(self.current_dir, f"results_{timestamp}.csv")
                df.to_csv(history_csv_path, index=False, sep=';', encoding="utf-8-sig")
                print(f"[BAŞARILI] {len(self.data_records)} adet analiz kaydı yerel dosyalara kaydedildi:")
                print(f" -> Güncel dosya: {csv_path}")
                print(f" -> Arşiv dosyası: {history_csv_path}")
            else:
                print("Kaydedilecek yüz verisi bulunamadı.")


if __name__ == "__main__":
    try:
        app = FacialExpressionAnalyzer()
        app.run()
    except Exception as e:
        import traceback
        print("\n========================================================")
        print("[KRİTİK HATA] Program beklenmeyen bir hata ile karşılaştı:")
        traceback.print_exc()
        print("========================================================")
        input("Pencereyi kapatmak için Enter'a basın...")
