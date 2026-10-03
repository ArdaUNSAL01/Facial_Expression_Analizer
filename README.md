# Facial Emotion and Micro-Expression Analyzer 🎭

Welcome to my project! This repository contains a real-time Python application that captures a live camera feed to analyze facial emotions and measure micro-expressions (like eyebrow movements) simultaneously. 

**Motivation & Ultimate Goal:** I built this project initially as a hands-on way to learn and integrate powerful Computer Vision and AI libraries. However, **my ultimate goal is to evolve this into a highly stable, comprehensive, and professional data measurement tool for human behavior and facial analytics.** I will be continuously adding new metrics, improving the stability, and expanding the scope of this tool in upcoming updates.

## 🚀 Features
* **Real-Time Camera Feed:** Captures smooth video using OpenCV.
* **Emotion Detection:** Analyzes the dominant emotion (Happy, Sad, Angry, Neutral, etc.) dynamically.
* **Micro-Expression Measurement:** Calculates the geometric distance between the eyebrow and the nose.
* **Scale-Invariant Normalization:** The expression distance is mathematically normalized using the Inter-Pupillary Distance (IPD). This ensures that moving closer to or further from the camera does not break the measurement logic.
* **Data Logging:** Automatically logs all frame data, distances, and emotions into a `results.csv` file for comprehensive data science analysis.

---

## 📚 Libraries Used & Their Logic

This project perfectly harmonizes 4 different libraries. Here is what I learned about them and how I used them:

1. **[MediaPipe](https://developers.google.com/mediapipe) (Modern Tasks API):** 
   * *The Logic:* Used for high-speed, lightweight facial landmark detection. It maps 468 3D points on the face in milliseconds. We use it to pinpoint the exact Cartesian coordinates (X, Y) of the eyebrows, nose, and eyes.
2. **[DeepFace](https://github.com/serengil/deepface):** 
   * *The Logic:* A deep learning facial recognition library. It is used here to classify the overall emotional state of the face from the raw camera frame. 
3. **[OpenCV](https://opencv.org/) (`cv2`):** 
   * *The Logic:* The backbone of the visual interface. It handles hardware communication, color space conversions (BGR to RGB), and drawing the text/UI directly onto the live feed.
4. **[Pandas](https://pandas.pydata.org/):** 
   * *The Logic:* Used for data structuring. All the micro-expression ratios and emotion tags are appended to a list of dictionaries, which Pandas seamlessly converts into a structured DataFrame and exports as a CSV file upon safe exit.

---

## 🛠️ How to Run

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe pandas deepface tf-keras
   ```
3. Run the Python script:
   ```bash
   python facial_expression_analyzer.py
   ```
4. Press **`q`** while on the camera window to safely quit and save your measurement data to `results.csv`.

*(Note: On the first run, the script will automatically download the required MediaPipe task model directly from Google's servers.)*

---

## 🗺️ Future Updates & Roadmap

To achieve my goal of building a comprehensive data measurement tool, I plan to implement the following updates over time:
- [ ] **Broader Behavioral Analytics:** Add Eye Aspect Ratio (EAR) for blink/drowsiness detection and Mouth Aspect Ratio (MAR) for speech/yawning detection.
- [ ] **Enhanced Stability & Performance:** Implement Python `threading` to move AI inference to a background thread, achieving a perfectly stable and lag-free 60 FPS video rendering.
- [ ] **Advanced Data Export:** Support for more detailed time-series data collection and advanced CSV/JSON logging structures.
- [ ] **Live Visualizations:** Create real-time graphs and dashboards alongside the camera feed to monitor data changes instantly.

---

## 🤝 AI Assistance & Transparency

In the spirit of honesty and modern software development, I would like to acknowledge that **Gemini 3.1 AI** was utilized as a pair-programming assistant during the development of this project. It provided valuable guidance in debugging, ensuring cross-library version compatibility (such as migrating to the new MediaPipe Tasks API), and structuring the code for optimal stability. 

---
*Feel free to explore the code, open issues, or suggest improvements. Happy coding!*
