#Facial Emotion and Micro-Expression Analyzer 🎭

Welcome to my project! This repository contains a real-time Python application that captures a live camera feed to analyze facial emotions and measure micro-expressions (like eyebrow movements) simultaneously. 

**Motivation & Ultimate Goal:** I built this project initially as a hands-on way to learn and integrate powerful Computer Vision and AI libraries. When I started, I did not yet have in-depth knowledge of Machine Learning algorithms and model training pipelines; therefore, I utilized **DeepFace** as an accessible high-level library to handle facial emotion recognition. 

However, **my next primary objective is to learn Machine Learning from the ground up, replace DeepFace completely, and develop my own lightweight ML model trained directly on facial expression and landmark datasets.** Ultimately, I aim to evolve this into an independent, highly stable, and comprehensive data measurement tool for human behavior and facial analytics—powered by custom-trained in-house models rather than heavy external wrappers.

## 🚀 Features
* **Real-Time Camera Feed:** Captures smooth video using OpenCV at a rock-solid 60 FPS.
* **Asynchronous Multi-Threading:** DeepFace emotion recognition runs in a background thread, preventing any video stutter or lag.
* **Dynamic Face Mesh & HUD:** Visualizes all 468 3D facial landmarks, color-coded measurement vectors, and a translucent dark HUD dashboard in real time.
* **Comprehensive Biometric Metrics:**
  * Dual Eyebrow Tracking (Left & Right micro-expressions)
  * Eye Aspect Ratio (EAR - Blink & drowsiness tracking)
  * Mouth Opening Ratio (MOR - Yawn & surprise tracking)
  * Smile Width Ratio (Happiness & expression intensity)
* **Emotion Detection:** Analyzes dominant emotion with optimized Face ROI cropping for higher sharpness and accuracy.
* **Scale-Invariant Normalization:** All distances are normalized via Inter-Pupillary Distance (IPD).
* **Data Logging:** Logs detailed per-frame time-series measurements and emotions to `results.csv`.

---

## ✨ Latest Updates (v2.0)

A major architectural upgrade has been implemented:
1. **Zero-Lag Video Stream:** Offloaded heavy DeepFace neural network inference to a daemon background thread (`threading.Thread`), unlocking silky-smooth 60 FPS video capture.
2. **Face ROI Cropping & 5x Acceleration:** The face region is automatically cropped from MediaPipe landmarks before feeding into DeepFace with `detector_backend='skip'`. This eliminates room/background noise, increases emotion sharpness, and delivers a 4-5x speedup.
3. **468-Point Mesh & Vector HUD:** The live camera window now renders the full 468-point facial mesh, a dynamic bounding box (`Face Tracked`), color-coded measurement lines (yellow for eyebrows, cyan for IPD baseline, magenta/orange for mouth), and a translucent HUD info panel.
4. **Rich Multidimensional Logging:** `results.csv` now captures individual left/right eyebrow ratios, mouth opening, smile width, and EAR alongside the preserved `Micro_Expression_Distance` metric.
5. **Technical Documentation:** An in-depth mathematical and algorithmic explanation guide has been added in [EKLEMELER_VE_MANTIK.md](EKLEMELER_VE_MANTIK.md).

---

## 📚 Libraries Used & Their Logic

This project perfectly harmonizes 4 different libraries. Here is what I learned about them and how I used them:

1. **[MediaPipe](https://developers.google.com/mediapipe) (Modern Tasks API):** 
   * *The Logic:* Used for high-speed, lightweight facial landmark detection. It maps 468 3D points on the face in milliseconds. We use it to pinpoint the exact Cartesian coordinates (X, Y) of the eyebrows, nose, and eyes.
2. **[DeepFace](https://github.com/serengil/deepface):** 
   * *The Logic:* A deep learning facial recognition library currently used to classify the overall emotional state of the face from the camera frame. *(Note: As part of my machine learning learning path, this will be replaced in v3.0 by a custom-trained lightweight model).*
3. **[OpenCV](https://opencv.org/) (`cv2`):** 
   * *The Logic:* The backbone of the visual interface. It handles hardware communication, color space conversions (BGR to RGB), and drawing the text/UI directly onto the live feed.
4. **[Pandas](https://pandas.pydata.org/):** 
   * *The Logic:* Used for data structuring. All the micro-expression ratios and emotion tags are appended to a list of dictionaries, which Pandas seamlessly converts into a structured DataFrame and exports as a CSV file upon safe exit.

---

## 🛠️ How to Run

1. Clone the repository.
2. Activate the virtual environment and install dependencies:
   ```bash
   source .venv/bin/activate
   pip install opencv-python mediapipe pandas deepface tf-keras
   ```
3. Run the application:
   ```bash
   ./run.sh
   # or: python facial_expression_analyzer.py
   ```
4. Press **`q`** while on the camera window to safely quit and save your measurement data to `results.csv`.

*(Note: On the first run, the script will automatically download the required MediaPipe task model directly from Google's servers.)*

---

## 🗺️ Future Updates & Roadmap

To achieve my goal of building a comprehensive data measurement tool, I plan to implement the following updates over time:
- [x] **Broader Behavioral Analytics:** Added Eye Aspect Ratio (EAR) for blink detection and Mouth Opening Ratio (MOR) / Smile Width. *(Completed in v2.0)*
- [x] **Enhanced Stability & Performance:** Implemented Python `threading` and Face ROI cropping, achieving lag-free 60 FPS rendering. *(Completed in v2.0)*
- [x] **Advanced Data Export:** Multi-metric time-series data collection with full backward compatibility in `results.csv`. *(Completed in v2.0)*
- [x] **Live Visualizations:** Real-time 468-mesh overlay, tracking bounding box, vector connections, and translucent HUD panel. *(Completed in v2.0)*
- [ ] **Phase 3: Replacing DeepFace with Custom Machine Learning Model (Upcoming Focus):**
  - **Machine Learning Learning Journey:** Learn core ML theory, training pipelines, and evaluation metrics from scratch.
  - **Feature Engineering:** Convert MediaPipe's 468 3D landmark coordinates and computed geometric ratios (EAR, MOR, Eyebrow distances, Smile width) into structured tabular feature vectors $(X \in \mathbb{R}^n)$.
  - **Dataset Training:** Train a lightweight classifier (e.g., Random Forest, SVM, or compact MLP) on facial expression benchmark datasets (such as CK+, JAFFE, or custom collected landmark logs).
  - **Zero Overhead & Microsecond Inference:** Completely eliminate the heavy TensorFlow/DeepFace footprint, achieving sub-millisecond CPU inference with minimal RAM usage.
- [ ] **Live Graphing Dashboard:** Real-time matplotlib/PyQt graphing alongside the camera window.

---

## 🤝 AI Assistance & Transparency

In the spirit of honesty and modern software development, I would like to acknowledge that **Gemini 3.1 AI** was utilized as a pair-programming assistant during the development of this project. It provided valuable guidance in debugging, ensuring cross-library version compatibility (such as migrating to the new MediaPipe Tasks API), and structuring the code for optimal stability. 

---
*Feel free to explore the code, open issues, or suggest improvements. Happy coding!*

Arda ÜNSAL

