# Wearable Human Activity Recognition (HAR) for Fitness Tracking

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://github.com/tensorflow/tensorflow)
[![Keras](https://img.shields.io/badge/Keras-3.0+-red.svg)](https://keras.io/)
[![Status](https://img.shields.io/badge/Status-Completed%20%26%20Live-green.svg)]()

An end-to-end Machine Learning project focused on **Human Activity Recognition (HAR)** using wearable inertial sensors (IMUs) and heart rate data. The goal is to build a robust system capable of automatically classifying fitness and sports activities (e.g., running, cycling, walking) to power next-generation fitness trackers.

---

## 🎯 Project Overview & Objective
Most commercial smartwatches track steps, but classifying complex fitness movements in real-time requires advanced sequence modeling. This project leverages the **PAMAP2 Physical Activity Monitoring Dataset** to train a lightweight, highly efficient **1D Convolutional Neural Network (1D-CNN)** suitable for edge deployment.

The project is structured in three completed phases:
1. **Exploratory Data Analysis (EDA) & Preprocessing:** Handling multi-sensor noise, missing values, and temporal windowing (overlapping sliding windows).
2. **Deep Learning Modeling:** Implementing and tuning a 1D-CNN architecture optimized for time-series sensor data.
3. **Interactive Web App Deployment:** Building and deploying a live Streamlit application where users can simulate wearable data streaming and visually inspect real-time predictions.

---

## 📊 Dataset: PAMAP2
The model is trained on the **PAMAP2 Dataset** from the UCI Machine Learning Repository. It contains data from 9 subjects wearing 3 Inertial Measurement Units (IMUs) (positioned on the wrist, chest, and ankle) and a heart rate monitor, performing 18 distinct daily and sporting activities.

To simulate a standard **smartwatch/wearable ecosystem**, this iteration primarily focuses on:
* 3-axis Accelerometer data from the **wrist/hand** sensor.
* Continuous **Heart Rate (BPM)** tracking.

---

## 🛠️ Tech Stack & Methods
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Plotly, Matplotlib, Seaborn
* **Deep Learning Framework:** TensorFlow / Keras 3 (with XLA environment tuning for cloud compatibility)
* **Preprocessing:** Scikit-Learn (LabelEncoder, StandardScaler)
* **Architecture:** 1D-CNN (Convolutional 1D, MaxPooling 1D, Dropout, Dense Layers)
* **Deployment:** Streamlit, Streamlit Community Cloud

---

## 🚀 Web App & Live Demo
The interactive dashboard is fully deployed and accessible live. It simulates a wearable device data stream, processing sliding temporal windows to output real-time activity classifications.

🔗 **[Live Streamlit Application](https://wearable-har-fitness-deeplearning-dexelt9a4fp4pglwfu5hic.streamlit.app/)**

### Cloud Optimization Notes:
The production application includes backend environment handling (`TF_XLA_FLAGS`) specifically tuned to guarantee memory stability and prevent segmentation faults on distributed cloud CPU infrastructures during neural network serialization.

---

## 📈 Project Roadmap & Achievements
- [x] Dataset download and environment setup.
- [x] Exploratory Data Analysis (EDA) and data cleaning (handling transient states and NaN values).
- [x] Implementation of the 5-second overlapping sliding window technique.
- [x] First working baseline with a 1D-CNN on a single subject.
- [x] Scaling the pipeline to aggregate data from subjects.
- [x] Model optimization, hyperparameter tuning, and evaluation.
- [x] Saving the optimized model and pipeline metadata (`.pkl`).
- [x] Building and deploying the Streamlit interactive web dashboard on Streamlit Cloud.

---

## 💻 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/AndreaCamasta/wearable-har-fitness-deeplearning.git](https://github.com/AndreaCamasta/wearable-har-fitness-deeplearning.git)
   cd wearable-har-fitness-deeplearning
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Lounch the streamlit app:
   ```
   streamlit run app.py
   ```   
