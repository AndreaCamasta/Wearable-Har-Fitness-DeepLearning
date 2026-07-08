# Wearable Human Activity Recognition (HAR) for Fitness Tracking

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://github.com/tensorflow/tensorflow)
[![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow.svg)]()

An end-to-end Machine Learning project focused on **Human Activity Recognition (HAR)** using wearable inertial sensors (IMUs) and heart rate data. The goal is to build a robust system capable of automatically classifying fitness and sports activities (e.g., running, cycling, walking) to power next-generation fitness trackers.

---

## 🎯 Project Overview & Objective
Most commercial smartwatches track steps, but classifying complex fitness movements in real-time requires advanced sequence modeling. This project leverages the **PAMAP2 Physical Activity Monitoring Dataset** to train a lightweight, highly efficient **1D Convolutional Neural Network (1D-CNN)** suitable for edge deployment.

The project is structured in three phases:
1. **Exploratory Data Analysis (EDA) & Preprocessing:** Handling multi-sensor noise, missing values, and temporal windowing (overlapping sliding windows).
2. **Deep Learning Modeling:** Implementing and tuning a 1D-CNN architecture optimized for time-series sensor data.
3. **Interactive Web App Deployment (Upcoming):** Building a Streamlit application where users can upload raw sensor snippets and visually inspect real-time predictions.

---

## 📊 Dataset: PAMAP2
The model is trained on the **PAMAP2 Dataset** from the UCI Machine Learning Repository. It contains data from 9 subjects wearing 3 Inertial Measurement Units (IMUs) (positioned on the wrist, chest, and ankle) and a heart rate monitor, performing 18 distinct daily and sporting activities.

To simulate a standard **smartwatch/wearable ecosystem**, this iteration primarily focuses on:
*   3-axis Accelerometer data from the **wrist/hand** sensor.
*   Continuous **Heart Rate (BPM)** tracking.

---

## 🛠️ Tech Stack & Methods
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Deep Learning Framework:** TensorFlow / Keras
*   **Preprocessing:** Scikit-Learn (LabelEncoder, StandardScaler, Train-Test Split)
*   **Architecture:** 1D-CNN (Convolutional 1D, MaxPooling 1D, Dropout, Dense Layers)
*   **Deployment (Planned):** Streamlit, Streamlit Community Cloud

---

## 🚧 Current Status & Roadmap
- [x] Dataset download and environment setup.
- [x] Exploratory Data Analysis (EDA) and data cleaning (handling transient states and NaN values).
- [x] Implementation of the 5-second overlapping sliding window technique.
- [x] First working baseline with a 1D-CNN on a single subject.
- [ ] Scaling the pipeline to aggregate data from all 9 subjects.
- [ ] Model optimization, hyperparameter tuning, and evaluation (Confusion Matrix, F1-Score).
- [ ] Saving the optimized model (`.h5` / `.onnx`).
- [ ] Building and deploying the Streamlit interactive web dashboard.

---

## 🚀 How to Run the Baseline
*(Temporary instructions while the final pipeline is being built)*
1. Open the exploratory notebook in Google Colab.
2. Ensure runtime is set to **GPU (T4)**.
3. Run the cells sequentially to download the PAMAP2 dataset, preprocess the data, and train the baseline 1D-CNN.
