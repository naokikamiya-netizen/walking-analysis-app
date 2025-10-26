# ==========================================================
# app.py - 歩行分析アプリ (v1.8 - 自動回転補正付き)
# ==========================================================
import streamlit as st
from scipy.signal import find_peaks
import cv2
import mediapipe as mp
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
import japanize_matplotlib

# --- 回転補正関数 ---
def auto_rotate_frame(frame):
    h, w = frame.shape[:2]
    if h > w:  # 縦長動画なら横向きに補正
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

# --- 歩行解析 ---
def analyze_walking(video_path, progress_bar, status_text):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = 0

    x_data, y_data = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = auto_rotate_frame(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            mid_x = (l_shoulder.x + r_shoulder.x) / 2
            mid_y = (l_shoulder.y + r_shoulder.y) / 2
            x_data.append(mid_x)
            y_data.append(mid_y)

        progress += 1
        progress_bar.progress(progress / total_frames)
        status_text.text(f"解析中: {progress}/{total_frames} フレーム")

    cap.release()
    return np.array(x_data), np.array(y_data)

# --- Streamlit UI ---
st.title("歩行分析アプリ v1.8（自動回転補正付き）")
uploaded_file = st.file_uploader("歩行動画をアップロード", type=["mp4", "mov", "avi"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmp_path = tmpfile.name

    st.video(tmp_path)
    progress_bar = st.progress(0)
    status_text = st.empty()

    x_data, y_data = analyze_walking(tmp_path, progress_bar, status_text)
    plt.figure(figsize=(8, 4))
    plt.plot(x_data, label="肩中心のX軌跡")
    plt.legend()
    plt.xlabel("フレーム")
    plt.ylabel("X位置")
    plt.title("体幹の左右移動")
    st.pyplot(plt)
