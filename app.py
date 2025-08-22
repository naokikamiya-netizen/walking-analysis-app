# ==========================================================
#  app.py - 歩行分析アプリ (真・最終完成版 - 構文エラー修正)
# ==========================================================
import streamlit as st
from scipy.signal import find_peaks
import cv2
import mediapipe as mp
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import tempfile

# --- フォントの設定 ---
FONT_PATH = 'NotoSansJP-Bold.otf'
if os.path.exists(FONT_PATH):
    font_prop = fm.FontProperties(fname=FONT_PATH)
else:
    font_prop = fm.FontProperties()
    st.warning(f"フォントファイル '{FONT_PATH}' が見つかりません。日本語が文字化けする可能性があります。")

# --- メインの分析ロジック ---
def analyze_walking(video_path, progress_bar, status_text):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    status_text.text("ステップ1/2: 分析データを収集中...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("エラー: 動画ファイルを開けませんでした。")
        return None, None
    
    success, image = cap.read()
    if not success:
        st.error("エラー: 動画フレームを読み込めませんでした。")
        return None, None
    
    frame_h, frame_w, _ = image.shape
    auto_rotate = frame_h > frame_w
    
    cap.set(c
