# ==========================================================
#  app.py - 歩行分析アプリ (v1.9 - 回転補正なし / アップロードしたまま分析)
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
import japanize_matplotlib  # 日本語表示のため


# ==========================================================
# 🎥 メイン分析関数
# ==========================================================
def analyze_walking(video_path, progress_bar, status_text):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    status_text.text("ステップ1/2: 分析データを収集中...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("エラー: 動画ファイルを開けませんでした。")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret:
        st.error("エラー: 動画を読み込めませんでした。")
        return None, None

    frame_h, frame_w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    all_angles, all_landmarks = [], []
    is_flipped, orientation_locked = False, False

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        progress_bar.progress((frame_count + 1) / total_frames * 0.5)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        all_landmarks.append(results.pose_landmarks)
        current_angle = 0

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            try:
                p_ls_raw = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                p_rs_raw = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                nose = lm[mp_pose.PoseLandmark.NOSE.value]

                if not orientation_locked and p_ls_raw.visibility > 0.7 and p_rs_raw.visibility > 0.7:
                    if nose.visibility < 0.1:
                        if p_rs_raw.x > p_ls_raw.x:
                            is_flipped = True
                        orientation_locked = True

                if p_ls_raw.visibility > 0.5 and p_rs_raw.visibility > 0.5:
                    p_ls, p_rs = (p_rs_raw, p_ls_raw) if is_flipped else (p_ls_raw, p_rs_raw)
                    dy = (p_rs.y - p_ls.y) * frame_h
                    dx = (p_rs.x - p_ls.x) * frame_w
                    angle = math.degrees(math.atan2(dy, dx))
                    if angle > 90:
                        angle -= 180
                    elif angle < -90:
                        angle += 180
                    if abs(angle) < 80:
                        current_angle = -angle
            except Exception:
                pass
        all_angles.append(current_angle)

    cap.release()
    pose.close()

    if not all_angles or len(all_angles) < int(fps):
        return None, None

    filtered = [all_angles[0]]
    for i in range(1, len(all_angles)):
        if abs(all_angles[i] - filtered[-1]) > 10:
            filtered.append(filtered[-1])
        else:
            filtered.append(all_angles[i])

    smoothed = pd.Series(filtered).rolling(window=11, min_periods=1, center=True).mean().tolist()
    angles = np.array(smoothed)

    num_static = min(int(fps * 1.0), 30)
    if len(angles) <= num_static:
        return None, None

    static_tilt = np.mean(angles[:num_static])
    dyn = angles[num_static:]
    left = dyn[dyn > 0]
    right = dyn[dyn < 0]

    summary = {
        'static_tilt': static_tilt,
        'avg_left_down_dynamic': np.mean(left) if len(left) > 0 else 0,
        'avg_right_down_dynamic': np.mean(right) if len(right) > 0 else 0
    }

    # ====== グラフと動画出力 ======
    status_text.text("ステップ2/2: 結果のビデオを生成中...")
    cap = cv2.VideoCapture(video_path)
    right_w = int(frame_w * 0.7)
    final_w = frame_w + right_w
    final_h = frame_h

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
