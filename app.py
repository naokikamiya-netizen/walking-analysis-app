# ==========================================================
#  app.py - 歩行分析アプリ (真・最終完成版 - Streamlit Cloud対応)
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

# --- メインの分析ロジック ---
def analyze_walking(video_path, progress_bar, status_text, should_rotate):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    status_text.text("ステップ1/2: 分析データを収集中...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("エラー: 動画ファイルを開けませんでした。")
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_angles, all_landmarks = [], []
    for frame_count in range(total_frames):
        success, image = cap.read()
        if not success: break
        progress_bar.progress((frame_count + 1) / total_frames * 0.5)
        if should_rotate: image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        all_landmarks.append(results.pose_landmarks)
        current_angle = 0
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                p_ls, p_rs = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                delta_y, delta_x = p_rs.y - p_ls.y, p_rs.x - p_ls.x
                angle = math.degrees(math.atan2(delta_y, delta_x))
                if angle > 90: angle -= 180
                elif angle < -90: angle += 180
                if abs(angle) < 80: current_angle = -angle
            except Exception: pass
        all_angles.append(current_angle)
    cap.release(); pose.close()
    if not all_angles or len(all_angles) < int(fps): return None, None
    filtered_angles = [all_angles[0]]
    spike_threshold = 10.0
    for i in range(1, len(all_angles)):
        if abs(all_angles[i] - filtered_angles[-1]) > spike_threshold:
            filtered_angles.append(filtered_angles[-1])
        else:
            filtered_angles.append(all_angles[i])
    angles_series = pd.Series(filtered_angles)
    smoothed_angles = angles_series.rolling(window=5, min_periods=1, center=True).mean().tolist()
    angles_np = np.array(smoothed_angles)
    num_static_frames = min(int(fps * 1.0), 30)
    if len(angles_np) <= num_static_frames: return None, None
    static_tilt = np.mean(angles_np[:num_static_frames])
    dynamic_angles_np = angles_np[num_static_frames:]
    left_down_angles = dynamic_angles_np[dynamic_angles_np > 0]
    right_down_angles = dynamic_angles_np[dynamic_angles_np < 0]
    summary = {
        'static_tilt': static_tilt,
        'avg_left_down_dynamic': np.mean(left_down_angles) if len(left_down_angles) > 0 else 0,
        'avg_right_down_dynamic': np.mean(right_down_angles) if len(right_down_angles) > 0 else 0
    }
    status_text.text("ステップ2/2: 結果のビデオを生成中...")
    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()
    if not success: return None, None
    if should_rotate: image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    frame_h, frame_w, _ = image.shape
    right_panel_w = int(frame_w * 0.7); final_w = frame_w + right_panel_w; final_h = frame_h
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4');
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (final_w, final_h))
    time_stamps, y_limit = np.arange(len(angles_np)) / fps, 15.0
    summary_h = int(final_h * 0.45)
    font_size = 20
    fig_s, ax_s = plt.subplots(figsize=(right_panel_w/100, summary_h/100), dpi=100, facecolor='#1E1E1E')
    ax_s.axis('off')
    static_label = "静的傾斜 (立位):"
    static_value_text = f"{abs(summary['static_tilt']):.2f} 度 ({'右' if summary['static_tilt'] < 0 else '左'}肩下がり)"
    ax_s.text(0.5, 0.85, static_label, color='white', fontsize=font_size, ha='center', va='center', transform=ax_s.transAxes, weight='bold')
    ax_s.text(0.5, 0.65, static_value_text, color='#FFC300', fontsize=font_size, ha='center', va='center', transform=ax_s.transAxes, weight='bold')
    texts_left = [(0.1, 0.35, "動的傾斜 (左):", '#33FF57', font_size)]
    texts_right = [(0.9, 0.35, f"{summary['avg_left_down_dynamic']:.2f} 度", '#33FF57', font_size)]
    texts_left.append((0.1, 0.1, "動的傾斜 (右):", '#33A8FF', font_size))
    texts_right.append((0.9, 0.1, f"{abs(summary['avg_right_down_dynamic']):.2f} 度", '#33A8FF', font_size))
    for x, y, text, color, size in texts_left: ax_s.text(x, y, text, color=color, fontsize=size, ha='left', va='center', transform=ax_s.transAxes, weight='bold')
    for x, y, text, color, size in texts_right: ax_s.text(x, y, text, color=color, fontsize=size, ha='right', va='center', transform=ax_s.transAxes, weight='bold')
    fig_s.tight_layout(pad=0); fig_s.canvas.draw(); summary_img_base = cv2.cvtColor(np.asarray(fig_s.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR); plt.close(fig_s)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(total_frames):
        success, image = cap.read()
        if not success: break
        progress_bar.progress(0.5 + (i + 1) / total_frames * 0.5)
        if should_rotate: image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if i < len(all_landmarks) and all_landmarks[i]:
            mp_drawing.draw_landmarks(image, all_landmarks[i], mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        graph_h = final_h - summary_h
        fig_g, ax_g = plt.subplots(figsize=(right_panel_w/100, graph_h/100), dpi=100)
        fig_g.set_facecolor('#1E1E1E'); ax_g.set_facecolor('#1E1E1E')
        ax_g.tick_params(colors='white', labelsize=10); [s.set_edgecolor('white') for s in ax_g.spines.values()]
        ax_g.plot(time_stamps, angles_np, color='#00FFFF', lw=2)
        if i < len(time_stamps): ax_g.plot(time_stamps[i], angles_np[i], 'o', markersize=8, color='#FF1493')
        ax_g.axh
