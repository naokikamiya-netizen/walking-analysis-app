# ==========================================================
#  app.py - 歩行分析アプリ (v1.7_no_rotate_fix2)
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
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    all_angles, all_landmarks = [], []

    orientation_locked = False
    is_flipped = False

    for frame_count in range(total_frames):
        success, image = cap.read()
        if not success:
            break

        progress_bar.progress((frame_count + 1) / total_frames * 0.5)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        all_landmarks.append(results.pose_landmarks)

        current_angle = 0
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                p_ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                p_rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                if p_ls.visibility > 0.5 and p_rs.visibility > 0.5:
                    delta_y = (p_rs.y - p_ls.y) * frame_h
                    delta_x = (p_rs.x - p_ls.x) * frame_w
                    angle = math.degrees(math.atan2(delta_y, delta_x))
                    if angle > 90: angle -= 180
                    elif angle < -90: angle += 180
                    if abs(angle) < 80: current_angle = -angle
            except Exception:
                pass
        all_angles.append(current_angle)

    cap.release()
    pose.close()

    if len(all_angles) < int(fps):
        return None, None

    # スパイク除去＋平滑化
    filtered = [all_angles[0]]
    for i in range(1, len(all_angles)):
        if abs(all_angles[i] - filtered[-1]) > 10:
            filtered.append(filtered[-1])
        else:
            filtered.append(all_angles[i])
    smoothed = pd.Series(filtered).rolling(window=11, min_periods=1, center=True).mean().tolist()

    angles_np = np.array(smoothed)
    num_static_frames = min(int(fps * 1.0), 30)
    static_tilt = np.mean(angles_np[:num_static_frames])
    dynamic = angles_np[num_static_frames:]
    left_down = dynamic[dynamic > 0]
    right_down = dynamic[dynamic < 0]
    summary = {
        "static_tilt": static_tilt,
        "avg_left_down_dynamic": np.mean(left_down) if len(left_down) > 0 else 0,
        "avg_right_down_dynamic": np.mean(right_down) if len(right_down) > 0 else 0
    }

    # === 結果ビデオ ===
    status_text.text("ステップ2/2: 結果のビデオを生成中...")
    cap = cv2.VideoCapture(video_path)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    right_panel_w = int(frame_w * 0.7)
    final_w, final_h = frame_w + right_panel_w, frame_h

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (final_w, final_h))

    time_stamps = np.arange(len(angles_np)) / fps
    y_limit = 15.0
    summary_h = int(final_h * 0.45)

    # Summaryパネル画像作成
    fig_s, ax_s = plt.subplots(figsize=(right_panel_w/100, summary_h/100), dpi=100, facecolor='#1E1E1E')
    ax_s.axis('off')
    static_label = "静的傾斜 (立位):"
    static_value_text = f"{abs(summary['static_tilt']):.2f} 度 ({'右' if summary['static_tilt'] < 0 else '左'}肩下がり)"
    ax_s.text(0.5, 0.85, static_label, color='white', fontsize=20, ha='center', va='center', weight='bold')
    ax_s.text(0.5, 0.65, static_value_text, color='#FFC300', fontsize=28, ha='center', va='center', weight='bold')
    ax_s.text(0.1, 0.35, "動的傾斜 (左):", color='#33FF57', fontsize=20, ha='left', va='center', weight='bold')
    ax_s.text(0.9, 0.35, f"{summary['avg_left_down_dynamic']:.2f} 度", color='#33FF57', fontsize=28, ha='right', va='center', weight='bold')
    ax_s.text(0.1, 0.1, "動的傾斜 (右):", color='#33A8FF', fontsize=20, ha='left', va='center', weight='bold')
    ax_s.text(0.9, 0.1, f"{abs(summary['avg_right_down_dynamic']):.2f} 度", color='#33A8FF', fontsize=28, ha='right', va='center', weight='bold')
    fig_s.tight_layout(pad=0)
    fig_s.canvas.draw()
    summary_img = cv2.cvtColor(np.asarray(fig_s.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    plt.close(fig_s)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(total_frames):
        success, image = cap.read()
        if not success:
            break
        progress_bar.progress(0.5 + (i + 1) / total_frames * 0.5)

        if i < len(all_landmarks) and all_landmarks[i]:
            mp_drawing.draw_landmarks(
                image, all_landmarks[i], mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # グラフ生成
        fig_g, ax_g = plt.subplots(figsize=(right_panel_w/100, (final_h - summary_h)/100), dpi=100)
        fig_g.set_facecolor('#1E1E1E')
        ax_g.set_facecolor('#1E1E1E')
        ax_g.tick_params(colors='white', labelsize=10)
        [s.set_edgecolor('white') for s in ax_g.spines.values()]
        ax_g.plot(time_stamps, angles_np, color='#00FFFF', lw=2)
        if i < len(time_stamps):
            ax_g.plot(time_stamps[i], angles_np[i], 'o', markersize=8, color='#FF1493')
        ax_g.axhline(0, color='red', linestyle='--', lw=1)
        ax_g.set_ylim(-y_limit, y_limit)
        ax_g.set_xlabel('時間(秒)', color='white', fontsize=12)
        ax_g.set_ylabel('角度(度)', color='white', fontsize=12)
        fig_g.tight_layout(pad=1.5)
        fig_g.canvas.draw()
        graph_img = cv2.cvtColor(np.asarray(fig_g.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close(fig_g)

        # サイズ揃え
        graph_img = cv2.resize(graph_img, (right_panel_w, final_h - summary_h))
        summary_img_resized = cv2.resize(summary_img, (right_panel_w, summary_h))
        right_panel = cv2.vconcat([graph_img, summary_img_resized])
        final_frame = cv2.hconcat([image, right_panel])
        out.write(final_frame)

    out.release()
    cap.release()
    status_text.text("完了！")
    return temp_output.name, summary
