# ==========================================================
#  app.py - 歩行分析アプリ (v1.9.1 安定版 - 回転なし + 黒画面対策)
# ==========================================================

import matplotlib
matplotlib.use("Agg")  # ← Streamlit Cloud環境の描画フリーズ防止
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

    # 平滑化フィルタ
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
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (final_w, final_h))

    times = np.arange(len(angles)) / fps
    y_limit = 15.0

    # --- サマリー画像 ---
    summary_h = int(final_h * 0.45)
    fig_s, ax_s = plt.subplots(figsize=(right_w/100, summary_h/100), dpi=100, facecolor='#1E1E1E')
    ax_s.axis('off')
    ax_s.text(0.5, 0.8, f"静的傾斜: {abs(static_tilt):.2f}° ({'右' if static_tilt < 0 else '左'})", color='#FFC300',
              ha='center', va='center', fontsize=26, weight='bold', transform=ax_s.transAxes)
    ax_s.text(0.3, 0.3, f"左傾き平均: {np.mean(left):.2f}°", color='#33FF57', fontsize=20, transform=ax_s.transAxes)
    ax_s.text(0.7, 0.3, f"右傾き平均: {abs(np.mean(right)):.2f}°", color='#33A8FF', fontsize=20, transform=ax_s.transAxes)
    fig_s.tight_layout(pad=0)
    fig_s.canvas.draw()
    summary_img = cv2.cvtColor(np.asarray(fig_s.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    plt.close(fig_s)

    # --- 動画フレーム結合 ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        progress_bar.progress(0.5 + (i + 1) / total_frames * 0.5)
        if i < len(all_landmarks) and all_landmarks[i]:
            mp_drawing.draw_landmarks(
                frame, all_landmarks[i], mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        graph_h = final_h - summary_h
        fig_g, ax_g = plt.subplots(figsize=(right_w/100, graph_h/100), dpi=100)
        fig_g.set_facecolor('#1E1E1E')
        ax_g.set_facecolor('#1E1E1E')
        ax_g.tick_params(colors='white')
        ax_g.plot(times, angles, color='#00FFFF', lw=2)
        ax_g.axhline(0, color='red', linestyle='--')
        if i < len(times):
            ax_g.plot(times[i], angles[i], 'o', color='#FF1493')
        ax_g.set_ylim(-y_limit, y_limit)
        ax_g.set_xlabel('時間(秒)', color='white')
        ax_g.set_ylabel('角度(度)', color='white')
        fig_g.tight_layout(pad=0.5)
        fig_g.canvas.draw()
        graph_img = cv2.cvtColor(np.asarray(fig_g.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close(fig_g)

        right_panel = cv2.vconcat([graph_img, summary_img])
        final_frame = cv2.hconcat([frame, right_panel])
        out.write(final_frame)

    out.release()
    cap.release()
    return temp_output.name, summary


# ==========================================================
# 🧩 表示関数
# ==========================================================
def display_results():
    st.success("🎉 分析が完了しました！")
    st.balloons()
    st.video(st.session_state.video_bytes)
    s = st.session_state.summary
    st.metric("静的傾斜", f"{abs(s['static_tilt']):.2f}° ({'右' if s['static_tilt'] < 0 else '左'})")
    st.metric("動的傾斜 左", f"{s['avg_left_down_dynamic']:.2f}°")
    st.metric("動的傾斜 右", f"{abs(s['avg_right_down_dynamic']):.2f}°")
    st.download_button("結果ビデオをダウンロード", st.session_state.video_bytes, "result.mp4", "video/mp4")
    if st.button("別の動画を分析する"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ==========================================================
# 🚀 Streamlit アプリ本体
# ==========================================================
def main_app():
    st.set_page_config(page_title="歩行分析アプリ", layout="wide")
    if 'page' not in st.session_state:
        st.session_state.page = 'main'

    if st.session_state.page == 'main':
        st.title("🚶‍♂️ 歩行分析アプリ")
        st.write("アップロードした動画をそのままの向きで分析します。")
        uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi", "m4v"])
        if uploaded_file:
            st.session_state.uploaded_file_data = uploaded_file.getvalue()
            st.session_state.page = "confirm"
            st.rerun()

    elif st.session_state.page == "confirm":
        st.title("分析前プレビュー")
        st.video(st.session_state.uploaded_file_data)
        if st.button("分析を開始", type="primary"):
            st.session_state.page = "analysis"
            st.rerun()

    elif st.session_state.page == "analysis":
        st.title("分析中...")
        progress = st.progress(0.0)
        status = st.empty()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(st.session_state.uploaded_file_data)
            video_path = tmp.name

        output, summary = analyze_walking(video_path, progress, status)
        if output and summary:
            with open(output, "rb") as f:
                st.session_state.video_bytes = f.read()
            st.session_state.summary = summary
            st.session_state.page = "results"
        else:
            st.session_state.page = "error"
        os.remove(video_path)
        if output and os.path.exists(output):
            os.remove(output)
        st.rerun()

    elif st.session_state.page == "results":
        display_results()

    elif st.session_state.page == "error":
        st.error("動画の分析に失敗しました。再度お試しください。")
        if st.button("最初に戻る"):
            st.session_state.page = "main"
            st.rerun()


if __name__ == "__main__":
    main_app()
