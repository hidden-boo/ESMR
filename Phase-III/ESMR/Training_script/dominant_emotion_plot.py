import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# === CONFIG ===
results_path = "Phase-III\ESMR\Output_training\episode_logs_by_user.json"
data_path = "Phase-II\TabTransfomers\Datasets\processed_engagement_with_clusters.json/"
plot_path = os.path.join(results_path, "dominant_emotion_trends")
os.makedirs(plot_path, exist_ok=True)

# === EMOTION ENCODING ===
emotion_order = ["excited", "happy", "neutral", "stressed", "disappointed", "frustrated", "anxious", "angry", "lonely"]
emotion_to_idx = {emo: i for i, emo in enumerate(emotion_order)}

# === LOAD EPISODE LOGS ===
episode_path = os.path.join(results_path, "episode_logs_by_user.json")
with open(episode_path, "r") as f:
    episode_logs_by_user = json.load(f)

# === LOAD LIGHTGBM PREDICTIONS ===
lgbm_path = os.path.join(data_path, "daily_recommendations.json")
with open(lgbm_path, "r") as f:
    lgbm_preds = json.load(f)

# === LOAD VIDEO METADATA FOR EMOTIONS ===
video_meta_path = os.path.join(results_path, "video_metadata_dict.json")
with open(video_meta_path, "r") as f:
    video_to_emotion = json.load(f)

# === PLOT DOMINANT EMOTION TRAJECTORY (RL vs LGBM) ===
def plot_dominant_emotion_trajectory(user_id, user_logs):
    days = []
    dominant_emotions_rl = []
    dominant_emotions_lgbm = []

    for entry in user_logs:
        day = entry.get("day")
        if not day:
            continue
        days.append(day)

        # RL user dominant emotion
        if "emotion_intensities" in entry:
            intensities = entry["emotion_intensities"]
            dominant = max(intensities.items(), key=lambda x: x[1])[0]
        else:
            dominant = entry.get("user_emotion", "neutral")
        dominant_emotions_rl.append(emotion_to_idx.get(dominant, -1))

        # LGBM video emotion
        try:
            lgbm_vids = lgbm_preds.get(user_id, [])[day - 1].get("videos", [])
            lgbm_emotions = [video_to_emotion.get(v["video_id"], "neutral") for v in lgbm_vids]
            lgbm_emotions = [e for e in lgbm_emotions if e in emotion_order]
            if lgbm_emotions:
                counts = defaultdict(int)
                for e in lgbm_emotions:
                    counts[e] += 1
                dominant_lgbm = max(counts.items(), key=lambda x: x[1])[0]
                dominant_emotions_lgbm.append(emotion_to_idx[dominant_lgbm])
            else:
                dominant_emotions_lgbm.append(None)
        except:
            dominant_emotions_lgbm.append(None)

    # === Plot ===
    if dominant_emotions_rl:
        plt.figure(figsize=(12, 4))
        plt.plot(days, dominant_emotions_rl, marker='o', label="RL User Emotion")
        if any(e is not None for e in dominant_emotions_lgbm):
            plt.plot(days, [e if e is not None else np.nan for e in dominant_emotions_lgbm], 
                     marker='x', linestyle='--', label="LGBM Video Emotion")
        plt.yticks(ticks=list(emotion_to_idx.values()), labels=emotion_order)
        plt.xlabel("Day")
        plt.ylabel("Dominant Emotion")
        plt.title(f"User {user_id} — Dominant Emotion Over Time (RL vs LGBM)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f"user_{user_id}_dominant_emotion_comparison.png"))
        plt.close()

# === GENERATE PLOTS ===
for i, (uid, logs) in enumerate(episode_logs_by_user.items()):
    if i >= 10:
        break
    plot_dominant_emotion_trajectory(uid, logs)

# === SUMMARIZE FINAL EMOTIONAL STATES ===
happy_counts = 0
positive_trajectories = 0
summary_emotions = defaultdict(int)

for uid, logs in episode_logs_by_user.items():
    dominant_seq = []
    for entry in logs:
        if "emotion_intensities" in entry:
            intensities = entry["emotion_intensities"]
            dominant = max(intensities.items(), key=lambda x: x[1])[0]
        elif "user_emotion" in entry and entry["user_emotion"] in emotion_order:
            dominant = entry["user_emotion"]
        else:
            continue
        dominant_seq.append(dominant)

    if any(d == "happy" for d in dominant_seq):
        happy_counts += 1
    if dominant_seq and emotion_to_idx.get(dominant_seq[-1], 3) < emotion_to_idx.get(dominant_seq[0], 3):
        positive_trajectories += 1
    if dominant_seq:
        summary_emotions[dominant_seq[-1]] += 1

summary = {
    "users_with_happy_dominance": happy_counts,
    "positive_trajectories": positive_trajectories,
    "final_dominant_emotions": dict(summary_emotions),
    "total_users": len(episode_logs_by_user)
}

with open(os.path.join(results_path, "dominant_emotion_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print("[Saved] All dominant emotion comparison plots and summary.")