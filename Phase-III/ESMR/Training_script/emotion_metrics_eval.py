import os
import json
from collections import defaultdict
import numpy as np

# === CONFIG ===
results_dir = "Phase-III\ESMR\Output_figures_training"
log_file = os.path.join(results_dir, "episode_logs_by_user.json")
out_file = os.path.join(results_dir, "emotion_metrics_by_user.json")

emotion_order = ["excited", "happy", "neutral", "stressed", "disappointed", "frustrated", "anxious", "angry", "lonely"]
negative_emotions = {"stressed", "disappointed", "frustrated", "anxious", "angry", "lonely"}

with open(log_file, "r") as f:
    episode_logs_by_user = json.load(f)

emotion_metrics_by_user = {}

def get_engagement_series(logs):
    return [entry.get("engagement", 0.0) for entry in logs]

def get_dominant_emotion_sequence(logs):
    dominant_seq = []
    for entry in logs:
        if "emotion_intensities" in entry:
            intensities = entry["emotion_intensities"]
            dominant = max(intensities.items(), key=lambda x: x[1])[0]
        elif "user_emotion" in entry:
            dominant = entry["user_emotion"]
        else:
            dominant = "neutral"
        dominant_seq.append(dominant)
    return dominant_seq

for uid, logs in episode_logs_by_user.items():
    dominant_seq = get_dominant_emotion_sequence(logs)
    rewards = [entry.get("reward", 0) for entry in logs]
    engagement_series = get_engagement_series(logs)

    neg_emotion_days = sum(1 for e in dominant_seq if e in negative_emotions)

    recovery_day = -1
    for i, e in enumerate(dominant_seq):
        if e in {"happy", "excited"}:
            recovery_day = i + 1  # day starts from 1
            break

    emotion_idx = [emotion_order.index(e) if e in emotion_order else 2 for e in dominant_seq]
    emotion_variance = float(np.std(emotion_idx))
    avg_reward = float(np.mean(rewards))
    reward_std = float(np.std(rewards))

    emotion_metrics_by_user[uid] = {
        "neg_emotion_days": neg_emotion_days,
        "time_to_recovery": recovery_day,
        "emotion_variance": emotion_variance,
        "avg_reward": avg_reward,
        "reward_std": reward_std,
        "trajectory": dominant_seq,
        "engagement_series": engagement_series,
        "engagement_avg": float(np.mean(engagement_series)) if engagement_series else None,
        "engagement_std": float(np.std(engagement_series)) if engagement_series else None
    }

# === SAVE ===
with open(out_file, "w") as f:
    json.dump(emotion_metrics_by_user, f, indent=2)

print("[Saved] Emotion metrics per user to:", out_file)