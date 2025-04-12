import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# === CONFIG ===
base_dir = "Phase-III\ESMR\Output_figures_training"
results_path = os.path.join(base_dir, "Rl_aware", "results")
data_path = os.path.join(base_dir, "Data")
plot_path = os.path.join(results_path, "plots", "additional_emotion_comparison")
os.makedirs(plot_path, exist_ok=True)

# === LOAD DATA ===
with open(os.path.join(results_path, "episode_logs_by_user.json"), "r") as f:
    episode_logs = json.load(f)
with open(os.path.join(data_path, "daily_recommendations.json"), "r") as f:
    lgbm_recs = json.load(f)
with open(os.path.join(results_path, "video_metadata_dict.json"), "r") as f:
    video_metadata = json.load(f)

# === SETUP ===
emotion_order = ["excited", "happy", "neutral", "stressed", "disappointed", "frustrated", "anxious", "angry", "lonely"]
emotion_to_idx = {emo: i for i, emo in enumerate(emotion_order)}
negative_emotions = {"stressed", "disappointed", "frustrated", "anxious", "angry", "lonely"}
valence_map = {"happy": 2, "excited": 1, "neutral": 0, "lonely": -1, "stressed": -2, "angry": -2, "disappointed": -1, "frustrated": -1, "anxious": -2}


# === PLOT DOMINANT EMOTION RL vs LGBM ===
def plot_rl_vs_lgbm_emotion(user_id, logs):
    lgbm_days = lgbm_recs.get(user_id, [])
    rl_emotions = []
    lgbm_emotions = []
    rl_mode_days = []
    days = []

    for entry in logs:
        day = entry.get("day")
        if day is None:
            continue
        days.append(day)

        # RL Dominant
        if "emotion_intensities" in entry:
            intensities = entry["emotion_intensities"]
            dominant = max(intensities.items(), key=lambda x: x[1])[0]
        else:
            dominant = entry.get("user_emotion", "neutral")
        rl_emotions.append(emotion_to_idx.get(dominant, -1))

        # RL Mode Highlighting
        if entry.get("mode") == "Causal":
            rl_mode_days.append(day)

        # LGBM Dominant
        try:
            top_vids = lgbm_days[day - 1].get("videos", [])
            emotion_counts = defaultdict(int)
            for v in top_vids:
                emo = video_metadata.get(v["video_id"], {}).get("emotion", "neutral")
                if emo in emotion_order:
                    emotion_counts[emo] += 1
            dominant_lgbm = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
            lgbm_emotions.append(emotion_to_idx.get(dominant_lgbm, -1))
        except:
            lgbm_emotions.append(None)

    if not rl_emotions:
        return

    plt.figure(figsize=(12, 4))
    plt.plot(days, rl_emotions, label="RL (Actual User Emotion)", marker='o', color='blue')
    if any(e is not None for e in lgbm_emotions):
        plt.plot(days, [e if e is not None else np.nan for e in lgbm_emotions], label="LGBM (Video Emotion)", linestyle='--', marker='x', color='orange')

    for d in rl_mode_days:
        plt.axvspan(d - 0.5, d + 0.5, color='blue', alpha=0.1)

    plt.xlabel("Day")
    plt.ylabel("Dominant Emotion")
    plt.yticks(ticks=list(emotion_to_idx.values()), labels=emotion_order)
    plt.title(f"User {user_id} — RL vs LGBM Emotion Trajectory")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f"user_{user_id}_rl_vs_lgbm_emotion.png"))
    plt.close()

# === PLOT EMOTION INTENSITY CURVES ===
def plot_emotion_intensities(user_id, logs):
    lgbm_days = lgbm_recs.get(user_id, [])
    rl_mode_days = []
    intensity_by_emotion = defaultdict(list)
    lgbm_emotion_trend = []
    days = []

    for entry in logs:
        day = entry.get("day")
        if day is None:
            continue
        days.append(day)

        if entry.get("mode") == "Causal":
            rl_mode_days.append(day)

        for emo in emotion_order:
            intensity = entry.get("emotion_intensities", {}).get(emo, 0.0)
            intensity_by_emotion[emo].append(intensity)

        try:
            top_vids = lgbm_days[day - 1].get("videos", [])
            emotion_counts = defaultdict(int)
            for v in top_vids:
                emo = video_metadata.get(v["video_id"], {}).get("emotion", "neutral")
                if emo in emotion_order:
                    emotion_counts[emo] += 1
            dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
            lgbm_emotion_trend.append(dominant)
        except:
            lgbm_emotion_trend.append("neutral")

    plt.figure(figsize=(12, 6))
    for emo in emotion_order:
        plt.plot(days, intensity_by_emotion[emo], label=f"RL-{emo}")

    lgbm_idx = [emotion_to_idx.get(e, -1) for e in lgbm_emotion_trend]
    plt.plot(days, lgbm_idx, linestyle='--', color='black', label="LGBM Dominant Emotion")

    for d in rl_mode_days:
        plt.axvspan(d - 0.5, d + 0.5, color='blue', alpha=0.05)

    plt.xlabel("Day")
    plt.ylabel("Emotion Intensity / Index")
    plt.title(f"User {user_id} — Emotion Intensity Curve (RL vs LGBM)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f"user_{user_id}_emotion_intensity_comparison.png"))
    plt.close()

# === RUN FOR FIRST 10 USERS ===
rl_neg_counts = []
lgbm_neg_counts = []
valence_gains = []

for i, (uid, logs) in enumerate(episode_logs.items()):
    if i >= 10:
        break
    plot_rl_vs_lgbm_emotion(uid, logs)
    plot_emotion_intensities(uid, logs)

    rl_neg = 0
    lgbm_neg = 0
    valence_seq = []
    lgbm_days = lgbm_recs.get(uid, [])
    for entry in logs:
        day = entry.get("day")
        if day is None:
            continue

        if "emotion_intensities" in entry:
            intensities = entry["emotion_intensities"]
            dominant = max(intensities.items(), key=lambda x: x[1])[0]
        else:
            dominant = entry.get("user_emotion", "neutral")
        valence_seq.append(valence_map.get(dominant, 0))
        if dominant in negative_emotions:
            rl_neg += 1

        try:
            top_vids = lgbm_days[day - 1].get("videos", [])
            emotion_counts = defaultdict(int)
            for v in top_vids:
                emo = video_metadata.get(v["video_id"], {}).get("emotion", "neutral")
                if emo in emotion_order:
                    emotion_counts[emo] += 1
            dominant_lgbm = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
            if dominant_lgbm in negative_emotions:
                lgbm_neg += 1
        except:
            continue
    
    if valence_seq:
        valence_gains.append(valence_seq[-1] - valence_seq[0])


    rl_neg_counts.append(rl_neg)
    lgbm_neg_counts.append(lgbm_neg)

# === PLOT NEGATIVE EMOTION COUNT COMPARISON ===
plt.figure(figsize=(6, 6))
x = np.arange(len(rl_neg_counts))
width = 0.35
plt.bar(x - width/2, rl_neg_counts, width, label="RL", color="blue")
plt.bar(x + width/2, lgbm_neg_counts, width, label="LGBM", color="orange")
plt.xlabel("User Index")
plt.ylabel("# Days with Negative Dominant Emotion")
plt.title("Negative Emotion Exposure: RL vs LGBM")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "negative_emotion_day_comparison_bar.png"))
plt.close()

# === PLOT VALENCE GAIN HISTOGRAM ===
plt.figure(figsize=(8, 4))
plt.hist(valence_gains, bins=10, color='purple', edgecolor='black')
plt.title("Distribution of Emotion Valence Gain (RL)")
plt.xlabel("Valence Gain (Final - Initial)")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "valence_gain_distribution.png"))
plt.close()

print("[Saved] All emotion comparison plots, intensity overlays, and summary histograms.")

print("[Saved] All emotion comparison plots and overlays.")