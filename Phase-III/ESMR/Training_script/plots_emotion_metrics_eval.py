import os
import json
import matplotlib.pyplot as plt
import numpy as np

# === CONFIG ===
results_path = "Phase-III\ESMR\Output_figures_training/"
plot_path = os.path.join(results_path, "Phase-III\ESMR\Output_figures_training\emotion_metrics_eval/")
os.makedirs(plot_path, exist_ok=True)

# === LOAD EMOTION METRICS ===
metrics_file = os.path.join(results_path, "emotion_metrics_by_user.json")
with open(metrics_file, "r") as f:
    emotion_metrics = json.load(f)

# === EXTRACT FIELDS ===
var_list = []
recovery_list = []
neg_days_list = []
engagement_means = []
engagement_stds = []

emotion_order = ["excited", "happy", "neutral", "stressed", "disappointed", "frustrated", "anxious", "angry", "lonely"]
emotion_to_idx = {emo: i for i, emo in enumerate(emotion_order)}

for uid, data in emotion_metrics.items():
    var_list.append(data.get("emotion_variance"))
    recovery_list.append(data.get("time_to_recovery"))
    neg_days_list.append(data.get("neg_emotion_days"))
    engagement_means.append(data.get("engagement_avg"))
    engagement_stds.append(data.get("engagement_std"))

# === PLOT HISTOGRAMS ===
def plot_hist(values, title, xlabel, filename, color):
    plt.figure(figsize=(8, 4))
    plt.hist([v for v in values if v is not None], bins=15, color=color, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("# Users")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, filename))
    plt.close()

plot_hist(var_list, "Emotion Variance Across Users", "Emotion Variance", "emotion_variance_hist.png", "skyblue")
plot_hist(recovery_list, "Time to Recovery Distribution", "Days Until Recovery", "recovery_time_hist.png", "salmon")
plot_hist(neg_days_list, "Negative Emotion Exposure", "# Negative Days", "neg_emotion_days_hist.png", "lightgreen")
plot_hist(engagement_means, "Average Engagement per User", "Engagement Mean", "engagement_mean_hist.png", "gold")
plot_hist(engagement_stds, "Engagement Volatility per User", "Engagement Std Dev", "engagement_std_hist.png", "mediumpurple")

# === TRAJECTORY OVERLAYS ===
for user_id, data in emotion_metrics.items():
    trajectory = data.get("trajectory", [])
    if not trajectory:
        continue
    indices = [emotion_to_idx.get(e, np.nan) for e in trajectory]
    plt.figure(figsize=(12, 3))
    plt.plot(range(1, len(indices) + 1), indices, marker='o')
    plt.yticks(ticks=list(emotion_to_idx.values()), labels=emotion_order)
    plt.xlabel("Day")
    plt.ylabel("Dominant Emotion")
    plt.title(f"User {user_id} — Emotion Trajectory")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f"user_{user_id}_emotion_trajectory_overlay.png"))
    plt.close()

print("[Saved] Emotion metric histograms and trajectory overlays to:", plot_path)
