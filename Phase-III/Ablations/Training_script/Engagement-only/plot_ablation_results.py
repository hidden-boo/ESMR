import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# === Paths ===
results_dir = "Phase-III\Ablations\Output_training" #chnage paths acc to the files

# === Load Logs ===
with open(os.path.join(results_dir, "episode_logs_by_user_engagement_only_ablation.json"), "r") as f:
    ablation_logs = json.load(f)

with open(os.path.join(results_dir, "episode_logs_by_user.json"), "r") as f:
    emotion_aware_logs = json.load(f)

# === Helper: Daily Average Emotion Valence ===
def compute_avg_emotions(logs, filter_rl_only=False):
    daily_emotions = {}
    for uid, user_logs in logs.items():
        for entry in user_logs:
            if filter_rl_only and entry.get("mode") != "Causal":
                continue
            day = entry["day"]
            dominant_emotion = entry.get("user_emotion", "neutral")
            valence = 1 if dominant_emotion in ['happy', 'excited'] else (-1 if dominant_emotion in ['stressed', 'anxious', 'disappointed', 'frustrated', 'angry', 'lonely'] else 0)
            daily_emotions.setdefault(day, []).append(valence)
    return [np.mean(daily_emotions[d]) if daily_emotions[d] else 0 for d in sorted(daily_emotions.keys())]

# === Compute ===
ablation_avg_emotions = compute_avg_emotions(ablation_logs)
emotion_aware_avg_emotions = compute_avg_emotions(emotion_aware_logs)

# === Plot 1: Emotional Trajectories ===
plt.figure(figsize=(10, 5))
plt.plot(ablation_avg_emotions, label='Engagement-Only RL Ablation', marker='o')
plt.plot(emotion_aware_avg_emotions, label='Emotion-Aware RL', marker='x')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.xlabel("Day")
plt.ylabel("Average Emotion Valence")
plt.title("Emotional Trajectories: Ablation vs Emotion-Aware RL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "emotional_trajectories_ablation_comparison.png"))
plt.show()

# === Plot 2: Valence Difference Per Day ===
valence_diff = np.array(emotion_aware_avg_emotions) - np.array(ablation_avg_emotions)
plt.figure(figsize=(10, 4))
plt.plot(valence_diff, marker='s', color='crimson')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.xlabel("Day")
plt.ylabel("Valence Improvement (Emotion-Aware - Ablation)")
plt.title("Daily Valence Advantage of Emotion-Aware RL")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "daily_valence_difference_emotion_vs_ablation.png"))
plt.show()

from collections import Counter

def plot_final_emotion_distribution(logs, label):
    final_emotions = [user_logs[-1]['user_emotion'] for user_logs in logs.values()]
    counts = Counter(final_emotions)
    emotions = ['happy', 'excited', 'neutral', 'stressed', 'disappointed', 'frustrated', 'anxious', 'angry', 'lonely']
    values = [counts.get(e, 0) for e in emotions]

    plt.bar(emotions, values)
    plt.xticks(rotation=45)
    plt.ylabel("Number of Users")
    plt.title(f"Final Emotional States — {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"final_emotion_distribution_{label.replace(' ', '_')}.png"))
    plt.show()

plot_final_emotion_distribution(ablation_logs, "Engagement-Only RL")
plot_final_emotion_distribution(emotion_aware_logs, "Emotion-Aware RL")

def plot_happy_user_ratio(logs, label):
    happy_ratios = []
    for d in range(30):
        happy_users = 0
        total_users = 0
        for user_logs in logs.values():
            for log in user_logs:
                if log["day"] == d:
                    total_users += 1
                    if log["user_emotion"] in ["happy", "excited"]:
                        happy_users += 1
        ratio = happy_users / total_users if total_users else 0
        happy_ratios.append(ratio)

    plt.plot(happy_ratios, label=label)

plot_happy_user_ratio(ablation_logs, "Engagement-Only RL")
plot_happy_user_ratio(emotion_aware_logs, "Emotion-Aware RL")
plt.xlabel("Day")
plt.ylabel("% Happy or Excited Users")
plt.title("Emotional Uplift Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "happy_user_ratio_comparison.png"))
plt.show()


# === Formalized Metrics ===
metrics_summary = {}

def calculate_metrics(logs, label):
    recovery_times, neg_days, valence_gains, engagement_scores = [], [], [], []

    for uid, user_logs in logs.items():
        emotions = [entry['user_emotion'] for entry in user_logs]
        engagement = [entry['engagement'] for entry in user_logs]

        recovery_time = next((d for d, e in enumerate(emotions) if e in ['happy', 'excited', 'neutral']), len(emotions))
        neg_count = sum(1 for e in emotions if e in ['stressed', 'disappointed', 'frustrated', 'anxious', 'angry', 'lonely'])

        initial_valence = -1 if emotions[0] in ['stressed', 'disappointed', 'frustrated', 'anxious', 'angry', 'lonely'] else 0
        final_valence = 1 if emotions[-1] in ['happy', 'excited'] else (0 if emotions[-1] == 'neutral' else -1)

        recovery_times.append(recovery_time)
        neg_days.append(neg_count)
        valence_gains.append(final_valence - initial_valence)
        engagement_scores.append(np.mean(engagement))

    metrics_summary[label] = {
        "Emotion Recovery Time (days)": round(np.mean(recovery_times), 2),
        "Negative Emotion Days": round(np.mean(neg_days), 2),
        "Avg Emotion Valence Gain": round(np.mean(valence_gains), 2),
        "Avg Engagement Score": round(np.mean(engagement_scores), 2),
    }

calculate_metrics(ablation_logs, "Engagement-Only RL")
calculate_metrics(emotion_aware_logs, "Emotion-Aware RL")

# === Print and Save Table ===
metrics_df = pd.DataFrame(metrics_summary)
print("\n=== Formalized Metrics Summary ===")
print(metrics_df.T)

metrics_df.T.to_csv(os.path.join(results_dir, "formalized_metrics_comparison.csv"))
