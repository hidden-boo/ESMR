# --- FULL PATCHED eval_plotting.py (COMPLETE) ---
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
from collections import Counter

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def smooth_rewards(rewards, window=50):
    return np.convolve(rewards, np.ones(window) / window, mode="valid") if len(rewards) >= window else rewards

# ============================== #
#       AGENT REWARD PLOTS       #
# ============================== #

def reward_summary(rewards_emotion, rewards_no_emotion=None, save_path="Phase-III\ESMR\Output_training/"):
    ensure_dir(save_path)
    smooth = smooth_rewards(rewards_emotion, window=50)
    smooth_reward_emotion = np.mean(smooth) if len(smooth) > 0 else 0
    avg_reward_emotion = np.mean(rewards_emotion)
    avg_reward_no_emotion = np.mean(rewards_no_emotion) if rewards_no_emotion else None
    summary = (
        "=== Evaluation Summary ===\n"
        f"Emotion-Aware    → Mean: {avg_reward_emotion:.2f}, Std: {np.std(rewards_emotion):.2f}, Smoothed: {smooth_reward_emotion:.2f}\n"
    )
    if rewards_no_emotion:
        summary += f"Emotion-Agnostic → Mean: {avg_reward_no_emotion:.2f}, Std: {np.std(rewards_no_emotion):.2f}\n"
    print(summary)
    with open(os.path.join(save_path, "reward_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print("[Saved] reward_summary.txt")

def plot_comparison(rewards_emotion, rewards_no_emotion=None, window=50, save_path="Phase-III\ESMR\Output_figures_training/"):
    ensure_dir(save_path)
    plt.figure(figsize=(10, 6))
    plt.plot(smooth_rewards(rewards_emotion, window), label="Emotion-Aware (Smoothed)", linewidth=2)
    plt.plot(rewards_emotion, alpha=0.2, label="Raw Emotion-Aware")
    if rewards_no_emotion:
        plt.plot(smooth_rewards(rewards_no_emotion, window), label="Emotion-Agnostic (Smoothed)", linewidth=2)
        plt.plot(rewards_no_emotion, alpha=0.2, label="Raw Emotion-Agnostic")
    plt.axhline(np.mean(rewards_emotion), color='blue', linestyle='--', label=f"Mean (Emotion-Aware): {np.mean(rewards_emotion):.2f}")
    if rewards_no_emotion:
        plt.axhline(np.mean(rewards_no_emotion), color='orange', linestyle='--', label=f"Mean (Emotion-Agnostic): {np.mean(rewards_no_emotion):.2f}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Comparison (Emotion-Aware vs Emotion-Agnostic)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "reward_comparison.png"))
    plt.close()
    print("[Saved] reward_comparison.png")

def plot_histogram(rewards_emotion, rewards_no_emotion=None, save_path="Phase-III\ESMR\Output_figures_training/"):
    ensure_dir(save_path)
    plt.figure(figsize=(8, 5))
    plt.hist(rewards_emotion, bins=30, alpha=0.6, label="Emotion-Aware")
    if rewards_no_emotion:
        plt.hist(rewards_no_emotion, bins=30, alpha=0.6, label="Emotion-Agnostic")
    plt.xlabel("Total Reward per Episode")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "reward_histogram.png"))
    plt.close()
    print("[Saved] reward_histogram.png")

# ============================== #
#         PER-USER PLOTS         #
# ============================== #

def plot_per_user_rewards(user_rewards_emotion, user_rewards_agnostic=None, save_path="Phase-III\ESMR\Output_figures_training\users"):
    ensure_dir(save_path)
    for user_id in user_rewards_emotion:
        plt.figure(figsize=(8, 5))
        plt.plot(user_rewards_emotion[user_id], label="Emotion-Aware")
        smoothed = smooth_rewards(user_rewards_emotion[user_id], window=5)
        plt.plot(smoothed, label="Smoothed", linestyle="--")
        if user_rewards_agnostic and user_id in user_rewards_agnostic:
            plt.plot(user_rewards_agnostic[user_id], label="Emotion-Agnostic")
        plt.xlabel("Day")
        plt.ylabel("Reward")
        plt.title(f"User {user_id} - Daily Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"user_{user_id}_reward_comparison.png"))
        plt.close()

def plot_user_emotion_trajectory(user_data_dict, rl_days_dict, user_id, save_path="Phase-III\ESMR\Output_figures_training\dominant_emotion_trends"):
    ensure_dir(save_path)
    user_days = user_data_dict[user_id]
    days = [entry["day"] for entry in user_days]
    
    # Initializing emotional states for all emotions
    happy = [entry.get("happy", 0) for entry in user_days]
    stressed = [entry.get("stressed", 0) for entry in user_days]
    disappointed = [entry.get("disappointed", 0) for entry in user_days]
    excited = [entry.get("excited", 0) for entry in user_days]
    frustrated = [entry.get("frustrated", 0) for entry in user_days]
    
    if len(days) < 3:
        print(f"[Warning] User {user_id} has short trajectory: {len(days)} days")
    
    # Plotting emotion trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(days, happy, label="Happy")
    plt.plot(days, stressed, label="Stressed")
    plt.plot(days, disappointed, label="Disappointed")
    plt.plot(days, excited, label="Excited")
    plt.plot(days, frustrated, label="Frustrated")
    
    # Plot vertical lines for RL days and GBM days
    for day in rl_days_dict.get(user_id, []):
        plt.axvline(x=day, color='red', linestyle='--', alpha=0.4, label="RL Day")
    
    for day in range(len(days)):
        if day not in rl_days_dict.get(user_id, []):
            plt.axvline(x=day, color='blue', linestyle='--', alpha=0.4, label="GBM Day")
    
    plt.xlabel("Day")
    plt.ylabel("Emotion Intensity")
    plt.title(f"User {user_id} - Emotion Trajectory (RL and GBM usage shown)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"user_{user_id}_emotion_trajectory.png"))
    plt.close()
    print(f"[Saved] user_{user_id}_emotion_trajectory.png")


# ============================== #
#       RL USAGE PLOTTING        #
# ============================== #

def plot_recommender_usage_distribution(rl_mode_days_dict, max_days=30, save_path="Phase-III\ESMR\Output_figures_training/"):
    ensure_dir(save_path)
    user_rl_days = {uid: len(days) for uid, days in rl_mode_days_dict.items()}
    counts = list(user_rl_days.values())
    plt.figure(figsize=(10, 5))
    plt.hist(counts, bins=range(0, max_days + 2), alpha=0.7, edgecolor='black')
    plt.xlabel("# of Days RL Recommender Used")
    plt.ylabel("# of Users")
    plt.title("Distribution of RL Usage Across Users")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "rl_usage_histogram.png"))
    plt.close()
    print("[Saved] rl_usage_histogram.png")
    total_users = len(rl_mode_days_dict)
    users_triggered_rl = sum(1 for c in counts if c > 0)
    avg_duration = np.mean([c for c in counts if c > 0]) if users_triggered_rl > 0 else 0
    print(f"\n[Summary] {users_triggered_rl}/{total_users} users triggered RL recommender ({100 * users_triggered_rl / total_users:.2f}%)")
    print(f"[Average RL Duration] {avg_duration:.2f} days")

# ============================== #
#     DOMINANT EMOTION SHIFTS    #
# ============================== #

def plot_dominant_emotion_shift(episode_logs_by_user, save_path="Phase-III\ESMR\Output_figures_training/"):
    ensure_dir(save_path)
    emotion_shift_counts = Counter()
    for user_id, logs in episode_logs_by_user.items():
        for i in range(1, len(logs)):
            if logs[i-1]["mode"] == "Causal":
                prev = logs[i-1]["user_emotion"]
                curr = logs[i]["user_emotion"]
                transition = f"{prev} → {curr}"
                emotion_shift_counts[transition] += 1
    transitions, counts = zip(*emotion_shift_counts.items()) if emotion_shift_counts else ([], [])
    plt.figure(figsize=(12, 6))
    plt.barh(transitions, counts, color="steelblue")
    plt.xlabel("Frequency")
    plt.title("Dominant Emotion Shifts on Day d+1 after RL Trigger")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "emotion_shift_dplus1.png"))
    plt.close()
    print("[Saved] emotion_shift_dplus1.png")

# ============================== #
#  NEGATIVE CONTENT DISTRIBUTION #
# ============================== #

def plot_negative_content_distribution(content_emotions_dict, video_metadata_dict, save_path="Phase-III\ESMR\Output_figures_training/"):
    negative_shares = []
    for user_id, daily_emotions in content_emotions_dict.items():
        for day_emotions in daily_emotions:
            if day_emotions:
                neg_count = 0
                total_intensity = 0
                for emo in day_emotions:
                    if emo in ["stressed", "anxious", "disappointed", "frustrated"]:
                        for vid, meta in video_metadata_dict.items():
                            if meta.get("emotion") == emo:
                                intensity = meta.get("emotion_intensity", 3)
                                neg_count += 1
                                total_intensity += intensity
                                break
                weighted_negative_exposure = total_intensity / neg_count if neg_count > 0 else 0
                negative_shares.append(weighted_negative_exposure)
    plt.figure(figsize=(10, 5))
    plt.hist(negative_shares, bins=20, alpha=0.75, edgecolor='black')
    plt.xlabel("% Negative Content Recommended (Weighted by Intensity)")
    plt.ylabel("# of Days")
    plt.title("Distribution of Negative Content Exposure (All Users, All Days) - With Intensity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "negative_content_exposure_with_intensity.png"))
    plt.close()
    print("[Saved] negative_content_exposure_with_intensity.png")
# ============================== #
#      CUMULATIVE REWARD PLOT    #
# ============================== #

def plot_cumulative_rewards(rewards_emotion, rewards_no_emotion=None, save_path="Phase-III\ESMR\Output_figures_training/"):
    ensure_dir(save_path)
    cum_emotion = np.cumsum(rewards_emotion)
    plt.figure(figsize=(10, 6))
    plt.plot(cum_emotion, label="Emotion-Aware", linewidth=2)
    if rewards_no_emotion:
        cum_agnostic = np.cumsum(rewards_no_emotion)
        plt.plot(cum_agnostic, label="Emotion-Agnostic", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "cumulative_rewards.png"))
    plt.close()
    print("[Saved] cumulative_rewards.png")

# ============================== #
#     ROLLING AVERAGE EMOTIONS   #
# ============================== #

def plot_rolling_emotion_avg(user_data_dict, user_id, window=3, save_path="Phase-III\ESMR\Output_figures_training\dominant_emotion_trends"):
    ensure_dir(save_path)
    user_days = user_data_dict[user_id]
    days = [entry["day"] for entry in user_days]
    emotions = ["happy", "stressed", "disappointed", "excited", "frustrated"]
    plt.figure(figsize=(10, 6))
    for emo in emotions:
        raw = [entry.get(emo, 0) for entry in user_days]
        rolling = np.convolve(raw, np.ones(window)/window, mode='valid')
        plt.plot(days[:len(rolling)], rolling, label=f"{emo} (rolling {window})")
    plt.xlabel("Day")
    plt.ylabel("Rolling Avg Emotion Intensity")
    plt.title(f"User {user_id} - Rolling Avg Emotions (window={window})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"user_{user_id}_rolling_emotion_avg.png"))
    plt.close()
    print(f"[Saved] user_{user_id}_rolling_emotion_avg.png")
