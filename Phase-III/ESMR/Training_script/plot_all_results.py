
import pickle
import json
import os
import pandas as pd
from eval_plotting import (
    reward_summary,
    plot_comparison,
    plot_histogram,
    plot_cumulative_rewards,
    plot_per_user_rewards,
    plot_rolling_emotion_avg,
    plot_dominant_emotion_shift,
    plot_negative_content_distribution,
    plot_recommender_usage_distribution,
    plot_user_emotion_trajectory,
    plot_rl_trigger_heatmap
)

# === Load paths ===
base_path = "Phase-III\ESMR\Output_figures_training/"
results_path = os.path.join(base_path, "results")

# === Load required data ===
with open(os.path.join(base_path, "user_rewards_hybrid_rl_lightgbm_dynamic.pkl"), "rb") as f:
    user_rewards_emotion = pickle.load(f)

with open(os.path.join(base_path, "q_table_hybrid_rl_lightgbm_dynamic.pkl"), "rb") as f:
    q_table = pickle.load(f)

with open(os.path.join(results_path, "rewards_per_epoch_hybrid_rl_lightgbm_dynamic.csv"), "r") as f:
    df = pd.read_csv(f)
rewards_emotion = df["Total_Reward"].tolist()

# === Load optional emotional analysis files ===
with open(os.path.join(results_path, "user_data_dict.json"), "r") as f:
    user_data_dict = json.load(f)
with open(os.path.join(results_path, "rl_mode_days_dict.json"), "r") as f:
    rl_days_dict = json.load(f)
with open(os.path.join(results_path, "episode_logs_by_user.json"), "r") as f:
    episode_logs_by_user = json.load(f)
with open(os.path.join(results_path, "content_emotions_dict.json"), "r") as f:
    content_emotions_dict = json.load(f)
with open(os.path.join(results_path, "video_metadata_dict.json"), "r") as f:
    video_metadata_dict = json.load(f)

# === Execute All Plots ===
reward_summary(rewards_emotion)
plot_comparison(rewards_emotion)
plot_histogram(rewards_emotion)
plot_cumulative_rewards(rewards_emotion)
plot_per_user_rewards(user_rewards_emotion)

for uid in list(user_data_dict.keys())[:5]:
    plot_rolling_emotion_avg(user_data_dict, uid)
    plot_user_emotion_trajectory(user_data_dict, rl_days_dict, uid)

plot_dominant_emotion_shift(episode_logs_by_user)
plot_negative_content_distribution(content_emotions_dict, video_metadata_dict)
plot_recommender_usage_distribution(rl_days_dict)
plot_rl_trigger_heatmap(rl_days_dict)
