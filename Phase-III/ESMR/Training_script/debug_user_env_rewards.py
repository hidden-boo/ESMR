import json
import numpy as np
from user_env import UserEnv
from load_env_data import load_env_data
from pprint import pprint

# Paths
DATA_PATH = "Phase-II\TabTransfomers\Datasets\processed_engagement_with_clusters.json"
CAUSAL_PATH = "Phase-II\Causal Discovery\Datasets_generated\top_causal_parents.csv"
RECS_PATH = "Phase-III\LightGBM\Top-N Recommendations\daily_recommendations.json"

def load_top_causal_parents(csv_path="top_causal_parents.csv", k=5):
    import pandas as pd
    df = pd.read_csv(csv_path)
    top_parents = {}
    for target in df["Target_Variable"].unique():
        filtered = df[df["Target_Variable"] == target].sort_values("Causal_Weight", key=abs, ascending=False).head(k)
        top_parents[target] = list(zip(filtered["Causal_Parent"], filtered["Causal_Weight"]))
    return top_parents

def debug_rewards_for_one_user(user_id=None):
    print("\n==== DEBUGGING REWARD FUNCTION ====")
    
    
    user_data_dict, video_metadata_dict, _, video_id_to_idx, _, user_negative_state_flag = load_env_data(DATA_PATH)
    print("\n==== Checking Engagement Score Parsing ====")
    sample_user = list(user_data_dict.keys())[0]
    for i, day_record in enumerate(user_data_dict[sample_user][:3]):
        print(f"User: {sample_user}, Day: {day_record['day']}, Engagement Score: {day_record.get('engagement_score')}")

    causal_top_parents = load_top_causal_parents(CAUSAL_PATH)
    
    with open(RECS_PATH, "r") as f:
        pre_generated_recommendations = json.load(f)

    # Use a fixed user for consistency
    if not user_id:
        user_id = list(user_data_dict.keys())[0]
    print(f"Using User ID: {user_id}")

    # Initialize environment for that user
    env = UserEnv(
        {user_id: user_data_dict[user_id]},
        video_metadata_dict,
        video_id_to_idx,
        use_emotion_reward=True,
        causal_top_parents=causal_top_parents,
        user_negative_state_flag=user_negative_state_flag,
        pre_generated_recommendations=pre_generated_recommendations,
        emotion_threshold=3,
        positive_threshold=0.5
    )

    env.reset(user_id=user_id)

    reward_trajectory = []

    for day in range(30):
        action = env.sample_action()
        _, reward, done, info = env.step(action)
        reward_trajectory.append(reward)

        print(f"\nDay {day}")
        print(f"→ Mode: {info['mode']}")
        print(f"→ Action (video_id): {info['video_id']}")
        print(f"→ Reward: {reward:.3f}")
        print(f"  ├─ Causal Bonus: {info.get('causal_bonus', 'N/A')}")
        print(f"  ├─ Threshold Bonus: {info.get('threshold_bonus', 'N/A')}")
        print(f"  ├─ Emotion Penalty: {env.episode_logs[-1].get('emotion_penalty', 'N/A')}")
        print(f"  ├─ User Emotion: {env.episode_logs[-1].get('user_emotion')}")
        print(f"  ├─ Video Emotion: {env.episode_logs[-1].get('emotion')}")
        print(f"  ├─ Engagement: {env.episode_logs[-1].get('engagement'):.2f}")
        if done:
            break

    print("\n==== Final Reward Trajectory ====")
    print(reward_trajectory)
    print("=================================\n")

    import matplotlib.pyplot as plt

    # Extract logs
    days = list(range(len(env.episode_logs)))
    rewards = [log["reward"] for log in env.episode_logs]
    engagements = [log["engagement"] for log in env.episode_logs]
    dominant_emotions = [log["user_emotion"] for log in env.episode_logs]
    video_emotions = [log["emotion"] for log in env.episode_logs]
    causal_bonus = [log.get("causal_bonus", 0) for log in env.episode_logs]
    threshold_bonus = [log.get("threshold_bonus", 0) for log in env.episode_logs]
    emotion_penalty = [log.get("emotion_penalty", 0) for log in env.episode_logs]
    modes = [log["mode"] for log in env.episode_logs]

    # Plot 1: Reward Trajectory
    plt.figure(figsize=(10, 5))
    plt.plot(days, rewards, marker='o')
    plt.title("Reward Trajectory Over Time")
    plt.xlabel("Day")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

    # Plot 2: Engagement Score vs Reward
    plt.figure(figsize=(10, 5))
    plt.scatter(engagements, rewards, c='blue')
    plt.title("Engagement Score vs Reward")
    plt.xlabel("Engagement Score")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

    # Plot 3: Causal + Threshold + Penalty Breakdown
    plt.figure(figsize=(10, 5))
    plt.plot(days, causal_bonus, label="Causal Bonus", linestyle='--', marker='x')
    plt.plot(days, threshold_bonus, label="Threshold Bonus", linestyle='--', marker='o')
    plt.plot(days, emotion_penalty, label="Emotion Penalty", linestyle='--', marker='s')
    plt.title("Reward Component Breakdown")
    plt.xlabel("Day")
    plt.ylabel("Component Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 4: Dominant Emotion Timeline
    plt.figure(figsize=(12, 3))
    plt.plot(days, dominant_emotions, marker='o')
    plt.xticks(days)
    plt.title("User Dominant Emotion Over Time")
    plt.xlabel("Day")
    plt.ylabel("Emotion")
    plt.grid(True)
    plt.show()

    # Plot 5: RL Mode Activation
    rl_days = [i for i, m in enumerate(modes) if m == "Causal"]
    plt.figure(figsize=(12, 2))
    for day in days:
        color = 'red' if day in rl_days else 'green'
        plt.bar(day, 1, color=color)
    plt.title("RL Mode Activation Timeline")
    plt.xlabel("Day")
    plt.yticks([])
    plt.legend(['Red = Causal (RL)', 'Green = LightGBM'])
    plt.show()


if __name__ == "__main__":
    debug_rewards_for_one_user()
