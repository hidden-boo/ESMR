import os
import json
import pickle
import pandas as pd
from collections import defaultdict
from user_env_no_rewards_ablation import UserEnv
from load_env_data import load_env_data
from train_q_learning_ablation import QLearningAgent
import numpy as np

def load_top_causal_parents(csv_path, top_k=3):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    parent_dict = defaultdict(list)
    for _, row in df.iterrows():
        target = row["Target_Variable"]
        parent = row["Causal_Parent"]
        weight = float(row["Causal_Weight"])
        parent_dict[target].append((parent, weight))
    for target in parent_dict:
        parent_dict[target] = sorted(parent_dict[target], key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return parent_dict

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(x) for x in obj]
    elif isinstance(obj, (np.int64, np.int32, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

def safe_json_dump(obj, path, label=None):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[OK] {label or os.path.basename(path)} saved: {size_mb:.2f} MB")
    except Exception as e:
        print(f"[ERROR] Failed to save {label or os.path.basename(path)}: {e}")

if __name__ == "__main__":
    print("=== RE-SIMULATING TO GENERATE LOG FILES ===")

    data_path = "processed_engagement_with_clusters.json" #path change acc to structure of files
    causal_path = "top_causal_parents.csv"#path change acc to structure of files
    lgbm_recs_path = "daily_recommendations.json"#path change acc to structure of files
    q_table_path = "q_table_engagement_only_rl_ablation.pkl"#path change acc to structure of files

    user_data_dict, video_metadata_dict, _, video_id_to_idx, _, user_negative_state_flag = load_env_data(data_path, emotion_threshold=3)
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)
    with open(lgbm_recs_path, "r") as f:
        pre_generated_recommendations = json.load(f)

    rl_mode_days_dict = {}
    episode_logs_by_user = {}
    content_emotions_dict = {}

    for uid in user_data_dict:
        user_env = UserEnv(
            {uid: user_data_dict[uid]},
            video_metadata_dict,
            video_id_to_idx,
            use_emotion_reward=False,          # Disable explicitly for ablation
            causal_top_parents=None,           # Disable explicitly for ablation
            user_negative_state_flag=user_negative_state_flag,
            pre_generated_recommendations=pre_generated_recommendations,
            emotion_threshold=3,
            positive_threshold=0.5
        )

        user_env.reset()
        for day in range(1, 31):
            state = user_env.get_state()
            state_id = tuple(round(val, 3) for val in state)
            q_values = q_table.get(state_id, {})
            action = max(q_values.items(), key=lambda x: x[1])[0] if q_values else user_env.sample_action()
            user_env.step(action)

        rl_mode_days_dict[uid] = user_env.rl_mode_days
        episode_logs_by_user[uid] = sanitize(user_env.episode_logs)
        content_emotions_dict[uid] = [
            [log["emotion"] for log in user_env.episode_logs if log["day"] == d]
            for d in range(user_env.max_days)
        ]

    out_dir = "Phase-III\Ablations\Ouptut_figures_training"
    os.makedirs(out_dir, exist_ok=True)

    rl_predictions_dict = {}

    for uid in user_data_dict:
        user_env = UserEnv(
            {uid: user_data_dict[uid]},
            video_metadata_dict,
            video_id_to_idx,
            use_emotion_reward=False,
            causal_top_parents=None,
            user_negative_state_flag=user_negative_state_flag,
            pre_generated_recommendations=pre_generated_recommendations,
            emotion_threshold=3,
            positive_threshold=0.5
        )
        user_env.reset()
        for _ in range(30):
            state = user_env.get_state()
            q_values = q_table.get(tuple(round(val, 3) for val in state), {})
            action = max(q_values.items(), key=lambda x: x[1])[0] if q_values else user_env.sample_action()
            user_env.step(action)

        # Save daily dominant emotion predictions from RL model
        # === Construct new RL predictions from episode logs ===
        rl_predictions_dict[uid] = []
        for log in user_env.episode_logs:
            top_k_videos = log.get("top_k_videos", [log["video_id"]])
            rl_predictions_dict[uid].append({
                "day": log["day"],
                "source": "RL" if log.get("mode") == "Causal" else "LGBM",
                "videos": top_k_videos,
                "dominant_emotion": log.get("dominant_emotion", log.get("emotion", "neutral")),
                "emotion_intensities": log.get("emotion_intensities", {})
            })


    safe_json_dump(user_data_dict, os.path.join(out_dir, "user_data_dict_engagement_only_ablation.json"))
    safe_json_dump(rl_mode_days_dict, os.path.join(out_dir, "rl_mode_days_dict_engagement_only_ablation.json"))
    safe_json_dump(episode_logs_by_user, os.path.join(out_dir, "episode_logs_by_user_engagement_only_ablation.json"))
    safe_json_dump(content_emotions_dict, os.path.join(out_dir, "content_emotions_dict_engagement_only_ablation.json"))
    safe_json_dump(video_metadata_dict, os.path.join(out_dir, "video_metadata_dict_engagement_only_ablation.json"))

    print(" Logs saved for evaluation.")
