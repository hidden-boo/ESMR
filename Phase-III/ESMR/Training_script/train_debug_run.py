
import json
import os
from train_q_learning import QLearningAgent
from user_env import UserEnv
from load_env_data import load_env_data

data_path = "Phase-II\TabTransfomers\Datasets\processed_engagement_with_clusters.json"
causal_csv_path = "Phase-II\Causal Discovery\Datasets_generated\top_causal_parents.csv"
lgbm_recs_path = "Phase-III\LightGBM\Top-N Recommendations\daily_recommendations.json"

user_data_dict, video_metadata_dict, _, video_id_to_idx, _, user_negative_state_flag = load_env_data(data_path)
causal_top_parents = {}
import pandas as pd
from collections import defaultdict

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

causal_top_parents = load_top_causal_parents(causal_csv_path)

with open(lgbm_recs_path, "r") as f:
    pre_generated_recommendations = json.load(f)

env = UserEnv(
    user_data_dict,
    video_metadata_dict,
    video_id_to_idx,
    use_emotion_reward=True,
    causal_top_parents=causal_top_parents,
    user_negative_state_flag=user_negative_state_flag,
    pre_generated_recommendations=pre_generated_recommendations,
    emotion_threshold=3,
    positive_threshold=0.5
)

agent = QLearningAgent(env, alpha=0.1, gamma=0.95, epsilon=0.2)
print("==== Starting short debug training ====")
episode_rewards, user_rewards_dict = agent.train(epochs=2)
print("==== Debug training complete ====")
