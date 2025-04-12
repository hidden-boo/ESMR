import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import csv
import json
import pandas as pd
import os

from user_env_no_rewards_ablation import UserEnv
from load_env_data import load_env_data

random.seed(42)
np.random.seed(42)

def load_top_causal_parents(csv_path, top_k=3):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if "Target_Variable" not in df.columns:
        raise ValueError("Column 'Target_Variable' not found in the CSV file.")
    parent_dict = defaultdict(list)
    for _, row in df.iterrows():
        target = row["Target_Variable"]
        parent = row["Causal_Parent"]
        weight = float(row["Causal_Weight"])
        parent_dict[target].append((parent, weight))
    for target in parent_dict:
        parent_dict[target] = sorted(parent_dict[target], key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return parent_dict


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0, min_epsilon=0.05, decay=0.99,
                 use_replay_buffer=True, replay_buffer_size=10000, batch_size=32):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.use_replay_buffer = use_replay_buffer
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.mode_counts = Counter()

        self.Q = defaultdict(lambda: defaultdict(float))
        self.episode_rewards = []
        self.threshold_bonus_log = []
        self.user_rewards_dict = defaultdict(list)
        self.video_ids = list(env.video_metadata.keys())
        self.replay_buffer = []
        self.state_visit_counts = Counter()

    def get_state_id(self, state):
        return tuple(np.round(state, 3))

    def sample_replay(self):
        return random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))

    def boltzmann_action_selection(self, q_vals, temperature=1.0):
        q_array = np.array(list(q_vals.values()))
        q_array = np.clip(q_array, -50, 50)
        q_array -= np.max(q_array)
        exp_q = np.exp(q_array / temperature)
        if np.any(np.isnan(exp_q)) or np.any(np.isinf(exp_q)):
            print("Warning: Detected NaN or Inf in exp_q, replacing with small values.")
            exp_q = np.nan_to_num(exp_q, nan=1e-10, posinf=1e10, neginf=1e-10)
        # Calculate the probabilities (softmax)
        prob_sum = np.sum(exp_q)
        if prob_sum == 0:
            probs = np.ones_like(exp_q) / len(exp_q)  # If all exp_q values are 0, distribute uniformly
        else:
            probs = exp_q / prob_sum
        
        # Ensure that the probabilities are valid (no NaN or negative values)
        probs = np.clip(probs, 0, 1)

        action = np.random.choice(list(q_vals.keys()), p=probs)
        return action
    
    mode_counts = Counter()
    def train(self, epochs=10):
        user_ids = list(self.env.user_data.keys())
        print("Training Q-learning agent on all users...")

        for epoch in range(epochs):
            print(f"=== Epoch {epoch + 1}/{epochs} ===")
            for uid in user_ids:
                user_env = UserEnv(
                    {uid: self.env.user_data[uid]},
                    self.env.video_metadata,
                    self.env.video_id_to_idx,
                    use_emotion_reward=self.env.use_emotion_reward,
                    causal_top_parents=self.env.causal_top_parents,
                    user_negative_state_flag=self.env.user_negative_state_flag,
                    pre_generated_recommendations=self.env.pre_generated_recommendations,
                    emotion_threshold=self.env.emotion_threshold,
                    positive_threshold=self.env.positive_threshold
                )
                state = user_env.reset()
                total_reward = 0
                daily_rewards = []
                daily_bonuses = []

                for _ in range(3):
                    user_env.step_random()
                    state = user_env.get_state()

                while not user_env.done:

                    mode = "Causal" if user_env.in_rl_mode else "LightGBM"
                    self.mode_counts[mode] += 1
                    state_id = self.get_state_id(state)
                    self.state_visit_counts[state_id] += 1
                    q_vals = {a: self.Q[state_id][a] for a in range(len(self.video_ids))}
                    action = self.boltzmann_action_selection(q_vals)
                    next_state, reward, done, info = user_env.step(action)
                    # === NEW: Store top-k videos and emotion intensities ===
                    if user_env.episode_logs:
                        q_vals = {a: self.Q[self.get_state_id(state)][a] for a in range(len(self.video_ids))}
                        top_k = 20
                        top_k_actions = sorted(q_vals.items(), key=lambda x: x[1], reverse=True)[:top_k]
                        top_k_videos = [self.video_ids[a] for a, _ in top_k_actions]

                        # Inject into the last episode log
                        user_env.episode_logs[-1]["top_k_videos"] = top_k_videos
                        user_env.episode_logs[-1]["emotion_intensities"] = {
                            emo: user_env.user_data[user_env.current_user][user_env.current_day - 1].get(emo, 0.0)
                            for emo in user_env.positive_emotions + user_env.negative_emotions
                        }

                    next_state_id = self.get_state_id(next_state) if next_state is not None else None

                    if q_vals:
                        q_array = np.array(list(q_vals.values()))
                        
                        # NEW: Prevent exploding values
                        if np.all(np.isclose(q_array, q_array[0])):
                            entropy = 0  # all values are the same → zero entropy
                        else:
                            try:
                                q_array = np.clip(q_array, -20, 20)  # stricter clipping
                                exp_q = np.exp(q_array - np.max(q_array))
                                probs = exp_q / (np.sum(exp_q) + 1e-8)
                                entropy = -np.sum(probs * np.log(probs + 1e-8))
                                reward += -0.05 * entropy
                            except:
                                print("[WARN] Entropy fallback triggered.")


                    if self.use_replay_buffer:
                        self.replay_buffer.append((state_id, action, reward, next_state_id))
                        if len(self.replay_buffer) > self.replay_buffer_size:
                            self.replay_buffer.pop(0)

                    best_future_q = max(self.Q[next_state_id].values()) if next_state_id and self.Q[next_state_id] else 0
                    self.Q[state_id][action] += self.alpha * (reward + self.gamma * best_future_q - self.Q[state_id][action])

                    state = next_state
                    total_reward += reward
                    daily_rewards.append(reward)
                    if info and "threshold_bonus" in info:
                        daily_bonuses.append(info["threshold_bonus"])

                self.episode_rewards.append(total_reward)
                self.threshold_bonus_log.append(np.sum(daily_bonuses))
                self.user_rewards_dict[uid] = daily_rewards

            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
            print(f"Epoch {epoch + 1}: Avg Reward = {np.mean(self.episode_rewards[-len(user_ids):]):.2f}, ε={self.epsilon:.3f}")

        print("Saving outputs...")
        print(f"Sample user reward data: {list(self.user_rewards_dict.items())[:1]}")
        print(f"Q-table states stored: {len(self.Q)}")
        self.save_outputs(label="engagement_only_rl_ablation")

        print("\n=== Recommendation Mode Usage Summary ===")
        total = sum(self.mode_counts.values())
        for m in ["Causal", "LightGBM"]:
            pct = 100 * self.mode_counts[m] / total if total else 0
            print(f"{m}: {self.mode_counts[m]} times ({pct:.2f}%)")


        if self.use_replay_buffer:
            for state_id, action, reward, next_state_id in self.sample_replay():
                best_future_q = max(self.Q[next_state_id].values()) if next_state_id and self.Q[next_state_id] else 0
                self.Q[state_id][action] += self.alpha * (reward + self.gamma * best_future_q - self.Q[state_id][action])

        return self.episode_rewards, self.user_rewards_dict



    def save_outputs(self, label="hybrid_rl_lightgbm_dynamic"):
        base_path = "Phase-III\Ablations\Output_training\Engagement-only"
        os.makedirs(base_path, exist_ok=True)
        plot_path = os.path.join(base_path, "plots")
        os.makedirs(plot_path, exist_ok=True)
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Q-Learning — Reward Trend ({label})")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f"q_learning_rewards_{label}.png"))
        plt.close()

        with open(os.path.join(base_path, f"q_table_{label}.pkl"), "wb") as f:
            pickle.dump(dict(self.Q), f)

        with open(os.path.join(base_path, f"user_rewards_{label}.pkl"), "wb") as f:
            pickle.dump(dict(self.user_rewards_dict), f)

        results_path = os.path.join(base_path, "results")
        os.makedirs(results_path, exist_ok=True)

        with open(os.path.join(results_path, f"rewards_per_epoch_{label}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Total_Reward", "Threshold_Bonus"])
            for i, (reward, bonus) in enumerate(zip(self.episode_rewards, self.threshold_bonus_log)):
                writer.writerow([i + 1, reward, bonus])

        with open(os.path.join(results_path, f"rare_states_{label}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["State_ID", "Visit_Count"])
            for state_id, count in self.state_visit_counts.items():
                writer.writerow([state_id, count])

        print(f"[Saved] Outputs saved in {base_path}")


if __name__ == "__main__":
    data_path = "processed_engagement_with_clusters.json"
    causal_csv_path = "top_causal_parents.csv"
    lgbm_recs_path = "daily_recommendations.json"

    user_data_dict, video_metadata_dict, _, video_id_to_idx, _, user_negative_state_flag = load_env_data(data_path)
    causal_top_parents = None

    with open(lgbm_recs_path, "r") as f:
        pre_generated_recommendations = json.load(f)

    env = UserEnv(
        user_data_dict,
        video_metadata_dict,
        video_id_to_idx,
        use_emotion_reward=False,        # Clearly disable emotion reward
        causal_top_parents=None,         # Clearly disable causal rewards
        user_negative_state_flag=user_negative_state_flag,
        pre_generated_recommendations=pre_generated_recommendations,
        emotion_threshold=3,
        positive_threshold=0.5
    )


    agent = QLearningAgent(env, alpha=0.1, gamma=0.95, epsilon=0.2)
    episode_rewards, user_rewards_dict = agent.train(epochs=10)
    agent.save_outputs(label="engagement_only_rl_ablation")
