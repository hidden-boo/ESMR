import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class UserEnv(gym.Env):
    def __init__(
        self,
        user_data_dict,
        video_metadata_dict,
        video_id_to_idx,
        use_emotion_reward=True,
        max_days=30,
        causal_top_parents=None,
        user_negative_state_flag=None,
        lightgbm_model=None,
        pre_generated_recommendations=None,
        emotion_threshold=3,
        positive_threshold=0.5
    ):
        super(UserEnv, self).__init__()

        self.user_data = user_data_dict
        self.video_metadata = video_metadata_dict
        self.video_id_to_idx = video_id_to_idx
        self.idx_to_video_id = {v: k for k, v in video_id_to_idx.items()}
        self.users = list(self.user_data.keys())

        self.use_emotion_reward = use_emotion_reward
        self.causal_top_parents = causal_top_parents or {}
        self.user_negative_state_flag = user_negative_state_flag or {}
        self.lightgbm_model = lightgbm_model
        self.pre_generated_recommendations = pre_generated_recommendations or {}

        self.emotion_threshold = emotion_threshold
        self.positive_threshold = positive_threshold

        self.max_days = max_days
        self.state_size = 21
        self.positive_emotions = ["happy", "excited"]
        self.negative_emotions = ["stressed", "disappointed", "frustrated", "anxious", "angry", "lonely"]
        self.vulnerable_states = ["stressed", "anxious", "disappointed", "frustrated"]

        self.emotion_mapping = {
            "happy": 1,
            "excited": 1,
            "neutral": 0,
            "stressed": -1,
            "disappointed": -1,
            "frustrated": -1,
            "anxious": -1,
            "angry": -1,
            "lonely": -1
        }

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.video_id_to_idx))
        self.previous_negative_streak_start = None

        self.reset()

    def reset(self, user_id=None):
        self.current_user = user_id if user_id else random.choice(self.users)
        self.current_day = 1
        self.engagement_history = []
        self.rl_mode_days = []
        self.daily_emotion_log = []
        self.episode_logs = []
        self.recent_emotions = []
        self.in_rl_mode = False
        self.done = False
        return self._get_state(self.current_day)

    def _has_sustained_negative_streak(self, threshold=3, intensity_threshold=0.2):
        negative_streak = 0
        for i in range(len(self.daily_emotion_log)-1, len(self.daily_emotion_log)-threshold-1, -1):
            if i < 0:
                break
            dominant_emotion = self.daily_emotion_log[i].get("dominant_emotion", "neutral")
            if dominant_emotion in self.negative_emotions:
                intensity = self.daily_emotion_log[i]["emotion_intensities"].get(dominant_emotion, 0)
                if intensity >= intensity_threshold:
                    negative_streak += 1
                else:
                    break
            else:
                break
        return negative_streak >= threshold

    def _has_positive_recovery(self, intensity_threshold=0.2):
        if not self.daily_emotion_log:
            return False
        recent_day = self.daily_emotion_log[-1]
        dominant_emotion = recent_day.get("dominant_emotion", "neutral")
        if dominant_emotion in self.positive_emotions:
            intensity = recent_day["emotion_intensities"].get(dominant_emotion, 0)
            return intensity >= intensity_threshold
        return False

    
    def _is_emotion_stable(self, window=3):
        if len(self.recent_emotions) < window:
            return False
        return all(e == "neutral" for e in self.recent_emotions[-window:])
    def _get_state(self, day_idx):
        user_days = self.user_data[self.current_user]
        if day_idx >= len(user_days):
            return np.zeros(self.state_size, dtype=np.float32)
        row = user_days[day_idx]
        state = {
            "happy": row.get("happy", 0),
            "stressed": row.get("stressed", 0),
            "frustrated": row.get("frustrated", 0),
            "disappointed": row.get("disappointed", 0),
            "excited": row.get("excited", 0),
            "happy_trend_avg3": row.get("happy_trend_avg3", 0),
            "stressed_trend_avg3": row.get("stressed_trend_avg3", 0),
            "frustrated_trend_avg3": row.get("frustrated_trend_avg3", 0),
            "disappointed_trend_avg3": row.get("disappointed_trend_avg3", 0),
            "churned": int(row.get("churned", 0)),
            "engagement_score": row.get("engagement_score", 0),
            "scrolling_time": row.get("scrolling_time", 0),
            "video_watching_duration": row.get("video_watching_duration", 0),
            "time_spent_daily": row.get("time_spent_daily", 0),
            "engagement_avg": np.mean(self.engagement_history[-3:]) if self.engagement_history else 0,
            "dominant_category": 0 if row.get("dominant_category", "") == "" else 1,
            "dominant_theme": 0 if row.get("dominant_theme", "") == "" else 1,
            "emotion_delta": row.get("emotion_delta", 0),
            "engagement_trend": row.get("engagement_trend", 0)
        }
        return np.array(list(state.values()), dtype=np.float32)

    def _update_emotional_state(self, user_day_data, video_meta):
        emo = video_meta.get("emotion", "neutral")
        intensity = min(max(video_meta.get("emotion_intensity", 0.5), 0), 1)

        if emo in self.positive_emotions:
            user_day_data["happy"] = min(user_day_data.get("happy", 0) + 0.1 * intensity, 1.0)
            if user_day_data["happy"] > 0.3:
                user_day_data["dominant_emotion"] = "happy"
        elif emo in self.negative_emotions:
            user_day_data[emo] = min(user_day_data.get(emo, 0) + 0.1 * intensity, 1.0)
            if user_day_data[emo] > 0.3:
                user_day_data["dominant_emotion"] = emo
            for e in self.negative_emotions:
                if e != emo:
                    user_day_data[e] = max(user_day_data.get(e, 0) - 0.05, 0)

    def step(self, action):
        user_day_data = next((r for r in self.user_data[self.current_user] if r.get("day") == self.current_day), {})
        current_emotion = user_day_data.get("dominant_emotion", "neutral")
        self.recent_emotions.append(current_emotion)
        #print(f"[DEBUG] Current Emotion: {current_emotion}")

        # Exploratory happy shift trigger
        # e.g., fallback to RL if engagement is < threshold for 3+ days
        if len(self.engagement_history) >= 3 and np.mean(self.engagement_history[-3:]) < 500:
            self.in_rl_mode = True
        else:
            self.in_rl_mode = False


        use_rl = self.in_rl_mode
        mode = "Causal" if use_rl else "LightGBM"

        # ===== Video Selection =====
        if use_rl:
            self.rl_mode_days.append(self.current_day)
            video_id = self.idx_to_video_id.get(action)
            video_meta = self.video_metadata.get(video_id, {})
        else:
            user_recs = self.pre_generated_recommendations.get(self.current_user, [])
            matched_day_entry = next((entry for entry in user_recs if entry["day"] == self.current_day), None)
            if matched_day_entry and matched_day_entry.get("videos"):
                top_video = matched_day_entry["videos"][0]
                video_id = top_video["video_id"]
                video_meta = self.video_metadata.get(video_id, {})
            else:
                video_id = self.idx_to_video_id.get(action)
                video_meta = self.video_metadata.get(video_id, {})

        #print(f"[DEBUG] Selected Video ID: {video_id}")
        #print(f"[DEBUG] Video Metadata: {video_meta}")

       # ===== Engagement-Only Reward Ablation =====
        engagement = user_day_data.get("engagement_score", 0)
        video_engagement_score = min(video_meta.get("engagement_score", 0.5), 120)

        reward = 0.4 * engagement + 0.4 * video_engagement_score

        churn_risk = int(user_day_data.get("churned", 0))
        if churn_risk:
            reward -= 0.5

        reward = np.tanh(reward / 100) * 400

         # ===== Logging and Progression =====
        self.daily_emotion_log.append({
            "day": self.current_day,
            "dominant_emotion": current_emotion,
            "video_emotion": video_meta.get("emotion", "neutral"),
            "mode": mode,
            "reward": reward,
            "emotion_intensities": {
                emo: user_day_data.get(emo, 0.0) 
                for emo in self.positive_emotions + self.negative_emotions
            }
})


        self.episode_logs.append({
            "day": self.current_day,
            "action": action,
            "video_id": video_id,
            "reward": reward,
            "engagement": engagement,
            "threshold_bonus": 0,  # explicitly zero for clarity
            "causal_bonus": 0,     # explicitly zero for clarity
            "emotion_penalty": 0,  # explicitly zero for clarity
            "mode": mode,
            "emotion": video_meta.get("emotion", "neutral"),
            "user_emotion": current_emotion,
            "emotion_intensities": {
                emo: user_day_data.get(emo, 0.0) for emo in self.positive_emotions + self.negative_emotions
            }
        })

        self.engagement_history.append(engagement)
        self.current_day += 1
        self.done = self.current_day >= self.max_days
        next_state = self._get_state(self.current_day) if not self.done else None

        return next_state, reward, self.done, {
            "threshold_bonus": 0, "causal_bonus": 0,
            "video_id": video_id, "mode": mode
        }




    def sample_action(self):
        return self.action_space.sample()

    def step_random(self):
        return self.step(self.sample_action())

    def get_state(self):
        return self._get_state(self.current_day)

    def get_rl_usage_summary(self):
        return {"user_id": self.current_user, "total_rl_days": len(self.rl_mode_days), "rl_days": self.rl_mode_days}

    def compute_reward_from_video_metadata(self, video_meta):
        engagement_score = video_meta.get("engagement_score", 0)
        emotion = video_meta.get("emotion", "neutral")
        intensity = min(max(video_meta.get("emotion_intensity", 0.5), 0), 1)
        if emotion in self.positive_emotions:
            reward = engagement_score * (1 + 0.3 * intensity)
        elif emotion in self.negative_emotions:
            reward = engagement_score * (1 - 0.4 * intensity)
        else:
            reward = engagement_score
        return np.clip(reward, -100, 150) 
