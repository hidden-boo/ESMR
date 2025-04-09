import json
import numpy as np

# Load data for Emotion-Aware and Engagement-Only models
with open("episode_logs_by_user.json", "r") as f:
    emotion_aware_logs = json.load(f)

with open("episode_logs_by_user_engagement_only_ablation.json", "r") as f:
    engagement_only_logs = json.load(f)

# Define emotion valence mappings
EMOTION_VALENCE = {
    "happy": 1,
    "excited": 0.8,
    "neutral": 0,
    "stressed": -0.7,
    "disappointed": -0.8,
    "frustrated": -0.9,
    "anxious": -0.6,
    "angry": -1,
    "lonely": -0.5
}

# Function to compute final-day emotion valence for users
def compute_positive_final_valence(logs):
    positive_count = 0
    total_users = len(logs)
    
    for user, user_logs in logs.items():
        final_emotion = user_logs[-1]["dominant_emotion"]
        final_valence = EMOTION_VALENCE.get(final_emotion, 0)
        
        if final_valence > 0:  # Positive final valence
            positive_count += 1
    
    return (positive_count / total_users) * 100

# Compute percentage of users with positive final valence
emotion_aware_positive_percent = compute_positive_final_valence(emotion_aware_logs)
engagement_only_positive_percent = compute_positive_final_valence(engagement_only_logs)

# Output summary
print("=== Final Emotion Valence Summary ===")
print(f"Emotion-Aware RL: {emotion_aware_positive_percent:.2f}% of users achieved positive final valence.")
print(f"Engagement-Only RL: {engagement_only_positive_percent:.2f}% of users achieved positive final valence.")
