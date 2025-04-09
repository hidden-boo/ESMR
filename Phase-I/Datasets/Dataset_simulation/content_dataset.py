import pandas as pd
import numpy as np

# Parameters
NUM_VIDEOS = 1000

# Define columns
columns = [
    "video_id", "emotion", "emotion_intensity", "video_duration",
    "engagement_score", "category", "theme"
]

# Define emotions with improved balancing
emotions = [
    "happy", "neutral", "excited", "anxious", "angry", "disappointed", "lonely", "stressed"
]

# Adjusted probabilities (less bias towards neutral)
emotion_probs = [0.22, 0.12, 0.15, 0.13, 0.11, 0.11, 0.08, 0.08]

# Emotion intensity scaling (Gaussian noise added)
intensity_map = {
    "happy": (3, 7),
    "neutral": (1, 3),
    "excited": (4, 8),
    "anxious": (6, 9),
    "angry": (5, 9),
    "disappointed": (4, 7),
    "lonely": (5, 8),
    "stressed": (6, 10)
}

# Categories & realistic distribution
categories = ["educational", "entertainment", "news", "inspirational"]
category_probs = [0.20, 0.40, 0.25, 0.15]  # Adjusted based on typical media consumption

# Simulate content dataset
np.random.seed(42)
video_ids = [f"V{str(i).zfill(4)}" for i in range(1, NUM_VIDEOS + 1)]

assigned_emotions = np.random.choice(emotions, size=NUM_VIDEOS, p=emotion_probs)
assigned_categories = np.random.choice(categories, size=NUM_VIDEOS, p=category_probs)

# Video durations: Skewed towards shorter videos (social media trend)
video_durations = np.clip(np.random.normal(30, 15, NUM_VIDEOS), 10, 90).astype(int)

# Engagement scores based on durations and category
engagement_base = {
    "educational": 0.9,
    "entertainment": 1.6,
    "news": 1.3,
    "inspirational": 1.1
}

# Virality Factor (Some content goes viral, ensuring proper assignment)
virality_factor = np.random.choice([1, 2, 3, 5, 10], size=NUM_VIDEOS, p=[0.70, 0.15, 0.07, 0.05, 0.03])

# Personalized Engagement Boost
user_interest_boost = np.random.normal(1, 0.15, NUM_VIDEOS)  # Small variations in engagement due to user preference

# Assign emotion intensity with proper scaling
emotion_intensity = np.array([
    np.clip(np.random.randint(intensity_map[emo][0], intensity_map[emo][1]) + np.random.normal(0, 1), 1, 10).astype(int)
    for emo in assigned_emotions
])

# Stronger impact of emotion intensity on engagement
intensity_impact = 1 + (emotion_intensity / 10 * 0.3)  # Adjusted weight for better effect

# Compute engagement scores with revised weighting
engagement_scores = np.clip(
    np.random.normal(
        loc=[video_durations[i] * engagement_base[assigned_categories[i]] * user_interest_boost[i] * intensity_impact[i]
             for i in range(NUM_VIDEOS)],
        scale=5
    ) * virality_factor, 5, 120  # Ensuring an upper limit of 120 for extreme cases
)

# Content themes with a trending factor (simulating real-world trends)
theme_mapping = {
    "educational": ["academic_stress", "motivation"],
    "entertainment": ["daily_humor", "relationships", "mental_health"],
    "news": ["politics", "mental_health"],
    "inspirational": ["motivation", "mental_health"]
}

assigned_themes = [np.random.choice(theme_mapping[cat]) for cat in assigned_categories]

# Simulate DataFrame
content_dataset = pd.DataFrame({
    "video_id": video_ids,
    "emotion": assigned_emotions,
    "emotion_intensity": emotion_intensity,
    "video_duration": video_durations,
    "engagement_score": engagement_scores,
    "category": assigned_categories,
    "theme": assigned_themes
})

# Save dataset
content_dataset.to_csv("Phase-I\Datasets\content_dataset.csv", index=False)

# Display first few rows
print(content_dataset.head())
