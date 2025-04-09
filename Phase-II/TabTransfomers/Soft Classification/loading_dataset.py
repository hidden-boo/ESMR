import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load JSON dataset
json_path = "Phase-II\TabTransfomers\Datasets\processed_engagement_with_clusters.json"
with open(json_path, 'r') as file:
    data = json.load(file)

df = pd.json_normalize(data)

# Explicit Engagement Features (from your original dataset)
engagement_features = [
    "scrolling_time", "video_watching_duration", "post_story_views",
    "time_spent_daily", "daily_logins", "posting_frequency",
    "liking_behavior", "commenting_activity", "sharing_behavior",
    "educational_time_spent", "entertainment_time_spent", "news_time_spent",
    "inspirational_time_spent", "engagement_score",
    "previous_day_engagement", "previous_week_avg_engagement",
    "engagement_growth_rate", "liking_trend", "commenting_trend", "sharing_trend",
    "scrolling_watching_ratio", "education_ratio", "entertainment_ratio",
    "news_ratio", "inspiration_ratio", "days_since_last_engagement"
]

# Feature Extraction from Videos + Engagement Features
def extract_combined_features(row):
    features = {}

    # Video interactions (watched/skipped)
    watched_videos = row['videos_watched']
    skipped_videos = row['skipped_videos_with_metadata']

    # Derived video metrics
    features['num_videos_watched'] = len(watched_videos)
    features['num_videos_skipped'] = len(skipped_videos)
    features['skip_rate'] = (features['num_videos_skipped'] /
                             (features['num_videos_watched'] + features['num_videos_skipped'])
                             ) if (features['num_videos_watched'] + features['num_videos_skipped']) else 0
    features['churned'] = 1 if features['num_videos_watched'] == 0 else 0

    # Predominant categorical video metadata (from watched videos)
    features['top_theme'] = pd.Series([vid['theme'] for vid in watched_videos]).mode()[0] if watched_videos else 'none'
    features['top_category'] = pd.Series([vid['category'] for vid in watched_videos]).mode()[0] if watched_videos else 'none'
    features['top_video_emotion'] = pd.Series([vid['emotion'] for vid in watched_videos]).mode()[0] if watched_videos else 'neutral'

    # Include all explicit numeric engagement features from original df
    for feat in engagement_features:
        features[feat] = row.get(feat, 0)  # Safe fallback if missing

    return pd.Series(features)

# Apply extraction
features_df = df.apply(extract_combined_features, axis=1)

# Target Label
features_df['user_emotion_label'] = df['user_emotion_label']

# Encoding categorical features
categorical_cols = ['top_theme', 'top_category', 'top_video_emotion', 'churned']
encoders = {col: LabelEncoder().fit(features_df[col]) for col in categorical_cols}
for col, encoder in encoders.items():
    features_df[col] = encoder.transform(features_df[col])

# Numeric columns (all engagement features + derived numeric features)
numeric_cols = engagement_features + ['num_videos_watched', 'num_videos_skipped', 'skip_rate']

# Scale numeric features
scaler = StandardScaler()
features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])

# Final dataset
X = features_df.drop('user_emotion_label', axis=1)
y = LabelEncoder().fit_transform(features_df['user_emotion_label'])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(" final preprocessing pipeline with explicit engagement features complete.")


