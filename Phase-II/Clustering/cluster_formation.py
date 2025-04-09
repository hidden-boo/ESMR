import pandas as pd
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load preprocessed engagement dataset
df = pd.read_csv("Phase-I\Video_assignement\Datasets\final_dataset_combined_ready.csv")

# Select relevant engagement features (excluding categorical/text fields)
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

X = df[engagement_features]

# Standardize features (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering (5 clusters)
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df["engagement_cluster"] = kmeans.fit_predict(X_scaled)

# Refine the cluster-emotion mapping based on observed engagement patterns
cluster_to_emotion = {
    0: "happy",        # High engagement, active interactions
    1: "lonely",       # No engagement, churned users
    2: "disappointed", # Moderate engagement, passive users
    3: "excited",      # High engagement, interactive users
    4: "frustrated"    # High scrolling, indecisive users
}

df["user_emotion"] = df["engagement_cluster"].map(cluster_to_emotion)

# Fix 'stressed' assignment (not just churned users)
stressed_condition = (
    (df["engagement_growth_rate"] < -500) |  # Sharp engagement drop
    ((df["scrolling_time"] > 1000) & (df["video_watching_duration"] < 300)) |  # Doomscrolling
    ((df["news_ratio"] > 0.2) | (df["education_ratio"] > 0.2))  # High news consumption
)

df.loc[stressed_condition, "user_emotion"] = "stressed"

# Assign 'churned' users separately (not an emotion)
df.loc[(df["engagement_score"] == 0) & (df["scrolling_time"] == 0), "user_emotion"] = "churned"

# Update numerical emotion labels
emotion_mapping = {
    "happy": 1.0,
    "neutral": 0.5,
    "excited": 0.8,
    "anxious": 0.3,
    "frustrated": 0.2,
    "disappointed": 0.4,
    "lonely": 0.25,
    "stressed": 0.1,
    "churned": -1  # Mark churned separately (not an emotion)
}

df["user_emotion_label"] = df["user_emotion"].map(emotion_mapping)

# Save the updated dataset
df.to_csv("Phase-II\Clustering\Generated_dataset\validated_emotion_clusters.csv", index=False)

# Save labels for LLM training
torch.save(torch.tensor(df["user_emotion_label"].values, dtype=torch.float32),
           "Phase-II\Clustering\user_emotion_labels_9_classes.pt")

print(" Clustering & Emotion Mapping Completed! Users grouped based on engagement behavior.")
