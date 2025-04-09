import pandas as pd

# Load dataset
file_path = "Phase-I\Video_assignement\Datasets\assigned_videos_11_timestamps.csv"  # Replace with your actual file path
user_dataset = pd.read_csv(file_path)

# Ensure dataset is sorted for temporal calculations
user_dataset = user_dataset.sort_values(by=["user_id", "day"])

# If there's no direct "total_engagement_score", we approximate it
user_dataset["engagement_score"] = (
    user_dataset["scrolling_time"] +
    user_dataset["video_watching_duration"] +
    (user_dataset["liking_behavior"] * 2) +  # Likes contribute to engagement
    (user_dataset["commenting_activity"] * 3) +  # Comments indicate deeper engagement
    (user_dataset["sharing_behavior"] * 4)  # Shares signal strong emotional connection
)

# Previous Day Engagement
user_dataset["previous_day_engagement"] = user_dataset.groupby("user_id")["engagement_score"].shift(1)

# Previous Week Average Engagement (rolling mean over last 7 days)
user_dataset["previous_week_avg_engagement"] = (
    user_dataset.groupby("user_id")["engagement_score"]
    .rolling(window=7, min_periods=1)
    .mean()
    .shift(1)
    .reset_index(level=0, drop=True)
)

# Engagement Growth Rate (difference from previous day)
user_dataset["engagement_growth_rate"] = user_dataset["engagement_score"] - user_dataset["previous_day_engagement"]

# Trend indicators for liking, commenting, and sharing behavior
user_dataset["liking_trend"] = user_dataset.groupby("user_id")["liking_behavior"].diff().fillna(0)
user_dataset["commenting_trend"] = user_dataset.groupby("user_id")["commenting_activity"].diff().fillna(0)
user_dataset["sharing_trend"] = user_dataset.groupby("user_id")["sharing_behavior"].diff().fillna(0)

# Ratio of scrolling to watching (higher ratio = low engagement)
user_dataset["scrolling_watching_ratio"] = user_dataset["scrolling_time"] / (user_dataset["video_watching_duration"] + 1)
user_dataset["scrolling_watching_ratio"].fillna(0, inplace=True)

# How much time was spent on different content types
user_dataset["education_ratio"] = user_dataset["educational_time_spent"] / user_dataset["time_spent_daily"]
user_dataset["entertainment_ratio"] = user_dataset["entertainment_time_spent"] / user_dataset["time_spent_daily"]
user_dataset["news_ratio"] = user_dataset["news_time_spent"] / user_dataset["time_spent_daily"]
user_dataset["inspiration_ratio"] = user_dataset["inspirational_time_spent"] / user_dataset["time_spent_daily"]

# Fill NaN values in ratios (if any) with 0
user_dataset.fillna(0, inplace=True)

user_dataset["days_since_last_engagement"] = user_dataset.groupby("user_id")["day"].diff().fillna(0)


output_path = "Phase-I\Video_assignement\Datasets\final_dataset_combined_ready.csv"
user_dataset.to_csv(output_path, index=False)
print("Updated dataset saved successfully!")

added_features = [
    "engagement_score", "previous_day_engagement", "previous_week_avg_engagement",
    "engagement_growth_rate", "liking_trend", "commenting_trend", "sharing_trend",
    "scrolling_watching_ratio", "education_ratio", "entertainment_ratio",
    "news_ratio", "inspiration_ratio", "days_since_last_engagement"
]

print(f"New Features Added: {added_features}")
print(user_dataset.head(10))  # Display first few rows to verify changes
