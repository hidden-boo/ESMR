import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Load the validated dataset
df = pd.read_csv("Phase-II\Clustering\Generated_dataset\validated_emotion_clusters.csv")

# Set Seaborn style
sns.set(style="whitegrid")

# Define engagement features to analyze
engagement_features = [
    "engagement_score", "scrolling_time", "video_watching_duration",
    "previous_day_engagement", "previous_week_avg_engagement",
    "liking_behavior", "commenting_activity", "sharing_behavior"
]

# Generate Boxplots
for feature in engagement_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="user_emotion", y=feature, data=df, palette="Set2")
    plt.title(f" Distribution of {feature} Across User Emotions")
    plt.xlabel("User Emotion")
    plt.ylabel(feature)
    plt.xticks(rotation=30)
    plt.show()

# Perform ANOVA test for each engagement feature
anova_results = {}
for feature in engagement_features:
    groups = [df[df["user_emotion"] == emotion][feature] for emotion in df["user_emotion"].unique()]
    f_stat, p_value = f_oneway(*groups)
    anova_results[feature] = (f_stat, p_value)

# Print results
print(" ANOVA Test Results for Engagement Features Across Emotions:")
for feature, (f_stat, p_val) in anova_results.items():
    print(f"{feature}: F-stat={f_stat:.2f}, p-value={p_val:.5f}")

# Identify features where emotions are NOT well-separated
low_p_values = {k: v for k, v in anova_results.items() if v[1] < 0.05}
if len(low_p_values) == len(engagement_features):
    print(" Emotions significantly differ based on engagement trends!")
else:
    print(" Some engagement features do not significantly differentiate emotions.")

# Count the number of instances per emotion
emotion_counts = df["user_emotion"].value_counts()

# Plot emotion distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette="coolwarm")
plt.xlabel("User Emotion")
plt.ylabel("Count")
plt.title(" Emotion Distribution in Clustered Dataset")
plt.xticks(rotation=45)
plt.show()

# Print emotion counts
print("Emotion Distribution:")
print(emotion_counts)

# Check engagement behavior for 'stressed' users
stressed_users = df[df["user_emotion"] == "stressed"]

# Print statistical summary
print("Engagement Trends for Stressed Users:")
print(stressed_users[["engagement_score", "scrolling_time", "news_ratio"]].describe())

# Boxplot to visualize 'stressed' users' engagement
plt.figure(figsize=(8, 5))
sns.boxplot(x="user_emotion", y="scrolling_time", data=stressed_users, palette="Reds")
plt.title("Scrolling Time Distribution for Stressed Users")
plt.xlabel("User Emotion")
plt.ylabel("Scrolling Time")
plt.show()

