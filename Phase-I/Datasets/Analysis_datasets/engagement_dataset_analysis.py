import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

engagement_df = pd.read_csv("Phase-I\Datasets\engagement_dataset.csv")

print("Basic Info on Engagement Dataset:")
print(engagement_df.info())

print("Sample Data:")
print(engagement_df.head())

print("Summary Statistics:")
print(engagement_df.describe())

print("Missing Values:")
print(engagement_df.isnull().sum())

# Distribution of Engagement Metrics
plt.figure(figsize=(12, 6))
sns.histplot(engagement_df["scrolling_time"], bins=50, kde=True, color="blue", label="Scrolling Time")
sns.histplot(engagement_df["video_watching_duration"], bins=50, kde=True, color="green", label="Video Duration")
plt.legend()
plt.title("Distribution of Scrolling Time & Video Watching Duration")
plt.xlabel("Time (minutes)")
plt.ylabel("Frequency")
plt.show()

# Boxplot to Detect Outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=engagement_df[["scrolling_time", "video_watching_duration", "time_spent_daily"]])
plt.title("Boxplot of Engagement Metrics")
plt.ylabel("Engagement Time (minutes)")
plt.show()

# Churned vs. Non-Churned Users Analysis
plt.figure(figsize=(8, 5))
sns.countplot(x="churned", data=engagement_df, palette="Set2")
plt.title("Churned vs. Non-Churned Users")
plt.xlabel("Churned (1 = Yes, 0 = No)")
plt.ylabel("User Count")
plt.show()


# Churn Rate Over Time (Daily Churned Users Percentage)
plt.figure(figsize=(10, 5))

# Calculate the percentage of users who churned each day
churned_users_per_day = engagement_df.groupby("day")["churned"].mean()

plt.plot(churned_users_per_day, marker="o", linestyle="-", color="red", label="Churn Rate")
plt.title("Churn Rate Over Time")
plt.xlabel("Day")
plt.ylabel("Percentage of Users Churned")
plt.legend()
plt.show()

# Correlation Heatmap (Dropping Categorical Columns)
plt.figure(figsize=(10, 6))

# Drop categorical columns before computing correlation
numeric_engagement_df = engagement_df.drop(columns=["user_id", "user_profile", "churned"])
sns.heatmap(numeric_engagement_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Between Engagement Features")
plt.show()
