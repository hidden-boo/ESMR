import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = "Phase-I\Video_assignement\Datasets\final_dataset_combined_ready.csv"  # Update this with your actual file path
user_dataset = pd.read_csv(file_path)

# Ensure dataset is sorted by user and day
user_dataset = user_dataset.sort_values(by=["user_id", "day"])

# 1. Overview of the Dataset


# Check for missing values
missing_values = user_dataset.isnull().sum()
print(" Missing Values:\n", missing_values)

# Summary statistics
summary_stats = user_dataset.describe()
print(" Summary Statistics:\n", summary_stats)

# 2. Engagement Trends Analysis


# Distribution of engagement scores
plt.figure(figsize=(8,5))
sns.histplot(user_dataset["engagement_score"], bins=30, kde=True)
plt.title("Distribution of Engagement Scores")
plt.xlabel("Engagement Score")
plt.ylabel("Frequency")
plt.show()

# Previous Engagement Trends (Descriptive Stats)
print(" Previous Engagement Trends:\n", user_dataset[["previous_day_engagement", "previous_week_avg_engagement"]].describe())

# Identify users with increasing vs. decreasing engagement
user_dataset["engagement_trend"] = user_dataset["engagement_growth_rate"].apply(lambda x: "Increasing" if x > 0 else "Decreasing")
engagement_trend_counts = user_dataset["engagement_trend"].value_counts()
print(" Engagement Trend Counts:\n", engagement_trend_counts)

#  3. Interaction-Based Insights


# Interaction Trends (Descriptive Stats)
print(" Interaction Trends:\n", user_dataset[["liking_trend", "commenting_trend", "sharing_trend"]].describe())

# Correlations between interactions and engagement score
interaction_correlation = user_dataset[["engagement_score", "liking_behavior", "commenting_activity", "sharing_behavior"]].corr()
print(" Interaction Correlations:\n", interaction_correlation)

# Heatmap for Correlations
plt.figure(figsize=(8,6))
sns.heatmap(interaction_correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Engagement & Interactions")
plt.show()

#  4. Consumption Patterns

# Scrolling vs. Watching Ratio Analysis
print(" Scrolling vs Watching Stats:\n", user_dataset["scrolling_watching_ratio"].describe())

# Content Type Consumption Analysis
content_type_ratios = user_dataset[["education_ratio", "entertainment_ratio", "news_ratio", "inspiration_ratio"]].describe()
print(" Content Type Ratios:\n", content_type_ratios)

# Visualizing Content Preferences
plt.figure(figsize=(8,5))
sns.boxplot(data=user_dataset[["education_ratio", "entertainment_ratio", "news_ratio", "inspiration_ratio"]])
plt.title("Distribution of Time Spent on Content Types")
plt.ylabel("Ratio of Total Time")
plt.show()

# 5. Churn & Activity Trends


# Distribution of days since last engagement
print(" Churn & Activity Stats:\n", user_dataset["days_since_last_engagement"].describe())

# Identify users at high risk of churn (low engagement over time)
high_risk_churn_users = user_dataset[user_dataset["previous_week_avg_engagement"] < user_dataset["engagement_score"].quantile(0.25)]
print(" High-Risk Churn Users Sample:\n", high_risk_churn_users.head())


print(" Analysis Complete! Check the plots and summaries above.")
