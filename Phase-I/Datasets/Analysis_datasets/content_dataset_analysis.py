import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

content_df = pd.read_csv("Phase-I\Datasets\content_dataset.csv")

print("Basic Info on Content Dataset:")
print(content_df.info())

print(" Sample Data:")
print(content_df.head())

print("Summary Statistics:")
print(content_df.describe())

print("Missing Values:")
print(content_df.isnull().sum())

# Plot distribution of engagement scores
plt.figure(figsize=(12, 6))
sns.histplot(content_df["engagement_score"], bins=50, kde=True, color="purple")
plt.title("Distribution of Engagement Scores")
plt.xlabel("Engagement Score")
plt.ylabel("Frequency")
plt.show()

# Plot category distribution
plt.figure(figsize=(10, 5))
sns.countplot(y="category", data=content_df, palette="viridis")
plt.title("Distribution of Content Categories")
plt.ylabel("Content Category")
plt.xlabel("Count")
plt.show()

# Emotion category distribution
plt.figure(figsize=(10, 5))
sns.countplot(y="emotion", data=content_df, palette="coolwarm")
plt.title("Distribution of Emotion Labels in Content")
plt.ylabel("Emotion Label")
plt.xlabel("Count")
plt.show()

# Engagement score vs. video duration
plt.figure(figsize=(12, 6))
sns.scatterplot(x="video_duration", y="engagement_score", data=content_df, alpha=0.5)
plt.title("Engagement Score vs. Video Duration")
plt.xlabel("Video Duration (seconds)")
plt.ylabel("Engagement Score")
plt.show()

# Boxplot of engagement scores by emotion label
plt.figure(figsize=(12, 6))
sns.boxplot(x="emotion", y="engagement_score", data=content_df, palette="Set3")
plt.title("Engagement Scores by Emotion Label")
plt.xticks(rotation=45)
plt.show()

# ANOVA Test: Does Emotion Affect Engagement?
anova_emotion = stats.f_oneway(
    *[content_df[content_df["emotion"] == e]["engagement_score"] for e in content_df["emotion"].unique()]
)
print(f"ANOVA Test for Emotion & Engagement: p-value = {anova_emotion.pvalue:.5f}")

# ANOVA Test: Does Category Affect Engagement?
anova_category = stats.f_oneway(
    *[content_df[content_df["category"] == c]["engagement_score"] for c in content_df["category"].unique()]
)
print(f"ANOVA Test for Category & Engagement: p-value = {anova_category.pvalue:.5f}")

# T-Test: Do Viral Videos Have Higher Engagement?
viral_videos = content_df[content_df["engagement_score"] > 80]["engagement_score"]
non_viral_videos = content_df[content_df["engagement_score"] <= 80]["engagement_score"]
t_stat, p_value = stats.ttest_ind(viral_videos, non_viral_videos, equal_var=False)

print(f" T-Test for Virality & Engagement: t-stat = {t_stat:.5f}, p-value = {p_value:.5f}")

# Correlation Analysis
print("Correlation Between Features:")
correlation_matrix = content_df[['video_duration', 'emotion_intensity', 'engagement_score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations")
plt.show()