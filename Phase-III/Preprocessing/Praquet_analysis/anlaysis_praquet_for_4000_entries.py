import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#Loading dataset

# Path to your dataset
part_files = sorted(glob.glob("Phase-III\Preprocessing\Batch_runs\merged_partials\*.parquet"))

# Empty list to collect samples
sampled_dfs = []

for path in part_files:
    try:
        df = pd.read_parquet(path)  # Load full data
        df_sample = df.sample(n=1000, random_state=42)  # Take 1000-row sample
        sampled_dfs.append(df_sample)
        print(f" Sampled {df_sample.shape[0]} rows from {os.path.basename(path)}")
    except Exception as e:
        print(f" Failed on {path}: {e}")

# Combine all sampled data
sampled_df = pd.concat(sampled_dfs, ignore_index=True)
print(f"\n Final Sampled Data Shape: {sampled_df.shape}")


#Analysis
summary_df = pd.DataFrame(label_stats)
summary_df.set_index("file")[["pos", "skip", "not_seen"]].plot(kind="bar", stacked=True, figsize=(12, 4))
plt.title("Label Distribution Across All Batches")
plt.ylabel("Proportion")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Aggregate engagement rate per emotion
df_filtered = sampled_df[sampled_df['label'].isin([0, 1])]
emotion_stats = df_filtered.groupby("predicted_emotion")['label'].mean()

# Plot
plt.figure(figsize=(10,5))
emotion_stats.sort_values().plot(kind='bar', color='royalblue')
plt.title(" Engagement Rate by Predicted Emotion")
plt.ylabel("Engagement Rate (Watched %)")
plt.xlabel("Emotion")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

df_filtered.groupby("category")['label'].mean().sort_values().plot(
    kind='barh', figsize=(10,6), color='coral'
)
plt.title(" Engagement Rate by Content Category")
plt.xlabel("Engagement Rate")
plt.ylabel("Category")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

# Pivot table for engagement rate by category-emotion
pivot = df_filtered.pivot_table(index='category', columns='predicted_emotion', values='label', aggfunc='mean')

plt.figure(figsize=(12,6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
plt.title(" Engagement Rate by Category and Emotion")
plt.xlabel("Predicted Emotion")
plt.ylabel("Category")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df_filtered["video_duration"], df_filtered["label"], alpha=0.3, color="green")
plt.title("Engagement Rate vs Video Duration")
plt.xlabel("Video Duration (seconds)")
plt.ylabel("Engagement Rate (Watched %)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

df_filtered["predicted_emotion"].value_counts()
