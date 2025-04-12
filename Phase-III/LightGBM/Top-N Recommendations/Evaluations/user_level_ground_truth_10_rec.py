import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIG 
K = 10
TOPK_PATH = f"top_{K}_recommendations.csv"
DATASET_DIR = r"G:\My Drive\ACM\Datasets\Final\merged_partials"

# Load Top-K Predictions
print(" Loading model predictions...")
topk_df = pd.read_csv(TOPK_PATH)

# Load Ground Truth (label == 1) 
print(" Loading ground truth from parquet files...")
part_files = sorted(glob.glob(f"{DATASET_DIR}/*.parquet"))
dfs = [pd.read_parquet(path) for path in part_files]
full_df = pd.concat(dfs, ignore_index=True)

ground_truth = full_df[full_df["label"] == 1][["user_id", "video_id", "day"]]

# Compute per-user stats
print(" Computing user-level hit/recall...")

# How many videos each user actually watched
watched_counts = ground_truth.groupby("user_id")["video_id"].nunique().reset_index()
watched_counts.columns = ["user_id", "num_watched"]

# How many hits we got in Top-K
topk_hits = topk_df.merge(ground_truth, on=["user_id", "video_id"], how="left")
topk_hits["hit"] = topk_hits["day"].notnull().astype(int)
hit_counts = topk_hits.groupby("user_id")["hit"].sum().reset_index()
hit_counts.columns = ["user_id", "num_hits"]

# Merge and compute recall@K
recall_df = pd.merge(watched_counts, hit_counts, on="user_id", how="left").fillna(0)
recall_df["recall@K"] = recall_df["num_hits"] / recall_df["num_watched"]
recall_df["recall@K"] = recall_df["recall@K"].clip(upper=1.0)

# Save CSV 
recall_df.to_csv(f"user_recall_analysis_top_{K}.csv", index=False)
print(f" Saved per-user recall summary to user_recall_analysis_top_{K}.csv")

# Visualization
plt.figure(figsize=(8, 4))
sns.histplot(recall_df["num_watched"], bins=30, kde=False)
plt.title("Distribution of #Videos Watched per User")
plt.xlabel("# Videos Watched")
plt.ylabel("User Count")
plt.tight_layout()
plt.savefig(f"hist_watched_per_user_top_{K}.png")
print(f" Saved plot: hist_watched_per_user_top_{K}.png")

plt.figure(figsize=(8, 4))
sns.histplot(recall_df["recall@K"], bins=30, kde=False)
plt.title(f"Recall@{K} Distribution")
plt.xlabel("Recall@K")
plt.ylabel("User Count")
plt.tight_layout()
plt.savefig(f"hist_recall_distribution_top_{K}.png")
print(f" Saved plot: hist_recall_distribution_top_{K}.png")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=recall_df, x="num_watched", y="recall@K", alpha=0.5)
plt.title(f"Recall@{K} vs. #Videos Watched")
plt.xlabel("# Videos Watched")
plt.ylabel("Recall@K")
plt.tight_layout()
plt.savefig(f"scatter_recall_vs_watched_top_{K}.png")
print(f" Saved plot: scatter_recall_vs_watched_top_{K}.png")

print(" Analysis complete!")
