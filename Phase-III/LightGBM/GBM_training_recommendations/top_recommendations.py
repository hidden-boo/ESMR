import pandas as pd
import glob
import lightgbm as lgb
import json
import os

# Config 
MODEL_PATH = "Phase-III\LightGBM\base_recommendation_model.txt"
K_list = [5, 10, 20]  
dataset_dir = "Phase-III\Preprocessing\Batch_runs\merged_partials"

#Load Model
print("Loading trained model...")
model = lgb.Booster(model_file=MODEL_PATH)

# Load Data
print("Loading full parquet data...")
part_files = sorted(glob.glob(f"{dataset_dir}/*.parquet"))
dfs = [pd.read_parquet(path) for path in part_files]
data = pd.concat(dfs, ignore_index=True)
print(f"Total data shape: {data.shape}")

#  Filter Candidates
candidate_data = data[data["label"] == -1].copy()
print(f"Candidate pool shape: {candidate_data.shape}")

#  Define Features 
feature_cols = [
    "video_duration", "video_engagement_score", "scrolling_time", "video_watching_duration",
    "post_story_views", "time_spent_daily", "daily_logins", "posting_frequency", "liking_behavior",
    "commenting_activity", "sharing_behavior", "previous_day_engagement", "previous_week_avg_engagement",
    "engagement_score", "engagement_growth_rate", "liking_trend", "commenting_trend", "sharing_trend",
    "scrolling_watching_ratio", "education_ratio", "entertainment_ratio", "news_ratio", "inspiration_ratio"
]

cat_features = ["user_profile", "category", "theme", "predicted_emotion"]

for col in cat_features:
    if col in candidate_data.columns:
        candidate_data[col] = candidate_data[col].astype("category")

X_candidate = candidate_data[feature_cols + cat_features]
print("Generating prediction scores...")
candidate_data["pred_score"] = model.predict(X_candidate)

#  Loop over multiple K values
for K in K_list:
    print(f"Generating Top-{K} recommendations...")
    topk_df = (
        candidate_data
        .sort_values(by=["user_id", "pred_score"], ascending=[True, False])
        .groupby("user_id")
        .head(K)
        .loc[:, ["user_id", "video_id", "pred_score"]]
    )

    # Save as CSV
    csv_path = f"top_{K}_recommendations.csv"
    topk_df.to_csv(csv_path, index=False)
    print(f" Saved CSV to: {csv_path}")

    # Save as JSON
    json_path = f"top_{K}_recommendations.json"
    topk_json = (
        topk_df
        .groupby("user_id")
        .apply(lambda x: x[["video_id", "pred_score"]].to_dict(orient="records"))
        .to_dict()
    )
    with open(json_path, "w") as f:
        json.dump(topk_json, f, indent=2)
    print(f" Saved JSON to: {json_path}")

print(" All Top-K recommendation files generated successfully!")
