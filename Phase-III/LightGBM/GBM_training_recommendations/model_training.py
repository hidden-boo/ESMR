import pandas as pd
import glob
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# === Step 1: Load Full Dataset ===
print("Loading all parquet files...")
dataset_dir = "Phase-III\Preprocessing\Batch_runs\merged_partials"
part_files = sorted(glob.glob(f"{dataset_dir}/*.parquet"))

if not part_files:
    raise ValueError("No .parquet files found in the directory!")

dfs = []
for path in part_files:
    df = pd.read_parquet(path)
    df = df[df["label"].isin([0, 1])]  # Filter out -1s
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Full dataset loaded. Shape: {data.shape}")

# === Step 2: Feature Selection ===
feature_cols = [
    "video_duration", "video_engagement_score", "scrolling_time", "video_watching_duration",
    "post_story_views", "time_spent_daily", "daily_logins", "posting_frequency", "liking_behavior",
    "commenting_activity", "sharing_behavior", "previous_day_engagement", "previous_week_avg_engagement",
    "engagement_score", "engagement_growth_rate", "liking_trend", "commenting_trend", "sharing_trend",
    "scrolling_watching_ratio", "education_ratio", "entertainment_ratio", "news_ratio", "inspiration_ratio"
]

cat_features = ["user_profile", "category", "theme", "predicted_emotion"]

# Convert categorical columns
for col in cat_features:
    if col in data.columns:
        data[col] = data[col].astype("category")

# === Step 3: Prepare Data ===
X = data[feature_cols + cat_features]
y = data["label"]

# === Step 4: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Step 5: LightGBM Dataset ===
lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train, categorical_feature=cat_features)

# === Step 6: LightGBM GPU Parameters ===
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "verbosity": -1,
    "device": "gpu"  #  GPU Acceleration
}

# === Step 7: Train Model ===
print(" Training LightGBM model with GPU...")
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=10)]
)
print(" Model training complete!")

# === Step 8: Predict & Evaluate ===
y_probs = model.predict(X_test)
y_preds = (y_probs >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_preds)
auc = roc_auc_score(y_test, y_probs)

print(f" Evaluation Results:")
print(f" Accuracy: {accuracy:.4f}")
print(f" AUC:      {auc:.4f}")

# === Step 9: Save Model ===
model.save_model("base_recommendation_model.txt")
print(" Model saved as 'base_recommendation_model.txt'")
