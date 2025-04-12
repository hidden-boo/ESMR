import pandas as pd

# === Load Your JSON File ===
input_path = "Phase-II\TabTransfomers\Datasets\soft_predictions.json"
emotion_df = pd.read_json(input_path)

# === Sort by user and day to compute trends correctly ===
emotion_df = emotion_df.sort_values(by=["user_id", "day"]).copy()

# === Emotions + churn to compute trends for ===
trend_features = ["happy", "stressed", "frustrated", "disappointed", "churned"]

# === Step 1: Day-to-day deltas ===
for feature in trend_features:
    emotion_df[f"{feature}_trend"] = emotion_df.groupby("user_id")[feature].diff()

# === Step 2: 3-day rolling average of deltas (smoothed trend) ===
for feature in trend_features:
    emotion_df[f"{feature}_trend_avg3"] = (
        emotion_df.groupby("user_id")[feature]
        .transform(lambda x: x.diff().rolling(window=3).mean())
    )

# === Step 3: Save updated file as JSON ===
output_path = "Phase-II\TabTransfomers\Datasets\soft_predictions_with_trends.json"
emotion_df.to_json(output_path, orient="records", indent=2)

print(f"Updated JSON with emotion + churn trends saved to:\n{output_path}")


