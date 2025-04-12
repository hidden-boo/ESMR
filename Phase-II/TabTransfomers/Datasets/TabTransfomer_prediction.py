# Load the dataset (since we didn't keep user_id & day in results_df)
import json
import pandas as pd

# Load the main dataset (to extract user_id and day)
dataset_file_path = "Phase-II\TabTransfomers\Datasets\processed_engagement_with_clusters.json"
with open(dataset_file_path, "r") as f:
    dataset = json.load(f)

# Convert to DataFrame
dataset_df = pd.DataFrame(dataset)

# Ensure correct order before merging
dataset_df = dataset_df.sort_values(["user_id", "day"]).reset_index(drop=True)

# Ensure results_df is the same length as dataset_df
results_df = results_df.sort_values("true_label").reset_index(drop=True)  # Adjust ordering

# Add 'user_id' and 'day' to results_df
results_df["user_id"] = dataset_df["user_id"]
results_df["day"] = dataset_df["day"]

# Ensure predictions are in human-readable format
emotion_classes = ["happy", "excited", "stressed", "frustrated", "disappointed", "churned"]
results_df["predicted_emotion"] = results_df[emotion_classes].idxmax(axis=1)  # Get highest probability emotion

print(" Now 'results_df' has user_id, day, and predicted_emotion!")

# HARD CLASSIFICATIONS 
# Select relevant columns
hard_predictions = results_df[["user_id", "day", "predicted_emotion"]].to_dict(orient="records")

# Save as JSON
with open("Phase-II\TabTransfomers\Datasets\hard_predictions.json", "w") as f:
    json.dump(hard_predictions, f, indent=4)

print(" Hard predictions saved: 'hard_predictions.json'")

# SOFT CLASSIFICATIONS 
# Select relevant columns for soft predictions
soft_predictions = results_df[["user_id", "day", "predicted_emotion"] + emotion_classes].to_dict(orient="records")

# Save as JSON
with open("Phase-II\TabTransfomers\Datasets\soft_predictions.json", "w") as f:
    json.dump(soft_predictions, f, indent=4)

print(" Soft predictions saved: 'soft_predictions.json'")

# MERGED PREDICTIONS
# Merge hard predictions
dataset_df = dataset_df.merge(results_df[["user_id", "day", "predicted_emotion"]], on=["user_id", "day"], how="left")

# Merge soft predictions (all probabilities)
dataset_df = dataset_df.merge(results_df[["user_id", "day"] + emotion_classes], on=["user_id", "day"], how="left")

# Save full dataset with predictions
dataset_with_predictions = dataset_df.to_dict(orient="records")
with open("Phase-II\TabTransfomers\Datasets\dataset_with_predictions.json", "w") as f:
    json.dump(dataset_with_predictions, f, indent=4)

print(" Full dataset with predictions saved: 'dataset_with_predictions.json'")
