import pandas as pd
import lightgbm as lgb
import glob

# Step 1: Load Preprocessed Data from Parquet Files

# Path to the directory containing your .parq files
dataset_dir = "Phase-III\Preprocessing\Batch_runs\merged_partials"

# Load the parquet files (assuming all files in the folder are the ones you need)
part_files = sorted(glob.glob(f"{dataset_dir}/*.parquet"))

# Read all files and concatenate into a single DataFrame
data = pd.concat([pd.read_parquet(file) for file in part_files], ignore_index=True)

# Check the first few rows to confirm
print("Data Loaded Successfully!")
print(data.head())

# Step 2: Train the LightGBM Model

# Prepare feature columns and target variable
X = data.drop(columns=['user_id', 'video_id', 'label'])  # Drop ID columns and label column
y = data['label']

# Convert to LightGBM Dataset format
train_data = lgb.Dataset(X, label=y)

# Set parameters for LightGBM (using GPU)
params = {
    'objective': 'binary',  # Binary classification (0 or 1)
    'metric': 'auc',        # AUC as evaluation metric
    'device': 'gpu',        # Use GPU
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 5,
    'feature_fraction': 0.9,
    'verbose': -1
}

# Train the model
print("Training LightGBM model with GPU...")
booster = lgb.train(params, train_data, num_boost_round=100)

# Save the trained model
booster.save_model('Phase-III\LightGBM\base_recommendation_model.txt')

print("Model training complete. Saved as 'base_recommendation_model.txt'.")

# Step 3: Generate Top-K Recommendations

# Make predictions (for all users and videos)
predictions = booster.predict(X)

# Add predictions to the original dataset
data['predicted_engagement'] = predictions

# Sort by predicted engagement to get Top-K recommendations
K = 10  # Adjust K as needed
top_k_recommendations = data.groupby('user_id').apply(lambda x: x.nlargest(K, 'predicted_engagement'))

# Save the recommendations to a JSON file
top_k_recommendations[['user_id', 'video_id', 'predicted_engagement']].to_json('Phase-III\LightGBM\Top-N Recommendations\daily_recommendations.json', orient='records', lines=True)

print("Top-K recommendations saved to 'top_k_recommendations.json'.")

