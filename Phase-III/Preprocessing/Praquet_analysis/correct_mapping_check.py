import pandas as pd
import glob

# Path to your merged Parquet files
MERGED_DIR = "Phase-III\Preprocessing\Batch_runs\merged_partials"
part_files = sorted(glob.glob(f"{MERGED_DIR}/*.parquet"))

# Load a small sample
df_sample = pd.read_parquet(part_files[0])  # Load first file
print(" Sample Data Loaded. Shape:", df_sample.shape)

# Check unique videos in the dataset
unique_videos = df_sample["video_id"].nunique()
total_rows = df_sample.shape[0]

print(f"Unique Videos in Parquet: {unique_videos}")
print(f" Total Rows in Parquet: {total_rows}")

# Show a few unique video IDs
print(" Sample Video IDs:", df_sample["video_id"].unique()[:10])

# Load original content dataset (if available)
content_path = "/content/drive/My Drive/ACM/Datasets/Final/content_dataset.csv"
content_df = pd.read_csv(content_path)

# Compare
missing_videos = set(content_df["video_id"]) - set(df_sample["video_id"])
print(f" Missing Videos: {len(missing_videos)}")

if missing_videos:
    print(" Some videos might be missing in Parquet files.")
else:
    print(" All videos seem correctly mapped!")
