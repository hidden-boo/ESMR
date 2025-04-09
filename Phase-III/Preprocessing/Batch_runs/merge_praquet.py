import pandas as pd
import glob
import os

# Paths
BATCH_DIR = "Phase-III\Preprocessing\Batch_runs\engagement_batches"
MERGED_DIR = "Phase-III\Preprocessing\Batch_runs\merged_partials"
os.makedirs(MERGED_DIR, exist_ok=True)

# All parquet files
batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, "*.parquet")))
print(f" Found {len(batch_files)} files")

# Batch merging config
MERGE_SIZE = 5  # How many files to merge at once
master_columns = None

for i in range(0, len(batch_files), MERGE_SIZE):
    part_files = batch_files[i:i + MERGE_SIZE]
    print(f" Merging files {i}–{i + MERGE_SIZE - 1}")

    part_dfs = []
    for f in part_files:
        df = pd.read_parquet(f)

        # Initialize master columns
        if master_columns is None:
            master_columns = list(df.columns)

        # Add missing columns if any
        for col in master_columns:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[master_columns]  # enforce column order
        part_dfs.append(df)

    part_df = pd.concat(part_dfs, ignore_index=True)
    part_path = os.path.join(MERGED_DIR, f"merged_part_{i // MERGE_SIZE + 1}.parquet")
    part_df.to_parquet(part_path, index=False)
    print(f" Saved partial merge: {part_path}, shape: {part_df.shape}")
