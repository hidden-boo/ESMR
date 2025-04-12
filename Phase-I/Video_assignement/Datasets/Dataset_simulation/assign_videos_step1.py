import pandas as pd
import numpy as np
import random
from numba import jit, prange
from multiprocessing import Pool, cpu_count

# Load datasets
content_df = pd.read_csv("Phase-I\Datasets\content_dataset.csv")
engagement_df = pd.read_csv("Phase-I\Datasets\engagement_dataset.csv")

# Debugging: Print dataset info
print("Content Dataset Loaded:")
print(content_df.info())
print(content_df.head())

print("Engagement Dataset Loaded:")
print(engagement_df.info())
print(engagement_df.head())

# Convert Categorical Columns to Integer Indices
category_map = {"educational": 0, "entertainment": 1, "news": 2, "inspirational": 3}
content_df["category"] = content_df["category"].map(category_map)

# Map Video IDs to Numeric Indices
video_id_map = {vid: idx for idx, vid in enumerate(content_df["video_id"].unique())}
content_df["video_index"] = content_df["video_id"].map(video_id_map)

# Ensure Content Data is Fully Numeric (Exclude 'video_id')
video_indices = content_df["video_index"].to_numpy(dtype=np.int64) 
content_np = content_df[['category', 'video_duration', 'engagement_score']].to_numpy(dtype=np.float64)

## -------NUMBA FOR FASTER ASSIGNMENET--------##
# Fast Numba-Compatible Video Assignment Function
@jit(nopython=True, parallel=True)
def assign_videos_np(available_time, category_indices, category_probs, content_np, video_indices):
    """
    Optimized video assignment function using Numba. Ensures:
    - Videos are selected based on category proportions.
    - Total assigned duration ≈ available_time.
    """
    total_time_used = 0
    assigned_videos = np.zeros(50, dtype=np.int64)  # Preallocate space for efficiency
    assigned_count = 0

    # Create cumulative probability array for category selection
    category_cumsum_probs = np.cumsum(category_probs)

    while total_time_used < available_time and assigned_count < 50:  # Limit to 50 videos max (for being able to get realstic results)
        # Use np.searchsorted() instead of np.random.choice()
        random_val = np.random.rand()  # Generate a uniform random value
        chosen_category_idx = category_indices[np.searchsorted(category_cumsum_probs, random_val)]

        # Filter videos matching the chosen category
        category_mask = content_np[:, 0] == chosen_category_idx
        category_videos = content_np[category_mask]
        category_video_indices = video_indices[category_mask]

        if len(category_videos) == 0:
            continue  # Skip if no videos found

        # Select a random video
        video_idx = np.random.randint(0, len(category_videos))
        video = category_videos[video_idx]
        video_id = category_video_indices[video_idx]

        if total_time_used + video[1] > available_time:
            break  # Stop if total time exceeds limit

        assigned_videos[assigned_count] = video_id  # Store numeric video index
        assigned_count += 1
        total_time_used += video[1]

    return assigned_videos[:assigned_count]  # Return only assigned videos

## -----Process a Batch of Users----##
def process_batch(batch_data):
    """
    Process a batch of users in parallel.
    """
    results = []
    for index, row in batch_data.iterrows():
        if row["churned"]:
            results.append(["churned"])
            continue

        available_time = row["video_watching_duration"]
        category_distribution = np.array([
            row["educational_time_spent"],
            row["entertainment_time_spent"],
            row["news_time_spent"],
            row["inspirational_time_spent"]
        ])

        total_category_time = np.sum(category_distribution)
        if total_category_time == 0:
            results.append([])
            continue

        # Convert dictionary to NumPy-compatible format
        category_indices = np.array([0, 1, 2, 3])  # Corresponding to category_map
        category_probs = category_distribution / total_category_time  # Normalize

        assigned_videos = assign_videos_np(available_time, category_indices, category_probs, content_np, video_indices)
        assigned_video_ids = [list(video_id_map.keys())[i] for i in assigned_videos]  # Convert indices back to video IDs
        results.append(assigned_video_ids)

    return results

## -----Parallel Processing----
if __name__ == '__main__':
    num_cores = min(cpu_count(), 4)  # Limit to 4 cores for stability
    print(f" Using {num_cores} cores for parallel processing...")

    batch_size = 500  # Process 500 users at a time
    total_users = len(engagement_df["user_id"].unique())
    user_batches = np.array_split(engagement_df, total_users // batch_size)

    final_results = []

    for batch_index, batch in enumerate(user_batches):
        print(f" Processing Batch {batch_index + 1}/{len(user_batches)}...")

        with Pool(processes=num_cores) as pool:
            batch_results = pool.map(process_batch, [batch])

        # Flatten and store results
        for result in batch_results:
            final_results.extend(result)

    # Assign back to DataFrame
    engagement_df["assigned_videos"] = final_results

    # Save the updated dataset
    engagement_df.to_csv("Phase-I\Video_assignement\Datasets\assigned_videos_parallel.csv", index=False)

    print(" Step 1-4 Completed: Videos Assigned Efficiently with Parallel Processing")

