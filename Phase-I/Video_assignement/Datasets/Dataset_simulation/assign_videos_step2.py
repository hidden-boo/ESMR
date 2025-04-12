import pandas as pd
import numpy as np
import random
from numba import jit, prange
from multiprocessing import Pool, cpu_count

# Load Datasets
engagement_df = pd.read_csv("Phase-I\Video_assignement\Datasets\assigned_videos_parallel.csv")
content_df = pd.read_csv("Phase-I\Datasets\content_dataset.csv")

# Convert Categorical Columns to Integer Indices
category_map = {"educational": 0, "entertainment": 1, "news": 2, "inspirational": 3}
content_df["category"] = content_df["category"].map(category_map)

# Map Video IDs to Numeric Indices
video_id_map = {vid: idx for idx, vid in enumerate(content_df["video_id"].unique())}
content_df["video_index"] = content_df["video_id"].map(video_id_map)

# Ensure Content Data is Fully Numeric
video_indices = content_df["video_index"].to_numpy(dtype=np.int64)
content_np = content_df[['category', 'video_duration', 'engagement_score']].to_numpy(dtype=np.float64)

# Sort Content Dataset by Engagement Score (For 70-30 Rule)
content_df_sorted = content_df.sort_values(by="engagement_score", ascending=False).reset_index(drop=True)

## --------- FOR FASTER ASSIGNMENT ------#
#------ Fast Numba-Compatible Video Assignment Function------#
@jit(nopython=True, parallel=True)
def assign_videos_np(available_time, category_indices, category_probs, content_np, video_indices):
    """
    Optimized video assignment function ensuring:
    - Videos fit within available watch time.
    - Skipped videos are correctly accounted for.
    - If below the limit, short videos are assigned to compensate.
    - Uses NumPy arrays for fast execution.
    """
    total_time_used = 0
    assigned_videos = np.zeros(50, dtype=np.int64)
    assigned_durations = np.zeros(50, dtype=np.float64)
    assigned_skip_flags = np.zeros(50, dtype=np.int64)
    assigned_count = 0

    category_cumsum_probs = np.cumsum(category_probs)

    while total_time_used < available_time and assigned_count < 50:
        random_val = np.random.rand()
        chosen_category_idx = category_indices[np.searchsorted(category_cumsum_probs, random_val)]

        category_mask = content_np[:, 0] == chosen_category_idx
        category_videos = content_np[category_mask]
        category_video_indices = video_indices[category_mask]

        if len(category_videos) == 0:
            continue

        # Apply 70-30 Rule
        if np.random.rand() < 0.7:
            video_idx = np.random.randint(0, min(len(category_videos), 10))
        else:
            video_idx = np.random.randint(0, len(category_videos))

        video = category_videos[video_idx]
        video_id = category_video_indices[video_idx]

        # Determine if the user skips the video
        video_engagement_score = video[2]
        skip_probability = max(0.1, 1 - (video_engagement_score / 10))
        is_skipped = np.random.rand() < skip_probability

        if is_skipped:
            watch_time = video[1] * 0.4
            assigned_skip_flags[assigned_count] = 1
        else:
            watch_time = video[1]
            assigned_skip_flags[assigned_count] = 0

        if total_time_used + watch_time > available_time:
            break

        assigned_videos[assigned_count] = video_id
        assigned_durations[assigned_count] = watch_time
        assigned_count += 1
        total_time_used += watch_time

    # Ensure ±10% tolerance for under-allocation
    while total_time_used < 0.9 * available_time and assigned_count < 50:
        extra_video_idx = np.random.randint(0, len(content_np))
        extra_video = content_np[extra_video_idx]

        extra_watch_time = extra_video[1]
        if total_time_used + extra_watch_time > available_time:
            break

        assigned_videos[assigned_count] = video_indices[extra_video_idx]
        assigned_durations[assigned_count] = extra_watch_time
        assigned_skip_flags[assigned_count] = 0
        assigned_count += 1
        total_time_used += extra_watch_time

    return assigned_videos[:assigned_count], assigned_durations[:assigned_count], assigned_skip_flags[:assigned_count]

#------ Process a Batch of Users------#
def process_batch(batch_data):
    """
    Process a batch of users in parallel.
    """
    results = []
    for index, row in batch_data.iterrows():
        if row["churned"]:
            results.append({"assigned_videos": ["churned"], "watch_time": 0, "skipped": []})
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
            results.append({"assigned_videos": [], "watch_time": 0, "skipped": []})
            continue

        # Balance Categories - Prevent Bias
        max_category = np.argmax(category_distribution)
        secondary_category_probs = category_distribution / total_category_time
        if category_distribution[max_category] > 0.8 * total_category_time:
            secondary_category_probs[max_category] = 0.7
            remaining_weight = 0.3
            for i in range(4):
                if i != max_category:
                    secondary_category_probs[i] += remaining_weight / 3
            secondary_category_probs /= np.sum(secondary_category_probs)

        category_indices = np.array([0, 1, 2, 3])
        assigned_videos, assigned_durations, assigned_skip_flags = assign_videos_np(
            available_time, category_indices, secondary_category_probs, content_np, video_indices
        )

        assigned_videos_with_metadata = []
        skipped_videos = []

        for vid, dur, skipped in zip(assigned_videos, assigned_durations, assigned_skip_flags):
            video_entry = {"video_id": list(video_id_map.keys())[int(vid)], "watch_time": dur}
            if skipped:
                skipped_videos.append(video_entry)
            else:
                assigned_videos_with_metadata.append(video_entry)

        results.append({"assigned_videos": assigned_videos_with_metadata, "watch_time": np.sum(assigned_durations), "skipped": skipped_videos})

    return results

##------ Parallel Processing------#
if __name__ == '__main__':
    num_cores = min(cpu_count(), 4)
    print(f" Using {num_cores} cores for parallel processing...")

    batch_size = 500
    total_users = len(engagement_df["user_id"].unique())
    user_batches = np.array_split(engagement_df, total_users // batch_size)

    final_results = []

    for batch_index, batch in enumerate(user_batches):
        print(f" Processing Batch {batch_index + 1}/{len(user_batches)}...")

        with Pool(processes=num_cores) as pool:
            batch_results = pool.map(process_batch, [batch])

        for result in batch_results:
            final_results.extend(result)

    engagement_df["assigned_videos"] = [res["assigned_videos"] for res in final_results]
    engagement_df["skipped_videos"] = [res["skipped"] for res in final_results]

    engagement_df.to_csv("Phase-I\Video_assignement\Datasets\assigned_videos_9_5_7.csv", index=False)

    print(" Step 9, 5, and 7 Completed: Videos Assigned Efficiently with Skipped Handling & Watch Time Correction")
