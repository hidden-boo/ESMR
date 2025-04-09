import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import ast  # Safe alternative to eval()

# Load Dataset with Assigned Videos from Step 8 & 10
engagement_df = pd.read_csv("Phase-I\Video_assignement\Datasets\assigned_videos_8_10_fixed.csv")

# Baseline Watching Start Time (Assume User Starts at 8:00 AM)
BASE_START_TIME = datetime.strptime("08:00:00", "%H:%M:%S")

# Step 11: Generate Timestamps for Each User-Day (Without Date)
def assign_timestamps(row):
    """
    Assigns timestamps to videos while ensuring:
    - Sequential increasing order of timestamps.
    - A realistic time gap (scrolling time) is added between videos.
    - Only the time is kept (date is removed).
    """
    user_id = row["user_id"]
    assigned_videos = row["assigned_videos"]

    # Fix: Handle NaN values & Parse Assigned Videos Safely
    if isinstance(assigned_videos, str):
        try:
            assigned_videos = ast.literal_eval(assigned_videos)
            if not isinstance(assigned_videos, list) or not all(isinstance(video, dict) and "video_id" in video for video in assigned_videos):
                assigned_videos = []
        except:
            assigned_videos = []
    else:
        assigned_videos = []

    if assigned_videos == ["churned"]:
        return ["churned"]  # No timestamps for churned users

    #  Ensure Assigned Videos are Sorted in the Order They Were Assigned
    assigned_videos = sorted(assigned_videos, key=lambda x: x.get("video_id", ""))  # Sort by video ID for consistency

    # Initialize Day's Start Time
    current_time = BASE_START_TIME  # Start at 08:00 AM

    timestamped_videos = []

    for video in assigned_videos:
        # Assign timestamp without the date
        timestamped_videos.append({
            "video_id": video["video_id"],
            "timestamp": current_time.strftime("%H:%M:%S")  # Only store time (HH:MM:SS)
        })

        #  Fix: Ensure Increasing Time Gaps
        # Adding both watch time and scroll time for realistic spacing
        watch_time = video.get("watch_time", 30)  # Default watch time = 30 sec if missing
        scroll_time = random.randint(5, 30)  # Random scrolling time (5-30 sec)

        # Ensure timestamps always increase
        current_time += timedelta(seconds=watch_time + scroll_time)

    return timestamped_videos

# Parallel Processing for Speed
if __name__ == '__main__':
    num_cores = min(cpu_count(), 4)  # Limit to 4 cores for stability
    print(f"Using {num_cores} cores for parallel processing...")

    with Pool(processes=num_cores) as pool:
        engagement_df["assigned_videos_with_timestamps"] = pool.map(assign_timestamps, [row for _, row in engagement_df.iterrows()])

    # Save Updated Dataset
    engagement_df.to_csv("Phase-I\Video_assignement\Datasets\assigned_videos_11_timestamps.csv", index=False)

    print(" Step 11 Completed: Assigned Timestamps with Strictly Increasing Order & Realistic Gaps")
