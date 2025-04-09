import pandas as pd
import numpy as np
import random
import ast  # Safer than eval()
from multiprocessing import Pool, cpu_count

# Load Dataset with Assigned Videos from Step 9, 5, 7
engagement_df = pd.read_csv("Phase-I\Video_assignement\Datasets\assigned_videos_9_5_7.csv")
content_df = pd.read_csv("Phase-I\Datasets\content_dataset.csv")

# Convert Categorical Columns to Integer Indices
category_map = {"educational": 0, "entertainment": 1, "news": 2, "inspirational": 3}
content_df["category"] = content_df["category"].map(category_map)

# Sort Content Dataset by Engagement Score for Trending/Fallbacks
content_df_sorted = content_df.sort_values(by="engagement_score", ascending=False).reset_index(drop=True)

# Track Previously Assigned Videos (Rolling Memory per User)
user_memory = {}  # Stores last 5 days of assigned videos per user

# Step 8 & Step 10: Video Assignment with Repetition Handling
def assign_videos_with_variation(row):
    """
    Assigns videos while ensuring:
    - Prevents excessive repeats (Step 8)
    - Handles fallback scenarios with trending videos (Step 10)
    - Accounts for skipped videos differently (higher chance of reappearance)
    """
    user_id = row["user_id"]
    day = row["day"]
    available_time = row["video_watching_duration"]

    if row["churned"]:
        return {"assigned_videos": ["churned"], "skipped_videos": []}  # No video assignment for churned users

    # Retrieve Past Assignments (Rolling Memory)
    if user_id not in user_memory:
        user_memory[user_id] = {"watched": [], "skipped": []}

    past_videos = user_memory[user_id]["watched"]  # Past watched videos
    skipped_videos = user_memory[user_id]["skipped"]  # Videos that were previously skipped

    # Fetch User’s Assigned Videos (From Step 9, 5, 7)
    assigned_videos = row["assigned_videos"]

    #  Handle NaN & Parse Assigned Videos Safely
    if isinstance(assigned_videos, str):
        try:
            assigned_videos = ast.literal_eval(assigned_videos)
            if not isinstance(assigned_videos, list) or not all(isinstance(video, dict) and "video_id" in video for video in assigned_videos):
                assigned_videos = []
        except:
            assigned_videos = []  # If conversion fails, assume empty list
    else:
        assigned_videos = []  # If already NaN or not a string, set to empty list

    # Step 8: Remove Repeats (Except Skipped Videos)
    updated_videos = []
    for video in assigned_videos:
        if video["video_id"] not in past_videos:  # No exact repeats
            updated_videos.append(video)
        elif video["video_id"] in skipped_videos and random.random() < 0.4:
            updated_videos.append(video)  # 40% chance to reassign skipped videos

    #  Step 10: Fallback Assignments if All Videos Were Repeated
    if len(updated_videos) == 0:
        fallback_videos = [vid for vid in content_df_sorted["video_id"].tolist() if vid not in updated_videos]
        if len(fallback_videos) > 0:
            updated_videos.extend([{"video_id": vid, "watch_time": 30} for vid in fallback_videos[:3]])
        else:
            updated_videos.append({"video_id": "default_placeholder", "watch_time": 30})  # Ensures non-empty output

    #  Avoid Duplicate Entries in Skipped Videos
    new_skipped = [vid["video_id"] for vid in assigned_videos if vid["video_id"] not in updated_videos]
    user_memory[user_id]["skipped"] = list(set(user_memory[user_id]["skipped"]).union(set(new_skipped)))

    # Trim memory to last 5 days (limit to 50 videos)
    user_memory[user_id]["watched"] = user_memory[user_id]["watched"][-50:]
    user_memory[user_id]["skipped"] = user_memory[user_id]["skipped"][-50:]

    return {"assigned_videos": updated_videos, "skipped_videos": user_memory[user_id]["skipped"]}

# Parallel Processing for Speed
if __name__ == '__main__':
    num_cores = min(cpu_count(), 4)  # Limit to 4 cores for stability
    print(f"Using {num_cores} cores for parallel processing...")

    with Pool(processes=num_cores) as pool:
        results = pool.map(assign_videos_with_variation, [row for _, row in engagement_df.iterrows()])

    # Ensure Results Always Contain 'skipped_videos' Key
    for res in results:
        if "skipped_videos" not in res:
            res["skipped_videos"] = []

    # Extract Assigned Videos & Skipped Videos
    engagement_df["assigned_videos"] = [res["assigned_videos"] for res in results]
    engagement_df["skipped_videos"] = [res["skipped_videos"] for res in results]

    # Save Updated Dataset
    engagement_df.to_csv("Phase-I\Video_assignement\Datasets\assigned_videos_8_10_fixed.csv", index=False)

    print(" Step 8 & 10 Completed: Prevented Repeats, Handled Skips, & Added Fallbacks")
