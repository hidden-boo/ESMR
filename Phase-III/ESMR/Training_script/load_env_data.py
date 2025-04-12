import json
from collections import defaultdict
import numpy as np

# Define emotion list and set for negative emotions
EMOTION_LIST = ["happy", "stressed", "disappointed", "angry", "anxious", "lonely", "excited", "neutral"]
NEGATIVE_EMOTIONS = {"stressed", "disappointed", "angry", "anxious", "lonely"}

def extract_emotion_vector(dominant_emotion):
    vector = []
    for emo in EMOTION_LIST:
        flag = 1 if dominant_emotion == emo else 0
        vector.append(flag)
    return vector

def compute_emotion_trend_averages(user_records):
    # Sort records by day for each user
    records = sorted(user_records, key=lambda x: x["day"])
    for i, record in enumerate(records):
        for emo in EMOTION_LIST:
            trend_vals = [records[j][emo] for j in range(max(0, i - 2), i + 1)]
            record[f"{emo}_trend_avg3"] = sum(trend_vals) / len(trend_vals)
    return records

def compute_user_negative_flags(user_emotional_state, emotion_threshold):
    flags = {}
    for user_id, emotions in user_emotional_state.items():
        negative_streak = 0
        for _, emotion in sorted(emotions, key=lambda x: x[0]):
            if emotion in NEGATIVE_EMOTIONS:
                negative_streak += 1
            else:
                negative_streak = 0
            if negative_streak >= emotion_threshold:
                flags[user_id] = True
                break
        else:
            flags[user_id] = False
    return flags

def load_env_data(json_path, content_metadata=None, emotion_threshold=5):
    # Load the raw data from the given JSON file
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    user_data_dict = defaultdict(list)
    video_metadata_dict = {}
    user_seen_videos = defaultdict(set)
    user_emotional_state = defaultdict(list)

    for entry in raw_data:
        user_id = entry["user_id"]
        day = entry["day"]
        videos_watched = entry.get("videos_watched", [])
        skipped_videos = entry.get("skipped_videos", [])

        # Process each video watched
        for video in videos_watched:
            video_id = video["video_id"]
            video_emotion = video.get("emotion", "neutral")
            user_emotional_state[user_id].append((day, video_emotion))

            # Process the video record, integrating causal features
            user_record = process_video_record(day, video_id, video_emotion, entry)
            user_data_dict[user_id].append(user_record)
            user_seen_videos[user_id].add(video_id)

            # Ensure video metadata is correctly added
            if video_id not in video_metadata_dict:
                video_metadata_dict[video_id] = {
                    "emotion": video_emotion,
                    "emotion_intensity": video.get("emotion_intensity", 3),
                    "engagement_score": video.get("video_engagement_score", 0.5),
                    "theme": video.get("theme", "unknown"),  # Default to "unknown"
                    "category": video.get("category", "unknown"),  # Default to "unknown"
                    "video_duration": video.get("video_duration", 30)  # Default to 30
                }

        # Process skipped videos (mark as disappointed)
        for video_id in skipped_videos:
            user_emotional_state[user_id].append((day, "disappointed"))
            meta = content_metadata.get(video_id, {}) if content_metadata else {}
            user_record = process_video_record(day, video_id, "disappointed", entry, is_skipped=True)
            user_data_dict[user_id].append(user_record)
            user_seen_videos[user_id].add(video_id)

            # If skipped video has no metadata, set defaults
            if video_id not in video_metadata_dict:
                video_metadata_dict[video_id] = {
                    "emotion": "disappointed",
                    "emotion_intensity": 2,
                    "engagement_score": 0.0,
                    "theme": meta.get("theme", "unknown"),
                    "category": meta.get("category", "unknown"),
                    "video_duration": meta.get("video_duration", 30)
                }

    # Debugging: Print loaded data
    #print(f"Loaded {len(user_data_dict)} users.")
    #print(f"Loaded {len(video_metadata_dict)} unique videos.")
    #print(f"Loaded {len(user_seen_videos)} unique users who have seen videos.")
    
    # Print some example user data to inspect
    #print("Sample user data (first user):", list(user_data_dict.items())[0])
    #print("Sample video metadata (first video):", list(video_metadata_dict.items())[0])


    # Calculate user flags based on negative emotions streak
    user_needs_rl_flag = compute_user_negative_flags(user_emotional_state, emotion_threshold)

    # Compute emotion trend averages (smooth emotional progression)
    for user_id in user_data_dict:
        user_data_dict[user_id] = compute_emotion_trend_averages(user_data_dict[user_id])

      # Compute emotion deltas (difference in emotions from day to day)
    user_data_dict = compute_emotion_deltas(user_data_dict)  # Add deltas here

    # Compute engagement trends (rolling window of engagement scores)
    user_data_dict = compute_engagement_trends(user_data_dict)  # Add engagement trends here

    # Debugging: Print after computed trends
    #print("Sample user data after trends (first user):", list(user_data_dict.items())[0])

    # Create mappings between video IDs and indices for easy reference
    all_video_ids = list(video_metadata_dict.keys())
    video_id_to_idx = {vid: idx for idx, vid in enumerate(all_video_ids)}
    idx_to_video_id = {idx: vid for vid, idx in video_id_to_idx.items()}

    # Return all the processed data, including flags for when RL should be used
    return (
        user_data_dict,
        video_metadata_dict,
        user_seen_videos,
        video_id_to_idx,
        idx_to_video_id,
        user_needs_rl_flag
    )

def compute_emotion_deltas(user_data_dict):
    """
    Compute the emotional change (delta) for each user between consecutive days.
    """
    for user_id, records in user_data_dict.items():
        previous_emotion_val = None
        for record in records:
            current_emotion_val = record["happy"]  # Can be generalized to a composite score based on all emotions
            if previous_emotion_val is not None:
                record["emotion_delta"] = current_emotion_val - previous_emotion_val
            previous_emotion_val = current_emotion_val
    return user_data_dict

def compute_engagement_trends(user_data_dict, window=3):
    """
    Compute the engagement trends for each user (average engagement over a rolling window).
    """
    for user_id, records in user_data_dict.items():
        engagement_history = [record["engagement_score"] for record in records]
        for i in range(len(records)):
            if i >= window:
                records[i]["engagement_trend"] = np.mean(engagement_history[i-window:i])
            else:
                records[i]["engagement_trend"] = np.mean(engagement_history[:i+1])
    return user_data_dict

def process_video_record(day, video_id, video_emotion, entry, is_skipped=False):
    """
    Process each video record to include features that will be used for causal reward shaping.
    """
    base_record = {
        "day": day,
        "video_id": video_id,
        "engagement_score": 0.0 if is_skipped else entry.get("engagement_score", 0.0),
        "churned": 1 if is_skipped else int(entry.get("churned", False)),
        "scrolling_time": entry.get("scrolling_time", 0),
        "video_watching_duration": entry.get("video_watching_duration", 0),
        "time_spent_daily": entry.get("time_spent_daily", 0),
        "dominant_emotion": video_emotion,
        "video_theme": entry.get("theme", "unknown"),
        "video_category": entry.get("category", "unknown"),
        "emotion_intensity": entry.get("emotion_intensity", 3),  # Store emotion intensity for reward shaping
    }
    emotion_vector = extract_emotion_vector(video_emotion)
    for idx, emo in enumerate(EMOTION_LIST):
        base_record[emo] = emotion_vector[idx]
    base_record["emotion_vector"] = emotion_vector
    return base_record

if __name__ == "__main__":
    # Path to your data
    data_path = "Phase-II\TabTransfomers\Datasets\processed_engagement_with_clusters.json"
    # Load the data
    user_data_dict, video_metadata_dict, _, video_id_to_idx, _, user_negative_state_flag = load_env_data(data_path)

    # Print outputs for debugging
    #print(f"Number of Users Loaded: {len(user_data_dict)}")
    #print(f"Number of Videos Loaded: {len(video_metadata_dict)}")
    #print(f"Sample User Data (first 2 users): {list(user_data_dict.items())[:2]}")
    #print(f"Sample Video Metadata (first 2 videos): {list(video_metadata_dict.items())[:2]}")
