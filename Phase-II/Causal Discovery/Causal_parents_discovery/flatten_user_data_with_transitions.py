import json
import pandas as pd
from collections import defaultdict, Counter
import os

def load_flattened_df_with_transitions(json_path, output_csv_path):
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    emotion_list = ["happy", "stressed", "disappointed", "angry", "anxious", "lonely", "excited"]
    emotion_val_map = {
        "happy": 1.0,
        "excited": 0.7,
        "neutral": 0.0,
        "lonely": -0.4,
        "anxious": -0.6,
        "disappointed": -0.8,
        "stressed": -1.0,
        "angry": -0.9
    }

    flat_data = []

    for entry in raw_data:
        user_id = entry["user_id"]
        day = entry["day"]
        scrolling_time = entry.get("scrolling_time", 0)
        watching_time = entry.get("video_watching_duration", 0)
        time_spent_daily = entry.get("time_spent_daily", 0)
        churned = int(entry.get("churned", False))
        watched_videos = entry.get("videos_watched", [])

        # Compute dominant emotion
        emotion_counts = Counter([v.get("emotion", "neutral") for v in watched_videos])
        dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "neutral"

        for video in watched_videos:
            video_id = video["video_id"]
            video_emotion = video.get("emotion", "neutral")
            intensity = video.get("emotion_intensity", 3)
            engagement = video.get("video_engagement_score", 0.0)

            row = {
                "user_id": user_id,
                "day": day,
                "video_id": video_id,
                "video_emotion": video_emotion,
                "dominant_emotion": dominant_emotion,
                "emotion_val": emotion_val_map.get(video_emotion, 0.0),
                "emotion_intensity": intensity,
                "video_engagement_score": engagement,
                "scrolling_time": scrolling_time,
                "video_watching_duration": watching_time,
                "time_spent_daily": time_spent_daily,
                "churned": churned
            }

            for emo in emotion_list:
                row[f"user_{emo}"] = 1 if video_emotion == emo else 0

            flat_data.append(row)

        for skipped_id in entry.get("skipped_videos", []):
            row = {
                "user_id": user_id,
                "day": day,
                "video_id": skipped_id,
                "video_emotion": "disappointed",
                "dominant_emotion": "disappointed",
                "emotion_val": emotion_val_map["disappointed"],
                "emotion_intensity": 2,
                "video_engagement_score": 0.0,
                "scrolling_time": scrolling_time,
                "video_watching_duration": watching_time,
                "time_spent_daily": time_spent_daily,
                "churned": 1
            }

            for emo in emotion_list:
                row[f"user_{emo}"] = 1 if emo == "disappointed" else 0

            flat_data.append(row)

    df = pd.DataFrame(flat_data)
    df.sort_values(by=["user_id", "day"], inplace=True)

    # === Add transition features ===
    df["next_day_engagement"] = df.groupby("user_id")["video_engagement_score"].shift(-1)
    df["engagement_change"] = df["next_day_engagement"] - df["video_engagement_score"]

    df["next_day_intensity"] = df.groupby("user_id")["emotion_intensity"].shift(-1)
    df["intensity_change"] = df["next_day_intensity"] - df["emotion_intensity"]

    df["next_day_emotion_val"] = df.groupby("user_id")["emotion_val"].shift(-1)
    df["emotion_val_change"] = df["next_day_emotion_val"] - df["emotion_val"]

    df["next_day_dominant_emotion"] = df.groupby("user_id")["dominant_emotion"].shift(-1)

    for emo in emotion_list:
        df[f"next_day_{emo}"] = df.groupby("user_id")[f"user_{emo}"].shift(-1)

    # === Save ===
    df.to_csv(output_csv_path, index=False)
    print(f"[SAVED] Flattened data with transitions: {output_csv_path}")


# === RUN FLATTENER ===
if __name__ == "__main__":
    input_json = "Phase-II\TabTransfomers\Datasets\processed_engagement_with_clusters.json"
    output_csv = "Phase-II\Causal Discovery\Datasets_generated\flat_user_data_with_transitions.csv"
    load_flattened_df_with_transitions(input_json, output_csv)
