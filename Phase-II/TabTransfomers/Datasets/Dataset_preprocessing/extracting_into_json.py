import pandas as pd
import json
import ast

#  Load Datasets
engagement_data_path = "Phase-I\Video_assignement\Datasets\final_dataset_combined_ready.csv"
video_data_path = "Phase-I\Datasets\content_dataset.csv"
clusters_data_path = "Phase-II\Clustering\Generated_dataset\validated_emotion_clusters.csv"

# Load datasets
engagement_df = pd.read_csv(engagement_data_path)
video_df = pd.read_csv(video_data_path)
clusters_df = pd.read_csv(clusters_data_path)

# Process JSON-like String Columns
def parse_json_list(json_str):
    try:
        return ast.literal_eval(json_str)
    except:
        return []

engagement_df["assigned_videos_with_timestamps"] = engagement_df["assigned_videos_with_timestamps"].apply(parse_json_list)
engagement_df["skipped_videos"] = engagement_df["skipped_videos"].apply(parse_json_list)  # Keep skipped videos

#  Convert Video Dataset into Dictionary for Fast Lookup
video_metadata_dict = video_df.set_index("video_id").to_dict(orient="index")

#  Attach Video Metadata (For Both Watched & Skipped Videos)
def attach_video_metadata(video_list):
    enriched_videos = []
    for video in video_list:
        # Handle different formats (dict vs. string)
        if isinstance(video, dict):
            video_id = video.get("video_id")
            timestamp = video.get("timestamp", None)  # Only assigned videos have timestamps
        else:  # If it's a string, treat it as a skipped video
            video_id = video
            timestamp = None  # Skipped videos don't have timestamps

        if video_id in video_metadata_dict:
            enriched_videos.append({
                "video_id": video_id,
                "timestamp": timestamp,
                "theme": video_metadata_dict[video_id]["theme"],
                "category": video_metadata_dict[video_id]["category"],
                "emotion": video_metadata_dict[video_id]["emotion"],
                "emotion_intensity": video_metadata_dict[video_id]["emotion_intensity"],
                "video_duration": video_metadata_dict[video_id]["video_duration"],
                "video_engagement_score": video_metadata_dict[video_id]["engagement_score"]
            })
    return enriched_videos

# Apply metadata attachment to both watched & skipped videos
engagement_df["videos_watched"] = engagement_df["assigned_videos_with_timestamps"].apply(attach_video_metadata)
engagement_df["skipped_videos_with_metadata"] = engagement_df["skipped_videos"].apply(attach_video_metadata)

#  Drop Unnecessary Columns (Keep Skipped Videos Now!)
engagement_df.drop(columns=["assigned_videos_with_timestamps"], inplace=True)

# Merge Engagement Data with Cluster Labels Based on user_id & day
merged_df = engagement_df.merge(clusters_df[["user_id", "day", "engagement_cluster", "user_emotion", "user_emotion_label"]],
                                on=["user_id", "day"], how="left")

#  Convert to JSON
output_json_path = "Phase-II\TabTransfomers\Datasets\processed_engagement_with_clusters.json"
merged_df.to_json(output_json_path, orient="records", indent=4)

print(f"Processed dataset with clusters (including skipped videos) saved as JSON at: {output_json_path}")
