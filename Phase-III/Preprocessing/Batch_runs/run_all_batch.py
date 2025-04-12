import json
import pandas as pd
from tqdm import tqdm
import os

# CONFIG
BATCH_SIZE = 50  # Number of users per batch
OUTPUT_DIR = "Phase-III\Preprocessing\Batch_runs\engagement_batches"
os.makedirs(OUTPUT_DIR, exist_ok=True)

content_df = pd.read_csv("Phase-I\Datasets\content_dataset.csv")
all_video_ids = content_df['video_id'].unique()
video_lookup = content_df.set_index("video_id").to_dict(orient="index")

# Get unique user IDs
all_user_ids = list(set(entry['user_id'] for entry in user_data))

for i in range(0, len(all_user_ids), BATCH_SIZE):
    batch_users = set(all_user_ids[i:i + BATCH_SIZE])
    print(f" Processing batch {i // BATCH_SIZE + 1}: Users {i}–{i + BATCH_SIZE - 1}")

    full_rows = []
    errors = 0

    batch_data = [entry for entry in user_data if entry['user_id'] in batch_users]

    for entry in tqdm(batch_data):
        try:
            user_id = entry['user_id']
            day = entry['day']

            watched_videos = {v['video_id'] for v in entry.get('videos_watched', [])}
            skipped_videos = set(entry.get('skipped_videos', []))

            for video_id in all_video_ids:
                # Label
                if video_id in watched_videos:
                    label = 1
                elif video_id in skipped_videos:
                    label = 0
                else:
                    label = -1

                video_meta = video_lookup.get(video_id, None)
                if not video_meta:
                    errors += 1
                    continue

                row = {
                    'user_id': user_id,
                    'day': day,
                    'video_id': video_id,
                    'label': label,
                    'category': video_meta.get('category', None),
                    'theme': video_meta.get('theme', None),
                    'video_duration': video_meta.get('video_duration', None),
                    'video_engagement_score': video_meta.get('engagement_score', None),
                    'video_emotion': video_meta.get('emotion', None),
                    'video_emotion_intensity': video_meta.get('emotion_intensity', None),
                    'scrolling_time': entry['scrolling_time'],
                    'video_watching_duration': entry['video_watching_duration'],
                    'post_story_views': entry['post_story_views'],
                    'time_spent_daily': entry['time_spent_daily'],
                    'daily_logins': entry['daily_logins'],
                    'posting_frequency': entry['posting_frequency'],
                    'liking_behavior': entry['liking_behavior'],
                    'commenting_activity': entry['commenting_activity'],
                    'sharing_behavior': entry['sharing_behavior'],
                    'educational_time_spent': entry['educational_time_spent'],
                    'entertainment_time_spent': entry['entertainment_time_spent'],
                    'news_time_spent': entry['news_time_spent'],
                    'inspirational_time_spent': entry['inspirational_time_spent'],
                    'user_profile': entry['user_profile'],
                    'churned': int(entry['churned']),
                    'previous_day_engagement': entry['previous_day_engagement'],
                    'previous_week_avg_engagement': entry['previous_week_avg_engagement'],
                    'engagement_score': entry['engagement_score'],
                    'engagement_growth_rate': entry['engagement_growth_rate'],
                    'liking_trend': entry['liking_trend'],
                    'commenting_trend': entry['commenting_trend'],
                    'sharing_trend': entry['sharing_trend'],
                    'scrolling_watching_ratio': entry['scrolling_watching_ratio'],
                    'education_ratio': entry['education_ratio'],
                    'entertainment_ratio': entry['entertainment_ratio'],
                    'news_ratio': entry['news_ratio'],
                    'inspiration_ratio': entry['inspiration_ratio'],
                    'days_since_last_engagement': entry['days_since_last_engagement']
                }

                # Add emotion predictions if available
                emo_key = (user_id, day)
                if emo_key in emotion_lookup:
                    row.update(emotion_lookup[emo_key])

                full_rows.append(row)

        except Exception as e:
            print(f"[ERROR] user {entry.get('user_id')} day {entry.get('day')} — {e}")
            errors += 1
            continue

    # Save this batch
    batch_df = pd.DataFrame(full_rows)
    out_path = os.path.join(OUTPUT_DIR, f"engagement_batch_{i // BATCH_SIZE + 1}.csv")
    out_path = out_path.replace(".csv", ".parquet")
    batch_df.to_parquet(out_path, index=False)
    print(f" Saved batch to {out_path} — shape: {batch_df.shape}, errors: {errors}")
