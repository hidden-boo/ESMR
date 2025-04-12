import pandas as pd
import numpy as np

# === PARAMETERS ===
NUM_USERS = 1000       # Simulating a platform-scale user base
NUM_DAYS = 30          # Duration of user engagement simulation

# === FEATURE SCHEMA ===
columns = [
    "user_id", "day", "scrolling_time", "video_watching_duration", "post_story_views",
    "time_spent_daily", "daily_logins", "posting_frequency", "liking_behavior",
    "commenting_activity", "sharing_behavior", "educational_time_spent",
    "entertainment_time_spent", "news_time_spent", "inspirational_time_spent",
    "user_profile", "churned"
]

# === USER BEHAVIOR PROFILES ===
# Profiles encode engagement patterns with realistic time budgets, churn probabilities, and video time ranges
# Inspired by clustering and churn segmentation literature&#8203;:contentReference[oaicite:0]{index=0}
user_profiles = {
    "casual": {
        "scroll_multiplier": 2.5, "interaction_multiplier": 0.5,
        "churn_prob": 0.2, "reengage_prob": 0.3, "second_churn_prob": 0.4,
        "video_time_range": (300, 900)   # 5-15 minutes per session
    },
    "engaged": {
        "scroll_multiplier": 1.5, "interaction_multiplier": 1.0,
        "churn_prob": 0.08, "reengage_prob": 0.5, "second_churn_prob": 0.3,
        "video_time_range": (900, 1800)  # 15-30 minutes per session
    },
    "highly_active": {
        "scroll_multiplier": 1.0, "interaction_multiplier": 1.5,
        "churn_prob": 0.05, "reengage_prob": 0.7, "second_churn_prob": 0.15,
        "video_time_range": (1800, 3600)  # 30-60+ minutes
    }
}

# === USER ID INITIALIZATION ===
np.random.seed(42)
user_ids = [f"U{str(i).zfill(4)}" for i in range(1, NUM_USERS + 1)]
user_profiles_assigned = np.random.choice(["casual", "engaged", "highly_active"], size=NUM_USERS, p=[0.5, 0.4, 0.1])
user_profiles_map = dict(zip(user_ids, user_profiles_assigned))

simulated_data = []

# === MAIN SIMULATION LOOP ===
for user_id in user_ids:
    profile = user_profiles[user_profiles_map[user_id]]
    churned = False
    reengaged = False
    engagement_trend = np.random.uniform(-2, 2)

    for day in range(1, NUM_DAYS + 1):
        churn_prob = profile["churn_prob"] * (0.9 if engagement_trend >= 0 else 1.2)
        reengage_prob = profile["reengage_prob"] * (1.1 if engagement_trend >= 0 else 0.8)

        # — Churn / Re-engagement Logic —
        if not churned and day > 7 and np.random.rand() < churn_prob:
            churned = True
        if churned and np.random.rand() < reengage_prob:
            churned = False
            engagement_trend += np.random.randint(2, 5)
            reengaged = True
        if reengaged and np.random.rand() < profile["second_churn_prob"]:
            churned = True

        if churned:
            simulated_data.append([
                user_id, day, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                user_profiles_map[user_id], True
            ])
            continue

        # Weekend and random engagement spikes
        weekend_multiplier = 1.2 if day % 7 in [6, 0] else 1.0
        spike_multiplier = np.random.uniform(1.1, 1.8) if day in np.random.choice(range(5, 25), 3) else 1.0

        # === Scrolling and Video Engagement ===
        scrolling_time = np.clip(
            np.random.uniform(1.5, 3.0) * np.random.uniform(*profile["video_time_range"]) * profile["scroll_multiplier"],
            100, 2000
        )
        video_watching_duration = np.clip(
            np.random.uniform(*profile["video_time_range"]) * spike_multiplier,
            100, 3600
        )

        post_story_views = int(scrolling_time * np.random.uniform(0.4, 1.0))
        time_spent_daily = scrolling_time + video_watching_duration + np.random.randint(10, 50)

        # === Partial Video Watching (Skipped vs Watched) ===
        num_videos = int(video_watching_duration / 45)
        skipped_videos = int(num_videos * 0.4)
        watched_videos = num_videos - skipped_videos
        skipped_time = sum(np.random.uniform(0.2, 0.7) * 45 for _ in range(skipped_videos))
        adjusted_video_watching_duration = watched_videos * 45 + skipped_time

        # === Social Interaction Features ===
        daily_logins = np.random.poisson(3)
        posting_frequency = min(np.random.negative_binomial(1, 0.7), 8)
        liking_behavior = int(round(np.random.lognormal(1.5, 0.6) * profile["interaction_multiplier"] * (adjusted_video_watching_duration / 300)))
        commenting_activity = int(round(np.random.choice([0, 1, 3, 5], p=[0.5, 0.3, 0.15, 0.05]) * (adjusted_video_watching_duration / 600)))
        sharing_behavior = int(round(np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05]) * (adjusted_video_watching_duration / 900)))

        # — Normalize Scrolling Time (re-scaling per profile)
        if user_profiles_map[user_id] == "casual":
            scrolling_time = np.clip(video_watching_duration * np.random.uniform(1.2, 2.0), 100, 1500)
        elif user_profiles_map[user_id] == "engaged":
            scrolling_time = np.clip(video_watching_duration * np.random.uniform(0.8, 1.5), 100, 1200)
        else:
            scrolling_time = np.clip(video_watching_duration * np.random.uniform(0.6, 1.2), 100, 1000)

        # === Content Time Allocation: Dirichlet Sampling ===
        base_weights = np.array([1.2, 1.2, 1.0, 1.0])
        category_distribution = np.random.dirichlet(base_weights)
        content_engagement = {
            "educational": video_watching_duration * max(0.15, category_distribution[0]),
            "entertainment": video_watching_duration * max(0.15, category_distribution[1]),
            "news": video_watching_duration * max(0.15, category_distribution[2]),
            "inspirational": video_watching_duration * max(0.15, category_distribution[3]),
        }

        # Normalize durations
        total_category_time = sum(content_engagement.values())
        scaling_factor = video_watching_duration / total_category_time
        content_engagement = {cat: time * scaling_factor for cat, time in content_engagement.items()}

        # Update engagement trend
        engagement_today = (scrolling_time + adjusted_video_watching_duration) / 2
        engagement_trend = 0.8 * engagement_trend + 0.2 * np.sign(engagement_today - 300)

        # === Append Record ===
        simulated_data.append([
            user_id, day, scrolling_time, adjusted_video_watching_duration, post_story_views,
            time_spent_daily, daily_logins, posting_frequency, liking_behavior,
            commenting_activity, sharing_behavior,
            content_engagement["educational"], content_engagement["entertainment"],
            content_engagement["news"], content_engagement["inspirational"],
            user_profiles_map[user_id], churned
        ])

# === SAVE FINAL CSV ===
engagement_data = pd.DataFrame(simulated_data, columns=columns)
engagement_data.to_csv("Phase-I\Datasets\engagement_dataset.csv", index=False)
print(engagement_data.head())
