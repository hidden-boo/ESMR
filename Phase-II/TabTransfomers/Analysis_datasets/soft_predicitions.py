import pandas as pd
import matplotlib.pyplot as plt

# === Load the updated JSON with trend features ===
file_path = "Phase-II\TabTransfomers\Datasets\soft_predictions_with_trends.json"
df = pd.read_json(file_path)

# === Pick a sample user to analyze ===
sample_user_id = df["user_id"].unique()[0]
user_df = df[df["user_id"] == sample_user_id].sort_values("day")

# === Plot Emotion Trends with Larger Fonts ===
plt.figure(figsize=(14, 7))

plt.plot(user_df["day"], user_df["happy_trend_avg3"], label="Happy", linewidth=2)
plt.plot(user_df["day"], user_df["disappointed_trend_avg3"], label="Disappointed", linewidth=2)
plt.plot(user_df["day"], user_df["stressed_trend_avg3"], label="Stressed", linewidth=2)
plt.plot(user_df["day"], user_df["frustrated_trend_avg3"], label="Frustrated", linewidth=2)
plt.plot(user_df["day"], user_df["churned_trend_avg3"], label="Churned", linestyle="--", linewidth=2)

plt.title(f"Emotional & Churn Trends Over Time — User: {sample_user_id}", fontsize=35)
plt.xlabel("Day", fontsize=28)
plt.ylabel("3-Day Avg Trend", fontsize=28)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)

plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()

