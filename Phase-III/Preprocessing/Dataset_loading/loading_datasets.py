import json
import pandas as pd
from tqdm import tqdm

# Load user engagement data
with open('Phase-II\TabTransfomers\Datasets\processed_engagement_with_clusters.json') as f:
    user_data = json.load(f)

# Load emotion prediction data
with open('Phase-II\TabTransfomers\Datasets\soft_predictions.json') as f:
    emotion_data = json.load(f)

# Convert emotion predictions into a lookup DataFrame
emotion_df = pd.DataFrame(emotion_data)
emotion_lookup = emotion_df.set_index(['user_id', 'day']).to_dict(orient='index')

content_df = pd.read_csv("Phase-I\Datasets\content_dataset.csv")

