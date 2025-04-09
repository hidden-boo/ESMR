import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from lingam import DirectLiNGAM
import os

# === Load your flattened data ===
csv_path = "Phase-II\Causal Discovery\Datasets_generated\flat_user_data_with_transitions.csv"
print(f"[INFO] Loading data from: {csv_path}")
df = pd.read_csv(csv_path)

# === Define expanded feature set ===
features = [
    "emotion_val",
    "emotion_intensity",
    "video_engagement_score",
    "scrolling_time",
    "video_watching_duration",
    "time_spent_daily",
    "user_happy",
    "user_stressed",
    "user_disappointed",
    "user_excited",
    "churned",
    "next_day_engagement",
    "engagement_change",
    "intensity_change",
    "next_day_emotion_val",
    "emotion_val_change",
    "next_day_happy",
    "next_day_stressed",
    "next_day_disappointed"
]

# Drop non-numeric columns and handle missing values
df_subset = df[features].dropna().astype(float)

print(f"[INFO] Using {len(features)} features, shape: {df_subset.shape}")

# === Run DirectLiNGAM ===
model = DirectLiNGAM()
model.fit(df_subset.values)
adj_matrix = model.adjacency_matrix_

# ===  Visualize learned causal graph ===
def plot_dag(adj_matrix, var_names):
    G = nx.DiGraph()
    for i, name in enumerate(var_names):
        G.add_node(name)

    edge_count = 0
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            weight = adj_matrix[i, j]
            if abs(weight) > 1e-3:
                G.add_edge(var_names[i], var_names[j], weight=round(weight, 3))
                edge_count += 1

    print(f"[INFO] Total edges: {edge_count}")
    plt.figure(figsize=(18, 14))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=3000,
            edge_color='gray', font_size=10, arrows=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Causal Graph (DirectLiNGAM)", fontsize=16)
    plt.tight_layout()
    plt.show()

plot_dag(adj_matrix, df_subset.columns.tolist())

# ===  Save adjacency matrix and variable names ===
save_dir = "Phase-II\Causal Discovery\Results_analysis_causal_parents"
adj_matrix_df = pd.DataFrame(adj_matrix, columns=features, index=features)

adj_path = os.path.join(save_dir, "causal_adjacency_matrix.csv")
vars_path = os.path.join(save_dir, "causal_variable_names.txt")

adj_matrix_df.to_csv(adj_path)
print(f"[INFO] Saved adjacency matrix to {adj_path}")

with open(vars_path, "w") as f:
    for var in features:
        f.write(f"{var}\n")
print(f"[INFO] Saved variable names to {vars_path}")
