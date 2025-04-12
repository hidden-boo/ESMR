import pandas as pd
import numpy as np
import os

# ===  Set your local paths ===
base_path = "Phase-II\Causal Discovery\Results_analysis_causal_parents"
out_path = "Phase-II\Causal Discovery\Datasets_generated"
adj_path = os.path.join(base_path, "causal_adjacency_matrix.csv")
vars_path = os.path.join(base_path, "causal_variable_names.txt")
output_path = os.path.join(out_path, "top_causal_parents.csv")

print(f"[INFO] Loading adjacency matrix from: {adj_path}")
adj_matrix = pd.read_csv(adj_path, index_col=0).values

print(f"[INFO] Loading variable names from: {vars_path}")
with open(vars_path, "r") as f:
    var_names = [line.strip() for line in f.readlines()]

# === Define new causal targets ===
targets = [
    "next_day_happy",
    "next_day_stressed",
    "next_day_disappointed",
    "next_day_emotion_val",
    "emotion_val_change",
    "engagement_change"
]

causal_parent_dict = {}

# === Extract parents for each target ===
for target in targets:
    if target not in var_names:
        print(f"[WARN] Target {target} not found in variable names.")
        continue

    target_idx = var_names.index(target)
    incoming_edges = adj_matrix[:, target_idx]

    parents = []
    for i, weight in enumerate(incoming_edges):
        if abs(weight) > 1e-3:  # Filter out near-zero weights
            parents.append((var_names[i], round(weight, 4)))

    sorted_parents = sorted(parents, key=lambda x: -abs(x[1]))  # Sort by strength
    causal_parent_dict[target] = sorted_parents

# === Save results to CSV ===
rows = []
for target, parent_list in causal_parent_dict.items():
    for parent, weight in parent_list:
        rows.append({
            "Target_Variable": target,
            "Causal_Parent": parent,
            "Causal_Weight": weight
        })

df_out = pd.DataFrame(rows)
df_out.to_csv(output_path, index=False)

print(f"[INFO] Saved top causal parents to: {output_path}")
print("\n=== Top Causal Parents (preview) ===")
print(df_out.head(20))
