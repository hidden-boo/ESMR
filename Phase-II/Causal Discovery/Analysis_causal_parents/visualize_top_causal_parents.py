import pandas as pd
import matplotlib.pyplot as plt
import os

# === 1. Load Causal Parent Results ===
csv_path = "Phase-II\Causal Discovery\Datasets_generated\top_causal_parents.csv"
output_dir = "Phase-II\Causal Discovery\Results_analysis_causal_parents\causal_plots"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
targets = df["Target_Variable"].unique()

# === 2. Plot Top K Causal Parents for Each Target ===
TOP_K = 5

for target in targets:
    df_t = df[df["Target_Variable"] == target].sort_values(by="Causal_Weight", key=abs, ascending=False).head(TOP_K)

    plt.figure(figsize=(8, 5))
    bars = plt.barh(df_t["Causal_Parent"], df_t["Causal_Weight"], color="steelblue")
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f"Top {TOP_K} Causal Parents for {target}", fontsize=13)
    plt.xlabel("Causal Weight")
    plt.gca().invert_yaxis()  # Most important on top
    plt.tight_layout()

    fname = f"{target}_causal_parents.png".replace(" ", "_")
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path)
    print(f"[Saved] {out_path}")
    plt.close()
