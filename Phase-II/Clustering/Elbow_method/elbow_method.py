import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Load Clustered Data
data_path = "Phase-II\Clustering\Generated_dataset\validated_emotion_clusters.csv"
df = pd.read_csv(data_path)

#  Check Available Columns
print(f"Columns: {df.columns.tolist()}")

#  Ensure Required Columns Exist
required_columns = ["engagement_cluster", "user_emotion", "engagement_score"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing Columns: {missing}")

# Extract Relevant Features
features = df[["engagement_score"]].copy() 

# Elbow Method (Optimal K Selection)
distortions = []
K_range = range(2, 11)  # Test clusters from K=2 to K=10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features)
    distortions.append(sum(np.min(cdist(features, kmeans.cluster_centers_, "euclidean"), axis=1)) / features.shape[0])

#  Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, distortions, marker="o", linestyle="--")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Distortion")
plt.title("Elbow Method for Optimal K")
plt.show()

# Silhouette Score (Cluster Separation Quality)
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    silhouette_scores.append(silhouette_score(features, labels))

#  Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker="o", linestyle="--", color="green")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs Number of Clusters")
plt.show()

# Cluster Distribution (Do Clusters Form Distinct Emotion Groups?)
plt.figure(figsize=(10, 6))
sns.countplot(x="engagement_cluster", hue="user_emotion", data=df, palette="viridis")
plt.xlabel("Engagement Cluster")
plt.ylabel("Count")
plt.title("Cluster Distribution by User Emotion")
plt.legend(title="User Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

# Intra-cluster Variance & Inter-cluster Distance
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)  # Assuming K=5 from previous clustering
df["predicted_cluster"] = kmeans.fit_predict(features)

# Compute Variance within each cluster
intra_variance = df.groupby("predicted_cluster")["engagement_score"].var()
print(" Intra-cluster Variance:")
print(intra_variance)

#  Compute Inter-cluster Distances (Between Cluster Centers)
cluster_centers = kmeans.cluster_centers_
inter_cluster_distances = cdist(cluster_centers, cluster_centers, metric="euclidean")

print(" Inter-cluster Distance Matrix:")
print(pd.DataFrame(inter_cluster_distances, columns=range(len(cluster_centers)), index=range(len(cluster_centers))))
