import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(12, 6))
for i, emotion in enumerate(emotion_classes):
    plt.hist(results_df[emotion], bins=30, alpha=0.6, label=emotion)

plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Histogram of Predicted Probabilities per Emotion")
plt.legend()
plt.show()

# Optional for clean styling
plt.style.use("seaborn-v0_8-whitegrid")

plt.figure(figsize=(10, 7))  # Bigger figure

# Loop over emotion classes
for i, emotion in enumerate(emotion_classes):
    prob_true, prob_pred = calibration_curve(results_df["true_label"] == i,
                                             results_df[emotion],
                                             n_bins=10)
    plt.plot(prob_pred, prob_true,
             marker='o', label=emotion,
             linewidth=2.5, markersize=6)

# Perfect calibration line
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)

# Title and axis styling
plt.title("Calibration Curve", fontsize=35)
plt.xlabel("Predicted Probability", fontsize=28)
plt.ylabel("Observed Probability", fontsize=28)

# Ticks and legend
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=22)
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute entropy for each prediction (higher = more uncertainty)
entropy_values = entropy(val_probs.T, base=2)

plt.figure(figsize=(10, 5))
plt.hist(entropy_values, bins=30, alpha=0.7, color="blue")
plt.xlabel("Entropy of Softmax Probabilities")
plt.ylabel("Frequency")
plt.title("Distribution of Model Uncertainty (Entropy)")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=results_df[emotion_classes])
plt.xlabel("Emotion Classes")
plt.ylabel("Predicted Probability")
plt.title("Distribution of Predicted Probabilities Across Classes")
plt.show()
