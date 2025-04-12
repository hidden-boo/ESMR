import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your trained model
model = TabTransformer(
    num_numeric=len(numeric_cols),
    cat_cardinalities=cat_cardinalities,
    embed_dim=32,
    num_heads=4,
    num_layers=2,
    dropout=0.2,
    num_classes=len(np.unique(y_train))
).to(device)

model.load_state_dict(torch.load("Phase-II\TabTransfomers\best_tabtransformer_model.pth"))
model.eval()

# Probability Prediction Function
def predict_probabilities(model, data_loader, device, temperature=3.0):
    all_probs = []
    all_true_labels = []

    with torch.no_grad():
        for x_num, x_cat, y_true in data_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            logits = model(x_num, x_cat)

            # Apply temperature scaling
            scaled_logits = logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=1)

            all_probs.append(probabilities.cpu().numpy())
            all_true_labels.extend(y_true.numpy())

    all_probs = np.vstack(all_probs)
    return all_probs, np.array(all_true_labels)


# Compute probabilities on validation data
val_probs, val_true_labels = predict_probabilities(model, val_loader, device)

# Convert to DataFrame for easy interpretation
emotion_classes = ["happy", "excited", "stressed", "frustrated", "disappointed", "churned"]
results_df = pd.DataFrame(val_probs, columns=emotion_classes[:val_probs.shape[1]])
results_df["true_label"] = val_true_labels

print(results_df.head())
