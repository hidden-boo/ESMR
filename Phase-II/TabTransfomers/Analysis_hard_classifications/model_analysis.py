from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve


# Load best model
model.load_state_dict(torch.load("Phase-II\TabTransfomers\best_tabtransformer_model.pth"))
model.eval()

# Predict validation set
y_true, y_pred = [], []
with torch.no_grad():
    for x_num, x_cat, y in val_loader:
        logits = model(x_num.to(device), x_cat.to(device))
        predictions = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(predictions)
        y_true.extend(y.numpy())

# Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["happy", "excited", "stressed", "frustrated", "disappointed", "churned"])
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.show()


# Plot Losses
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

# Plot Accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Get probabilities
y_probs = []
with torch.no_grad():
    for x_num, x_cat, _ in val_loader:
        logits = model(x_num.to(device), x_cat.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        y_probs.append(probabilities)
y_probs = np.vstack(y_probs)

# Binarize true labels
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5])
n_classes = y_true_bin.shape[1]

# Plot ROC curves
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves per Class')
plt.legend(loc='best')
plt.show()

# Plot Precision-Recall curves
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
    avg_precision = average_precision_score(y_true_bin[:, i], y_probs[:, i])
    plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {avg_precision:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves per Class')
plt.legend(loc='best')
plt.show()

plt.figure(figsize=(10, 6))
for i in range(n_classes):
    prob_true, prob_pred = calibration_curve(y_true_bin[:, i], y_probs[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i}')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Probability')
plt.title('Calibration Curves')
plt.legend(loc='best')
plt.show()
