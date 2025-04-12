import torch
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X_numeric, X_categorical, y):
        self.X_numeric = torch.tensor(X_numeric.values, dtype=torch.float32)
        self.X_categorical = torch.tensor(X_categorical.values, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_numeric[idx], self.X_categorical[idx], self.y[idx]

# Prepare numeric and categorical features
categorical_cols = ['top_theme', 'top_category', 'top_video_emotion', 'churned']
numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

train_dataset = TabularDataset(X_train[numeric_cols], X_train[categorical_cols], y_train)
val_dataset = TabularDataset(X_val[numeric_cols], X_val[categorical_cols], y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Get cardinalities for embeddings
cat_cardinalities = [X_train[col].nunique() + 1 for col in categorical_cols]
