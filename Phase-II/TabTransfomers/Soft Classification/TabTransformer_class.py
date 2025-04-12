import torch
import torch.nn as nn
import torch.nn.functional as F

class TabTransformer(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, embed_dim=32, num_heads=4, num_layers=2, dropout=0.2, num_classes=3):
        super(TabTransformer, self).__init__()

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim) for cardinality in cat_cardinalities
        ])

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Linear layers for numeric features
        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim)
        )

        # Classification head
        total_embed_dim = embed_dim * (len(cat_cardinalities) + 1)
        self.classifier = nn.Sequential(
            nn.Linear(total_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_numeric, x_categorical):
        # Embed categorical features
        embed = [embedding(x_categorical[:, i]) for i, embedding in enumerate(self.embeddings)]
        embed = torch.stack(embed, dim=1)  # shape: (batch_size, num_cat, embed_dim)

        # Transformer Encoder (categorical embeddings)
        embed = self.transformer_encoder(embed)
        embed = embed.flatten(1)  # Flatten categorical embeddings

        # Numeric features projection
        numeric_embed = self.numeric_proj(x_numeric)

        # Combine numeric and categorical embeddings
        combined = torch.cat([embed, numeric_embed], dim=1)

        # Classification output
        logits = self.classifier(combined)
        return logits
