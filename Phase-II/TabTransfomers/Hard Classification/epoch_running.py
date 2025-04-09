device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TabTransformer(
    num_numeric=len(numeric_cols),
    cat_cardinalities=cat_cardinalities,
    embed_dim=32,
    num_heads=4,
    num_layers=2,
    dropout=0.2,
    num_classes=len(np.unique(y_train))
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Store metrics per epoch
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

epochs = 30
best_val_acc = 0
patience = 5  # Number of epochs to wait before early stopping
early_stop_counter = 0  # Counter to track epochs without improvement


for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "Phase-II\TabTransfomers\best_tabtransformer_model.pth")
        print("  Model saved.")
        early_stop_counter = 0  # Reset counter upon improvement
    else:
        early_stop_counter += 1  # Increment counter if no improvement

    # Early stopping condition
    if early_stop_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break
