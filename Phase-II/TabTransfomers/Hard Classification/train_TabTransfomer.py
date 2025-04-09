from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0

    for x_num, x_cat, y in tqdm(loader, desc="Training", leave=False):
        x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        print(logits)
        print(logits.shape)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for x_num, x_cat, y in loader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            logits = model(x_num, x_cat)
            loss = criterion(logits, y)

            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)
