import torch
import torch.nn.functional as F

def calculate_mia_score(model, train_loader, test_loader, device='cpu'):
    model.eval()

    def get_loss_dist(loader):
        losses = []
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.cross_entropy(output, target, reduction='none')
                losses.extend(loss.tolist())
        return torch.tensor(losses)

    train_losses = get_loss_dist(train_loader)
    test_losses = get_loss_dist(test_loader)

    threshold = test_losses.median()

    hits = (train_losses < threshold).sum().item()
    return hits / len(train_losses)