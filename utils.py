import torch

def train(model, loader, optimizer, epsilon, device):
    model.train()
    max_grad_norm = 1.0
    criterion = torch.nn.CrossEntropyLoss()

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()

        # Clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # 🔥 Stronger noise
        for param in model.parameters():
            if param.grad is not None:
                noise_std = (2 * max_grad_norm) / (epsilon + 1e-8)
                noise = torch.normal(0, noise_std, size=param.grad.shape).to(device)
                param.grad += noise

        optimizer.step()

    return model