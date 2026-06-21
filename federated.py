import torch

def compute_importance(model):
    importance_dict = {}
    total_imp = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Interpretability: Gradient * Weight magnitude[cite: 1]
            imp = torch.abs(param.grad * param.data)
            importance_dict[name] = imp.mean().item()
            total_imp.append(imp.view(-1))
    
    if len(total_imp) == 0:
        return 0.0, importance_dict
    
    total_imp = torch.cat(total_imp)
    return total_imp.sum().item(), importance_dict

def compute_eps(sens, total_eps=5.0):
    # Proportional allocation based on sensitivity[cite: 1]
    weights = [s + 1e-8 for s in sens]
    total_weight = sum(weights)
    return [(w / total_weight) * total_eps for w in weights]