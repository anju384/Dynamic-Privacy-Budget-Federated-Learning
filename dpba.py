import numpy as np

def compute_sensitivity(client_loader):
    counts = {0: 0, 1: 0}

    for _, target in client_loader:
        for t in target:
            counts[t.item()] += 1

    total = sum(counts.values())
    weights = {0: 0.2, 1: 0.8}

    sensitivity = (weights[0]*counts[0] + weights[1]*counts[1]) / total
    return sensitivity


def compute_dynamic_epsilon(sens_list, total_eps=5.0):
    weights = [s + 1e-8 for s in sens_list]
    total = sum(weights)

    eps = [(w / total) * total_eps for w in weights]
    return eps