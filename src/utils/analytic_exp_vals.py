import numpy as np


def compute_analytic_expectation(beta, gamma, i, j, n, edge_weights, available_node='i'):
    weight_ij = edge_weights.get((i, j), edge_weights.get((j, i)))
    prod = 1
    if available_node == 'i':
        for k in range(n):
            if k != j and k != i:
                weight = edge_weights.get((i, k), edge_weights.get((k, i)))
                prod *= np.cos(np.arctan(weight) * gamma)
    elif available_node == 'j':
        for k in range(n):
            if k != i and k != j:
                weight = edge_weights.get((j, k), edge_weights.get((k, j)))
                prod *= np.cos(np.arctan(weight) * gamma)

    expectation = np.sin(beta * np.pi) * np.sin(np.arctan(weight_ij) * gamma) * prod

    return expectation
