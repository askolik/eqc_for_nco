
import numpy as np
import networkx as nx


def compute_tour_length(nodes, tour):
    """
    Compute length of a tour, including return to start node.
    (If start node is already added as last node in tour, 0 will be added to tour length.)
    :param nodes: all nodes in the graph in form of (x, y) coordinates
    :param tour: list of node indices denoting a (potentially partial) tour
    :return: tour length
    """
    tour_length = 0
    for i in range(len(tour)):
        if i < len(tour)-1:
            tour_length += np.linalg.norm(np.asarray(nodes[tour[i]]) - np.asarray(nodes[tour[i+1]]))
        else:
            tour_length += np.linalg.norm(np.asarray(nodes[tour[-1]]) - np.asarray(nodes[tour[0]]))

    return tour_length