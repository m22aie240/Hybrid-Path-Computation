import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# Centralized Path Computation
def compute_optimal_path(G, source, destination):
    try:
        path = nx.shortest_path(G, source, destination)
        cost = nx.shortest_path_length(G, source, destination)
        return cost, path
    except nx.NetworkXNoPath:
        return float('inf'), []

# Distributed Path Computation
def compute_local_path(G, node, destination, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)

    if node == destination:
        return 0, [node]

    min_cost = float('inf')
    min_path = None
    for neighbor in G.neighbors(node):
        if neighbor not in visited:
            if neighbor == destination:
                return G[node][neighbor]['weight'], [node, neighbor]
            cost, path = compute_local_path(G, neighbor, destination, visited)
            if cost + G[node][neighbor]['weight'] < min_cost:
                min_cost = cost + G[node][neighbor]['weight']
                min_path = [node] + path

    visited.remove(node)
    return min_cost, min_path

# Hybrid Path Computation
def compute_hybrid_path(G, source, destination):
    centralized_cost, centralized_path = compute_optimal_path(G, source, destination)
    distributed_cost, distributed_path = compute_local_path(G, source, destination)

    if centralized_cost <= distributed_cost:
        return centralized_cost, centralized_path
    else:
        return distributed_cost, distributed_path

# Generate Random Graph
def generate_random_graph(num_nodes):
    G = nx.gnm_random_graph(num_nodes, num_nodes * 2)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.randint(1, 10)
    return G

# Print Graph
def print_graph(G):
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

# Parameters
num_nodes = 30
test_scenarios = [(0, num_nodes - 1), (1, num_nodes - 2), (2, num_nodes - 3), (0, num_nodes - 3), (1, num_nodes -3), (2, num_nodes-1), (0, num_nodes-2)]

# Collect Data
centralized_costs = []
distributed_costs = []
hybrid_costs = []

for source, destination in test_scenarios:
    G = generate_random_graph(num_nodes)
    print(f"Graph for Test Scenario ({source}, {destination}):")
    print_graph(G)

    centralized_cost, _ = compute_optimal_path(G, source, destination)
    if centralized_cost != float('inf'):
        centralized_costs.append(centralized_cost)

    distributed_cost, _ = compute_local_path(G, source, destination)
    if distributed_cost != float('inf'):
        distributed_costs.append(distributed_cost)

    hybrid_cost, _ = compute_hybrid_path(G, source, destination)
    hybrid_costs.append(hybrid_cost)

# Plot Comparative Analysis
x = np.arange(len(test_scenarios))

fig, ax = plt.subplots()

if centralized_costs:
    ax.plot(x, centralized_costs, label='Centralized', marker='o')
if distributed_costs:
    ax.plot(x, distributed_costs, label='Distributed', marker='o')
if hybrid_costs:
    ax.plot(x, hybrid_costs, label='Hybrid', marker='o')

ax.set_xlabel('Test Scenarios')
ax.set_ylabel('Cost')
ax.set_title('Comparative Analysis of Path Computation Algorithms')
ax.set_xticks(x)
ax.set_xticklabels(test_scenarios)
ax.legend()

plt.show()
