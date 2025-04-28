import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx

# 1. Define the neurons and their properties (excitatory vs inhibitory)
neurons = [
    {"voltage": 0, "connections": [], "type": "excitatory", "last_spike": -1},
    {"voltage": 0, "connections": [], "type": "inhibitory", "last_spike": -1},
    {"voltage": 0, "connections": [], "type": "excitatory", "last_spike": -1},
    {"voltage": 0, "connections": [], "type": "inhibitory", "last_spike": -1},
    {"voltage": 0, "connections": [], "type": "excitatory", "last_spike": -1},
    {"voltage": 0, "connections": [], "type": "inhibitory", "last_spike": -1},
    {"voltage": 0, "connections": [], "type": "excitatory", "last_spike": -1},
    {"voltage": 0, "connections": [], "type": "inhibitory", "last_spike": -1},
    {"voltage": 0, "connections": [], "type": "excitatory", "last_spike": -1},
    {"voltage": 0, "connections": [], "type": "inhibitory", "last_spike": -1}
]

# 2. Connect neurons randomly with weights
def connect_neurons():
    for i in range(len(neurons)):
        for j in range(i + 1, len(neurons)):
            weight = random.uniform(0.1, 0.5)
            neurons[i]["connections"].append({"target": j, "weight": weight})
            neurons[j]["connections"].append({"target": i, "weight": weight})

connect_neurons()

# 3. Simulate neuron firing and spike timing
def fire(neuron, current_time):
    if neuron["voltage"] > 0.5:  # Threshold for firing
        print(f"Neuron fired! Voltage: {neuron['voltage']}")
        neuron["voltage"] = 0  # Reset voltage after firing
        neuron["last_spike"] = current_time  # Store the last spike time
        return True
    return False

# 4. Propagate signals and simulate excitatory/inhibitory effects
def propagate(source_idx, current_time):
    source = neurons[source_idx]
    if fire(source, current_time):
        for conn in source["connections"]:
            target = neurons[conn["target"]]
            weight = conn["weight"]
            if neurons[conn["target"]]["type"] == "inhibitory":
                weight = -weight  # Inhibitory effect
            target["voltage"] += weight
            print(f"Signal sent to Neuron {conn['target']} with weight {weight}")

# 5. Hebbian Learning Rule
def hebbian_learning(pre_idx, post_idx, current_time):
    pre_neuron = neurons[pre_idx]
    post_neuron = neurons[post_idx]
    if fire(pre_neuron, current_time) and fire(post_neuron, current_time):
        for conn in pre_neuron["connections"]:
            if conn["target"] == post_idx:
                conn["weight"] += 0.05  # Hebbian learning rule
                print(f"Hebbian Learning: Strengthened {pre_idx}→{post_idx} to {conn['weight']}")

# 6. Spike-Timing Dependent Plasticity (STDP)
def stdp(pre_idx, post_idx, current_time):
    pre_neuron = neurons[pre_idx]
    post_neuron = neurons[post_idx]
    time_diff = pre_neuron["last_spike"] - post_neuron["last_spike"]
    for conn in pre_neuron["connections"]:
        if conn["target"] == post_idx:
            if time_diff > 0:  # LTP: Pre-fires before post
                conn["weight"] += 0.1
            elif time_diff < 0:  # LTD: Post-fires before pre
                conn["weight"] -= 0.1
            print(f"STDP: Updated {pre_idx}→{post_idx} weight to {conn['weight']}")

# 7. Reward-based Learning (Dopamine Effect)
def reward_based_learning(pre_idx, post_idx, intensity):
    reward = intensity * 0.1  # Dopamine release based on firing intensity
    for conn in neurons[pre_idx]["connections"]:
        if conn["target"] == post_idx:
            conn["weight"] += reward  # Apply reward to synapse
            print(f"Reward: Increased weight between {pre_idx}→{post_idx} to {conn['weight']}")

# 8. Long-Term Potentiation (LTP) and Long-Term Depression (LTD)
def ltp_ltd(pre_idx, post_idx, current_time):
    time_diff = abs(neurons[pre_idx]["last_spike"] - neurons[post_idx]["last_spike"])
    for conn in neurons[pre_idx]["connections"]:
        if conn["target"] == post_idx:
            if time_diff < 10:  # LTP condition (spike timing)
                conn["weight"] += 0.2  # Strengthen synapse
                print(f"LTP: Strengthened {pre_idx}→{post_idx} to {conn['weight']}")
            elif time_diff > 20:  # LTD condition
                conn["weight"] -= 0.2  # Weaken synapse
                print(f"LTD: Weakened {pre_idx}→{post_idx} to {conn['weight']}")

# 9. Memory System (Working Memory + Long-Term Memory)
memory = []

def store_in_memory(neuron_idx):
    memory.append(neurons[neuron_idx]["voltage"])
    print(f"Memory: Stored voltage of Neuron {neuron_idx} as {neurons[neuron_idx]['voltage']}")

# 10. Sleep Phase (Reset neurons)
def sleep():
    print("Neurons are sleeping... Consolidating learning")
    for neuron in neurons:
        neuron["voltage"] = 0  # Reset voltage during sleep

# 11. Simulate the learning cycle
def simulate_learning_cycle(current_time):
    for i in range(len(neurons)):
        # Randomly excite neurons for learning
        neurons[i]["voltage"] = random.uniform(0.4, 0.6)
        propagate(i, current_time)

    # Apply Hebbian learning and STDP
    hebbian_learning(0, 1, current_time)
    stdp(1, 2, current_time)

    # Apply reward-based learning based on firing intensity
    reward_based_learning(0, 1, neurons[0]["voltage"])

    # Apply LTP and LTD
    ltp_ltd(0, 2, current_time)

    # Store in memory
    store_in_memory(random.randint(0, len(neurons)-1))

    # Sleep phase after 5 cycles
    if current_time % 5 == 0:
        sleep()

# 12. Plotting the weights after a few learning cycles
def plot_weights():
    weights = [conn["weight"] for conn in neurons[0]["connections"]]
    plt.bar(range(len(weights)), weights)
    plt.title("Connection Weights from Neuron 0")
    plt.savefig("weights.png")
    print("Saved plot to weights.png")

# 13. Visualize the network with real-time updates
def visualize_network():
    G = nx.DiGraph()  # Directed graph to show the direction of signal flow

    # Add nodes for each neuron
    for idx, neuron in enumerate(neurons):
        G.add_node(idx, type=neuron["type"])

    # Add edges for the connections
    for i, neuron in enumerate(neurons):
        for conn in neuron["connections"]:
            G.add_edge(i, conn["target"], weight=conn["weight"])

    # Visualization of the network
    pos = nx.spring_layout(G, seed=42)  # Positioning of nodes (force-directed layout)
    plt.figure(figsize=(10, 10))

    # Draw nodes
    node_colors = ['lightblue' if neurons[i]["type"] == 'excitatory' else 'salmon' for i in range(len(neurons))]
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, alpha=0.7)

    # Draw edges with varying thickness based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=[weight * 10 for weight in edge_weights], alpha=0.5, edge_color='gray')

    # Draw labels for neurons
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

    # Display weights on edges
    edge_labels = {(u, v): f"{round(G[u][v]['weight'], 2)}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # Title
    plt.title("Spiking Neural Network Visualization")
    plt.axis("off")
    plt.show()

# Running the simulation and visualization
if __name__ == "__main__":
    for t in range(10):
        simulate_learning_cycle(t)
        visualize_network()
        plot_weights()
