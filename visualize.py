import matplotlib.pyplot as plt
import networkx as nx
import neat

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False):
    """Draws the neural network using matplotlib and networkx."""
    from neat.graphs import feed_forward_layers

    if prune_unused:
        connections = [cg.key for cg in genome.connections.values() if cg.enabled or show_disabled]
        used_nodes = set()
        for connection in connections:
            used_nodes.add(connection[0])
            used_nodes.add(connection[1])
    else:
        used_nodes = set(genome.nodes.keys())

    layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, genome.connections)
    G = nx.DiGraph()
    for layer in layers:
        for node in layer:
            if node in used_nodes:
                G.add_node(node)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            if cg.key[0] in used_nodes and cg.key[1] in used_nodes:
                G.add_edge(cg.key[0], cg.key[1], weight=cg.weight)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    if filename:
        plt.savefig(filename)
    if view:
        plt.show()
    plt.close()

def plot_stats(statistics, ylog=False, view=False, filename=None):
    """Plot the statistics."""
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]

    plt.figure()
    plt.plot(generation, best_fitness, label="Best")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Fitness over Generations")
    plt.grid()
    plt.legend(loc="best")
    if filename:
        plt.savefig(filename)
    if view:
        plt.show()
    plt.close()

def plot_species(statistics, view=False, filename=None):
    """Visualize species diversity."""
    num_generations = len(statistics.most_fit_genomes)
    species_sizes = statistics.get_species_sizes()

    plt.figure()
    for species_id, sizes in species_sizes.items():
        plt.plot(range(num_generations), sizes, label=f"Species {species_id}")
    plt.xlabel("Generation")
    plt.ylabel("Number of Individuals")
    plt.title("Species Sizes")
    plt.grid()
    plt.legend(loc="best")
    if filename:
        plt.savefig(filename)
    if view:
        plt.show()
    plt.close()
