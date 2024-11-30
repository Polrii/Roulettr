import os

import neat
import visualize

import random
import json


def eval_genomes(genomes, config):
    # Create a list of random numbers
    numbers = []
    for i in range(1000):
        numbers.append(random.randint(0, 36))
    
    # Evaluate the genomes
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Test the NN
        genome.fitness = 1000  # We establish the balance as the fitness
        lost_rounds = 0
        for answer in numbers:
            output = net.activate([lost_rounds])
            if 0 < answer < 19:
                genome.fitness += output[0]
                lost_rounds = 0
            else:
                genome.fitness -= output[0]
                lost_rounds += 1
                

def save_nn(winner_genome, config, folder="Checkpoints", filename="winning_nn.json"):
    """
    Save the neural network in a human-readable JSON format.
    """
    os.makedirs(folder, exist_ok=True)  # Ensure the folder exists

    # Extract weights and biases from the network
    layers = {
        "biases": {
            node_id: {
                "bias": node.bias,
                "activation": node.activation,
                "response": node.response,
            }
            for node_id, node in winner_genome.nodes.items()
        },
        "connections": [
            {
                "from_node": conn.key[0],
                "to_node": conn.key[1],
                "weight": conn.weight,
                "enabled": conn.enabled,
            }
            for conn in winner_genome.connections.values()
        ]
    }

    # Save to a JSON file
    save_path = os.path.join(folder, filename)
    with open(save_path, "w") as f:
        json.dump(layers, f, indent=4)

    print(f"Winning neural network saved to {save_path}")


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    prefix = os.path.join('Checkpoints', 'neat-checkpoint-')
    p.add_reporter(neat.Checkpointer(generation_interval=500, filename_prefix=prefix))

    # Run for up to x generations.
    winner = p.run(eval_genomes, 100000000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    
    # Create the winning neural network
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Save the winning neural network in a readable format
    save_nn(winner, config)

    node_names = {-1: 'Consecutive Losses', 0: 'Bet'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')  --> Continues training from checkpoint
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)