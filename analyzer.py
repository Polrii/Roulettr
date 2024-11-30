import neat
import visualize
import os


def analyze_checkpoint(checkpoint_file, config_file):
    # Load the NEAT configuration
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_file
    )

    # Restore the population from the checkpoint
    population = neat.Checkpointer.restore_checkpoint(checkpoint_file)

    # Find the best genome in the population
    best_genome = None
    best_fitness = float('-inf')
    for genome_id, genome in population.population.items():
        if genome.fitness is not None and genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome = genome

    if best_genome is None:
        print("No genome found with a valid fitness score.")
        return
    
    # Load the best nn
    best_net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    print(f"Best genome ID: {best_genome.key}")
    print(f"Best fitness: {best_genome.fitness}")

    # Visualize the best genome
    node_names = {-1: 'Consecutive Losses', 0: 'Bet'}
    visualize.draw_net(config, best_genome, view=True, node_names=node_names)
    visualize.draw_net(config, best_genome, view=True, node_names=node_names, prune_unused=True)


if __name__ == "__main__":
    # Specify the paths to the config file and checkpoint file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    checkpoint_file = 'Checkpoints/neat-checkpoint-2547'

    # Analyze the checkpoint
    analyze_checkpoint(checkpoint_file, config_path)
