import random

import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace

# Define ALL possible movement combinations
MOVEMENTS = [
    ["NOOP"],  # 0: No operation (do nothing)
    ["right"],  # 1: Walk right
    ["right", "A"],  # 2: Walk right + jump
    ["right", "B"],  # 3: Walk right + run/speed
    ["right", "A", "B"],  # 4: Walk right + jump + run
    ["left"],  # 5: Walk left
    ["left", "A"],  # 6: Walk left + jump
    ["left", "B"],  # 7: Walk left + run
    ["left", "A", "B"],  # 8: Walk left + jump + run
    ["down"],  # 9: Crouch
    ["up"],  # 10: Look up / enter pipes
    ["A"],  # 11: Jump only
    ["B"],  # 12: Run only
]

# Genetic Algorithm Parameters
POPULATION_SIZE = 20
SEQUENCE_LENGTH = (
    100  # Number of moves in each individual's genome, probably more than we need
)
MUTATION_RATE = 0.3  # Probability of mutating each gene
ELITE_SIZE = 10  # Number of top individuals to keep (kill bottom 50)


class MarioIndividual:
    def __init__(self, genome=None):
        if genome is None:
            # Create random genome of actions
            self.genome = [
                random.randint(0, len(MOVEMENTS) - 1) for _ in range(SEQUENCE_LENGTH)
            ]
        else:
            self.genome = genome.copy()
        self.fitness = 0
        self.max_x = 0
        self.steps_survived = 0

    def mutate(self, mutation_rate=MUTATION_RATE):
        """Mutate the genome by randomly changing actions"""
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                self.genome[i] = random.randint(0, len(MOVEMENTS) - 1)
        return self

    def crossover(self, other):
        """Perform single-point crossover with another individual"""
        point = random.randint(0, len(self.genome))
        child_genome = self.genome[:point] + other.genome[point:]
        return MarioIndividual(child_genome)


def evaluate_fitness(individual, render=False):
    """Evaluate how far Mario gets with this individual's action sequence"""
    # Create environment
    try:
        env = gym_super_mario_bros.make(
            "SuperMarioBros-1-1-v0", render_mode="human" if render else None
        )
    except TypeError:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

    env = JoypadSpace(env, MOVEMENTS)

    obs = env.reset()
    done = False
    step = 0
    total_reward = 0
    max_x = 0

    while not done and step < len(individual.genome):
        action = individual.genome[step]

        # Take step
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Track progress
        x_pos = info.get("x_pos", step)
        if isinstance(x_pos, list):
            x_pos = x_pos[0]

        if x_pos > max_x:
            max_x = x_pos

        step += 1

    env.close()

    # Fitness based on distance traveled and steps survived
    individual.fitness = max_x + step * 0.1  # Small bonus for each step
    individual.max_x = max_x
    individual.steps_survived = step

    return individual.fitness


def create_initial_population(size):
    """Create a random initial population"""
    return [MarioIndividual() for _ in range(size)]


def select_survivors(population, elite_size):
    """Select the top elite_size individuals based on fitness"""
    # Sort by fitness (highest first)
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    return sorted_pop[:elite_size]


def create_next_generation(survivors, population_size):
    """Create next generation from survivors through mutation and crossover"""
    next_generation = survivors.copy()  # Keep the elites

    # Fill the rest with offspring
    while len(next_generation) < population_size:
        # Select two random parents from survivors
        parent1 = random.choice(survivors)
        parent2 = random.choice(survivors)

        # Create child through crossover
        child = parent1.crossover(parent2)

        # Mutate the child
        child.mutate()

        next_generation.append(child)

    return next_generation


def print_best_individual(individual, generation):
    """Print details of the best individual"""
    action_counts = {}
    for action in individual.genome[: min(50, individual.steps_survived)]:
        action_name = "+".join(MOVEMENTS[action]) if MOVEMENTS[action] else "NOOP"
        action_counts[action_name] = action_counts.get(action_name, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"GENERATION {generation} - BEST INDIVIDUAL")
    print(f"{'=' * 60}")
    print(f"Fitness: {individual.fitness:.2f}")
    print(f"Distance: {individual.max_x} units")
    print(f"Steps Survived: {individual.steps_survived}")
    print(f"\nTop 5 Actions Used in First 50 Steps:")
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for action, count in sorted_actions:
        print(f"  {action}: {count} times")

    # Show first 20 actions as preview
    print(f"\nFirst 20 Actions: ", end="")
    for i in range(min(20, individual.steps_survived)):
        action = individual.genome[i]
        if MOVEMENTS[action] == ["right"]:
            print("→", end="")
        elif MOVEMENTS[action] == ["right", "A"]:
            print("↗", end="")
        elif MOVEMENTS[action] == ["right", "B"]:
            print("⇒", end="")
        elif MOVEMENTS[action] == ["right", "A", "B"]:
            print("⇗", end="")
        elif MOVEMENTS[action] == ["left"]:
            print("←", end="")
        elif MOVEMENTS[action] == ["A"]:
            print("↑", end="")
        elif MOVEMENTS[action] == ["NOOP"]:
            print("·", end="")
        else:
            print("?", end="")
    print()


def run_genetic_algorithm(generations=100):
    """Main genetic algorithm loop"""
    print("=" * 60)
    print("SUPER MARIO BROS - GENETIC PROGRAMMING")
    print("=" * 60)
    print(f"\nPopulation Size: {POPULATION_SIZE}")
    print(f"Genome Length: {SEQUENCE_LENGTH} moves")
    print(f"Mutation Rate: {MUTATION_RATE * 100}%")
    print(f"Generations: {generations}")
    print(f"\nStarting evolution...")
    print("=" * 60)

    # Initialize population
    population = create_initial_population(POPULATION_SIZE)
    best_fitness_history = []

    for generation in range(generations):
        print(f"\n--- Generation {generation + 1} ---")

        # Evaluate fitness for all individuals
        fitnesses = []
        for i, individual in enumerate(population):
            fitness = evaluate_fitness(individual, render=False)
            fitnesses.append(fitness)

            if (i + 1) % 20 == 0:
                print(f"  Evaluated {i + 1}/{POPULATION_SIZE} individuals...")

        # Find best individual
        best_idx = np.argmax(fitnesses)
        best_individual = population[best_idx]
        best_fitness = fitnesses[best_idx]
        best_fitness_history.append(best_fitness)

        # Print progress
        avg_fitness = np.mean(fitnesses)
        print(f"\n  Generation {generation + 1} Results:")
        print(f"    Best Fitness: {best_fitness:.2f}")
        print(f"    Avg Fitness: {avg_fitness:.2f}")
        print(f"    Best Distance: {best_individual.max_x}")

        # Select survivors (kill bottom 50)
        survivors = select_survivors(population, ELITE_SIZE)
        print(f"    Survivors: {len(survivors)}")

        # Create next generation
        population = create_next_generation(survivors, POPULATION_SIZE)

        # Every 10 generations, show the best individual in action
        if (generation + 1) % 10 == 0:
            print_best_individual(best_individual, generation + 1)

            # Optional: Show the best individual playing
            if input(f"\nShow best individual playing? (y/n): ").lower() == "y":
                evaluate_fitness(best_individual, render=True)

    # Final results
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE - FINAL RESULTS")
    print("=" * 60)

    # Re-evaluate best individual with rendering
    final_best = max(population, key=lambda x: x.fitness)
    print_best_individual(final_best, "FINAL")

    print(f"\nFitness History:")
    for i, fitness in enumerate(best_fitness_history):
        print(f"  Gen {i + 1}: {fitness:.2f}")

    print(f"\n{'=' * 60}")
    print("PLAYING FINAL BEST INDIVIDUAL")
    print("=" * 60)
    evaluate_fitness(final_best, render=True)

    return final_best


if __name__ == "__main__":
    # Run the genetic algorithm
    best_mario = run_genetic_algorithm(generations=20)

    print("\nSimulations Complete, chud")
