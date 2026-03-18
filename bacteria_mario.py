import copy
import os
import random
import time

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
SEQUENCE_LENGTH = 1000  # Number of moves in each individual's genome
MUTATION_RATE = 0.7  # Probability of mutating each gene
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

    def copy(self):
        """Create a deep copy of this individual"""
        new_individual = MarioIndividual(self.genome)
        new_individual.fitness = self.fitness
        new_individual.max_x = self.max_x
        new_individual.steps_survived = self.steps_survived
        return new_individual


def evaluate_fitness(individual, render=False):
    """Evaluate how far Mario gets with this individual's action sequence"""

    if render:
        print("\n🎮 Attempting to open Mario game window...")
        print("📌 Tips:")
        print("   - The window might open behind your terminal")
        print("   - Check your taskbar/dock for a new window")
        print("   - Make sure pygame is installed: pip install pygame")
        print("   - Press Ctrl+C in this terminal to stop\n")
        time.sleep(3)

    # Try different rendering approaches
    try:
        # Approach 1: Try with render_mode="human" (newer gym API)
        if render:
            print("Method 1: Trying render_mode='human'...")
            env = gym_super_mario_bros.make(
                "SuperMarioBros-1-1-v0", render_mode="human"
            )
        else:
            env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    except TypeError:
        # Approach 2: Older gym API without render_mode
        if render:
            print("Method 2: Using older API with render() method...")
            env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        else:
            env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return 0

    # Apply action space
    env = JoypadSpace(env, MOVEMENTS)

    # For rendering, try to call render() explicitly if needed
    if render:
        try:
            # Some environments need an explicit render call
            env.render()
            print("✓ Render method called successfully")
        except:
            print("⚠️  Could not call render() explicitly, but continuing...")

    obs = env.reset()
    done = False
    step = 0
    total_reward = 0
    max_x = 0

    if render:
        print(f"🎮 Playing Mario with {len(individual.genome)} move sequence...")
        print("Watch the game window! (Press Ctrl+C in terminal to stop)\n")

    try:
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

            # Render if needed (for older API)
            if render:
                try:
                    env.render()
                except:
                    pass  # Already rendering via render_mode
                time.sleep(0.05)  # Control game speed

                # Print progress occasionally
                if step % 30 == 0:
                    print(f"  Step {step}: Position {x_pos}")

    except KeyboardInterrupt:
        if render:
            print("\n👋 Game interrupted by user")
    finally:
        env.close()

        if render:
            print(f"\n📊 Game Over - Steps: {step}, Max Position: {max_x}")
            time.sleep(2)

    # Only update the individual's stats if we're not rendering
    if not render:
        individual.fitness = max_x + step * 0.1
        individual.max_x = max_x
        individual.steps_survived = step

    return max_x + step * 0.1


def create_initial_population(size):
    """Create a random initial population"""
    return [MarioIndividual() for _ in range(size)]


def select_survivors(population, elite_size):
    """Select the top elite_size individuals based on fitness"""
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    return sorted_pop[:elite_size]


def create_next_generation(survivors, population_size):
    """Create next generation from survivors through mutation and crossover"""
    next_generation = [s.copy() for s in survivors]

    while len(next_generation) < population_size:
        parent1 = random.choice(survivors)
        parent2 = random.choice(survivors)
        child = parent1.crossover(parent2)
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


def test_rendering():
    """Test function to verify rendering works"""
    print("\n🔧 Testing rendering system...")
    print("Creating a simple test environment...")

    try:
        # Try to create a test environment with rendering
        test_env = gym_super_mario_bros.make(
            "SuperMarioBros-1-1-v0", render_mode="human"
        )
        test_env = JoypadSpace(test_env, MOVEMENTS)
        test_env.reset()
        test_env.render()
        print("✓ Test environment created successfully")
        print("✓ Render method called")
        print("If you don't see a window, check your taskbar/dock")
        time.sleep(3)
        test_env.close()
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def run_genetic_algorithm(generations=100):
    """Main genetic algorithm loop"""
    print("=" * 60)
    print("SUPER MARIO BROS - GENETIC PROGRAMMING")
    print("=" * 60)

    # Test rendering first
    print("\nFirst, let's test if rendering works:")
    rendering_works = test_rendering()
    if not rendering_works:
        print("\n⚠️  Rendering test failed. You may need to:")
        print("   - Install pygame: pip install pygame")
        print("   - Check your display settings")
        print("   - Try running in a different environment")

    print(f"\nPopulation Size: {POPULATION_SIZE}")
    print(f"Genome Length: {SEQUENCE_LENGTH} moves")
    print(f"Mutation Rate: {MUTATION_RATE * 100}%")
    print(f"Generations: {generations}")
    print(f"\nStarting evolution...")
    print("=" * 60)

    population = create_initial_population(POPULATION_SIZE)
    best_fitness_history = []
    best_individual_history = []

    for generation in range(generations):
        print(f"\n--- Generation {generation + 1} ---")

        fitnesses = []
        for i, individual in enumerate(population):
            fitness = evaluate_fitness(individual, render=False)
            fitnesses.append(fitness)

            if (i + 1) % 20 == 0:
                print(f"  Evaluated {i + 1}/{POPULATION_SIZE} individuals...")

        best_idx = np.argmax(fitnesses)
        best_individual = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]
        best_fitness_history.append(best_fitness)
        best_individual_history.append(best_individual)

        avg_fitness = np.mean(fitnesses)
        print(f"\n  Generation {generation + 1} Results:")
        print(f"    Best Fitness: {best_fitness:.2f}")
        print(f"    Avg Fitness: {avg_fitness:.2f}")
        print(f"    Best Distance: {best_individual.max_x}")

        survivors = select_survivors(population, ELITE_SIZE)
        print(f"    Survivors: {len(survivors)}")

        population = create_next_generation(survivors, POPULATION_SIZE)

        if (generation + 1) % 10 == 0:
            print_best_individual(best_individual, generation + 1)

            if input(f"\nShow best individual playing? (y/n): ").lower() == "y":
                render_individual = best_individual.copy()
                evaluate_fitness(render_individual, render=True)

    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE - FINAL RESULTS")
    print("=" * 60)

    final_best = max(best_individual_history, key=lambda x: x.fitness)
    print_best_individual(final_best, "FINAL")

    print(f"\nFitness History:")
    for i, fitness in enumerate(best_fitness_history):
        print(f"  Gen {i + 1}: {fitness:.2f}")

    print(f"\n{'=' * 60}")
    print("PLAYING FINAL BEST INDIVIDUAL")
    print("=" * 60)
    print("\n⚠️  The game window should open now. If it doesn't:")
    print("   1. Check your taskbar/dock for a new window")
    print("   2. Make sure pygame is installed")
    print("   3. Try running with pythonw instead of python")
    print("   4. Check if you're on a headless server (no display)")
    print("\nPress Enter to continue...")
    input()

    render_individual = final_best.copy()
    evaluate_fitness(render_individual, render=True)

    return final_best


if __name__ == "__main__":
    # Run the genetic algorithm
    best_mario = run_genetic_algorithm(generations=50)

    print("\nSimulations Complete, chud")
