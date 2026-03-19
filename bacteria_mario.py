import copy
import os
import pickle
import random
import time
from datetime import datetime

import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace

# Original movement combinations (13 actions)
BASE_MOVEMENTS = [
    ["NOOP"],  # 0: Do nothing
    ["right"],  # 1: Walk right
    ["right", "A"],  # 2: Walk right + jump
    ["right", "B"],  # 3: Walk right + run
    ["right", "A", "B"],  # 4: Walk right + jump + run
    ["down"],  # 9: Crouch
    ["up"],  # 10: Look up
    ["A"],  # 11: Jump only
    ["B"],  # 12: Run only
]

# Jump durations (frames to hold A button)
JUMP_DURATIONS = [1, 3, 6, 10, 15, 20, 25, 30]
NUM_BASE_ACTIONS = len(BASE_MOVEMENTS)

# Create expanded action space
MOVEMENTS = []
for duration in JUMP_DURATIONS:
    for base_action in BASE_MOVEMENTS:
        MOVEMENTS.append(
            {
                "base": base_action.copy(),
                "jump_duration": duration if any("A" in a for a in base_action) else 0,
            }
        )

print(f"Created {len(MOVEMENTS)} actions with variable jump heights")

# Genetic Algorithm Parameters
POPULATION_SIZE = 20
SEQUENCE_LENGTH = 300
MUTATION_RATE = 0.3
ELITE_SIZE = 10

class MarioIndividual:
    def __init__(self, genome=None):
        if genome is None:
            self.genome = [
                random.randint(0, len(MOVEMENTS) - 1) for _ in range(SEQUENCE_LENGTH)
            ]
        else:
            self.genome = genome.copy()
        self.fitness = 0
        self.max_x = 0
        self.steps_survived = 0
        self.total_frames = 0
        self.generation = 0
        self.id = random.randint(10000, 99999)

    def mutate(self, mutation_rate=MUTATION_RATE):
        """Mutate with bias against backward moves"""
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                # Decide whether to allow backward moves
                self.genome[i] = random.randint(0, len(MOVEMENTS) - 1)
        return self

    def crossover(self, other):
        point = random.randint(0, len(self.genome))
        child_genome = self.genome[:point] + other.genome[point:]
        return MarioIndividual(child_genome)

    def copy(self):
        new_individual = MarioIndividual(self.genome)
        new_individual.fitness = self.fitness
        new_individual.max_x = self.max_x
        new_individual.steps_survived = self.steps_survived
        new_individual.total_frames = self.total_frames
        new_individual.generation = self.generation
        new_individual.id = self.id
        return new_individual

    def save(self, filename):
        """Save individual to file"""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"  Saved individual to {filename}")


def evaluate_fitness(individual, render=False, record=False):
    """Evaluate how far Mario gets with this individual's action sequence"""
    if render:
        print("\n🎮 Opening Mario game window...")
        time.sleep(2)

    try:
        if render:
            env = gym_super_mario_bros.make(
                "SuperMarioBros-1-1-v0", render_mode="human"
            )
        else:
            env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    except TypeError:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

    env = JoypadSpace(env, BASE_MOVEMENTS)

    obs = env.reset()
    done = False
    action_index = 0
    total_frames = 0
    max_x = 0

    # For recording
    frames = [] if record else None

    if render:
        try:
            env.render()
        except:
            pass

    try:
        while not done and action_index < len(individual.genome):
            encoded_action = individual.genome[action_index]
            action_info = MOVEMENTS[encoded_action]

            base_action = action_info["base"]
            jump_duration = action_info["jump_duration"]
            base_idx = BASE_MOVEMENTS.index(base_action)

            if jump_duration > 0:
                if render and jump_duration > 0:
                    jump_height = "LOW"
                    if jump_duration > 20:
                        jump_height = "VERY HIGH"
                    elif jump_duration > 12:
                        jump_height = "HIGH"
                    elif jump_duration > 6:
                        jump_height = "MEDIUM"
                    print(
                        f"\n  Action {action_index}: {base_action} - {jump_height} JUMP"
                    )

                for frame in range(jump_duration):
                    obs, reward, done, info = env.step(base_idx)
                    total_frames += 1

                    if record:
                        frames.append(obs.copy())

                    x_pos = info.get("x_pos", 0)
                    if isinstance(x_pos, list):
                        x_pos = x_pos[0] if x_pos else 0
                    if x_pos > max_x:
                        max_x = x_pos

                    if render:
                        env.render()
                        time.sleep(0.02)

                    if done:
                        break
            else:
                if render:
                    print(f"\n  Action {action_index}: {base_action}")

                obs, reward, done, info = env.step(base_idx)
                total_frames += 1

                if record:
                    frames.append(obs.copy())

                x_pos = info.get("x_pos", 0)
                if isinstance(x_pos, list):
                    x_pos = x_pos[0] if x_pos else 0
                if x_pos > max_x:
                    max_x = x_pos

                if render:
                    env.render()
                    time.sleep(0.02)

            action_index += 1

            if render and action_index % 10 == 0:
                print(f"  Progress: {action_index}/{len(individual.genome)}")

    except KeyboardInterrupt:
        if render:
            print("\n Game interrupted by user")
    finally:
        env.close()

        if render:
            print(f"\nGame Over - Position: {max_x}")

    # Update stats only in non-render mode
    if not render:
        individual.fitness = max_x + action_index * 0.1
        individual.max_x = max_x
        individual.steps_survived = action_index
        individual.total_frames = total_frames

    return max_x + action_index * 0.1, frames


def create_initial_population(size):
    return [MarioIndividual() for _ in range(size)]


def select_survivors(population, elite_size):
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    return sorted_pop[:elite_size]


def create_next_generation(survivors, population_size, current_gen):
    next_generation = []

    # Keep elites
    for s in survivors:
        elite_copy = s.copy()
        elite_copy.generation = current_gen
        next_generation.append(elite_copy)

    # Fill the rest with offspring
    while len(next_generation) < population_size:
        parent1 = random.choice(survivors)
        parent2 = random.choice(survivors)
        child = parent1.crossover(parent2)
        child.mutate()
        child.generation = current_gen
        child.id = random.randint(10000, 99999)
        next_generation.append(child)

    return next_generation


def save_best_individual(individual, generation, fitness_history):
    """Save the best individual and its stats"""
    # Create saves directory if it doesn't exist
    os.makedirs("saves", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the individual
    filename = f"saves/best_gen_{generation}_{timestamp}.pkl"
    individual.save(filename)

    # Save stats
    stats_filename = f"saves/best_gen_{generation}_{timestamp}_stats.txt"
    with open(stats_filename, "w") as f:
        f.write(f"Generation: {generation}\n")
        f.write(f"Fitness: {individual.fitness:.2f}\n")
        f.write(f"Distance: {individual.max_x}\n")
        f.write(f"Actions Taken: {individual.steps_survived}\n")
        f.write(f"Total Frames: {individual.total_frames}\n")
        f.write(f"\nFitness History:\n")
        for i, fit in enumerate(fitness_history):
            f.write(f"  Gen {i + 1}: {fit:.2f}\n")

    print(f"  Saved stats to {stats_filename}")
    return filename


def print_best_individual(individual, generation):
    """Print details of the best individual"""
    action_counts = {}
    jump_heights = []
    backward_count = 0

    for action_idx in individual.genome[: min(50, individual.steps_survived)]:
        action_info = MOVEMENTS[action_idx]
        base_action = "+".join(action_info["base"]) if action_info["base"] else "NOOP"
        jump_duration = action_info["jump_duration"]

        action_key = f"{base_action}" + (
            f" (jump:{jump_duration})" if jump_duration > 0 else ""
        )
        action_counts[action_key] = action_counts.get(action_key, 0) + 1

        if jump_duration > 0:
            jump_heights.append(jump_duration)

        # Count backward moves
        if base_action and base_action.startswith("left"):
            backward_count += 1

    print(f"\n{'=' * 60}")
    print(f"GENERATION {generation} - BEST INDIVIDUAL (ID: {individual.id})")
    print(f"{'=' * 60}")
    print(f"Fitness: {individual.fitness:.2f}")
    print(f"Distance: {individual.max_x} units")
    print(f"Actions Taken: {individual.steps_survived}")
    print(f"Total Game Frames: {individual.total_frames}")
    print(f"Backward Moves (first 50): {backward_count}")

    if jump_heights:
        avg_jump = sum(jump_heights) / len(jump_heights)
        max_jump = max(jump_heights)
        min_jump = min(jump_heights)
        print(f"\nJump Analysis:")
        print(f"  Average Jump Duration: {avg_jump:.1f} frames")
        print(f"  Maximum Jump Duration: {max_jump} frames")
        print(f"  Minimum Jump Duration: {min_jump} frames")
        print(f"  Total Jumps: {len(jump_heights)}")

    print(f"\nTop 5 Actions Used in First 50 Steps:")
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for action, count in sorted_actions:
        print(f"  {action}: {count} times")

    # Show first 20 actions
    print(f"\nFirst 20 Actions: ", end="")
    for i in range(min(20, individual.steps_survived)):
        action_idx = individual.genome[i]
        action_info = MOVEMENTS[action_idx]
        base_action = action_info["base"]
        jump_duration = action_info["jump_duration"]

        if jump_duration > 0:
            if jump_duration > 20:
                print("V", end="")
            elif jump_duration > 12:
                print("H", end="")
            elif jump_duration > 6:
                print("M", end="")
            else:
                print("L", end="")
        elif base_action == ["right"]:
            print(">", end="")
        elif base_action == ["left"]:
            print("<", end="")
        elif base_action == ["NOOP"]:
            print(".", end="")
        elif base_action == ["A"]:
            print("^", end="")
        elif base_action == ["B"]:
            print("R", end="")
        elif base_action == ["right", "B"]:
            print("+", end="")
        elif base_action == ["right", "A", "B"]:
            print("#", end="")
        else:
            print("?", end="")
    print()


def run_genetic_algorithm(generations=50):
    """Main genetic algorithm loop"""
    print("=" * 60)
    print("SUPER MARIO BROS - GENETIC PROGRAMMING")
    print("WITH BIASED MUTATION AGAINST BACKWARD MOVES")
    print("=" * 60)

    print(f"\nPopulation Size: {POPULATION_SIZE}")
    print(f"Genome Length: {SEQUENCE_LENGTH} actions")
    print(f"Mutation Rate: {MUTATION_RATE * 100}%")
    print(f"Jump Durations: {JUMP_DURATIONS} frames")
    print(f"Total Actions: {len(MOVEMENTS)}")
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
            fitness, _ = evaluate_fitness(individual, render=False)
            fitnesses.append(fitness)

            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i + 1}/{POPULATION_SIZE} individuals...")

        best_idx = np.argmax(fitnesses)
        best_individual = population[best_idx].copy()
        best_individual.generation = generation + 1
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

        population = create_next_generation(survivors, POPULATION_SIZE, generation + 2)

        if (generation + 1) % 10 == 0:
            print_best_individual(best_individual, generation + 1)

            # Auto-save every 10 generations
            save_best_individual(best_individual, generation + 1, best_fitness_history)

            if generations - generation <= 10:
                response = input(f"\nShow best individual playing? (y/n): ").lower()
                if response == "y":
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

    # Save final best
    saved_file = save_best_individual(final_best, "FINAL", best_fitness_history)

    print(f"\n{'=' * 60}")
    print("PLAYING AND RECORDING FINAL BEST INDIVIDUAL")
    print("=" * 60)
    print("\nPress Enter to watch and record the final evolved Mario...")
    input()

    render_individual = final_best.copy()
    fitness, frames = evaluate_fitness(render_individual, render=True, record=True)

    # Save frames if we want to create a video later
    if frames:
        frames_file = (
            f"saves/final_best_frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        with open(frames_file, "wb") as f:
            pickle.dump(frames, f)
        print(f"\nSaved {len(frames)} frames to {frames_file}")
        print("You can use these frames to create a video with OpenCV later.")

    print(f"\nBest individual saved to: {saved_file}")
    print("\nSimulations Complete")

    return final_best


def load_best_individual(filename):
    """Load a saved individual"""
    with open(filename, "rb") as f:
        individual = pickle.load(f)
    print(f"Loaded individual from {filename}")
    print(f"  Generation: {individual.generation}")
    print(f"  Fitness: {individual.fitness:.2f}")
    print(f"  Distance: {individual.max_x}")
    return individual


if __name__ == "__main__":
    # Run the genetic algorithm
    best_mario = run_genetic_algorithm(generations=20)
