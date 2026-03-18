import copy
import random
import time

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
    ["left"],  # 5: Walk left
    ["left", "A"],  # 6: Walk left + jump
    ["left", "B"],  # 7: Walk left + run
    ["left", "A", "B"],  # 8: Walk left + jump + run
    ["down"],  # 9: Crouch
    ["up"],  # 10: Look up
    ["A"],  # 11: Jump only
    ["B"],  # 12: Run only
]

# Jump durations (frames to hold A button) - this actually controls jump height
JUMP_DURATIONS = [1, 3, 6, 10, 15, 20, 25, 30]  # Different jump heights
NUM_BASE_ACTIONS = len(BASE_MOVEMENTS)

# Create expanded action space by encoding jump duration into action index
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
print(f"  Base actions: {NUM_BASE_ACTIONS}")
print(f"  Jump durations: {JUMP_DURATIONS}")

# Genetic Algorithm Parameters
POPULATION_SIZE = 20
SEQUENCE_LENGTH = 300  # Reduced since each action now takes multiple frames
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
        self.total_frames = 0  # Track actual game frames

    def mutate(self, mutation_rate=MUTATION_RATE):
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
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
        return new_individual


def evaluate_fitness(individual, render=False):
    """Evaluate how far Mario gets with this individual's action sequence"""
    # Create environment with rendering if requested
    if render:
        print("\n🎮 Opening Mario game window...")
        print("The window may open behind your terminal. Check your taskbar.")
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
    action_index = 0  # Track which action in the genome we're on
    total_frames = 0  # Track total game frames
    max_x = 0

    if render:
        try:
            env.render()
        except:
            pass

    try:
        while not done and action_index < len(individual.genome):
            # Get the encoded action
            encoded_action = individual.genome[action_index]
            action_info = MOVEMENTS[encoded_action]

            # Extract base action and find its index
            base_action = action_info["base"]
            jump_duration = action_info["jump_duration"]
            base_idx = BASE_MOVEMENTS.index(base_action)

            if jump_duration > 0:
                # THIS IS THE KEY - Hold the button for jump_duration frames
                jump_height = "LOW"
                if jump_duration > 20:
                    jump_height = "VERY HIGH"
                elif jump_duration > 12:
                    jump_height = "HIGH"
                elif jump_duration > 6:
                    jump_height = "MEDIUM"

                if render and jump_duration > 0:
                    print(
                        f"\n  Action {action_index}: {base_action} - {jump_height} JUMP (holding for {jump_duration} frames)"
                    )

                # Hold the jump for the specified duration
                for frame in range(jump_duration):
                    obs, reward, done, info = env.step(base_idx)
                    total_frames += 1

                    # Track progress
                    x_pos = info.get("x_pos", 0)
                    if isinstance(x_pos, list):
                        x_pos = x_pos[0] if x_pos else 0
                    if x_pos > max_x:
                        max_x = x_pos

                    if render:
                        env.render()
                        time.sleep(0.02)  # Make it viewable

                    if done:
                        break
            else:
                # Non-jump action - just do it once
                if render:
                    print(f"\n  Action {action_index}: {base_action}")

                obs, reward, done, info = env.step(base_idx)
                total_frames += 1

                x_pos = info.get("x_pos", 0)
                if isinstance(x_pos, list):
                    x_pos = x_pos[0] if x_pos else 0
                if x_pos > max_x:
                    max_x = x_pos

                if render:
                    env.render()
                    time.sleep(0.02)

            action_index += 1  # Move to next action in genome

            if render and action_index % 10 == 0:
                print(
                    f"  Progress: Action {action_index}/{len(individual.genome)}, Position: {x_pos}"
                )

    except KeyboardInterrupt:
        if render:
            print("\n Game interrupted by user")
    finally:
        env.close()

        if render:
            print(
                f"\nGame Over - Actions: {action_index}, Frames: {total_frames}, Max Position: {max_x}"
            )
            time.sleep(2)

    # Update stats only in non-render mode
    if not render:
        individual.fitness = max_x + action_index * 0.1
        individual.max_x = max_x
        individual.steps_survived = action_index
        individual.total_frames = total_frames

    return max_x + action_index * 0.1


def create_initial_population(size):
    return [MarioIndividual() for _ in range(size)]


def select_survivors(population, elite_size):
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    return sorted_pop[:elite_size]


def create_next_generation(survivors, population_size):
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
    jump_heights = []

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

    print(f"\n{'=' * 60}")
    print(f"GENERATION {generation} - BEST INDIVIDUAL")
    print(f"{'=' * 60}")
    print(f"Fitness: {individual.fitness:.2f}")
    print(f"Distance: {individual.max_x} units")
    print(f"Actions Taken: {individual.steps_survived}")
    print(f"Total Game Frames: {individual.total_frames}")
    print(
        f"Avg Frames per Action: {individual.total_frames / individual.steps_survived:.1f}"
    )

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

    # Show first 20 actions with simple ASCII
    print(f"\nFirst 20 Actions: ", end="")
    for i in range(min(20, individual.steps_survived)):
        action_idx = individual.genome[i]
        action_info = MOVEMENTS[action_idx]
        base_action = action_info["base"]
        jump_duration = action_info["jump_duration"]

        if jump_duration > 0:
            if jump_duration > 20:
                print("V", end="")  # Very High jump
            elif jump_duration > 12:
                print("H", end="")  # High jump
            elif jump_duration > 6:
                print("M", end="")  # Medium jump
            else:
                print("L", end="")  # Low jump
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
    print("WITH VARIABLE JUMP HEIGHTS (ACTUALLY VARIED)")
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
            fitness = evaluate_fitness(individual, render=False)
            fitnesses.append(fitness)

            if (i + 1) % 5 == 0:
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

    print(f"\n{'=' * 60}")
    print("PLAYING FINAL BEST INDIVIDUAL")
    print("=" * 60)
    print("\nPress Enter to watch the final evolved Mario...")
    input()

    render_individual = final_best.copy()
    evaluate_fitness(render_individual, render=True)

    return final_best


if __name__ == "__main__":
    # Run the genetic algorithm
    best_mario = run_genetic_algorithm(generations=10)

    print("\nSimulations Complete")
