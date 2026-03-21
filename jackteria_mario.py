import copy
import os
import pickle
import random
import time
from datetime import datetime
import json

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

# Identify backward-moving actions (left, left+A, left+B, left+A+B)
BACKWARD_ACTIONS = []
for i, action in enumerate(MOVEMENTS):
    base = action["base"]
    if base and base[0] == "left":  # Any action starting with "left" is backward
        BACKWARD_ACTIONS.append(i)

print(f"Created {len(MOVEMENTS)} actions with variable jump heights")
print(f"  Backward actions: {len(BACKWARD_ACTIONS)} (will be mutated less frequently)")

# Genetic Algorithm Parameters
POPULATION_SIZE = 20
SEQUENCE_LENGTH = 300
MUTATION_RATE = 0.3
ELITE_SIZE = 10
BACKWARD_MUTATION_PENALTY = 0.5  # Half as likely to mutate to backward moves

# Structural mutation parameters
STRUCTURAL_MUTATION_RATE = 0.15  # Chance for structural mutation per individual
MAX_DELETE_SIZE = 5  # Maximum number of moves to delete
MAX_INSERT_SIZE = 5  # Maximum number of moves to insert
MIN_GENOME_LENGTH = 50  # Minimum allowed genome length
MAX_GENOME_LENGTH = 600  # Maximum allowed genome length

# Fitness weighting parameters
DISTANCE_WEIGHT = 0.5  # Weight for distance traveled
SPEED_WEIGHT = 1.0  # Weight for speed (distance per action)
COMPLETION_BONUS = 1000  # Bonus for completing the level
MIN_SPEED_BONUS = 0.1  # Minimum speed bonus to avoid division by zero


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
        self.speed_score = 0  # Track speed component separately
        self.generation = 0
        self.id = random.randint(10000, 99999)

    def mutate(self, mutation_rate=MUTATION_RATE):
        """Mutate with bias against backward moves and structural mutations"""
        # First, decide if this individual gets a structural mutation
        if random.random() < STRUCTURAL_MUTATION_RATE:
            self.structural_mutation()

        # Then apply point mutations
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                # Decide whether to allow backward moves
                if random.random() < BACKWARD_MUTATION_PENALTY:
                    # Bias toward forward/neutral moves - exclude backward actions
                    valid_actions = [
                        a for a in range(len(MOVEMENTS)) if a not in BACKWARD_ACTIONS
                    ]
                    self.genome[i] = random.choice(valid_actions)
                else:
                    # Normal mutation - any action possible
                    self.genome[i] = random.randint(0, len(MOVEMENTS) - 1)
        return self

    def structural_mutation(self):
        """Apply structural mutations: delete or insert chunks of moves"""
        mutation_type = random.choice(["delete", "insert"])

        if mutation_type == "delete" and len(self.genome) > MIN_GENOME_LENGTH:
            # Delete 1-5 consecutive moves
            delete_size = random.randint(1, MAX_DELETE_SIZE)
            if len(self.genome) > delete_size:
                start_idx = random.randint(0, len(self.genome) - delete_size)
                del self.genome[start_idx : start_idx + delete_size]

        elif mutation_type == "insert" and len(self.genome) < MAX_GENOME_LENGTH:
            # Insert 1-5 new random moves
            insert_size = random.randint(1, MAX_INSERT_SIZE)
            insert_pos = random.randint(0, len(self.genome))

            # Create chunk of new moves (biased against backward moves)
            new_chunk = []
            for _ in range(insert_size):
                if random.random() < BACKWARD_MUTATION_PENALTY:
                    # Bias toward forward/neutral moves
                    valid_actions = [
                        a for a in range(len(MOVEMENTS)) if a not in BACKWARD_ACTIONS
                    ]
                    new_chunk.append(random.choice(valid_actions))
                else:
                    new_chunk.append(random.randint(0, len(MOVEMENTS) - 1))

            # Insert the chunk
            self.genome[insert_pos:insert_pos] = new_chunk

    def crossover(self, other):
        """Crossover with possible different genome lengths"""
        # Choose crossover point for first parent
        point1 = random.randint(0, len(self.genome))

        # Choose crossover point for second parent (adjusted for length)
        point2 = random.randint(0, len(other.genome))

        # Create child by combining segments
        child_genome = self.genome[:point1] + other.genome[point2:]

        return MarioIndividual(child_genome)

    def copy(self):
        new_individual = MarioIndividual(self.genome)
        new_individual.fitness = self.fitness
        new_individual.max_x = self.max_x
        new_individual.steps_survived = self.steps_survived
        new_individual.total_frames = self.total_frames
        new_individual.speed_score = self.speed_score
        new_individual.generation = self.generation
        new_individual.id = self.id
        return new_individual

    def save(self, filename):
        """Save individual to file"""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"  Saved individual to {filename}")
        print(f"  Genome length: {len(self.genome)}")


def calculate_fitness(max_x, actions_taken, total_frames=None):
    """
    Calculate fitness based on distance and time/speed.

    Args:
        max_x: Maximum x position reached
        actions_taken: Number of actions executed
        total_frames: Total game frames (if available, use for more precise time)

    Returns:
        fitness, speed_score
    """
    # Base distance score
    distance_score = max_x * DISTANCE_WEIGHT

    # Calculate speed (distance per action or per frame)
    if total_frames is not None and total_frames > 0:
        # More precise: use actual frames/time
        speed = max_x / total_frames
        speed_score = speed * SPEED_WEIGHT * 100  # Scale to be meaningful
        speed_metric = f"{speed:.3f} distance/frame"
    else:
        # Use actions as proxy for time
        actions = max(actions_taken, 1)  # Avoid division by zero
        speed = max_x / actions
        speed_score = speed * SPEED_WEIGHT * 10  # Scale to be meaningful
        speed_metric = f"{speed:.2f} distance/action"

    # Completion bonus for reaching the end of the level
    # Level 1-1 ends around ~3072 units
    completion_bonus = COMPLETION_BONUS if max_x > 3000 else 0

    # Final fitness
    fitness = distance_score + speed_score + completion_bonus
 
    return fitness, speed_score


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
    start_time = time.time() if render else None

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
            elapsed_time = time.time() - start_time if start_time else 0
            print(f"\nGame Over - Position: {max_x}")
            print(f"Time Elapsed: {elapsed_time:.2f} seconds")
            print(f"Actions Used: {action_index}")
            print(f"Total Frames: {total_frames}")
    
    fitness = None
    # Update stats with new fitness calculation
    if not render:
        # Calculate fitness using both distance and speed
        fitness, speed_score = calculate_fitness(max_x, action_index, total_frames)

        individual.fitness = fitness
        individual.max_x = max_x
        individual.steps_survived = action_index
        individual.total_frames = total_frames
        individual.speed_score = speed_score

    return fitness, frames


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
        f.write(f"Speed Score: {individual.speed_score:.2f}\n")
        f.write(f"Actions Taken: {individual.steps_survived}\n")
        f.write(f"Total Frames: {individual.total_frames}\n")
        f.write(f"Genome Length: {len(individual.genome)}\n")

        # Calculate and display actual speed
        if individual.total_frames > 0:
            actual_speed = individual.max_x / individual.total_frames
            f.write(f"Actual Speed: {actual_speed:.3f} distance/frame\n")

        f.write(f"\nFitness Components:\n")
        f.write(f"  Distance Score: {individual.max_x * DISTANCE_WEIGHT:.2f}\n")
        f.write(f"  Speed Score: {individual.speed_score:.2f}\n")
        f.write(
            f"  Completion Bonus: {COMPLETION_BONUS if individual.max_x > 3000 else 0}\n"
        )

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

    # Calculate actual speed
    actual_speed = 0
    if individual.total_frames > 0:
        actual_speed = individual.max_x / individual.total_frames
    elif individual.steps_survived > 0:
        actual_speed = individual.max_x / individual.steps_survived

    print(f"\n{'=' * 60}")
    print(f"GENERATION {generation} - BEST INDIVIDUAL (ID: {individual.id})")
    print(f"{'=' * 60}")
    print(f"Fitness: {individual.fitness:.2f}")
    print(f"  Distance Score: {individual.max_x * DISTANCE_WEIGHT:.2f}")
    print(f"  Speed Score: {individual.speed_score:.2f}")
    print(f"  Completion Bonus: {COMPLETION_BONUS if individual.max_x > 3000 else 0}")
    print(f"\nPerformance:")
    print(f"  Distance: {individual.max_x} units")
    print(f"  Actions Taken: {individual.steps_survived}")
    print(f"  Total Frames: {individual.total_frames}")
    print(f"  Speed: {actual_speed:.3f} units/frame")
    print(f"  Genome Length: {len(individual.genome)}")
    print(f"  Backward Moves (first 50): {backward_count}")

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

def MarioIndividual_to_dict(individual):
    dic = {}
    dic["genome"] = [int(i) for i in individual.genome]
    dic["fitness"] = int(individual.fitness)
    dic["max_x"] = int(individual.max_x) 
    dic["steps_survived"] = int(individual.steps_survived)
    dic["total_frames"] = int(individual.total_frames)
    dic["generation"] = int(individual.generation)
    dic["id"] = int(individual.id)
    return dic


def write_individual_to_file(individual, individuals_file):
    # Have this somewhere  individuals_file = open(output_file_path, "a")
    json.dump(MarioIndividual_to_dict(individual), individuals_file)
    individuals_file.write("\n")
    individuals_file.flush()

def conclude_run(individuals_file, best_individual):
    individuals_file.write("--------------------------------------------------------------\n")
    write_individual_to_file(best_individual, individuals_file)
    individuals_file.write("--------------------------------------------------------------\n")
    individuals_file.flush()

def dict_to_MarioIndividual(dict):
    individual = MarioIndividual()
    individual.genome = dict["genome"]
    individual.fitness = dict["fitness"]
    individual.max_x = dict["max_x"]
    individual.steps_survived = dict["steps_survived"]
    individual.total_frames = dict["total_frames"]
    individual.generation = dict["generation"]
    individual.id = dict["id"]
    return individual

def read_individuals_from_file(individuals_file):
    individuals = []
    best_individuals = []
    individuals_file.seek(0)  
    for line in individuals_file:
        if line[0] != "-":
            individuals.append(dict_to_MarioIndividual(json.loads(line)))
        else:
            # skip the ------
            line = next(individuals_file, None)
            best_individuals.append(dict_to_MarioIndividual(json.loads(line)))
            # skip the ------
            next(individuals_file, None)
   
    return individuals, best_individuals

def display_individual(individual):
    evaluate_fitness(individual, render=True, record=True)

def display_best_individuals(individuals_file_path):
    with open(f"{individuals_file_path}", "r") as individuals_file:
        individuals, best_individuals = read_individuals_from_file(individuals_file)
        for b in best_individuals:
            print("Called")
            display_individual(b)
            print("Done")
            
def run_genetic_algorithm(generations=50, individual_file_path="saved_individuals"):
    """Main genetic algorithm loop"""
    print("=" * 60)
    print("SUPER MARIO BROS - GENETIC PROGRAMMING")
    print("WITH DISTANCE + SPEED OPTIMIZATION")
    print("=" * 60)

    print(f"\nFitness Configuration:")
    print(f"  Distance Weight: {DISTANCE_WEIGHT}")
    print(f"  Speed Weight: {SPEED_WEIGHT}")
    print(f"  Completion Bonus: {COMPLETION_BONUS}")
    print(f"\nPopulation Size: {POPULATION_SIZE}")
    print(f"Initial Genome Length: {SEQUENCE_LENGTH} actions")
    print(f"Point Mutation Rate: {MUTATION_RATE * 100}%")
    print(f"Structural Mutation Rate: {STRUCTURAL_MUTATION_RATE * 100}%")
    print(f"  Delete Chunk Size: 1-{MAX_DELETE_SIZE} moves")
    print(f"  Insert Chunk Size: 1-{MAX_INSERT_SIZE} moves")
    print(f"  Min/Max Genome Length: {MIN_GENOME_LENGTH}/{MAX_GENOME_LENGTH}")
    print(f"Backward Mutation Penalty: {BACKWARD_MUTATION_PENALTY * 100}% less likely")
    print(f"Jump Durations: {JUMP_DURATIONS} frames")
    print(f"Total Actions: {len(MOVEMENTS)}")
    print(f"Generations: {generations}")
    print(f"\nStarting evolution...")
    print("=" * 60)

    population = create_initial_population(POPULATION_SIZE)
    best_fitness_history = []
    best_individual_history = []
    length_history = []
    speed_history = []
    individuals_file = open(f"{individual_file_path}", "a")

    best_individual = None
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
        length_history.append(len(best_individual.genome))

        # Track speed of best individual
        if best_individual.total_frames > 0:
            speed_history.append(best_individual.max_x / best_individual.total_frames)
        else:
            speed_history.append(0)

        avg_fitness = np.mean(fitnesses)
        avg_length = np.mean([len(ind.genome) for ind in population])

        # Calculate average speed of population
        avg_speed = 0
        speed_count = 0
        for ind in population:
            if ind.total_frames > 0:
                avg_speed += ind.max_x / ind.total_frames
                speed_count += 1
        avg_speed = avg_speed / speed_count if speed_count > 0 else 0

        for i in population:
            write_individual_to_file(i, individuals_file)
            
        print(f"\n  Generation {generation + 1} Results:")
        print(f"    Best Fitness: {best_fitness:.2f}")
        print(f"    Avg Fitness: {avg_fitness:.2f}")
        print(f"    Best Distance: {best_individual.max_x}")
        print(f"    Best Speed: {speed_history[-1]:.3f} units/frame")
        print(f"    Avg Speed: {avg_speed:.3f} units/frame")
        print(f"    Best Genome Length: {len(best_individual.genome)}")
        print(f"    Avg Genome Length: {avg_length:.1f}")

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
        
    conclude_run(individuals_file, best_individual)
        
    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE - FINAL RESULTS")
    print("=" * 60)

    final_best = max(best_individual_history, key=lambda x: x.fitness)
    print_best_individual(final_best, "FINAL")

    print(f"\nFitness History:")
    for i, fitness in enumerate(best_fitness_history):
        print(f"  Gen {i + 1}: {fitness:.2f}")

    print(f"\nSpeed Evolution (units/frame):")
    for i, speed in enumerate(speed_history):
        if speed > 0:
            print(f"  Gen {i + 1}: {speed:.3f}")

    print(f"\nGenome Length Evolution:")
    for i, length in enumerate(length_history):
        print(f"  Gen {i + 1}: {length}")

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
    print(f"  Speed Score: {individual.speed_score:.2f}")
    print(f"  Genome Length: {len(individual.genome)}")
    return individual


if __name__ == "__main__":
    # Path for individuals (the extension is added in the program)
    individual_file_path="saved_individuals.jsonl"
    
    # Run the genetic algorithm
    best_mario = run_genetic_algorithm(generations=60, individual_file_path=individual_file_path)
    
    # Display the best individuals (you can run this multiple times)
    display_best_individuals(individual_file_path)
