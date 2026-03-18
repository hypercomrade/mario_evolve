# basic_mario.py
import time

import gym_super_mario_bros
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

# Enable rendering by setting render_mode
try:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode="human")
except TypeError:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

# Apply ALL possible moves to the environment
env = JoypadSpace(env, MOVEMENTS)

print("=" * 50)
print("SUPER MARIO BROS - AI ENVIRONMENT")
print("=" * 50)
print(f"\nAction Space: {len(MOVEMENTS)} possible moves")
print("\nAvailable Actions:")
for i, moves in enumerate(MOVEMENTS):
    action_name = "+".join(moves) if moves else "NOOP"
    print(f"  {i:2d}: {action_name:15s} - ", end="")

    # Add descriptions for common moves
    if moves == ["NOOP"]:
        print("Do nothing")
    elif moves == ["right"]:
        print("Walk right")
    elif moves == ["right", "A"]:
        print("Walk right + jump (basic jump)")
    elif moves == ["right", "B"]:
        print("Walk right + run (fast)")
    elif moves == ["right", "A", "B"]:
        print("Walk right + jump + run (running jump)")
    elif moves == ["left"]:
        print("Walk left")
    elif moves == ["left", "A"]:
        print("Walk left + jump")
    elif moves == ["A"]:
        print("Jump in place")
    elif moves == ["B"]:
        print("Run in place")
    elif moves == ["down"]:
        print("Crouch / go down pipes")
    elif moves == ["up"]:
        print("Look up / enter pipes")
    else:
        print("")

print("\nCurrent Strategy: Walking right only (Action 1)")
print("A game window should appear. Press Ctrl+C to stop")
print("=" * 50)
time.sleep(3)

try:
    obs = env.reset()
    done = False
    step = 0

    while not done:
        # CURRENT STRATEGY: Just walk right (action 1)
        # FUTURE AI: You can change this to any action 0-12
        action = 1  # ['right']

        # Take step
        obs, reward, done, info = env.step(action)

        # Render
        if hasattr(env, "render"):
            env.render()

        # Print progress
        if step % 30 == 0:
            x_pos = info.get("x_pos", step)
            if isinstance(x_pos, list):
                x_pos = x_pos[0]

            # Get additional info
            world = info.get("world", 1)
            stage = info.get("stage", 1)
            score = info.get("score", 0)
            coins = info.get("coins", 0)
            time_left = info.get("time", 400)

            action_name = "+".join(MOVEMENTS[action]) if MOVEMENTS[action] else "NOOP"

            print(
                f"Step {step:4d} | Action: {action_name:12s} | "
                f"X: {x_pos:4d} | World: {world}-{stage} | "
                f"Score: {score:6d} | Coins: {coins:2d} | Time: {time_left:3d}"
            )

        time.sleep(0.05)
        step += 1

    print(f"\nGame ended after {step} steps")
    if info.get("flag_get", False):
        print("VICTORY! Mario reached the flag!")
        print(f"Final Score: {info.get('score', 0)}")
        print(f"Coins Collected: {info.get('coins', 0)}")
        print(f"Time Remaining: {info.get('time', 0)}")
    else:
        print("GAME OVER - Mario died")
        print(f"Final Position: World {info.get('world', 1)}-{info.get('stage', 1)}")
        print(
            f"Distance: {info.get('x_pos', [0])[0] if isinstance(info.get('x_pos'), list) else info.get('x_pos', 0)}"
        )

except KeyboardInterrupt:
    print(f"\n\nTraining interrupted after {step} steps")
finally:
    env.close()
    print("\nEnvironment closed.")
