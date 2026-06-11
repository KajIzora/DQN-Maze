#!/usr/bin/env python3
"""Run manual control mode for the maze environment."""
import argparse
import math

from maze_rl.maze_env import MazeEnv


def main():
    parser = argparse.ArgumentParser(description="Manual control mode for maze environment.")
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI rendering.")
    parser.add_argument("--forward-distance", type=float, default=0.15, help="Per-step forward translation distance for the environment.")
    args = parser.parse_args()

    env = MazeEnv(gui=args.gui, forward_distance=args.forward_distance)

    print("Manual control mode. Enter N/E/S/W to move; Q to quit.")
    state = env.reset()
    dir_to_yaw = {
        "n": math.pi / 2,
        "e": 0.0,
        "s": -math.pi / 2,
        "w": math.pi,
    }
    action_deltas = {
        0: 0.0,
        1: math.pi / 2,
        2: -math.pi / 2,
        3: math.pi,
    }
    try:
        while True:
            print(f"Position x={state[0]:.3f}, y={state[1]:.3f}")
            cmd = input("Command (N/E/S/W or Q to quit): ").strip().lower()
            if not cmd:
                continue
            key = cmd[0]
            if key == "q":
                print("Exiting manual mode.")
                break
            if key not in dir_to_yaw:
                print("Invalid command. Use N, E, S, W, or Q.")
                continue
            desired_yaw = dir_to_yaw[key]
            current_yaw = state[2]
            delta = MazeEnv._wrap_angle(desired_yaw - current_yaw)
            action = min(
                action_deltas.items(),
                key=lambda kv: abs(MazeEnv._wrap_angle(delta - kv[1])),
            )[0]
            state, reward, done, info = env.step(action)
            print(f" → moved with action {action}, reward={reward:.3f}")
            print(f"   new position x={state[0]:.3f}, y={state[1]:.3f}")
            if info.get("collision"):
                print("   collision detected.")
            if done:
                print("Goal reached! Resetting environment.")
                state = env.reset()
    except KeyboardInterrupt:
        print("\nManual control interrupted by user.")

    env.close()


if __name__ == "__main__":
    main()

