#!/usr/bin/env python3
"""Run random policy rollouts on the maze environment."""
import argparse

import numpy as np

from maze_rl.maze_env import MazeEnv


def main():
    parser = argparse.ArgumentParser(description="Run random policy rollouts on maze environment.")
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI rendering.")
    parser.add_argument(
        "--demo-steps",
        type=int,
        default=2000,
        help="Number of random demo steps when not training.",
    )
    parser.add_argument(
        "--forward-distance",
        type=float,
        default=0.05,
        help="Per-step forward translation distance for the environment.",
    )
    args = parser.parse_args()

    env = MazeEnv(gui=args.gui, forward_distance=args.forward_distance)

    print("MazeEnv standalone demo. Running random policy for diagnostics...")
    state = env.reset()
    for step in range(args.demo_steps):
        action = np.random.choice(env.action_dim)
        state, reward, done, _ = env.step(action)
        if done:
            print(f"Goal reached at step {step}!")
            state = env.reset()

    env.close()


if __name__ == "__main__":
    main()

