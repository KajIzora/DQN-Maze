#!/usr/bin/env python3
"""Load and run a trained DQN model in the maze environment."""

import argparse
import pickle
import time

import numpy as np
import torch

from maze_rl.dqn import QNetwork, select_device
from maze_rl.maze_env import MazeEnv


def load_model(model_path: str) -> dict:
    """Load a trained DQN checkpoint from a pickle file."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def build_policy_net(model_data: dict, device: torch.device) -> QNetwork:
    """Reconstruct the policy network from saved checkpoint metadata."""
    training_params = model_data.get("training_params", {})
    state_dim = model_data.get("state_dim", training_params.get("state_dim"))
    action_dim = model_data.get("action_dim", training_params.get("action_dim"))
    hidden_sizes = model_data.get("hidden_sizes", training_params.get("hidden_sizes", (128, 128)))

    if state_dim is None or action_dim is None:
        raise ValueError(
            "Checkpoint is missing state_dim/action_dim. "
            "Expected top-level keys or training_params entries."
        )

    state_dict = model_data.get("policy_net_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint is missing policy_net_state_dict.")

    policy_net = QNetwork(state_dim, action_dim, tuple(hidden_sizes)).to(device)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()
    return policy_net


def greedy_action(policy_net: QNetwork, state: np.ndarray, device: torch.device) -> int:
    """Select the highest-Q action (no exploration)."""
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_t)
    return int(torch.argmax(q_values, dim=1).item())


def run_episode(
    env: MazeEnv,
    policy_net: QNetwork,
    device: torch.device,
    max_steps: int = 1000,
    delay: float = 0.0,
    verbose: bool = True,
) -> dict:
    """Run a single episode using the trained DQN policy."""
    state = env.reset()
    total_reward = 0.0
    steps = 0
    trajectory = [state.copy()]

    for _ in range(max_steps):
        action = greedy_action(policy_net, state, device)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        steps += 1
        trajectory.append(next_state.copy())

        if verbose:
            print(
                f"Step {steps}: pos=({next_state[0]:.3f}, {next_state[1]:.3f}), "
                f"yaw={next_state[2]:.3f}, reward={reward:.2f}, "
                f"action={action}, collision={info.get('collision', False)}"
            )

        if delay > 0:
            time.sleep(delay)

        if done:
            if verbose:
                if env._goal_reached:
                    print(f"Goal reached in {steps} steps! Total reward: {total_reward:.2f}")
                else:
                    print(f"Episode terminated at step {steps}. Total reward: {total_reward:.2f}")
            break

        state = next_state

    return {
        "steps": steps,
        "total_reward": total_reward,
        "success": env._goal_reached if hasattr(env, "_goal_reached") else False,
        "trajectory": trajectory,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a trained DQN model in the maze environment.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/dqn/dqn.pkl",
        help="Path to the trained DQN pickle file.",
    )
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI rendering.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode.")
    parser.add_argument(
        "--forward-distance",
        type=float,
        default=0.05,
        help="Per-step forward translation distance (should match training).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. cpu, cuda, mps). Defaults to best available.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Delay between steps in seconds. Defaults to 0.05 with --gui, else 0.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity.")
    args = parser.parse_args()

    if args.delay is None:
        args.delay = 0.05 if args.gui else 0.0

    print(f"Loading model from {args.model}...")
    model_data = load_model(args.model)

    device = select_device(args.device)
    policy_net = build_policy_net(model_data, device)

    training_params = model_data.get("training_params", {})
    episode = model_data.get("episode")
    episode_rewards = model_data.get("episode_rewards", [])

    print("Model loaded successfully!")
    print(f"  Device: {device}")
    if episode is not None:
        print(f"  Checkpoint episode: {episode}")
    if episode_rewards:
        print(f"  Training episodes recorded: {len(episode_rewards)}")
        print(f"  Best training reward: {max(episode_rewards):.2f}")
    if training_params:
        print(f"  Training params: {training_params}")
    print()

    env = MazeEnv(gui=args.gui, forward_distance=args.forward_distance)

    results = []
    for episode_idx in range(args.episodes):
        if not args.quiet:
            print(f"=== Episode {episode_idx + 1}/{args.episodes} ===")

        result = run_episode(
            env,
            policy_net,
            device,
            max_steps=args.max_steps,
            delay=args.delay,
            verbose=not args.quiet,
        )
        results.append(result)

        if not args.quiet:
            print()

    print("=" * 50)
    print("Summary:")
    print(f"  Episodes run: {len(results)}")
    successes = sum(1 for r in results if r["success"])
    print(f"  Successful episodes: {successes}/{len(results)}")
    if results:
        avg_steps = np.mean([r["steps"] for r in results])
        avg_reward = np.mean([r["total_reward"] for r in results])
        print(f"  Average steps per episode: {avg_steps:.1f}")
        print(f"  Average reward per episode: {avg_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
