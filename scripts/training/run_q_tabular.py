#!/usr/bin/env python3
"""Run tabular Q-learning training on the maze environment."""
import argparse
import json
import math
import os
import pickle
from collections import defaultdict
from typing import Optional
import time

import numpy as np
from tqdm import tqdm

from maze_rl.maze_env import MazeEnv


def discretize_state(state: np.ndarray, cell: float = 0.10) -> tuple[int, int, int]:
    """
    Map continuous (x, y, yaw) to coarse discrete bins. Matches tabular Q-learning defaults.
    """
    x, y, yaw = float(state[0]), float(state[1]), float(state[2])
    i = int(round(x / cell))
    j = int(round(y / cell))
    yaw = yaw % (2.0 * math.pi)
    heading_bin = int(round(yaw / (math.pi / 2.0))) % 4
    return i, j, heading_bin


def _linear_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return float(end)
    frac = min(max(step / float(total_steps - 1), 0.0), 1.0)
    return start + frac * (end - start)


def _epsilon_greedy_tabular(
    q_table: defaultdict,
    state_key: tuple[int, int, int],
    action_dim: int,
    epsilon: float,
    rng: np.random.Generator,
    ) -> int:
    if rng.random() < epsilon or state_key not in q_table:
        return int(rng.integers(low=0, high=action_dim))
    values = q_table[state_key]
    max_q = np.max(values)
    best = np.flatnonzero(values == max_q)
    choice = int(best[rng.integers(low=0, high=len(best))])
    return choice


def _get_log_path(checkpoint_path: Optional[str]) -> Optional[str]:
    """Get the log file path based on checkpoint path."""
    if not checkpoint_path:
        return None
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not checkpoint_dir:
        checkpoint_dir = "."
    log_path = os.path.join(checkpoint_dir, "log.csv")
    return log_path


def _get_actions_path(checkpoint_path: Optional[str]) -> Optional[str]:
    """Get the actions file path based on checkpoint path."""
    if not checkpoint_path:
        return None
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not checkpoint_dir:
        checkpoint_dir = "."
    actions_path = os.path.join(checkpoint_dir, "actions.json")
    return actions_path


def _init_log_file(log_path: str, training_params: dict) -> None:
    """Initialize the log file with header and training parameters."""
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    with open(log_path, "w") as f:
        # Write training parameters header
        f.write("# Training Parameters\n")
        for key, value in training_params.items():
            f.write(f"# {key}: {value}\n")
        f.write("#\n")
        # Write column headers
        f.write("episode,steps,reward,epsilon,alpha,success,success_rate,avg_recent_len,episode_time,elapsed_time\n")


def _log_episode(
    log_path: str,
    episode: int,
    steps: int,
    reward: float,
    epsilon: float,
    alpha: float,
    success: bool,
    success_rate: float,
    avg_recent_len: float,
    episode_time: float,
    elapsed_time: float,) -> None:
    """Log episode metrics to the log file."""
    if not log_path:
        return
    
    with open(log_path, "a") as f:
        f.write(
            f"{episode},{steps},{reward:.4f},{epsilon:.4f},{alpha:.4f},"
            f"{int(success)},{success_rate:.2f},{avg_recent_len:.2f},"
            f"{episode_time:.4f},{elapsed_time:.4f}\n"
        )


def _save_checkpoint(
    q_table: defaultdict,
    episode: int,
    checkpoint_path: str,
    cell_size: float,
    episode_lengths: list[int],
    episode_rewards: list[float],
    successes: int,
    training_params: dict,
) -> None:
    """Save a checkpoint of the Q-table and training state."""
    # Create checkpoint filename with episode number
    base_path, ext = os.path.splitext(checkpoint_path)
    checkpoint_file = f"{base_path}_ep{episode}{ext}"
    
    # Create directory if it doesn't exist
    checkpoint_dir = os.path.dirname(checkpoint_file)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    save_data = {
        "q_table": dict(q_table),  # Convert defaultdict to dict for saving
        "episode": episode,
        "cell_size": cell_size,
        "episode_lengths": episode_lengths.copy(),
        "episode_rewards": episode_rewards.copy(),
        "successes": successes,
        "training_params": training_params,
    }
    with open(checkpoint_file, "wb") as f:
        pickle.dump(save_data, f)


def train_tabular_q(
    env: MazeEnv,
    *,
    episodes: int = 400,
    max_steps: int = 400,
    alpha_start: float = 0.5,
    alpha_end: float = 0.05,
    gamma: float = 0.75,
    eps_start: float = 0.80,
    eps_end: float = 0.05,
    cell_size: float = 0.10,
    ray_stop_tol: float = 0.05,
    no_progress_patience: int = 40,
    progress_tol: float = 0.01,
    seed: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: Optional[int] = None,
    training_params: Optional[dict] = None,
    ) -> dict:
    """
    Train a coarse tabular Q-learning policy on MazeEnv using discretized states.
    """
    rng = np.random.default_rng(seed)
    if hasattr(env, "_ray_stop_clearance"):
        env._ray_stop_clearance = float(ray_stop_tol)

    action_dim = env.action_dim
    q_table: defaultdict = defaultdict(lambda: np.zeros(action_dim, dtype=np.float32))
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    successes = 0

    training_start_time = time.time()
    
    # Initialize log file if checkpoint path is provided
    log_path = _get_log_path(checkpoint_path)
    if log_path:
        _init_log_file(log_path, training_params or {})
        print(f"Logging to: {log_path}")
    
    # Initialize actions file if checkpoint path is provided
    actions_path = _get_actions_path(checkpoint_path)
    all_episode_actions = {}
    if actions_path:
        actions_dir = os.path.dirname(actions_path)
        if actions_dir and not os.path.exists(actions_dir):
            os.makedirs(actions_dir, exist_ok=True)
        print(f"Action sequences will be saved to: {actions_path}")
    
    # Create progress bar
    pbar = tqdm(
        range(episodes),
        desc="Training",
        unit="ep",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    for ep in pbar:
        state = env.reset()
        state_disc = discretize_state(state, cell=cell_size)

        eps = _linear_schedule(ep, episodes, eps_start, eps_end)
        alpha = _linear_schedule(ep, episodes, alpha_start, alpha_end)

        if hasattr(env, "_last_goal_distance") and env._last_goal_distance is not None:
            last_goal_distance = float(env._last_goal_distance)
        else:
            last_goal_distance = None

        total_reward = 0.0
        steps = 0
        no_progress = 0
        done = False
        episode_actions = []  # Track actions for this episode

        episode_start_time = time.time()
        for step_idx in range(max_steps):
            action = _epsilon_greedy_tabular(q_table, state_disc, action_dim, eps, rng)
            episode_actions.append(int(action))  # Store action
            next_state, reward, done, info = env.step(action)
            time.sleep(0.01)
            total_reward += reward

            next_disc = discretize_state(next_state, cell=cell_size)
            best_next = float(np.max(q_table[next_disc]))
            td_target = reward + gamma * best_next * (0.0 if done else 1.0)
            td_error = td_target - q_table[state_disc][action]
            q_table[state_disc][action] += alpha * td_error

            if hasattr(env, "_last_goal_distance") and env._last_goal_distance is not None:
                curr_goal_distance = float(env._last_goal_distance)
                if last_goal_distance is not None:
                    if (last_goal_distance - curr_goal_distance) < progress_tol:
                        no_progress += 1
                    else:
                        no_progress = 0
                last_goal_distance = curr_goal_distance

            steps += 1
            state_disc = next_disc

            if done or no_progress >= no_progress_patience:
                break

        episode_time = time.time() - episode_start_time
        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
        episode_success = done and getattr(env, "_goal_reached", False)
        if episode_success:
            successes += 1

        # Calculate statistics for progress bar
        recent = episode_lengths[-20:] if len(episode_lengths) >= 20 else episode_lengths
        avg_recent = float(np.mean(recent)) if recent else float("nan")
        success_rate = (successes / (ep + 1)) * 100 if ep > 0 else 0.0
        
        # Calculate estimated time remaining
        elapsed_time = time.time() - training_start_time
        episodes_completed = ep + 1
        if episodes_completed > 0:
            avg_time_per_episode = elapsed_time / episodes_completed
            remaining_episodes = episodes - episodes_completed
            estimated_remaining = avg_time_per_episode * remaining_episodes
            estimated_remaining_str = f"{estimated_remaining/60:.1f}m" if estimated_remaining > 60 else f"{estimated_remaining:.1f}s"
        else:
            estimated_remaining_str = "calculating..."

        # Save episode actions and update JSON file immediately
        if actions_path:
            all_episode_actions[ep + 1] = episode_actions
            # Save/update actions file after each episode
            with open(actions_path, "w") as f:
                json.dump(all_episode_actions, f, indent=2)
        
        # Log episode metrics to file
        if log_path:
            _log_episode(
                log_path,
                ep + 1,
                steps,
                total_reward,
                eps,
                alpha,
                episode_success,
                success_rate,
                avg_recent,
                episode_time,
                elapsed_time,
            )

        # Update progress bar description with current stats
        pbar.set_postfix({
            "ε": f"{eps:.2f}",
            "α": f"{alpha:.2f}",
            "success": f"{successes}",
            "ETA": estimated_remaining_str
        })
        
        # Save checkpoint if enabled and it's time
        if checkpoint_path and checkpoint_every and (ep + 1) % checkpoint_every == 0:
            _save_checkpoint(
                q_table,
                ep + 1,
                checkpoint_path,
                cell_size,
                episode_lengths,
                episode_rewards,
                successes,
                training_params or {},
            )
    
    pbar.close()

    # Final save of actions (in case any were missed, though they should all be saved)
    if actions_path and all_episode_actions:
        with open(actions_path, "w") as f:
            json.dump(all_episode_actions, f, indent=2)
        print(f"Action sequences saved to: {actions_path}")

    trailing = episode_lengths[-50:] if episode_lengths else []
    trailing_avg = float(np.mean(trailing)) if trailing else float("nan")
    print(
        f"\nTabular training done. Successes: {successes}/{episodes}. "
        f"Avg episode length (last 50): {trailing_avg:.1f}"
    )
    return {
        "q_table": q_table,
        "episode_lengths": episode_lengths,
        "episode_rewards": episode_rewards,
        "successes": successes,
    }


def main():
    parser = argparse.ArgumentParser(description="Train tabular Q-learning on maze environment.")
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI rendering.")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum environment steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the trainer.")
    parser.add_argument("--forward-distance", type=float, default=0.20, help="Per-step forward translation distance for the environment.")
    parser.add_argument("--tabular-alpha-start", type=float, default=0.5, help="Initial learning rate for tabular Q-learning.")
    parser.add_argument("--tabular-alpha-end", type=float, default=0.05, help="Final learning rate for tabular Q-learning.")
    parser.add_argument("--tabular-eps-start", type=float, default=0.80, help="Initial epsilon for tabular epsilon-greedy policy.")
    parser.add_argument("--tabular-eps-end", type=float, default=0.05, help="Final epsilon for tabular epsilon-greedy policy.")
    parser.add_argument("--tabular-gamma", type=float, default=0.99, help="Discount factor for tabular Q-learning.")
    parser.add_argument("--tabular-cell-size", type=float, default=0.01, help="Discretization cell size (meters) for tabular state mapping.")
    parser.add_argument("--tabular-no-progress-patience", type=int, default=5000, help="Terminate episode early if progress stagnates for this many steps.")
    parser.add_argument("--tabular-progress-tol", type=float, default=0.01, help="Required improvement in goal distance to reset no-progress counter.")
    parser.add_argument("--tabular-ray-stop-tol", type=float, default=0.01, help="Clearance threshold used by tabular training when checking wall proximity.")
    parser.add_argument("--save-model", type=str, default=None, help="Path to save the trained model (pickle format).")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Save a checkpoint every N episodes during training. Checkpoints are saved as <save-model>_ep<N>.pkl")
    args = parser.parse_args()

    env = MazeEnv(gui=args.gui, forward_distance=args.forward_distance)

    print("Starting tabular Q-learning training demo...")
    # Prepare training parameters for checkpoint saving
    training_params = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "alpha_start": args.tabular_alpha_start,
        "alpha_end": args.tabular_alpha_end,
        "gamma": args.tabular_gamma,
        "eps_start": args.tabular_eps_start,
        "eps_end": args.tabular_eps_end,
        "cell_size": args.tabular_cell_size,
        "seed": args.seed,
    }
    # Only enable checkpoints if save_model is provided
    checkpoint_path = args.save_model if args.save_model else None
    checkpoint_every = args.checkpoint_every if (args.save_model and args.checkpoint_every) else None
    
    if checkpoint_every:
        print(f"Checkpoints will be saved every {checkpoint_every} episodes to {checkpoint_path}")
    
    results = train_tabular_q(
        env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        alpha_start=args.tabular_alpha_start,
        alpha_end=args.tabular_alpha_end,
        gamma=args.tabular_gamma,
        eps_start=args.tabular_eps_start,
        eps_end=args.tabular_eps_end,
        cell_size=args.tabular_cell_size,
        ray_stop_tol=args.tabular_ray_stop_tol,
        no_progress_patience=args.tabular_no_progress_patience,
        progress_tol=args.tabular_progress_tol,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        training_params=training_params,
    )
    
    if args.save_model:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(args.save_model)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Save final Q-table and training metadata
        save_data = {
            "q_table": dict(results["q_table"]),  # Convert defaultdict to dict for saving
            "cell_size": args.tabular_cell_size,
            "episode_lengths": results["episode_lengths"],
            "episode_rewards": results["episode_rewards"],
            "successes": results["successes"],
            "training_params": training_params,
        }
        with open(args.save_model, "wb") as f:
            pickle.dump(save_data, f)
        print(f"Final model saved to {args.save_model}")

    env.close()


if __name__ == "__main__":
    main()
