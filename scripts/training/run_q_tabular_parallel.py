#!/usr/bin/env python3
"""Run parallel tabular Q-learning training on the maze environment."""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np

from maze_rl.maze_env import MazeEnv
from run_q_tabular import (
    discretize_state,
    _epsilon_greedy_tabular,
    _linear_schedule,
    _get_log_path,
    _get_actions_path,
    _init_log_file,
    _log_episode,
    _save_checkpoint,
)


StateKey = Tuple[int, int, int]


@dataclass
class ParallelConfig:
    episodes: int
    max_steps: int
    num_workers: int
    alpha_start: float
    alpha_end: float
    gamma: float
    eps_start: float
    eps_end: float
    cell_size: float
    seed: int
    checkpoint_path: str | None
    ray_stop_tol: float
    no_progress_patience: int
    progress_tol: float
    forward_distance: float
    action_dim: int
    checkpoint_interval: int
    broadcast_every: int
    save_actions: bool
    action_log_interval: int


def actor_process(
    worker_id: int,
    cfg: ParallelConfig,
    param_conn,  # mp.Pipe end for receiving Q snapshots
    queue: mp.Queue,
):
    """Actor process that runs environment episodes and sends transitions to learner."""
    env = MazeEnv(
        gui=False,
        forward_distance=cfg.forward_distance,
    )
    # Set ray stop tolerance if available
    if hasattr(env, "_ray_stop_clearance"):
        env._ray_stop_clearance = float(cfg.ray_stop_tol)

    rng = np.random.default_rng(cfg.seed + worker_id)

    # Local Q-table copy (state_key → np.ndarray)
    q_local = defaultdict(lambda: np.zeros(cfg.action_dim, dtype=np.float32))

    episodes_per_worker = cfg.episodes // cfg.num_workers
    # Handle remainder episodes
    if worker_id < (cfg.episodes % cfg.num_workers):
        episodes_per_worker += 1

    for ep in range(episodes_per_worker):
        ep_start = time.time()
        state = env.reset()
        state_key = discretize_state(state, cell=cfg.cell_size)

        total_reward = 0.0
        steps = 0
        success = False
        no_progress = 0
        last_goal_distance = None
        episode_actions = [] if cfg.save_actions else None
        episode_transitions = []

        # Optional: check if learner sent an updated Q-table snapshot
        try:
            while param_conn.poll():
                q_snapshot = param_conn.recv()
                # Convert dict back to defaultdict
                q_local = defaultdict(lambda: np.zeros(cfg.action_dim, dtype=np.float32))
                for k, v in q_snapshot.items():
                    q_local[k] = v.copy()  # Make a copy of the array
        except (EOFError, BrokenPipeError):
            break

        # Check for updates periodically during episode (every 100 steps)
        update_check_interval = 100

        # Compute epsilon schedule based on global episode (approximate)
        # We'll use the worker's local episode count as a proxy
        global_episode_approx = worker_id * episodes_per_worker + ep
        eps = _linear_schedule(
            global_episode_approx, cfg.episodes, cfg.eps_start, cfg.eps_end
        )

        for t in range(cfg.max_steps):
            # Periodically check for Q-table updates during episode
            if t % update_check_interval == 0 and t > 0:
                try:
                    while param_conn.poll():
                        q_snapshot = param_conn.recv()
                        # Convert dict back to defaultdict
                        q_local = defaultdict(lambda: np.zeros(cfg.action_dim, dtype=np.float32))
                        for k, v in q_snapshot.items():
                            q_local[k] = v.copy()  # Make a copy of the array
                except (EOFError, BrokenPipeError):
                    break

            action = _epsilon_greedy_tabular(
                q_local, state_key, cfg.action_dim, eps, rng
            )
            if episode_actions is not None:
                episode_actions.append(int(action))
            next_state, reward, done, info = env.step(action)
            next_key = discretize_state(next_state, cell=cfg.cell_size)

            total_reward += reward
            steps += 1

            # Track progress for early termination
            if hasattr(env, "_last_goal_distance") and env._last_goal_distance is not None:
                curr_goal_distance = float(env._last_goal_distance)
                if last_goal_distance is not None:
                    if (last_goal_distance - curr_goal_distance) < cfg.progress_tol:
                        no_progress += 1
                    else:
                        no_progress = 0
                last_goal_distance = curr_goal_distance

            episode_transitions.append(
                (state_key, int(action), float(reward), next_key, bool(done))
            )

            state_key = next_key

            if done or no_progress >= cfg.no_progress_patience:
                success = done and getattr(env, "_goal_reached", False)
                break

        episode_time = time.time() - ep_start

        queue.put(
            {
                "type": "episode_end",
                "worker_id": worker_id,
                "episode_idx": ep,
                "steps": steps,
                "reward": total_reward,
                "success": success,
                "episode_time": episode_time,
                "actions": episode_actions,
                "transitions": episode_transitions,
            }
        )

    env.close()


def run_parallel_training(cfg: ParallelConfig):
    """Main learner process that collects transitions and updates Q-table."""
    ctx = mp.get_context("spawn")  # safer on macOS
    queue = ctx.Queue(maxsize=10000)

    # Pipes for param broadcast
    # Pipe(duplex=False) returns (read_conn, write_conn)
    # Parent needs write_conn (to send), child needs read_conn (to receive)
    child_conns, parent_conns = zip(
        *[ctx.Pipe(duplex=False) for _ in range(cfg.num_workers)]
    )

    # Canonical Q-table in main proc
    q_table = defaultdict(lambda: np.zeros(cfg.action_dim, dtype=np.float32))

    # Logging setup
    training_params = {
        "episodes": cfg.episodes,
        "max_steps": cfg.max_steps,
        "num_workers": cfg.num_workers,
        "alpha_start": cfg.alpha_start,
        "alpha_end": cfg.alpha_end,
        "gamma": cfg.gamma,
        "eps_start": cfg.eps_start,
        "eps_end": cfg.eps_end,
        "cell_size": cfg.cell_size,
        "seed": cfg.seed,
        "ray_stop_tol": cfg.ray_stop_tol,
        "no_progress_patience": cfg.no_progress_patience,
        "progress_tol": cfg.progress_tol,
        "forward_distance": cfg.forward_distance,
    }

    log_path = _get_log_path(cfg.checkpoint_path)
    actions_path = _get_actions_path(cfg.checkpoint_path) if cfg.save_actions else None
    if log_path:
        _init_log_file(log_path, training_params)
        print(f"Logging to: {log_path}")
    
    # Initialize actions file if checkpoint path is provided
    all_episode_actions = {}
    if actions_path:
        actions_dir = os.path.dirname(actions_path)
        if actions_dir and not os.path.exists(actions_dir):
            os.makedirs(actions_dir, exist_ok=True)
        print(f"Action sequences will be saved to: {actions_path}")

    # Start workers
    procs = []
    for wid in range(cfg.num_workers):
        p = ctx.Process(
            target=actor_process,
            args=(wid, cfg, child_conns[wid], queue),
        )
        p.start()
        procs.append(p)

    global_step = 0
    successes = 0
    episode_lengths = []
    episode_rewards = []

    # Main learner loop
    total_episodes = cfg.episodes
    finished_episodes = 0
    training_start_time = time.time()
    last_print_time = training_start_time
    last_print_episode = 0

    print(f"Starting parallel training with {cfg.num_workers} workers...")
    print(f"Total episodes: {total_episodes}")

    try:
        while finished_episodes < total_episodes:
            msg = queue.get()

            if msg["type"] == "transition":
                transitions = [
                    (
                        msg["state"],
                        msg["action"],
                        msg["reward"],
                        msg["next_state"],
                        msg["done"],
                    )
                ]

                total_steps = cfg.episodes * cfg.max_steps
                for s, a, r, s_next, done in transitions:
                    alpha = _linear_schedule(
                        global_step, total_steps, cfg.alpha_start, cfg.alpha_end
                    )
                    best_next = float(np.max(q_table[s_next]))
                    td_target = r + cfg.gamma * best_next * (0.0 if done else 1.0)
                    td_error = td_target - q_table[s][a]
                    q_table[s][a] += alpha * td_error
                    global_step += 1

            elif msg["type"] == "episode_end":
                transitions = msg.get("transitions", [])
                total_steps = cfg.episodes * cfg.max_steps
                for s, a, r, s_next, done in transitions:
                    alpha = _linear_schedule(
                        global_step, total_steps, cfg.alpha_start, cfg.alpha_end
                    )
                    best_next = float(np.max(q_table[s_next]))
                    td_target = r + cfg.gamma * best_next * (0.0 if done else 1.0)
                    td_error = td_target - q_table[s][a]
                    q_table[s][a] += alpha * td_error
                    global_step += 1

                finished_episodes += 1
                steps = msg["steps"]
                rew = msg["reward"]
                success = msg["success"]
                episode_actions = msg.get("actions", [])
                episode_lengths.append(steps)
                episode_rewards.append(rew)
                if success:
                    successes += 1
                
                # Save episode actions (use finished_episodes as episode number)
                if actions_path and episode_actions:
                    all_episode_actions[finished_episodes] = episode_actions
                    should_flush_actions = (
                        cfg.action_log_interval > 0
                        and finished_episodes % cfg.action_log_interval == 0
                    )
                    if should_flush_actions:
                        with open(actions_path, "w") as f:
                            json.dump(all_episode_actions, f)

                # Compute logging stats
                success_rate = (successes / max(1, finished_episodes)) * 100.0
                avg_recent_len = (
                    np.mean(episode_lengths[-100:]) if episode_lengths else 0.0
                )
                eps_current = _linear_schedule(
                    finished_episodes, cfg.episodes, cfg.eps_start, cfg.eps_end
                )
                # Use current alpha based on global_step
                total_steps = cfg.episodes * cfg.max_steps
                alpha_current = _linear_schedule(
                    global_step, total_steps, cfg.alpha_start, cfg.alpha_end
                )

                elapsed_time = time.time() - training_start_time

                if log_path:
                    _log_episode(
                        log_path=log_path,
                        episode=finished_episodes,
                        steps=steps,
                        reward=rew,
                        epsilon=eps_current,
                        alpha=alpha_current,
                        success=success,
                        success_rate=success_rate,
                        avg_recent_len=avg_recent_len,
                        episode_time=msg["episode_time"],
                        elapsed_time=elapsed_time,
                    )

                # Periodic checkpoint
                if (
                    cfg.checkpoint_path
                    and finished_episodes % cfg.checkpoint_interval == 0
                ):
                    _save_checkpoint(
                        q_table=q_table,
                        episode=finished_episodes,
                        checkpoint_path=cfg.checkpoint_path,
                        cell_size=cfg.cell_size,
                        episode_lengths=episode_lengths,
                        episode_rewards=episode_rewards,
                        successes=successes,
                        training_params=training_params,
                    )
                    print(
                        f"Checkpoint saved at episode {finished_episodes}/{total_episodes}"
                    )

                # Optional: broadcast updated Q-table snapshot every so often
                if cfg.broadcast_every > 0 and finished_episodes % cfg.broadcast_every == 0:
                    snapshot = dict(q_table)  # Convert defaultdict to dict for sending
                    print(f"Broadcasting Q-table snapshot ({len(snapshot)} states) to {cfg.num_workers} workers...")
                    
                    # Send asynchronously in a thread to avoid blocking the main loop
                    def broadcast_snapshot(snap, conns):
                        for pc in conns:
                            try:
                                pc.send(snap)
                            except (BrokenPipeError, OSError):
                                pass  # Worker may have finished
                    
                    broadcast_thread = threading.Thread(
                        target=broadcast_snapshot, args=(snapshot, parent_conns)
                    )
                    broadcast_thread.daemon = True
                    broadcast_thread.start()
                    # Don't wait for completion - let it run in background
                    print("Broadcast initiated (async).")

                # Progress update
                if finished_episodes % 10 == 0:
                    current_time = time.time()
                    episodes_since_last_print = finished_episodes - last_print_episode
                    time_since_last_print = current_time - last_print_time
                    time_per_episode = time_since_last_print / episodes_since_last_print if episodes_since_last_print > 0 else 0.0
                    
                    # Format total elapsed time nicely
                    total_elapsed = elapsed_time
                    hours = int(total_elapsed // 3600)
                    minutes = int((total_elapsed % 3600) // 60)
                    seconds = int(total_elapsed % 60)
                    if hours > 0:
                        elapsed_str = f"{hours}h{minutes}m{seconds}s"
                    elif minutes > 0:
                        elapsed_str = f"{minutes}m{seconds}s"
                    else:
                        elapsed_str = f"{seconds}s"
                    
                    print(
                        f"Episode {finished_episodes}/{total_episodes} | "
                        f"Successes: {successes} ({success_rate:.1f}%) | "
                        f"Avg length: {avg_recent_len:.1f} | "
                        f"ε: {eps_current:.3f} | α: {alpha_current:.3f} | "
                        f"Time/ep: {time_per_episode:.2f}s | "
                        f"Elapsed: {elapsed_str}"
                    )
                    
                    last_print_time = current_time
                    last_print_episode = finished_episodes

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Clean up workers
        for p in procs:
            p.terminate()
            p.join(timeout=1.0)
            if p.is_alive():
                p.kill()

    # Final save of actions (in case any were missed)
    if actions_path and all_episode_actions:
        with open(actions_path, "w") as f:
            json.dump(all_episode_actions, f, indent=2)
        print(f"Action sequences saved to: {actions_path}")
    
    return q_table, {
        "episode_lengths": episode_lengths,
        "episode_rewards": episode_rewards,
        "successes": successes,
        "training_params": training_params,
    }


def main():
    parser = argparse.ArgumentParser(description="Train tabular Q-learning on maze environment (parallel version).")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum environment steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the trainer.")
    parser.add_argument("--forward-distance", type=float, default=0.15, help="Per-step forward translation distance for the environment.")
    parser.add_argument("--tabular-alpha-start", type=float, default=0.5, help="Initial learning rate for tabular Q-learning.")
    parser.add_argument("--tabular-alpha-end", type=float, default=0.05, help="Final learning rate for tabular Q-learning.")
    parser.add_argument("--tabular-eps-start", type=float, default=1.0, help="Initial epsilon for tabular epsilon-greedy policy.")
    parser.add_argument("--tabular-eps-end", type=float, default=0.05, help="Final epsilon for tabular epsilon-greedy policy.")
    parser.add_argument("--tabular-gamma", type=float, default=0.99, help="Discount factor for tabular Q-learning.")
    parser.add_argument("--tabular-cell-size", type=float, default=0.01, help="Discretization cell size (meters) for tabular state mapping.")
    parser.add_argument("--tabular-no-progress-patience", type=int, default=5000, help="Terminate episode early if progress stagnates for this many steps.")
    parser.add_argument("--tabular-progress-tol", type=float, default=0.01, help="Required improvement in goal distance to reset no-progress counter.")
    parser.add_argument("--tabular-ray-stop-tol", type=float, default=0.01, help="Clearance threshold used by tabular training when checking wall proximity.")
    parser.add_argument("--save-model", type=str, default=None, help="Path to save the trained model (pickle format).")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save a checkpoint every N episodes during training. Checkpoints are saved as <save-model>_ep<N>.pkl")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of parallel actor processes.")
    parser.add_argument("--broadcast-every", type=int, default=60, help="Episodes between sending Q-table snapshots to actors. Use 0 to disable broadcasts.")
    parser.add_argument("--save-actions", action="store_true", help="Save per-episode action sequences for replay. Disabled by default because it is expensive on long runs.")
    parser.add_argument("--action-log-every", type=int, default=None, help="Flush actions JSON every N episodes. Defaults to checkpoint interval; use 0 to write only at the end.")
    args = parser.parse_args()

    # Set default seed if not provided
    if args.seed is None:
        args.seed = int(time.time()) % (2**31)

    # Create a test env to get action_dim
    test_env = MazeEnv(gui=False, forward_distance=args.forward_distance)
    action_dim = test_env.action_dim
    test_env.close()

    # Build ParallelConfig
    cfg = ParallelConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
        alpha_start=args.tabular_alpha_start,
        alpha_end=args.tabular_alpha_end,
        gamma=args.tabular_gamma,
        eps_start=args.tabular_eps_start,
        eps_end=args.tabular_eps_end,
        cell_size=args.tabular_cell_size,
        seed=args.seed,
        checkpoint_path=args.save_model if args.save_model else None,
        ray_stop_tol=args.tabular_ray_stop_tol,
        no_progress_patience=args.tabular_no_progress_patience,
        progress_tol=args.tabular_progress_tol,
        forward_distance=args.forward_distance,
        action_dim=action_dim,
        checkpoint_interval=args.checkpoint_every,
        broadcast_every=args.broadcast_every,
        save_actions=args.save_actions,
        action_log_interval=(
            args.checkpoint_every
            if args.action_log_every is None
            else args.action_log_every
        ),
    )

    print("Starting parallel tabular Q-learning training...")
    print(f"Configuration: {cfg.num_workers} workers, {cfg.episodes} episodes")

    results = run_parallel_training(cfg)
    q_table, stats = results

    if args.save_model:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(args.save_model)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Save final Q-table and training metadata
        save_data = {
            "q_table": dict(q_table),  # Convert defaultdict to dict for saving
            "cell_size": cfg.cell_size,
            "episode_lengths": stats["episode_lengths"],
            "episode_rewards": stats["episode_rewards"],
            "successes": stats["successes"],
            "training_params": stats["training_params"],
        }
        with open(args.save_model, "wb") as f:
            pickle.dump(save_data, f)
        print(f"Final model saved to {args.save_model}")

    # Print summary
    trailing = stats["episode_lengths"][-50:] if stats["episode_lengths"] else []
    trailing_avg = float(np.mean(trailing)) if trailing else float("nan")
    print(
        f"\nParallel training done. Successes: {stats['successes']}/{cfg.episodes}. "
        f"Avg episode length (last 50): {trailing_avg:.1f}"
    )


if __name__ == "__main__":
    main()
