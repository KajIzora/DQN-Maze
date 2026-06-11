#!/usr/bin/env python3
"""Run DQN training on the maze environment."""
import argparse
import math
import os
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from maze_rl.dqn import QNetwork, select_device
from maze_rl.maze_env import MazeEnv


# Q-learning trainer defaults
Q_TRAIN_MAX_EPISODES = 200000
Q_TRAIN_MAX_STEPS = 2000
Q_TRAIN_DISCOUNT = 0.99
Q_TRAIN_LR = 1e-3
Q_TRAIN_BATCH_SIZE = 64
Q_TRAIN_REPLAY_CAPACITY = 20000
Q_TRAIN_START_EPS = 1.0
Q_TRAIN_END_EPS = 0.05
Q_TRAIN_EPS_DECAY_EPISODES = 4000
Q_TRAIN_TARGET_UPDATE_EVERY = 10
Q_TRAIN_WARMUP_STEPS = 2000
Q_TRAIN_PRINT_EVERY = 1


def create_q_networks(
    state_dim: int,
    action_dim: int,
    *,
    hidden_sizes: tuple[int, ...] = (128, 128),
    device: Optional[torch.device] = None,
    learning_rate: float = Q_TRAIN_LR,
) -> tuple[QNetwork, QNetwork, torch.optim.Optimizer, torch.device]:
    """Instantiate policy & target networks with shared architecture."""
    device = device or select_device()
    policy = QNetwork(state_dim, action_dim, hidden_sizes).to(device)
    target = QNetwork(state_dim, action_dim, hidden_sizes).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    return policy, target, optimizer, device


def epsilon_by_episode(
    episode: int,
    *,
    start_eps: float = Q_TRAIN_START_EPS,
    end_eps: float = Q_TRAIN_END_EPS,
    decay_episodes: int = Q_TRAIN_EPS_DECAY_EPISODES,
) -> float:
    """Linearly decay epsilon over the configured horizon."""
    if episode >= decay_episodes:
        return end_eps
    slope = (end_eps - start_eps) / max(1, decay_episodes)
    return start_eps + slope * episode


def epsilon_greedy_action(
    policy_net: QNetwork,
    state: np.ndarray,
    epsilon: float,
    *,
    action_dim: int,
    device: torch.device,
    rng: np.random.Generator,
    ) -> int:
    """Sample an action using epsilon-greedy strategy."""
    if rng.random() < epsilon:
        return int(rng.integers(low=0, high=action_dim))

    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_t)
    action = int(torch.argmax(q_values, dim=1).item())
    return action


class ReplayBuffer:
    """Fixed-size replay buffer for off-policy Q-learning."""

    def __init__(self, state_dim: int, capacity: int, rng: Optional[np.random.Generator] = None):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self._states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.bool_)
        self._write_idx = 0
        self._size = 0
        self._rng = rng or np.random.default_rng()

    def __len__(self) -> int:
        return self._size

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        idx = self._write_idx
        self._states[idx] = state
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_states[idx] = next_state
        self._dones[idx] = done

        self._write_idx = (self._write_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        if self._size < batch_size:
            raise ValueError("Not enough samples in buffer.")
        idxs = self._rng.integers(low=0, high=self._size, size=batch_size)
        states = torch.as_tensor(self._states[idxs], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self._actions[idxs], dtype=torch.int64, device=device)
        rewards = torch.as_tensor(self._rewards[idxs], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(self._next_states[idxs], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self._dones[idxs], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones


def _save_dqn_checkpoint(
    policy_net: nn.Module,
    episode: int,
    checkpoint_path: str,
    episode_rewards: list[float],
    losses: list[float],
    training_params: dict,
) -> None:
    """Save a checkpoint of the DQN model and training state."""
    # Create checkpoint filename with episode number
    base_path, ext = os.path.splitext(checkpoint_path)
    checkpoint_file = f"{base_path}_ep{episode}{ext}"
    
    # Create directory if it doesn't exist
    checkpoint_dir = os.path.dirname(checkpoint_file)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    save_data = {
        "policy_net_state_dict": policy_net.state_dict(),
        "episode": episode,
        "episode_rewards": episode_rewards.copy(),
        "losses": losses.copy(),
        "training_params": training_params,
    }
    with open(checkpoint_file, "wb") as f:
        pickle.dump(save_data, f)
    print(f"  Checkpoint saved: {checkpoint_file}")


def train_q_learning(
    env: MazeEnv,
    *,
    episodes: int = Q_TRAIN_MAX_EPISODES,
    max_steps: int = Q_TRAIN_MAX_STEPS,
    gamma: float = Q_TRAIN_DISCOUNT,
    batch_size: int = Q_TRAIN_BATCH_SIZE,
    replay_capacity: int = Q_TRAIN_REPLAY_CAPACITY,
    warmup_steps: int = Q_TRAIN_WARMUP_STEPS,
    train_every: int = 1,
    target_update_every: int = Q_TRAIN_TARGET_UPDATE_EVERY,
    print_every: int = Q_TRAIN_PRINT_EVERY,
    device: Optional[torch.device] = None,
    hidden_sizes: tuple[int, ...] = (128, 128),
    learning_rate: float = Q_TRAIN_LR,
    eps_start: float = Q_TRAIN_START_EPS,
    eps_end: float = Q_TRAIN_END_EPS,
    eps_decay_episodes: int = Q_TRAIN_EPS_DECAY_EPISODES,
    seed: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: Optional[int] = None,
    training_params: Optional[dict] = None,
    ) -> dict:
    """Run a Q-learning training loop on the provided MazeEnv instance."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    initial_state = env.reset()
    state_dim = initial_state.shape[0]
    action_dim = env.action_dim

    policy_net, target_net, optimizer, device = create_q_networks(
        state_dim,
        action_dim,
        hidden_sizes=hidden_sizes,
        device=device,
        learning_rate=learning_rate,
    )
    replay_buffer = ReplayBuffer(state_dim, replay_capacity, rng=rng)
    train_every = max(1, int(train_every))

    total_steps = 0
    episode_rewards: list[float] = []
    losses: list[float] = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_epsilon = epsilon_by_episode(
            episode,
            start_eps=eps_start,
            end_eps=eps_end,
            decay_episodes=eps_decay_episodes,
        )

        for step in range(max_steps):
            action = epsilon_greedy_action(
                policy_net,
                state,
                episode_epsilon,
                action_dim=action_dim,
                device=device,
                rng=rng,
            )
            next_state, reward, done, _ = env.step(action)

            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_steps += 1

            if (
                total_steps >= warmup_steps
                and len(replay_buffer) >= batch_size
                and total_steps % train_every == 0
            ):
                batch = replay_buffer.sample(batch_size, device)
                states_t, actions_t, rewards_t, next_states_t, dones_t = batch

                with torch.no_grad():
                    next_q_values = target_net(next_states_t).max(dim=1)[0]
                    targets = rewards_t + gamma * (1.0 - dones_t) * next_q_values

                q_values = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

                losses.append(float(loss.item()))

            if done:
                break

        if (episode + 1) % target_update_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(episode_reward)

        if (episode + 1) % print_every == 0 or episode == 0:
            avg_last = np.mean(episode_rewards[-print_every:]) if episode_rewards else 0.0
            last_loss = np.mean(losses[-print_every:]) if losses else 0.0
            print(
                f"[Episode {episode + 1}/{episodes}] "
                f"epsilon={episode_epsilon:.3f} "
                f"reward={episode_reward:.2f} "
                f"avg_reward={avg_last:.2f} "
                f"loss={last_loss:.4f}"
            )
        
        # Save checkpoint if enabled and it's time
        if checkpoint_path and checkpoint_every and (episode + 1) % checkpoint_every == 0:
            _save_dqn_checkpoint(
                policy_net,
                episode + 1,
                checkpoint_path,
                episode_rewards,
                losses,
                training_params or {},
            )

    return {
        "policy_net": policy_net,
        "target_net": target_net,
        "optimizer": optimizer,
        "device": device,
        "episode_rewards": episode_rewards,
        "losses": losses,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DQN on maze environment.")
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI rendering.")
    parser.add_argument("--episodes", type=int, default=Q_TRAIN_MAX_EPISODES, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=Q_TRAIN_MAX_STEPS, help="Maximum environment steps per episode.")
    parser.add_argument("--print-every", type=int, default=Q_TRAIN_PRINT_EVERY, help="Logging interval in episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the trainer.")
    parser.add_argument("--device", type=str, default="auto", help="Preferred torch device identifier: auto, cpu, cuda, or mps.")
    parser.add_argument("--forward-distance", type=float, default=0.05, help="Per-step forward translation distance for the environment.")
    parser.add_argument("--batch-size", type=int, default=Q_TRAIN_BATCH_SIZE, help="DQN replay batch size.")
    parser.add_argument("--learning-rate", type=float, default=Q_TRAIN_LR, help="DQN optimizer learning rate.")
    parser.add_argument("--train-every", type=int, default=1, help="Run one optimizer update every N environment steps.")
    parser.add_argument("--eps-start", type=float, default=Q_TRAIN_START_EPS, help="Initial epsilon for DQN epsilon-greedy policy.")
    parser.add_argument("--eps-end", type=float, default=Q_TRAIN_END_EPS, help="Final epsilon for DQN epsilon-greedy policy.")
    parser.add_argument("--eps-decay-episodes", type=int, default=Q_TRAIN_EPS_DECAY_EPISODES, help="Number of episodes used to linearly decay epsilon.")
    parser.add_argument("--torch-threads", type=int, default=None, help="Set PyTorch CPU thread count before training.")
    parser.add_argument("--save-model", type=str, default=None, help="Path to save the trained model (pickle format).")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Save a checkpoint every N episodes during training. Checkpoints are saved as <save-model>_ep<N>.pkl")
    args = parser.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(max(1, args.torch_threads))

    device = None if args.device == "auto" else select_device(args.device)

    env = MazeEnv(gui=args.gui, forward_distance=args.forward_distance)

    print("Starting DQN training demo...")
    # Get state and action dimensions
    initial_state = env.reset()
    state_dim = initial_state.shape[0]
    action_dim = env.action_dim
    
    # Prepare training parameters for checkpoint saving
    training_params = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "gamma": Q_TRAIN_DISCOUNT,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "replay_capacity": Q_TRAIN_REPLAY_CAPACITY,
        "warmup_steps": Q_TRAIN_WARMUP_STEPS,
        "train_every": args.train_every,
        "target_update_every": Q_TRAIN_TARGET_UPDATE_EVERY,
        "eps_start": args.eps_start,
        "eps_end": args.eps_end,
        "eps_decay_episodes": args.eps_decay_episodes,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_sizes": (128, 128),  # Default hidden sizes
        "seed": args.seed,
    }
    # Only enable checkpoints if save_model is provided
    checkpoint_path = args.save_model if args.save_model else None
    checkpoint_every = args.checkpoint_every if (args.save_model and args.checkpoint_every) else None
    
    if checkpoint_every:
        print(f"Checkpoints will be saved every {checkpoint_every} episodes to {checkpoint_path}")
    
    results = train_q_learning(
        env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        train_every=args.train_every,
        print_every=args.print_every,
        device=device,
        learning_rate=args.learning_rate,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_episodes=args.eps_decay_episodes,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        training_params=training_params,
    )
    rewards = np.array(results["episode_rewards"], dtype=np.float32)
    mean_tail = float(rewards[-10:].mean()) if len(rewards) >= 10 else float(rewards.mean())
    print(
        f"Training complete. Episodes={args.episodes}, "
        f"best_reward={rewards.max():.2f}, "
        f"mean_last_10={mean_tail:.2f}"
    )
    
    if args.save_model:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(args.save_model)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Save final DQN model and training metadata
        save_data = {
            "policy_net_state_dict": results["policy_net"].state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_sizes": training_params["hidden_sizes"],
            "episode_rewards": results["episode_rewards"],
            "losses": results["losses"],
            "training_params": training_params,
        }
        with open(args.save_model, "wb") as f:
            pickle.dump(save_data, f)
        print(f"Final model saved to {args.save_model}")

    env.close()


if __name__ == "__main__":
    main()
