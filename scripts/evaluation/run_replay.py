#!/usr/bin/env python3
"""Replay saved action sequences in the maze environment with GUI visualization."""
import argparse
import json
import time

from maze_rl.maze_env import MazeEnv


def load_actions(actions_path: str) -> dict:
    """Load action sequences from JSON file."""
    with open(actions_path, "r") as f:
        return json.load(f)


def replay_episode(
    env: MazeEnv,
    actions: list[int],
    episode_num: int,
    delay: float = 0.1,
    verbose: bool = True,
) -> dict:
    """Replay a single episode's action sequence."""
    state = env.reset()
    total_reward = 0.0
    steps = 0
    trajectory = [state.copy()]
    
    if verbose:
        print(f"\n=== Replaying Episode {episode_num} ({len(actions)} actions) ===")
    
    for step_idx, action in enumerate(actions):
        if action not in (0, 1, 2, 3):
            print(f"Warning: Invalid action {action} at step {step_idx}, skipping")
            continue
        
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        trajectory.append(next_state.copy())
        
        if verbose:
            print(
                f"Step {steps}: action={action}, "
                f"pos=({next_state[0]:.3f}, {next_state[1]:.3f}), "
                f"yaw={next_state[2]:.3f}, reward={reward:.2f}, "
                f"collision={info.get('collision', False)}"
            )
        
        # Add delay for visualization
        if delay > 0:
            time.sleep(delay)
        
        if done:
            if verbose:
                if env._goal_reached:
                    print(f"✓ Goal reached in {steps} steps! Total reward: {total_reward:.2f}")
                else:
                    print(f"Episode terminated at step {steps}. Total reward: {total_reward:.2f}")
            break
    
    return {
        "steps": steps,
        "total_reward": total_reward,
        "success": env._goal_reached if hasattr(env, "_goal_reached") else False,
        "trajectory": trajectory,
    }


def main():
    parser = argparse.ArgumentParser(description="Replay saved action sequences in the maze environment.")
    parser.add_argument("--actions", type=str, required=True, help="Path to the actions JSON file (e.g., models/tabular/actions.json).")
    parser.add_argument("--episode", type=int, default=None, help="Specific episode number to replay. If not provided, replays all episodes.")
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI rendering (visualize the robot).")
    parser.add_argument("--forward-distance", type=float, default=0.15, help="Per-step forward translation distance for the environment (should match training).")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between steps in seconds (for visualization). Set to 0 for fastest replay.")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity.")
    args = parser.parse_args()
    
    # Load action sequences
    print(f"Loading actions from {args.actions}...")
    try:
        all_actions = load_actions(args.actions)
    except FileNotFoundError:
        print(f"Error: Actions file not found: {args.actions}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in actions file: {e}")
        return
    
    print(f"Loaded {len(all_actions)} episode(s)")
    
    # Create environment
    env = MazeEnv(gui=args.gui, forward_distance=args.forward_distance)
    
    # Determine which episodes to replay
    if args.episode is not None:
        # JSON keys are strings, so convert episode number to string for lookup
        episode_key = str(args.episode)
        if episode_key not in all_actions:
            print(f"Error: Episode {args.episode} not found in actions file.")
            print(f"Available episodes: {sorted([int(k) for k in all_actions.keys()])}")
            env.close()
            return
        episodes_to_replay = [args.episode]
    else:
        episodes_to_replay = sorted([int(ep) for ep in all_actions.keys()])
        print(f"Replaying all {len(episodes_to_replay)} episodes...")
    
    # Replay episodes
    results = []
    for ep_num in episodes_to_replay:
        actions = all_actions[str(ep_num)]  # JSON keys are strings
        result = replay_episode(
            env,
            actions,
            ep_num,
            delay=args.delay,
            verbose=not args.quiet,
        )
        results.append((ep_num, result))
        
        if not args.quiet:
            print()
    
    # Print summary
    print("=" * 60)
    print("Replay Summary:")
    print(f"  Episodes replayed: {len(results)}")
    successes = sum(1 for _, r in results if r["success"])
    print(f"  Successful episodes: {successes}/{len(results)}")
    if results:
        avg_steps = sum(r["steps"] for _, r in results) / len(results)
        avg_reward = sum(r["total_reward"] for _, r in results) / len(results)
        print(f"  Average steps per episode: {avg_steps:.1f}")
        print(f"  Average reward per episode: {avg_reward:.2f}")
    
    env.close()


if __name__ == "__main__":
    main()

