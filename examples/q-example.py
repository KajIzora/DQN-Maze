import numpy as np
import matplotlib.pyplot as plt
import csv

# ============================================================
# Q-learning Gridworld Demo
# ------------------------------------------------------------
# - 5x5 grid with a start and goal
# - Agent learns with tabular Q-learning
# - Tweak the parameters below (ALPHA, GAMMA, EPSILON, etc.)
# - At the end, you'll see:
#     * Greedy policy (arrows)
#     * A sample greedy path
#     * Q-value evolution for tracked states over episodes
#     * Value function snapshots at different training stages
#     * Q-table statistics (mean, max, min) over training
#     * Final heatmap of V(s) = max_a Q(s,a)
# ============================================================

# ------------------------
# CONFIG: tweak these knobs
# ------------------------
GRID_HEIGHT = 5
GRID_WIDTH  = 5

START = (4, 0)    # (row, col)
GOAL  = (0, 4)

# Walls the agent cannot move into
WALLS = {(1, 1), (1, 2), (2, 1)}

ALPHA    = 0.1    # learning rate
GAMMA    = 0.99   # discount factor
EPSILON  = 0.2    # exploration rate
EPISODES = 400    # number of training episodes
MAX_STEPS = 50    # max steps per episode

STEP_REWARD = -0.01   # encourages shorter paths
GOAL_REWARD = 1.0
WALL_REWARD = -0.1    # bumping into a wall / invalid move

# ------------------------
# Helper functions
# ------------------------
def state_to_idx(row: int, col: int) -> int:
    return row * GRID_WIDTH + col

def idx_to_state(idx: int):
    return divmod(idx, GRID_WIDTH)

def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH

def step_env(state, action):
    """One step in the gridworld.
    Actions: 0=up, 1=right, 2=down, 3=left
    Returns: next_state, reward, done
    """
    row, col = state

    # If already at terminal state, stay there
    if state == GOAL:
        return state, 0.0, True

    # Propose movement
    if action == 0:      # up
        new_row, new_col = row - 1, col
    elif action == 1:    # right
        new_row, new_col = row, col + 1
    elif action == 2:    # down
        new_row, new_col = row + 1, col
    elif action == 3:    # left
        new_row, new_col = row, col - 1
    else:
        raise ValueError("Invalid action")

    # Check legality
    if (not in_bounds(new_row, new_col)) or ((new_row, new_col) in WALLS):
        # Illegal move: stay put and give penalty
        reward = STEP_REWARD + WALL_REWARD
        return (row, col), reward, False

    # Legal move
    next_state = (new_row, new_col)
    if next_state == GOAL:
        reward = GOAL_REWARD
        done = True
    else:
        reward = STEP_REWARD
        done = False

    return next_state, reward, done

# ------------------------
# Q-learning training loop
# ------------------------
def train_q():
    n_states = GRID_HEIGHT * GRID_WIDTH
    n_actions = 4  # up, right, down, left

    Q = np.zeros((n_states, n_actions), dtype=float)
    
    # Track Q-table evolution over time
    snapshot_interval = max(1, EPISODES // 5)  # Take 5 snapshots
    Q_snapshots = []  # Store Q-table at different episodes
    snapshot_episodes = []
    
    # Track Q-values for specific states over all episodes
    # Track START state and a few states near the goal
    tracked_states = [
        START,
        (0, 3),  # One step before goal
        (1, 4),  # Below goal
        (3, 0),  # Near start
    ]
    tracked_state_indices = [state_to_idx(*s) for s in tracked_states]
    # Create a mapping from state tuple to Q_history
    Q_history = {s_idx: {a: [] for a in range(n_actions)} 
                 for s_idx in tracked_state_indices}
    episode_numbers = []

    for ep in range(EPISODES):
        state = START
        for t in range(MAX_STEPS):
            s_idx = state_to_idx(*state)

            # ε-greedy action selection
            if np.random.rand() < EPSILON:
                action = np.random.randint(n_actions)
            else:
                action = int(np.argmax(Q[s_idx]))

            next_state, reward, done = step_env(state, action)
            ns_idx = state_to_idx(*next_state)

            # TD target
            if done:
                td_target = reward
            else:
                td_target = reward + GAMMA * np.max(Q[ns_idx])

            # TD error
            td_error = td_target - Q[s_idx, action]

            # Q update
            Q[s_idx, action] += ALPHA * td_error

            state = next_state
            if done:
                break
        
        # Record Q-values for tracked states at end of each episode
        if ep % 10 == 0:  # Sample every 10 episodes to avoid too much data
            episode_numbers.append(ep)
            for s_idx in tracked_state_indices:
                for a in range(n_actions):
                    Q_history[s_idx][a].append(Q[s_idx, a])
        
        # Take snapshots at regular intervals
        if ep % snapshot_interval == 0 or ep == EPISODES - 1:
            Q_snapshots.append(Q.copy())
            snapshot_episodes.append(ep)

    return Q, Q_snapshots, snapshot_episodes, Q_history, episode_numbers

# ------------------------
# Visualization helpers
# ------------------------
def greedy_policy(Q):
    n_states, _ = Q.shape
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        policy[s] = int(np.argmax(Q[s]))
    return policy

def print_policy(policy):
    arrow_for_action = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    print("Greedy policy after training:")
    for r in range(GRID_HEIGHT):
        row_chars = []
        for c in range(GRID_WIDTH):
            if (r, c) == GOAL:
                row_chars.append("G")
            elif (r, c) == START:
                row_chars.append("S")
            elif (r, c) in WALLS:
                row_chars.append("█")
            else:
                s_idx = state_to_idx(r, c)
                a = policy[s_idx]
                row_chars.append(arrow_for_action[a])
        print(" ".join(row_chars))
    print()

def plot_values(Q):
    """Plot V(s) = max_a Q(s,a) as a heatmap."""
    V = np.max(Q, axis=1).reshape(GRID_HEIGHT, GRID_WIDTH)
    fig, ax = plt.subplots()
    im = ax.imshow(V)
    ax.set_title("State values V(s) = max_a Q(s,a)")
    ax.set_xticks(range(GRID_WIDTH))
    ax.set_yticks(range(GRID_HEIGHT))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

def plot_q_evolution(Q_history, episode_numbers, tracked_states):
    """Plot how Q-values for specific states change over episodes."""
    action_names = ["Up", "Right", "Down", "Left"]
    colors = ['blue', 'green', 'red', 'orange']
    
    n_states = len(tracked_states)
    fig, axes = plt.subplots(n_states, 1, figsize=(10, 3 * n_states))
    if n_states == 1:
        axes = [axes]
    
    # Match tracked_states with their indices in Q_history
    for idx, state in enumerate(tracked_states):
        s_idx = state_to_idx(*state)
        ax = axes[idx]
        for a in range(4):
            ax.plot(episode_numbers, Q_history[s_idx][a], 
                   label=f"{action_names[a]}", color=colors[a], linewidth=2)
        
        ax.set_title(f"Q-values for state {state} (row, col)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Q-value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_value_snapshots(Q_snapshots, snapshot_episodes):
    """Plot value function snapshots at different training stages."""
    n_snapshots = len(Q_snapshots)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4))
    if n_snapshots == 1:
        axes = [axes]
    
    # Find global min/max for consistent color scale
    all_values = [np.max(Q, axis=1) for Q in Q_snapshots]
    vmin = min(np.min(v) for v in all_values)
    vmax = max(np.max(v) for v in all_values)
    
    for idx, (Q, ep) in enumerate(zip(Q_snapshots, snapshot_episodes)):
        V = np.max(Q, axis=1).reshape(GRID_HEIGHT, GRID_WIDTH)
        im = axes[idx].imshow(V, vmin=vmin, vmax=vmax)
        axes[idx].set_title(f"Episode {ep}")
        axes[idx].set_xticks(range(GRID_WIDTH))
        axes[idx].set_yticks(range(GRID_HEIGHT))
        axes[idx].set_xlabel("Column")
        axes[idx].set_ylabel("Row")
        plt.colorbar(im, ax=axes[idx])
    
    plt.suptitle("Value Function Evolution: V(s) = max_a Q(s,a)", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_average_q_value(Q_snapshots, snapshot_episodes):
    """Plot average Q-value across all states over training."""
    avg_q_values = [np.mean(Q) for Q in Q_snapshots]
    max_q_values = [np.max(Q) for Q in Q_snapshots]
    min_q_values = [np.min(Q) for Q in Q_snapshots]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(snapshot_episodes, avg_q_values, 'b-', linewidth=2, label='Mean Q-value')
    ax.plot(snapshot_episodes, max_q_values, 'g--', linewidth=1.5, label='Max Q-value')
    ax.plot(snapshot_episodes, min_q_values, 'r--', linewidth=1.5, label='Min Q-value')
    ax.fill_between(snapshot_episodes, min_q_values, max_q_values, alpha=0.2, color='gray')
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Q-value")
    ax.set_title("Q-table Statistics Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def run_greedy_episode(Q, max_steps=30):
    """Run one greedy episode and print the path."""
    state = START
    path = [state]
    for _ in range(max_steps):
        if state == GOAL:
            break
        s_idx = state_to_idx(*state)
        action = int(np.argmax(Q[s_idx]))
        next_state, _, done = step_env(state, action)
        path.append(next_state)
        state = next_state
        if done:
            break
    print("Example greedy path from START to GOAL (row, col):")
    print(path)
    print()

def export_q_table_to_csv(Q, filename="q_table.csv"):
    """Export the Q-table to a CSV file.
    
    Format: Each row represents a state with columns:
    - row, col: State coordinates
    - Q_Up, Q_Right, Q_Down, Q_Left: Q-values for each action
    - V_value: max_a Q(s,a) (value function)
    - best_action: The action with the highest Q-value
    - is_start: Whether this is the start state
    - is_goal: Whether this is the goal state
    - is_wall: Whether this is a wall state
    """
    n_states = GRID_HEIGHT * GRID_WIDTH
    action_names = ["Up", "Right", "Down", "Left"]
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['row', 'col', 'Q_Up', 'Q_Right', 'Q_Down', 'Q_Left', 
                  'V_value', 'best_action', 'is_start', 'is_goal', 'is_wall']
        writer.writerow(header)
        
        # Write data for each state
        for s_idx in range(n_states):
            row, col = idx_to_state(s_idx)
            state = (row, col)
            
            # Get Q-values for this state
            q_values = Q[s_idx]
            v_value = np.max(q_values)
            best_action_idx = int(np.argmax(q_values))
            best_action = action_names[best_action_idx]
            
            # Determine state type
            is_start = (state == START)
            is_goal = (state == GOAL)
            is_wall = (state in WALLS)
            
            # Write row (round Q-values to 2 decimal places)
            row_data = [
                row, col,
                round(q_values[0], 2),  # Q_Up
                round(q_values[1], 2),  # Q_Right
                round(q_values[2], 2),  # Q_Down
                round(q_values[3], 2),  # Q_Left
                round(v_value, 2),      # V_value
                best_action,
                is_start,
                is_goal,
                is_wall
            ]
            writer.writerow(row_data)
    
    print(f"Q-table exported to {filename}")
    print()

if __name__ == "__main__":
    print("Training Q-learning agent...")
    Q, Q_snapshots, snapshot_episodes, Q_history, episode_numbers = train_q()
    
    # Define tracked states for visualization (same as in train_q)
    tracked_states = [
        START,
        (0, 3),  # One step before goal
        (1, 4),  # Below goal
        (3, 0),  # Near start
    ]
    
    print("\n" + "="*60)
    print("VISUALIZING Q-TABLE EVOLUTION")
    print("="*60 + "\n")
    
    # Show final results
    policy = greedy_policy(Q)
    print_policy(policy)
    run_greedy_episode(Q)
    
    # Export Q-table to CSV
    export_q_table_to_csv(Q, "q_table.csv")
    
    # Show Q-table evolution visualizations
    print("Plotting Q-value evolution for tracked states...")
    plot_q_evolution(Q_history, episode_numbers, tracked_states)
    
    print("Plotting value function snapshots at different training stages...")
    plot_value_snapshots(Q_snapshots, snapshot_episodes)
    
    print("Plotting Q-table statistics over training...")
    plot_average_q_value(Q_snapshots, snapshot_episodes)
    
    print("Plotting final value function...")
    plot_values(Q)
