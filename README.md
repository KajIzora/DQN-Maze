# Maze Q-Learning (Parallel)

PyBullet simulation of a 12×12 maze with a differential-drive-style robot. Train and evaluate agents with **tabular Q-learning** (single- and multi-process) or **DQN**.

## Requirements

- Python 3.10+
- macOS/Linux recommended (PyBullet GUI); headless training works without a display

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e . --no-deps
```

Run all commands from the repository root so model paths resolve correctly.

## Project layout

```
maze-q-learn-paralell/
├── src/maze_rl/          # Core package (environment, DQN network)
├── scripts/
│   ├── training/         # Training entry points
│   ├── evaluation/       # Evaluate saved models and replay logs
│   ├── run_manual.py     # Keyboard control
│   └── run_random.py     # Random policy baseline
├── examples/             # Standalone demos (no package import needed)
└── models/
    ├── tabular/          # Tabular Q-table checkpoints and logs
    └── dqn/                # DQN checkpoints
```

| Path | Description |
|------|-------------|
| `src/maze_rl/maze_env.py` | Gym-style maze environment (PyBullet) |
| `src/maze_rl/dqn.py` | Shared DQN network and device helpers |
| `examples/maze_simple.py` | Standalone maze viewer (no RL) |
| `examples/q-example.py` | Small 5×5 gridworld Q-learning demo (matplotlib) |
| `scripts/training/run_q_tabular.py` | Tabular Q-learning training |
| `scripts/training/run_q_tabular_parallel.py` | Parallel tabular Q-learning (actor–learner) |
| `scripts/training/run_dqn.py` | Deep Q-network training (PyTorch) |
| `scripts/evaluation/run_trained_model.py` | Evaluate a saved tabular Q-table |
| `scripts/evaluation/run_trained_dqn.py` | Evaluate a saved DQN checkpoint |
| `scripts/evaluation/run_replay.py` | Replay logged action sequences |

Model checkpoints (`.pkl`) are gitignored; train locally or copy your own into `models/`.

## Environment

- **State**: position `(x, y)`, heading, and raycast distances to walls.
- **Actions** (4): go straight, turn right 90°, turn left 90°, turn around 180°.
- **Reward**: step penalty, optional collision penalty, goal bonus when entering the goal zone.

View the maze only:

```bash
python examples/maze_simple.py
```

Manual control:

```bash
python scripts/run_manual.py --gui
```

## Training

### Tabular Q-learning (single process)

```bash
python scripts/training/run_q_tabular.py \
  --save-model models/tabular/q-tabular.pkl \
  --checkpoint-every 100
```

Add `--gui` to watch training. Useful flags: `--episodes`, `--max-steps`, `--forward-distance`, `--tabular-ray-stop-tol`.

### Tabular Q-learning (parallel)

Uses multiple worker processes that collect transitions while a central learner updates the Q-table.

```bash
python scripts/training/run_q_tabular_parallel.py \
  --save-model models/tabular/q-tabular-parallel.pkl \
  --checkpoint-every 100 \
  --num-workers 10
```

### DQN

```bash
python scripts/training/run_dqn.py \
  --device mps \
  --checkpoint-every 100 \
  --max-steps 500 \
  --torch-threads 12 \
  --batch-size 128 \
  --train-every 4 \
  --print-every 25 \
  --eps-decay-episodes 750 \
  --save-model models/dqn/dqn.pkl \
  --forward-distance 0.5 \
  --episodes 1000
```

Use `--device cpu` or `--device cuda` if MPS is unavailable.

## Evaluation

Tabular policy:

```bash
python scripts/evaluation/run_trained_model.py --gui \
  --model models/tabular/q-tabular-parallel.pkl \
  --episodes 10 \
  --max-steps 1000
```

DQN policy:

```bash
python scripts/evaluation/run_trained_dqn.py --gui \
  --model models/dqn/dqn.pkl \
  --episodes 10 \
  --max-steps 10000
```

Replay saved actions (from `models/tabular/actions.json`):

```bash
python scripts/evaluation/run_replay.py --gui \
  --actions models/tabular/actions.json \
  --episode 4
```

## VS Code

`.vscode/launch.json` includes debug configurations for training, evaluation, and manual control.

## Push to GitHub

```bash
git add .
git commit -m "Initial commit: maze Q-learning with PyBullet"
git remote add origin https://github.com/YOUR_USERNAME/maze-q-learn-paralell.git
git push -u origin main
```

Replace the remote URL with your repository. After cloning elsewhere, run `pip install -r requirements.txt`, `pip install -e . --no-deps`, and the training scripts to generate `.pkl` checkpoints locally.
