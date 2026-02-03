# ⚽ Kloppy Football AI: Centralized Planner

A deep learning framework for learning centralized multi-agent planning from real-world football tracking data. This project trains a neural network to predict player movements and passing decisions by learning from professional football matches.

## 📋 Overview

This library processes Metrica Sports and SkillCorner tracking data and trains a centralized planner that can:
- **Predict player movements** based on game state (m/s)
- **Decide passing actions** (pass/no-pass, passer, receiver)
- **Handle multi-agent coordination** in an attacker-centric framework

The trained model learns strategic patterns from real professional football matches, including positioning, spacing, and tactical decision-making.

## 🏗️ Project Structure

```
kloppy/
├── scripts/              # Data processing pipeline
│   ├── track2event.py           # Generate pass events from tracking data
│   ├── data_processing.py       # Create attacker-centric training data
│   ├── check_data.py            # Visualize processed data
│   ├── check_pass_event.py      # Verify event quality
│   ├── metrica.py               # Visualize raw tracking data
│   └── 3v3pick.py               # Extract 3v3 scenarios
│   ├── skillcorner_probe.py           # Load SkillCorner local data & inspect format
│   ├── skillcorner_track2event.py     # Generate pass events from SkillCorner dynamic events
│   ├── skillcorner_data_processing.py # Create SkillCorner attacker-centric npz
│   ├── skillcorner_batch_process.py   # Batch process SkillCorner matches
│   ├── skillcorner_compare_events.py  # Compare dynamic vs tracking-inferred events
│   ├── skillcorner_frame_viewer.py    # Frame-by-frame viewer (SkillCorner)
│   ├── eval_move_quality.py           # One-step & multi-step move eval
│   ├── eval_move_one_step.py          # Train-split one-step eval
│   ├── diag_slot_hungarian.py         # Slot confusion diagnostic (Hungarian)
│   ├── diag_slot_single.py            # Slot swap diagnostic
│   ├── diag_slot_stats.py             # Slot stats / heatmap stats
│   └── diag_act_move_consistency.py   # act_move vs Δpos consistency
│
├── train/                # Training framework
│   ├── train.py                 # Main training script
│   ├── train_overfit.py         # Overfit test on small dataset
│   ├── model.py                 # Neural network architecture
│   └── dataset.py               # PyTorch dataset loader
│
├── data/                 # Generated data (auto-created)
│   ├── metrica_match*_inferred_events.csv
│   └── metrica_match*_train_attacker_centric.npz
│   └── skillcorner_processed/
│       ├── events/
│       └── npz/
│
└── output/               # Visualizations (auto-created)
    └── *.gif
```

## 🚀 Quick Start

### Prerequisites

```bash
# Create conda environment
conda create -n kloppy python=3.10
conda activate kloppy

# Install dependencies
pip install kloppy pandas numpy matplotlib mplsoccer torch tqdm wandb
```

### Data Processing Pipeline

Run these scripts in order from the `scripts/` directory:

```bash
cd kloppy/scripts

# Step 1: Generate pass events from tracking data (geometry-based)
python track2event.py

# Step 2: Create attacker-centric training tensors
python data_processing.py

# Step 3: Verify data quality (optional)
python check_data.py
python check_pass_event.py
```

### SkillCorner Batch Pipeline (local open-data)

```bash
cd kloppy/scripts

# Generate dynamic-event passes, then npz (10 matches)
python skillcorner_batch_process.py --sample_rate 0.1
```

Outputs go to:
- `data/skillcorner_processed/events/*.csv`
- `data/skillcorner_processed/npz/*.npz`

### Training

```bash
cd kloppy/train

# Train the model
python train.py

# Test overfitting capability (optional)
python train_overfit.py
```

## 📊 Data Format

### Input: Metrica Tracking Data
- **Source**: Metrica Sports open dataset
- **Format**: 25 fps tracking data with x,y positions for all 22 players + ball
- **Coordinate System**: 105m × 68m pitch

### Output: Attacker-Centric Tensors (current format)

**Observation** `(T, 23, 3)`:
- 11 teammates: `[x, y, has_ball]`
- 11 opponents: `[x, y, has_ball]`  
- 1 ball: `[x, y, 0]`
- Coordinates normalized to `[-1, 1]` with attack direction always left-to-right

**Movement**:
- `act_move`: `(T, 11, 2)` velocities in **m/s**
- `dt`: `(T,)` seconds per frame (stored in npz)

**Passing (global)**:
- `pass_flag`: `(T,)` 0/1 (pass_or_not)
- `passer_id`: `(T,)` 0..10
- `receiver_id`: `(T,)` 0..10

Notes:
- `pass_dir` and `dir_loss` were removed.
- All movement units are **m/s**; regeneration required if `move_unit` is missing.
- SkillCorner NPZs are generated via dynamic events (preferred) and include `dt`.

## 🔧 Recent Training/Model Updates (2026-02)

These changes are already integrated in `train/` and used in recent runs:

- **Move-only mode**: training can disable pass heads (`train_move_only=True`) to focus on movement first.
- **Multi-step rollout loss**: predict a horizon of velocities and integrate to positions for **ADE/FDE** supervision.
- **Residual velocity prediction**: model predicts a delta on top of last-step velocity baseline.
- **High-speed weighting**: errors for high-speed frames (e.g., >4 m/s) receive higher weight.
- **Direction & magnitude terms**: move loss includes vector MSE + magnitude error + direction loss.
- **SkillCorner pipeline**: dynamic-event-based batch processing writes to:
  - `data/skillcorner_processed/events/`
  - `data/skillcorner_processed/npz/`

Important: **eval scripts must match the training config** (e.g., `obs_history`, `move_horizon`,
residual-on/off), otherwise MSE can look wildly incorrect.

## 🧠 Model Architecture

The centralized planner uses:
- **Input**: Game state (23 agents × 3 features)
- **Architecture**: Multi-layer Transformer with attention over all agents
- **Output**: 
  - Movement velocities for all teammates (optionally multi-step horizon)
  - Pass/no-pass probability
  - Passer & receiver classification

## 📈 Key Features

### 1. Geometry-Based Event Inference (`track2event.py`)
Automatically detects pass events using:
- **State machine** for possession detection
- **Distance thresholds** (2m possession radius)
- **Temporal smoothing** to filter noise

### 2. Attacker-Centric Perspective (`data_processing.py`, `skillcorner_data_processing.py`)
- **Dynamic coordinate flipping**: Always attack left-to-right
- **Possession-based switching**: Attacker = current ball possessor's team
- **Slot management**: Handle player substitutions smoothly

### 3. Automatic Direction Detection
- Analyzes team positioning to determine attack direction
- Works for both periods (handles side-switching at halftime)
- No manual annotation needed

## 🎨 Visualization Tools

### Check Processed Data
```bash
python scripts/check_data.py
```
Visualizes normalized training data in `[-1, 1]` space with:
- Player positions and velocities
- Pass triggers and directions
- Ball position

### Check Raw Tracking
```bash
python scripts/metrica.py
```
Shows original Metrica data with coordinate axes for debugging.

### Verify Pass Events
```bash
python scripts/check_pass_event.py
```
Overlays detected pass events on tracking data to verify quality.

### SkillCorner Frame Viewer (frame-by-frame)
```bash
python scripts/skillcorner_frame_viewer.py --match_id 2017461 --start_frame 0 --num_frames 2000
```

### Model Rollout Visualization
```bash
python vis/vis.py --match_id 2017461 --checkpoint checkpoints/<run>/best_model.pth \
  --sample_rate 0.1 --start_frame 0 --num_frames 2000 --fps 10 --save_gif \
  --gif_path output/skillcorner_vis_2017461.gif
```

## ⚙️ Configuration

### Data Processing (`data_processing.py`)
```python
POSSESSION_DIST_THR = 2.0      # Ball possession distance (meters)
POSSESSION_HYSTERESIS = 5      # Frames before switching possession
LOOKAHEAD_FRAMES = 10          # Future frames for pass target
KICK_SECTORS = 12              # Pass direction discretization
```

### Visualization (`check_data.py`, `metrica.py`)
```python
MATCH_ID = 1                   # Which match to process
START_FRAME = 0                # Starting frame
NUM_FRAMES = 1000              # Number of frames to visualize
FPS = 25                       # Playback frame rate
```

## 🔬 Diagnostics / Evaluation

```bash
# One-step movement quality
python scripts/eval_move_quality.py --data data/skillcorner_processed/npz --checkpoint checkpoints/<run>/best_model.pth

# Train-split one-step eval
python scripts/eval_move_one_step.py --data data/skillcorner_processed/npz --checkpoint checkpoints/<run>/best_model.pth \
  --obs_history 5 --val_ratio 0.1 --seed 42

# Slot diagnostics
python scripts/diag_slot_hungarian.py --data data/skillcorner_processed/npz
python scripts/diag_slot_single.py --data data/skillcorner_processed/npz --out_json output/slot_swap_diag.json
python scripts/diag_act_move_consistency.py --data data/skillcorner_processed/npz --speed_thr 0.5
```

Notes:
- `eval_move_quality.py` uses the **model config** to interpret outputs (move horizon, residual).
- For rollout-based training, prefer ADE/FDE metrics in addition to one-step MSE.

## 📝 Usage Example

```python
# Load processed data
import numpy as np
data = np.load('data/metrica_match1_train_attacker_centric.npz')
obs = data['obs']           # (T, 23, 3) - game states
act_move = data['act_move'] # (T, 11, 2) - m/s
pass_flag = data['pass_flag']
passer_id = data['passer_id']
receiver_id = data['receiver_id']
dt = data['dt']

# Train model
from train.model import SoccerPolicy
from train.dataset import SoccerDataset

model = SoccerPolicy(input_dim=3, hidden_dim=256, num_agents=23)
dataset = SoccerDataset(obs_history=5, move_horizon=1, file_paths=[...])
# ... training loop ...
```

## ✅ Checkpoints

Checkpoints are saved per run:
```
checkpoints/<run_name>/best_model.pth
```

## 🔬 Research Applications

This framework can be used for:
- **Imitation Learning**: Clone professional player behavior
- **Tactical Analysis**: Extract learned patterns and strategies
- **Multi-Agent RL**: Use as initialization for policy learning
- **Game Simulation**: Generate realistic football trajectories
- **Coaching Tools**: Analyze optimal positioning and passing decisions

## 📚 Dataset Citation

This project uses the Metrica Sports open tracking data:

```
Metrica Sports. (2019). Sample tracking data. 
https://github.com/metrica-sports/sample-data
```

## 🛠️ Troubleshooting

### Common Issues

**Problem**: Events not detected
- **Solution**: Adjust `POSSESSION_DIST` threshold in `track2event.py`

**Problem**: Wrong attack direction
- **Solution**: The auto-detection analyzes team positioning. If incorrect, check `data_processing.py` flip logic.

**Problem**: Visualization shows players off-field
- **Solution**: Ensure coordinate transformation is applied (`standardized=False` for metric coordinates)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional datasets (StatsBomb, Wyscout, etc.)
- More sophisticated event detection
- Advanced model architectures (GNNs, Transformers)
- Real-time inference optimization

## 📄 License

This project is for research and educational purposes. Please cite if used in academic work.

## 🙏 Acknowledgments

- **Kloppy**: Python library for loading tracking data
- **Metrica Sports**: Open tracking dataset
- **mplsoccer**: Football pitch visualization

---

**Note**: This is a research project for learning centralized planning from real football data. The trained model captures tactical patterns from professional matches and can be used as a baseline for multi-agent coordination research.
