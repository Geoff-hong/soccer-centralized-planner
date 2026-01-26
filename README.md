# ⚽ Kloppy Football AI: Centralized Planner

A deep learning framework for learning centralized multi-agent planning from real-world football tracking data. This project trains a neural network to predict player movements and passing decisions by learning from professional football matches.

## 📋 Overview

This library processes Metrica Sports tracking data and trains a centralized planner that can:
- **Predict player movements** based on game state
- **Decide passing actions** (when and where to pass)
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

### Output: Attacker-Centric Tensors

**Observation** `(T, 23, 3)`:
- 11 teammates: `[x, y, has_ball]`
- 11 opponents: `[x, y, has_ball]`  
- 1 ball: `[x, y, 0]`
- Coordinates normalized to `[-1, 1]` with attack direction always left-to-right

**Action** `(T, 11, 4)`:
- For each teammate: `[vx, vy, pass_trigger, pass_direction]`
- `pass_trigger`: 0/1 indicating pass action
- `pass_direction`: discretized into 12 sectors (30° each)

## 🧠 Model Architecture

The centralized planner uses:
- **Input**: Game state (23 agents × 3 features)
- **Architecture**: Multi-layer Transformer with attention over all agents
- **Output**: 
  - Movement velocities for all teammates
  - Pass trigger probability
  - Pass direction (12-class classification)

## 📈 Key Features

### 1. Geometry-Based Event Inference (`track2event.py`)
Automatically detects pass events using:
- **State machine** for possession detection
- **Distance thresholds** (2m possession radius)
- **Temporal smoothing** to filter noise

### 2. Attacker-Centric Perspective (`data_processing.py`)
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

## 📝 Usage Example

```python
# Load processed data
import numpy as np
data = np.load('data/metrica_match1_train_attacker_centric.npz')
obs = data['obs']    # (T, 23, 3) - game states
acts = data['acts']  # (T, 11, 4) - ground truth actions

# Train model
from train.model import CentralizedPlanner
from train.dataset import FootballDataset

model = CentralizedPlanner(input_dim=3, hidden_dim=256, num_agents=23)
dataset = FootballDataset(obs, acts)
# ... training loop ...
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
