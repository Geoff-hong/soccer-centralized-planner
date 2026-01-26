import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 配置参数
# ==========================================
DATA_PATH = "data/metrica_match2_train_attacker_centric.npz"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 500  # 过拟合需要多跑一点
D_MODEL = 64  # 为了测试，模型弄小一点
N_HEAD = 4
N_LAYERS = 2
KICK_ZONES = 12

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. 定义模型 (Simple Transformer Policy)
# ==========================================
class SoccerPolicy(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        # 1. Embedding
        # Input features: [x, y, has_ball] -> 3 dims
        self.input_proj = nn.Linear(3, d_model)
        
        # Positional Encoding: 代表 "身份" (Agent Identity)
        # 我们有 23 个 slot (11 home + 11 away + 1 ball)
        self.agent_pos_embed = nn.Parameter(torch.randn(1, 23, d_model))
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Action Heads (只预测前 11 个 agent)
        # Head A: Movement (Vx, Vy) -> Continuous
        self.head_move = nn.Linear(d_model, 2)
        
        # Head B: Kick Trigger (Logit) -> Binary
        self.head_trigger = nn.Linear(d_model, 1)
        
        # Head C: Kick Direction (Logits) -> Multi-class
        self.head_kick = nn.Linear(d_model, KICK_ZONES)

    def forward(self, x):
        # x shape: [Batch, 23, 3]
        
        # Embedding
        x = self.input_proj(x)  # [B, 23, 64]
        x = x + self.agent_pos_embed # Add Identity context
        
        # Transformer Processing
        feat = self.transformer(x) # [B, 23, 64]
        
        # 只取前 11 个 (Teammates) 进行动作预测
        teammate_feat = feat[:, :11, :] # [B, 11, 64]
        
        # Outputs
        pred_vel = self.head_move(teammate_feat)       # [B, 11, 2]
        pred_trigger = self.head_trigger(teammate_feat)# [B, 11, 1]
        pred_kick = self.head_kick(teammate_feat)      # [B, 11, 12]
        
        return pred_vel, pred_trigger, pred_kick

# ==========================================
# 3. 数据加载
# ==========================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Run data_processing.py first.")

data = np.load(DATA_PATH)
obs_all = data['obs'] # [N, 23, 3]
acts_all = data['acts'] # [N, 11, 4]

# --- 关键：为了验证 Pipeline，只取前 1000 帧进行过拟合 ---
# 这样我们可以快速验证模型是否具备“记忆”能力
LIMIT = 1000
obs_train = torch.FloatTensor(obs_all[:LIMIT]).to(device)
acts_train = torch.FloatTensor(acts_all[:LIMIT]).to(device)

print(f"Training Data Shape: Obs {obs_train.shape}, Acts {acts_train.shape}")

dataset = TensorDataset(obs_train, acts_train)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 4. 训练循环 (Training Loop)
# ==========================================
model = SoccerPolicy(d_model=D_MODEL, nhead=N_HEAD, num_layers=N_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Loss Functions
criterion_mse = nn.MSELoss() # For velocity
# Pos_weight: 踢球是稀疏事件(1%的时间)，如果不加权重，模型会全猜0
# 这里简单给一个权重，比如 10 倍
pos_weight = torch.tensor([100.0]).to(device) 
criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # For trigger
criterion_ce = nn.CrossEntropyLoss() # For kick direction

loss_history = []

print("\nStarting Overfitting Sanity Check...")

for epoch in range(EPOCHS):
    total_loss = 0
    
    for batch_obs, batch_acts in dataloader:
        optimizer.zero_grad()
        
        # Forward
        # batch_acts shape: [B, 11, 4] -> vx, vy, trigger, kick_cls
        pred_vel, pred_trigger, pred_kick = model(batch_obs)
        
        # --- Targets ---
        target_vel = batch_acts[:, :, 0:2]     # [B, 11, 2]
        target_trigger = batch_acts[:, :, 2:3] # [B, 11, 1]
        target_kick = batch_acts[:, :, 3].long() # [B, 11] (Indices)
        
        # --- Loss A: Movement (MSE) ---
        loss_move = criterion_mse(pred_vel, target_vel)
        
        # --- Loss B: Trigger (BCE) ---
        loss_trigger = criterion_bce(pred_trigger, target_trigger)
        
        # --- Loss C: Kick Direction (Masked CE) ---
        # 关键：只计算那些真实数据中 trigger=1 的样本
        # Flatten everything to handle batch processing
        flat_pred_kick = pred_kick.reshape(-1, KICK_ZONES) # [B*11, 12]
        flat_target_kick = target_kick.reshape(-1)         # [B*11]
        flat_trigger = target_trigger.reshape(-1)          # [B*11]
        
        # Mask: 找出 trigger == 1 的索引
        kick_mask = (flat_trigger > 0.5)
        
        if kick_mask.sum() > 0:
            # 只有当本个 Batch 里有踢球动作时才算这个 Loss
            loss_kick = criterion_ce(flat_pred_kick[kick_mask], flat_target_kick[kick_mask])
        else:
            loss_kick = torch.tensor(0.0).to(device)
            
        # Combine
        loss = loss_move + loss_trigger + loss_kick
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.5f} | Move: {loss_move:.5f}, Trig: {loss_trigger:.5f}, Kick: {loss_kick:.5f}")

# ==========================================
# 5. 结果可视化
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title("Overfitting Loss Curve (Should go to near 0)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("output/train_loss_curve.png")
print("\nTraining Done. Loss curve saved to train_loss_curve.png")

# 简单验证一下预测
model.eval()
with torch.no_grad():
    # 找一个有 Kick 的帧来测试
    # 搜索 acts_train 里的 trigger
    idx = (acts_train[:, :, 2] > 0.5).nonzero(as_tuple=False)
    if len(idx) > 0:
        sample_idx = idx[0][0].item() # 选第一个包含 kick 的帧
        player_idx = idx[0][1].item()
        
        obs_sample = obs_train[sample_idx:sample_idx+1] # [1, 23, 3]
        p_vel, p_trig, p_kick = model(obs_sample)
        
        trig_prob = torch.sigmoid(p_trig[0, player_idx]).item()
        kick_cls = torch.argmax(p_kick[0, player_idx]).item()
        
        real_cls = acts_train[sample_idx, player_idx, 3].item()
        
        print(f"\n[Sanity Check] Frame {sample_idx}, Player {player_idx}")
        print(f" > Ground Truth: Kick Class {int(real_cls)}")
        print(f" > Prediction:   Kick Class {kick_cls} (Conf: {trig_prob:.2f})")
        
        if int(real_cls) == kick_cls and trig_prob > 0.5:
            print(" > SUCCESS: Model memorized the kick!")
        else:
            print(" > FAILURE: Model failed to memorize.")
    else:
        print("\nNo kick events found in the training subset to verify.")