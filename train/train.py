import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import wandb
from tqdm import tqdm

# 引入我们刚才定义的模块
from model import SoccerPolicy
from dataset import SoccerDataset

# ===========================
# Hyperparameters & Config
# ===========================
CONFIG = {
    "project_name": "soccer-behavior-cloning",
    "run_name": "transformer_baseline_v1",
    
    # Paths
    "data_path": "data/metrica_match2_train_attacker_centric.npz",
    "save_dir": "checkpoints",
    
    # Training
    "epochs": 100,
    "batch_size": 128,
    "lr": 3e-4,          # Adam 默认偏好
    "weight_decay": 1e-4, # 正则化
    
    # Model
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dropout": 0.1,
    
    # Loss Weights
    "pos_weight": 25.0,  # 对 Trigger=1 的正样本加权
    "lambda_move": 1.0,  # 移动 Loss 权重
    "lambda_trig": 2.0,  # 踢球触发 Loss 权重
    "lambda_kick": 1.0,  # 踢球方向 Loss 权重
    
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def calculate_metrics(pred_trigger, target_trigger, threshold=0.5):
    """计算 Recall, Precision, F1"""
    probs = torch.sigmoid(pred_trigger)
    preds = (probs > threshold).float()
    
    preds_flat = preds.view(-1)
    targets_flat = target_trigger.view(-1)
    
    tp = ((preds_flat == 1) & (targets_flat == 1)).sum().item()
    fp = ((preds_flat == 1) & (targets_flat == 0)).sum().item()
    fn = ((preds_flat == 0) & (targets_flat == 1)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return precision, recall, f1

def train():
    # 1. Init WandB
    wandb.init(project=CONFIG["project_name"], name=CONFIG["run_name"], config=CONFIG)
    config = wandb.config # 使用 wandb 的 config 对象
    
    os.makedirs(config.save_dir, exist_ok=True)
    device = torch.device(config.device)
    print(f"Training on {device}")

    # 2. Data
    dataset = SoccerDataset(config.data_path)
    
    # 拆分 90% Train, 10% Val
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 3. Model
    model = SoccerPolicy(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(device)
    
    # 记录模型结构
    wandb.watch(model, log="all", log_freq=100)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True) # 监控 Val F1

    # Losses
    criterion_mse = nn.MSELoss()
    trig_pos_weight = torch.tensor([config.pos_weight]).to(device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=trig_pos_weight)
    criterion_ce = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_val_f1 = 0.0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for obs, acts in pbar:
            obs, acts = obs.to(device), acts.to(device)
            
            # Forward
            pred_vel, pred_trig, pred_kick = model(obs)
            
            # Targets
            t_vel = acts[:, :, 0:2]      # [B, 11, 2]
            t_trig = acts[:, :, 2:3]     # [B, 11, 1]
            t_kick = acts[:, :, 3].long() # [B, 11]
            
            # --- Losses ---
            # 1. Movement
            loss_move = criterion_mse(pred_vel, t_vel)
            
            # 2. Trigger (Kick or Not)
            loss_trig = criterion_bce(pred_trig, t_trig)
            
            # 3. Kick Direction (Masked: only calc when GT says kick)
            # Flatten for easier indexing
            flat_pred_kick = pred_kick.reshape(-1, 12)
            flat_t_kick = t_kick.reshape(-1)
            flat_t_trig = t_trig.reshape(-1)
            
            kick_mask = (flat_t_trig > 0.5)
            if kick_mask.sum() > 0:
                loss_kick = criterion_ce(flat_pred_kick[kick_mask], flat_t_kick[kick_mask])
            else:
                loss_kick = torch.tensor(0.0).to(device)
            
            # Weighted Sum
            total_loss = (config.lambda_move * loss_move) + \
                         (config.lambda_trig * loss_trig) + \
                         (config.lambda_kick * loss_kick)
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪防止爆炸
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            # Real-time update on progress bar
            pbar.set_postfix({"Loss": f"{total_loss.item():.3f}"})

        # --- Validation ---
        model.eval()
        val_prec_list, val_rec_list, val_f1_list = [], [], []
        val_losses = []
        
        with torch.no_grad():
            for obs, acts in val_loader:
                obs, acts = obs.to(device), acts.to(device)
                pred_vel, pred_trig, pred_kick = model(obs)
                
                # Metrics (只看 Trigger 准不准)
                p, r, f1 = calculate_metrics(pred_trig, acts[:, :, 2:3])
                val_prec_list.append(p)
                val_rec_list.append(r)
                val_f1_list.append(f1)
                
                # Val Loss (简化计算)
                # ...此处略去详细 loss 计算以节省资源，主要关注指标...

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_prec = np.mean(val_prec_list)
        avg_val_rec = np.mean(val_rec_list)
        avg_val_f1 = np.mean(val_f1_list)
        
        # Logging to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "val/precision": avg_val_prec,
            "val/recall": avg_val_rec,
            "val/f1": avg_val_f1,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        print(f" > Val Recall: {avg_val_rec:.4f} | Val F1: {avg_val_f1:.4f}")

        # Scheduler Step
        scheduler.step(avg_val_f1)

        # Save Best Model
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            save_path = os.path.join(config.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path) # Upload to cloud
            print(f" [Saved Best Model] F1: {best_val_f1:.4f}")

    wandb.finish()

if __name__ == "__main__":
    train()