import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np
import os
import wandb
from tqdm import tqdm

# 引入模块
from model import SoccerPolicy
from dataset import SoccerDataset

# ===========================
# 获取项目路径
# ===========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ===========================
# Hyperparameters & Config
# ===========================
CONFIG = {
    "project_name": "soccer-behavior-cloning",
    "run_name": "transformer_sampler_v2", # 修改名字以区分
    
    # Paths
    "data_path": os.path.join(PROJECT_ROOT, "data", "metrica_match2_train_attacker_centric.npz"),
    "save_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
    
    # Training
    "epochs": 100,
    "batch_size": 128,
    "lr": 1e-4,           # 稍微调低一点，配合 Sampler
    "weight_decay": 1e-4,
    
    # Model
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dropout": 0.1,
    
    # Loss Weights
    # 注意：有了 Sampler 后，pos_weight 可以适当降低，不用 25 那么激进了
    "pos_weight": 5.0,    
    "lambda_move": 1.0,
    "lambda_trig": 2.0,
    "lambda_kick": 1.0,
    
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
    config = wandb.config
    
    os.makedirs(config.save_dir, exist_ok=True)
    device = torch.device(config.device)
    print(f"Training on {device}")

    # 2. Data & Sampler Setup
    full_dataset = SoccerDataset(config.data_path)
    
    # Split
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    # --- 关键修改：构建 WeightedRandomSampler ---
    print("Computing Sampler Weights (Solving Class Imbalance)...")
    
    # 获取整个数据集的权重
    all_weights = full_dataset.get_sample_weights() # 需要 dataset.py 支持此方法
    
    # 提取训练集对应的权重
    train_indices = train_set.indices
    train_weights = torch.as_tensor(all_weights[train_indices], dtype=torch.double)
    
    # 创建采样器 (replacement=True 允许重复采样，这是过采样的核心)
    sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)
    
    # DataLoader (shuffle 必须为 False)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 3. Model
    model = SoccerPolicy(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(device)
    
    wandb.watch(model, log="all", log_freq=100)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # 修改 Scheduler：监控 Loss 而不是 F1，避免 F1=0 时 LR 归零
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Losses
    criterion_mse = nn.MSELoss()
    trig_pos_weight = torch.tensor([config.pos_weight]).to(device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=trig_pos_weight)
    criterion_ce = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_val_recall = 0.0 # 改为监控 Recall，因为这对我们最重要

    for epoch in range(config.epochs):
        model.train()
        
        # 分项 Loss 记录
        log_loss_move = []
        log_loss_trig = []
        log_loss_kick = []
        debug_logits = [] # 调试 Logits
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for obs, acts in pbar:
            obs, acts = obs.to(device), acts.to(device)
            
            # Forward
            pred_vel, pred_trig, pred_kick = model(obs)
            
            # 调试：记录 Logits 均值
            debug_logits.append(pred_trig.detach().mean().item())
            
            # Targets
            t_vel = acts[:, :, 0:2]
            t_trig = acts[:, :, 2:3]
            t_kick = acts[:, :, 3].long()
            
            # --- Losses ---
            loss_move = criterion_mse(pred_vel, t_vel)
            loss_trig = criterion_bce(pred_trig, t_trig)
            
            # Kick Direction
            flat_pred_kick = pred_kick.reshape(-1, 12)
            flat_t_kick = t_kick.reshape(-1)
            flat_t_trig = t_trig.reshape(-1)
            
            kick_mask = (flat_t_trig > 0.5)
            if kick_mask.sum() > 0:
                loss_kick = criterion_ce(flat_pred_kick[kick_mask], flat_t_kick[kick_mask])
            else:
                loss_kick = torch.tensor(0.0).to(device)
            
            total_loss = (config.lambda_move * loss_move) + \
                         (config.lambda_trig * loss_trig) + \
                         (config.lambda_kick * loss_kick)
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 记录
            log_loss_move.append(loss_move.item())
            log_loss_trig.append(loss_trig.item())
            log_loss_kick.append(loss_kick.item())
            
            pbar.set_postfix({
                "L_Trig": f"{loss_trig.item():.3f}",
                "L_Kick": f"{loss_kick.item():.3f}"
            })

        # --- Validation ---
        model.eval()
        val_prec_list, val_rec_list, val_f1_list = [], [], []
        val_loss_list = []
        
        with torch.no_grad():
            for obs, acts in val_loader:
                obs, acts = obs.to(device), acts.to(device)
                pred_vel, pred_trig, pred_kick = model(obs)
                
                # Metrics
                p, r, f1 = calculate_metrics(pred_trig, acts[:, :, 2:3])
                val_prec_list.append(p)
                val_rec_list.append(r)
                val_f1_list.append(f1)
                
                # Val Loss 用于 Scheduler
                # (这里简单计算 Trigger Loss 作为主要监控指标)
                v_loss = criterion_bce(pred_trig, acts[:, :, 2:3])
                val_loss_list.append(v_loss.item())

        # Aggregates
        avg_logit = np.mean(debug_logits)
        avg_val_loss = np.mean(val_loss_list)
        avg_val_rec = np.mean(val_rec_list)
        avg_val_prec = np.mean(val_prec_list)
        
        # Logging to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train/loss_move": np.mean(log_loss_move),
            "train/loss_trig": np.mean(log_loss_trig),
            "train/loss_kick": np.mean(log_loss_kick),
            "debug/mean_logit": avg_logit, # 核心调试指标
            "val/loss": avg_val_loss,
            "val/precision": avg_val_prec,
            "val/recall": avg_val_rec,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        print(f" > Epoch {epoch+1} | Val Recall: {avg_val_rec:.4f} | Mean Logit: {avg_logit:.2f}")

        # Scheduler Step (Monitor Val Loss)
        scheduler.step(avg_val_loss)

        # Save Best Model (Based on Recall)
        if avg_val_rec > best_val_recall:
            best_val_recall = avg_val_rec
            save_path = os.path.join(config.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path)
            print(f" [Saved Best Model] Recall: {best_val_recall:.4f}")

    wandb.finish()

if __name__ == "__main__":
    train()