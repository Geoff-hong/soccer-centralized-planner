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
    "run_name": "transformer_pass_global_skillcorner10",
    
    # Paths
    "data_path": os.path.join(PROJECT_ROOT, "data", "skillcorner_processed", "npz"),
    "save_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
    
    # Training
    "epochs": 50,
    "batch_size": 256,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "val_ratio": 0.1,
    "seed": 42,
    "split_by_file": True,
    "use_weighted_sampler": False,
    
    # Model
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dropout": 0.3,       # 【关键修改】增加 Dropout 防止过拟合
    "obs_history": 5,     # 堆叠过去 K 帧
    
    # Loss Weights (移动为主，传球为辅)
    "pos_weight": 2.0,     # pass_or_not 的正例权重（auto_pos_weight=False 时生效）
    "auto_pos_weight": True,
    "lambda_move": 1.0,
    "train_move_only": True,
    # Move loss shaping
    "move_speed_thr": 4.0,      # 高速阈值 (m/s)
    "move_high_weight": 3.0,    # 高速段权重
    "move_cos_weight": 0.5,     # 方向一致性权重
    "move_mag_weight": 0.5,     # 速度幅值误差权重
    "move_dir_speed_thr": 0.5,  # 仅在目标速度>阈值时计算方向损失
    "lambda_pass": 0.0,
    "lambda_passer": 0.0,
    "lambda_receiver": 0.0,
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Passer Mask
    "passer_topk": 3,         # 只在离球最近的 K 人里训练 passer_id
    "trigger_dist_thr": 3.0,  # 用于距离计算的阈值（与数据处理对齐）

    # Scheduler
    "patience": 15        # 【关键修改】增加耐心值
}

def metrics_from_logits(logits, targets, prob_threshold=0.5):
    """基于全量 logits/targets 计算 Precision/Recall/F1"""
    probs = torch.sigmoid(logits)
    preds = (probs > prob_threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = ((preds_flat == 1) & (targets_flat == 1)).sum().item()
    fp = ((preds_flat == 1) & (targets_flat == 0)).sum().item()
    fn = ((preds_flat == 0) & (targets_flat == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return precision, recall, f1, tp, fp, fn

def compute_move_loss(pred_vel, move, config):
    """
    Move loss = weighted vector MSE + mag error + direction cosine penalty.
    pred_vel/move: [B, 11, 2]
    """
    # Vector MSE (speed-weighted)
    speed_t = torch.norm(move, dim=2)  # [B, 11]
    weight = torch.ones_like(speed_t)
    weight = weight + (speed_t > config.move_speed_thr).float() * config.move_high_weight
    vec_err = (pred_vel - move) ** 2
    vec_mse = (vec_err.sum(dim=2) * weight).mean()

    # Magnitude loss
    speed_p = torch.norm(pred_vel, dim=2)
    mag_err = torch.abs(speed_p - speed_t)
    mag_loss = mag_err.mean()

    # Direction cosine loss (magnitude-independent, mask by target speed)
    pred_norm = pred_vel / (speed_p.unsqueeze(-1) + 1e-8)
    target_norm = move / (speed_t.unsqueeze(-1) + 1e-8)
    cos = (pred_norm * target_norm).sum(dim=2).clamp(-1.0, 1.0)
    dir_mask = (speed_t > config.move_dir_speed_thr)
    if dir_mask.any():
        dir_loss = (1.0 - cos[dir_mask]).mean()
    else:
        dir_loss = torch.tensor(0.0, device=pred_vel.device)

    total = vec_mse + config.move_mag_weight * mag_loss + config.move_cos_weight * dir_loss
    return total, vec_mse, mag_loss, dir_loss

def train():
    # 1. Init WandB
    wandb.init(project=CONFIG["project_name"], name=CONFIG["run_name"], config=CONFIG)
    config = wandb.config
    
    os.makedirs(config.save_dir, exist_ok=True)
    run_name = getattr(wandb.run, "name", None) or "run"
    run_id = getattr(wandb.run, "id", "runid")
    run_tag = f"{run_name}_{run_id}".replace("/", "_")
    run_save_dir = os.path.join(config.save_dir, run_tag)
    os.makedirs(run_save_dir, exist_ok=True)
    device = torch.device(config.device)
    print(f"Training on {device}")

    # 2. Data & Sampler Setup
    if config.split_by_file and os.path.isdir(config.data_path):
        # 按比赛文件拆分，避免泄露
        npz_files = [os.path.join(config.data_path, f) for f in os.listdir(config.data_path) if f.endswith(".npz")]
        npz_files.sort()
        if not npz_files:
            raise FileNotFoundError(f"No npz files found in {config.data_path}")
        rng = np.random.default_rng(config.seed)
        rng.shuffle(npz_files)
        val_count = max(1, int(len(npz_files) * config.val_ratio))
        val_files = npz_files[:val_count]
        train_files = npz_files[val_count:]
        print(f"Split by file: train={len(train_files)} val={len(val_files)}")
        train_set = SoccerDataset(obs_history=config.obs_history, file_paths=train_files)
        val_set = SoccerDataset(obs_history=config.obs_history, file_paths=val_files)
    else:
        full_dataset = SoccerDataset(config.data_path, obs_history=config.obs_history)
        val_size = int(len(full_dataset) * config.val_ratio)
        train_size = len(full_dataset) - val_size
        train_set, val_set = random_split(full_dataset, [train_size, val_size])

    if config.use_weighted_sampler:
        print("Computing Sampler Weights (Solving Class Imbalance)...")
        train_weights = train_set.get_sample_weights()
        sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)
        train_loader = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 3. Model
    model = SoccerPolicy(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
        input_dim=3 * config.obs_history
    ).to(device)
    
    wandb.watch(model, log="all", log_freq=100)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Scheduler: 监控 Val Loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config.patience)

    # Losses
    criterion_mse = nn.MSELoss()
    if not config.train_move_only:
        # 自动计算 pos_weight（负/正比），否则使用配置值
        if config.auto_pos_weight:
            n_kicks = train_set.n_kicks
            n_total = train_set.total_frames
            n_no = max(n_total - n_kicks, 1)
            pos_weight_val = float(n_no / max(n_kicks, 1))
        else:
            pos_weight_val = config.pos_weight
        pass_pos_weight = torch.tensor([pos_weight_val]).to(device)
        print(f"Using pos_weight for pass_or_not: {pos_weight_val:.3f}")
        criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pass_pos_weight)
        criterion_ce = nn.CrossEntropyLoss()
    else:
        criterion_bce = None
        criterion_ce = None

    # 4. Training Loop
    if config.train_move_only:
        best_val_metric = float("inf")
        best_metric_name = "val_loss"
    else:
        best_val_metric = -float("inf")
        best_metric_name = "f1"

    for epoch in range(config.epochs):
        model.train()
        
        log_loss_move = []
        log_loss_move_vec = []
        log_loss_move_mag = []
        log_loss_move_dir = []
        log_loss_pass = []
        log_loss_passer = []
        log_loss_receiver = []
        debug_logits = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for obs, move, pass_flag, passer_id, pass_dir, receiver_id in pbar:
            obs = obs.to(device)
            move = move.to(device)
            pass_flag = pass_flag.to(device)
            passer_id = passer_id.to(device)
            pass_dir = pass_dir.to(device)
            receiver_id = receiver_id.to(device)
            
            # Forward
            pred_vel, pred_pass, pred_passer, pred_receiver = model(obs)
            
            # 记录 Logits 用于调试
            debug_logits.append(pred_pass.detach().mean().item())
            
            # 计算距离 (用于 top-k passer mask)
            k = config.obs_history
            latest = obs[:, :, (k - 1) * 3 : k * 3]  # [B, 23, 3]
            att_pos_norm = latest[:, :11, 0:2]
            ball_pos_norm = latest[:, 22, 0:2].unsqueeze(1)
            att_pos = att_pos_norm.clone()
            att_pos[:, :, 0] = att_pos[:, :, 0] * 52.5 + 52.5
            att_pos[:, :, 1] = att_pos[:, :, 1] * 34.0 + 34.0
            ball_pos = ball_pos_norm.clone()
            ball_pos[:, :, 0] = ball_pos[:, :, 0] * 52.5 + 52.5
            ball_pos[:, :, 1] = ball_pos[:, :, 1] * 34.0 + 34.0
            dist = torch.norm(att_pos - ball_pos, dim=2)
            topk_idx = dist.topk(config.passer_topk, largest=False).indices  # [B, K]
            topk_mask = torch.zeros_like(dist, dtype=torch.bool)
            topk_mask.scatter_(1, topk_idx, True)

            # --- Losses ---
            loss_move, loss_move_vec, loss_move_mag, loss_move_dir = compute_move_loss(pred_vel, move, config)
            # pass_or_not 仅在有人接近球时计算
            if not config.train_move_only:
                possession_mask = (dist.min(dim=1).values < config.trigger_dist_thr)
                if possession_mask.any():
                    loss_pass = criterion_bce(pred_pass.view(-1)[possession_mask], pass_flag.view(-1)[possession_mask])
                else:
                    loss_pass = torch.tensor(0.0, device=device)
            else:
                loss_pass = torch.tensor(0.0, device=device)

            # Passer ID Loss (仅在 pass_flag=1 & passer_id 有效 & passer 在 top-k 内)
            if not config.train_move_only:
                passer_mask = (pass_flag > 0.5) & (passer_id >= 0)
                if passer_mask.any():
                    # 判断真实 passer 是否在 top-k
                    gather_idx = passer_id.clamp(min=0).unsqueeze(1)
                    in_topk = topk_mask.gather(1, gather_idx).squeeze(1)
                    valid_passer = passer_mask & in_topk
                    if valid_passer.any():
                        logits = pred_passer[valid_passer]
                        mask_logits = topk_mask[valid_passer]
                        logits = logits.masked_fill(~mask_logits, -1e9)
                        loss_passer = criterion_ce(logits, passer_id[valid_passer])
                    else:
                        loss_passer = torch.tensor(0.0, device=device)
                else:
                    loss_passer = torch.tensor(0.0, device=device)
            else:
                loss_passer = torch.tensor(0.0, device=device)

            # Receiver Loss
            if not config.train_move_only:
                recv_mask = (pass_flag > 0.5) & (receiver_id >= 0)
                if recv_mask.any():
                    loss_receiver = criterion_ce(pred_receiver[recv_mask], receiver_id[recv_mask])
                else:
                    loss_receiver = torch.tensor(0.0, device=device)
            else:
                loss_receiver = torch.tensor(0.0, device=device)

            total_loss = (config.lambda_move * loss_move) + \
                         (config.lambda_pass * loss_pass) + \
                         (config.lambda_passer * loss_passer) + \
                         (config.lambda_receiver * loss_receiver)
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Record
            log_loss_move.append(loss_move.item())
            log_loss_move_vec.append(loss_move_vec.item())
            log_loss_move_mag.append(loss_move_mag.item())
            log_loss_move_dir.append(loss_move_dir.item())
            log_loss_pass.append(loss_pass.item())
            log_loss_passer.append(loss_passer.item())
            log_loss_receiver.append(loss_receiver.item())
            
            if not config.train_move_only:
                pbar.set_postfix({
                    "L_Pass": f"{loss_pass.item():.3f}",
                    "L_Passer": f"{loss_passer.item():.3f}"
                })
            else:
                pbar.set_postfix({
                    "L_Move": f"{loss_move.item():.3f}"
                })

        # --- Validation ---
        model.eval()
        val_loss_list = []

        # 收集 pass logits
        all_val_logits = []
        all_val_targets = []

        # 统计 move loss (val)
        val_loss_move = []
        val_loss_move_vec = []
        val_loss_move_mag = []
        val_loss_move_dir = []

        # 额外统计
        passer_correct = 0
        passer_total = 0
        recv_correct = 0
        recv_total = 0

        with torch.no_grad():
            for obs, move, pass_flag, passer_id, pass_dir, receiver_id in val_loader:
                obs = obs.to(device)
                move = move.to(device)
                pass_flag = pass_flag.to(device)
                passer_id = passer_id.to(device)
                pass_dir = pass_dir.to(device)
                receiver_id = receiver_id.to(device)

                pred_vel, pred_pass, pred_passer, pred_receiver = model(obs)

                # top-k mask
                k = config.obs_history
                latest = obs[:, :, (k - 1) * 3 : k * 3]  # [B, 23, 3]
                att_pos_norm = latest[:, :11, 0:2]
                ball_pos_norm = latest[:, 22, 0:2].unsqueeze(1)
                att_pos = att_pos_norm.clone()
                att_pos[:, :, 0] = att_pos[:, :, 0] * 52.5 + 52.5
                att_pos[:, :, 1] = att_pos[:, :, 1] * 34.0 + 34.0
                ball_pos = ball_pos_norm.clone()
                ball_pos[:, :, 0] = ball_pos[:, :, 0] * 52.5 + 52.5
                ball_pos[:, :, 1] = ball_pos[:, :, 1] * 34.0 + 34.0
                dist = torch.norm(att_pos - ball_pos, dim=2)
                topk_idx = dist.topk(config.passer_topk, largest=False).indices
                topk_mask = torch.zeros_like(dist, dtype=torch.bool)
                topk_mask.scatter_(1, topk_idx, True)

                # 计算 val loss (同训练配方)
                loss_move, loss_move_vec, loss_move_mag, loss_move_dir = compute_move_loss(pred_vel, move, config)
                val_loss_move.append(loss_move.item())
                val_loss_move_vec.append(loss_move_vec.item())
                val_loss_move_mag.append(loss_move_mag.item())
                val_loss_move_dir.append(loss_move_dir.item())
                if not config.train_move_only:
                    possession_mask = (dist.min(dim=1).values < config.trigger_dist_thr)
                    if possession_mask.any():
                        loss_pass = criterion_bce(pred_pass.view(-1)[possession_mask], pass_flag.view(-1)[possession_mask])
                    else:
                        loss_pass = torch.tensor(0.0, device=device)
                else:
                    loss_pass = torch.tensor(0.0, device=device)

                # passer loss
                if not config.train_move_only:
                    passer_mask = (pass_flag > 0.5) & (passer_id >= 0)
                    if passer_mask.any():
                        gather_idx = passer_id.clamp(min=0).unsqueeze(1)
                        in_topk = topk_mask.gather(1, gather_idx).squeeze(1)
                        valid_passer = passer_mask & in_topk
                        if valid_passer.any():
                            logits = pred_passer[valid_passer]
                            mask_logits = topk_mask[valid_passer]
                            logits = logits.masked_fill(~mask_logits, -1e9)
                            loss_passer = criterion_ce(logits, passer_id[valid_passer])

                            # passer acc
                            pred_idx = torch.argmax(pred_passer[valid_passer], dim=1)
                            passer_correct += (pred_idx == passer_id[valid_passer]).sum().item()
                            passer_total += valid_passer.sum().item()
                        else:
                            loss_passer = torch.tensor(0.0, device=device)
                    else:
                        loss_passer = torch.tensor(0.0, device=device)
                else:
                    loss_passer = torch.tensor(0.0, device=device)

                # receiver loss + acc
                if not config.train_move_only:
                    recv_mask = (pass_flag > 0.5) & (receiver_id >= 0)
                    if recv_mask.any():
                        loss_receiver = criterion_ce(pred_receiver[recv_mask], receiver_id[recv_mask])
                        pred_recv_idx = torch.argmax(pred_receiver[recv_mask], dim=1)
                        recv_correct += (pred_recv_idx == receiver_id[recv_mask]).sum().item()
                        recv_total += recv_mask.sum().item()
                    else:
                        loss_receiver = torch.tensor(0.0, device=device)
                else:
                    loss_receiver = torch.tensor(0.0, device=device)

                v_loss = (config.lambda_move * loss_move) + \
                         (config.lambda_pass * loss_pass) + \
                         (config.lambda_passer * loss_passer) + \
                         (config.lambda_receiver * loss_receiver)
                val_loss_list.append(v_loss.item())

                # Collect for pass metrics
                if not config.train_move_only:
                    all_val_logits.append(pred_pass.detach().cpu())
                    all_val_targets.append(pass_flag.detach().cpu())

        # Aggregates
        avg_logit_train = np.mean(debug_logits)
        avg_val_loss = np.mean(val_loss_list)
        avg_val_move = np.mean(val_loss_move) if val_loss_move else float("nan")
        avg_val_move_vec = np.mean(val_loss_move_vec) if val_loss_move_vec else float("nan")
        avg_val_move_mag = np.mean(val_loss_move_mag) if val_loss_move_mag else float("nan")
        avg_val_move_dir = np.mean(val_loss_move_dir) if val_loss_move_dir else float("nan")

        if not config.train_move_only:
            # --- Threshold Analysis (Pass/Not) ---
            all_val_logits = torch.cat(all_val_logits).view(-1)
            all_val_targets = torch.cat(all_val_targets).view(-1)

            # 全量聚合指标 (Threshold=0.5)
            p, r, f1, tp, fp, fn = metrics_from_logits(all_val_logits, all_val_targets, prob_threshold=0.5)

            passer_acc = passer_correct / max(passer_total, 1)
            recv_acc = recv_correct / max(recv_total, 1)

            print(f"\n[Epoch {epoch+1} Analysis]")
            print(f" > Mean Logit (Train): {avg_logit_train:.2f}")
            print(f" > Logit Std (Val):    {all_val_logits.std():.2f}")
            print(f" > Logit Max (Val):    {all_val_logits.max():.2f}")
            print(f" > TP/FP/FN (Val@0.5): {tp}/{fp}/{fn}")
            print(f" > Passer Acc: {passer_acc:.4f} | Receiver Acc: {recv_acc:.4f}")

            test_thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
            for t in test_thresholds:
                p_t, r_t, f1_t, tp_t, fp_t, fn_t = metrics_from_logits(all_val_logits, all_val_targets, prob_threshold=t)
                print(f"   [Prob {t:>5}] Recall: {r_t:.4f} | Prec: {p_t:.4f} | F1: {f1_t:.4f} | TP/FP/FN: {tp_t}/{fp_t}/{fn_t}")
        else:
            p = r = f1 = tp = fp = fn = float("nan")
            passer_acc = recv_acc = float("nan")
            print(f"\n[Epoch {epoch+1} Analysis]")
            print(f" > Mean Logit (Train): {avg_logit_train:.2f}")
            print(" > Pass metrics skipped (train_move_only=True)")
            print(f" > Val Move: {avg_val_move:.4f} | Vec {avg_val_move_vec:.4f} | Mag {avg_val_move_mag:.4f} | Dir {avg_val_move_dir:.4f}")

        # Logging to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train/loss_move": np.mean(log_loss_move),
            "train/loss_move_vec": np.mean(log_loss_move_vec),
            "train/loss_move_mag": np.mean(log_loss_move_mag),
            "train/loss_move_dir": np.mean(log_loss_move_dir),
            "train/loss_pass": np.mean(log_loss_pass),
            "train/loss_passer": np.mean(log_loss_passer),
            "train/loss_receiver": np.mean(log_loss_receiver),
            "debug/mean_logit": avg_logit_train,
            "debug/val_logit_max": (all_val_logits.max().item() if not config.train_move_only else float("nan")),
            "debug/val_logit_std": (all_val_logits.std().item() if not config.train_move_only else float("nan")),
            "val/loss": avg_val_loss,
            "val/loss_move": avg_val_move,
            "val/loss_move_vec": avg_val_move_vec,
            "val/loss_move_mag": avg_val_move_mag,
            "val/loss_move_dir": avg_val_move_dir,
            "val/precision": p,
            "val/recall": r,
            "val/passer_acc": passer_acc,
            "val/receiver_acc": recv_acc,
            "lr": optimizer.param_groups[0]['lr']
        })

        # Scheduler Step
        scheduler.step(avg_val_loss)

        # Save Best Model
        current_metric = avg_val_loss if config.train_move_only else f1
        improved = (current_metric < best_val_metric) if config.train_move_only else (current_metric > best_val_metric)
        if improved:
            best_val_metric = current_metric
            save_path = os.path.join(run_save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path)
            if config.train_move_only:
                print(f" [Saved Best Model] val_loss: {best_val_metric:.4f}")
            else:
                print(f" [Saved Best Model] F1: {best_val_metric:.4f}")

    wandb.finish()

if __name__ == "__main__":
    train()
