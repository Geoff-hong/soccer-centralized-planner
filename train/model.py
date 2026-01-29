import torch
import torch.nn as nn
import math

class SoccerPolicy(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, dropout=0.1, input_dim=3):
        """
        Args:
            d_model: 内部特征维度 (建议 128 或 256)
            nhead: 注意力头数
            num_layers: Transformer 层数
            dropout: 防止过拟合
            input_dim: 输入特征维度（默认 3, 堆叠历史则为 3 * K）
        """
        super().__init__()
        
        # ===========================
        # 1. Embedding Layer
        # ===========================
        # Input: [x, y, has_ball] -> 3 features (或堆叠历史)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Learnable Positional Encoding (用于区分 23 个不同的 Agent)
        # 0-10: Teammates, 11-21: Opponents, 22: Ball
        self.agent_pos_embed = nn.Parameter(torch.randn(1, 23, d_model))
        
        # ===========================
        # 2. Backbone (Transformer)
        # ===========================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-Norm 通常收敛更好
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ===========================
        # 3. Action Heads (Decoders)
        # ===========================
        # 我们只预测前 11 个 Agent (Teammates) 的动作
        
        # Head A: Movement (Vx, Vy) -> 回归
        self.head_move = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )
        
        # Head B: Pass/Not (Global, Logits)
        self.head_pass = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        # Head C: Passer ID (11-class)
        self.head_passer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        # Head D: Pass Direction (12-class, Global)
        self.head_dir = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 12)
        )

        # Head E: Receiver ID (11-class)
        self.head_receiver = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        """
        x: [Batch, 23, 3]
        Returns: 
            pred_vel: [Batch, 11, 2]
            pred_pass: [Batch, 1]
            pred_passer: [Batch, 11]
            pred_dir: [Batch, 12]
            pred_receiver: [Batch, 11]
        """
        B, N, F = x.shape
        
        # 1. Embed & Add Identity
        x = self.input_proj(x)
        x = x + self.agent_pos_embed # Broadcasting [1, 23, D] -> [B, 23, D]
        
        # 2. Transformer Feature Extraction
        # Output: [B, 23, D]
        feat = self.transformer(x)
        
        # 3. Slice Teammates (First 11 agents)
        teammate_feat = feat[:, :11, :] # [B, 11, D]

        # 4. Global Feature (use ball token as global context)
        global_feat = feat[:, 22, :] # [B, D]

        # 5. Heads
        pred_vel = self.head_move(teammate_feat)
        pred_pass = self.head_pass(global_feat)  # [B, 1]
        pred_passer = self.head_passer(teammate_feat).squeeze(-1)  # [B, 11]
        pred_dir = self.head_dir(global_feat)  # [B, 12]
        pred_receiver = self.head_receiver(teammate_feat).squeeze(-1)  # [B, 11]

        return pred_vel, pred_pass, pred_passer, pred_dir, pred_receiver
