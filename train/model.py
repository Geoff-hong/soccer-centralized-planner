import torch
import torch.nn as nn
import math

class SoccerPolicy(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        """
        Args:
            d_model: 内部特征维度 (建议 128 或 256)
            nhead: 注意力头数
            num_layers: Transformer 层数
            dropout: 防止过拟合
        """
        super().__init__()
        
        # ===========================
        # 1. Embedding Layer
        # ===========================
        # Input: [x, y, has_ball] -> 3 features
        self.input_proj = nn.Linear(3, d_model)
        
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
        
        # Head B: Kick Trigger (Logits) -> 二分类
        self.head_trigger = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Head C: Kick Direction (Logits) -> 12 分类
        self.head_kick = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 12)
        )

    def forward(self, x):
        """
        x: [Batch, 23, 3]
        Returns: 
            pred_vel: [Batch, 11, 2]
            pred_trigger: [Batch, 11, 1]
            pred_kick: [Batch, 11, 12]
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
        
        # 4. Heads
        pred_vel = self.head_move(teammate_feat)
        pred_trigger = self.head_trigger(teammate_feat)
        pred_kick = self.head_kick(teammate_feat)
        
        return pred_vel, pred_trigger, pred_kick