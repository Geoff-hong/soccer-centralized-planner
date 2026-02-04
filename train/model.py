import torch
import torch.nn as nn
import math

class SoccerPolicy(nn.Module):
    def __init__(
        self,
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.1,
        input_dim=3,
        move_horizon=1,
        num_agents=23,
        num_attackers=11,
    ):
        """
        Args:
            d_model: 内部特征维度 (建议 128 或 256)
            nhead: 注意力头数
            num_layers: Transformer 层数
            dropout: 防止过拟合
            input_dim: 输入特征维度（默认 3, 堆叠历史则为 3 * K）
            move_horizon: 预测未来 K 帧速度
        """
        super().__init__()
        self.move_horizon = max(1, int(move_horizon))
        self.num_agents = int(num_agents)
        self.num_attackers = int(num_attackers)
        if self.num_agents < self.num_attackers + 1:
            raise ValueError("num_agents must be >= num_attackers + 1 (ball)")
        
        # ===========================
        # 1. Embedding Layer
        # ===========================
        # Input: [x, y, has_ball] -> 3 features (或堆叠历史)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Learnable Positional Encoding (用于区分不同的 Agent)
        # 0..(num_attackers-1): Teammates, num_attackers..(num_agents-2): Opponents, last: Ball
        self.agent_pos_embed = nn.Parameter(torch.randn(1, self.num_agents, d_model))
        
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
        # 我们只预测前 num_attackers 个 Agent (Teammates) 的动作
        
        # Head A: Movement (Vx, Vy) -> 回归，支持 multi-step
        self.head_move = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2 * self.move_horizon)
        )
        
        # Head B: Pass/Not (Global, Logits)
        self.head_pass = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        # Head C: Passer ID (A-class)
        self.head_passer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        # Head D: Receiver ID (A-class)
        self.head_receiver = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        """
        x: [Batch, N, F]
        Returns: 
            pred_vel: [Batch, K, num_attackers, 2]
            pred_pass: [Batch, 1]
            pred_passer: [Batch, num_attackers]
            pred_receiver: [Batch, num_attackers]
        """
        B, N, F = x.shape
        if N != self.num_agents:
            raise ValueError(f"Input agent count {N} != model.num_agents {self.num_agents}")
        ball_idx = self.num_agents - 1
        
        # 1. Embed & Add Identity
        x = self.input_proj(x)
        x = x + self.agent_pos_embed # Broadcasting [1, N, D] -> [B, N, D]
        
        # 2. Transformer Feature Extraction
        # Output: [B, N, D]
        feat = self.transformer(x)
        
        # 3. Slice Teammates (First num_attackers agents)
        teammate_feat = feat[:, : self.num_attackers, :] # [B, A, D]

        # 4. Global Feature (use ball token as global context)
        global_feat = feat[:, ball_idx, :] # [B, D]

        # 5. Heads
        pred_vel = self.head_move(teammate_feat)
        # [B, A, K*2] -> [B, A, K, 2] -> [B, K, A, 2]
        pred_vel = pred_vel.view(B, self.num_attackers, self.move_horizon, 2).permute(0, 2, 1, 3).contiguous()
        pred_pass = self.head_pass(global_feat)  # [B, 1]
        pred_passer = self.head_passer(teammate_feat).squeeze(-1)  # [B, A]
        pred_receiver = self.head_receiver(teammate_feat).squeeze(-1)  # [B, A]

        return pred_vel, pred_pass, pred_passer, pred_receiver
