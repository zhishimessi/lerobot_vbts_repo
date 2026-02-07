"""
触觉传感器神经网络模型

使用MLP将RGB颜色差值映射到梯度值(gx, gy)
"""

import torch
from torch import nn


class MLPGradientEncoder(nn.Module):
    """
    MLP梯度编码器
    
    输入: [B, G, R, X, Y] 或 [B, G, R] (5维或3维)
    输出: [gx, gy] (2维梯度)
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, output_dim: int = 2):
        """
        初始化MLP模型
        
        Args:
            input_dim: 输入维度，5表示[B,G,R,X,Y]，3表示[B,G,R]
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（默认2，表示gx和gy）
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class MLPGradientEncoderLarge(nn.Module):
    """
    更大的MLP梯度编码器（用于更复杂的映射，如仅用颜色预测梯度，没有像素位置）
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, output_dim: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


def load_model(model_path: str, input_dim: int = 5, device: str = 'cpu') -> MLPGradientEncoder:
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        input_dim: 输入维度
        device: 设备（'cpu' 或 'cuda'）
        
    Returns:
        加载好的模型
    """
    model = MLPGradientEncoder(input_dim=input_dim)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model
