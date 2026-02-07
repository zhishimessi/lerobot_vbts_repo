"""
MLP模型训练脚本

训练神经网络将颜色差值映射到梯度值
输入: [B, G, R, X, Y] (5维)
输出: [gx, gy] (2维)

使用方法:
1. 先运行 6_1_collect_mlp_dataset.py 收集数据
2. 运行本脚本训练模型
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt

# 确保可以导入 lerobot 模块
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from lerobot.cameras.tactile_cam.model import MLPGradientEncoder


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_num = 0
    
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.shape[0]
        total_num += labels.shape[0]
    
    return total_loss / total_num


def eval_epoch(model, data_loader, loss_fn, device):
    """评估一个epoch"""
    model.eval()
    total_loss = 0
    total_num = 0
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item() * labels.shape[0]
            total_num += labels.shape[0]
    
    return total_loss / total_num


def train(dataset_path: str, model_save_path: str, 
          input_dim: int = 5, hidden_dim: int = 32,
          num_epochs: int = 100, batch_size: int = 64,
          learning_rate: float = 0.001, weight_decay: float = 1e-5,
          train_ratio: float = 0.8, device: str = None):
    """
    训练MLP模型
    
    Args:
        dataset_path: 数据集路径 (.npy)
        model_save_path: 模型保存路径 (.pt)
        input_dim: 输入维度 (5: BGR+XY, 3: BGR only)
        hidden_dim: 隐藏层维度
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        weight_decay: 权重衰减
        train_ratio: 训练集比例
        device: 设备 ('cuda' or 'cpu')
    """
    # 设置设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] 使用设备: {device}")
    
    # 加载数据
    print(f"[INFO] 加载数据集: {dataset_path}")
    data = np.load(dataset_path)
    print(f"[INFO] 数据形状: {data.shape}")
    
    # 数据格式: [dB, dG, dR, X, Y, gx, gy]
    # 注意：dB, dG, dR 是颜色差分值，已经归一化到 [-1, 1] 范围
    if input_dim == 5:
        features = data[:, :5].copy()  # dB, dG, dR, X, Y
    else:
        features = data[:, :3].copy()  # dB, dG, dR only
    labels = data[:, 5:7]      # gx, gy
    
    # 归一化特征
    # 颜色差分值已经是 /255.0 归一化的，不需要再处理
    print(f"[INFO] 颜色差分范围: [{features[:, :3].min():.3f}, {features[:, :3].max():.3f}]")
    
    # XY: 归一化到 [0, 1]（如果使用）
    if input_dim == 5:
        x_max = features[:, 3].max()
        y_max = features[:, 4].max()
        features[:, 3] = features[:, 3] / x_max
        features[:, 4] = features[:, 4] / y_max
        print(f"[INFO] X范围: [0, {x_max}], Y范围: [0, {y_max}]")
    
    print(f"[INFO] 特征范围: [{features.min():.3f}, {features.max():.3f}]")
    print(f"[INFO] 标签范围: gx=[{labels[:, 0].min():.3f}, {labels[:, 0].max():.3f}], "
          f"gy=[{labels[:, 1].min():.3f}, {labels[:, 1].max():.3f}]")
    
    # 转换为张量
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    # 创建数据集
    dataset = TensorDataset(features_tensor, labels_tensor)
    
    # 划分训练集和验证集
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"[INFO] 训练集: {train_size}, 验证集: {val_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = MLPGradientEncoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=2)
    model.to(device)
    print(f"[INFO] 模型结构:\n{model}")
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    
    # 训练记录
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    print(f"\n[INFO] 开始训练，共 {num_epochs} 轮")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    print("=" * 60)
    print(f"[INFO] 训练完成！最佳验证损失: {best_val_loss:.6f}")
    print(f"[INFO] 模型已保存: {model_save_path}")
    
    # 保存归一化参数
    norm_params = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
    }
    if input_dim == 5:
        norm_params['x_max'] = x_max
        norm_params['y_max'] = y_max
    
    norm_path = model_save_path.replace('.pt', '_norm.npz')
    np.savez(norm_path, **norm_params)
    print(f"[INFO] 归一化参数已保存: {norm_path}")
    
    return model, history


def plot_training_history(history: dict, save_path: str = None):
    """绘制训练历史"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('MLP Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] 训练曲线已保存: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    # 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data", "mlp_calibration")
    model_dir = os.path.join(current_dir, "load")
    
    dataset_path = os.path.join(data_dir, "mlp_dataset.npy")
    model_path = os.path.join(model_dir, "mlp_gradient_model.pt")
    plot_path = os.path.join(data_dir, "training_history.png")
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"[ERROR] 数据集不存在: {dataset_path}")
        print("[INFO] 请先运行 6_1_collect_mlp_dataset.py 收集数据")
        return
    
    # 训练参数
    config = {
        'input_dim': 5,          # 输入维度: 5(BGR+XY) 或 3(BGR only)
        'hidden_dim': 32,        # 隐藏层维度
        'num_epochs': 200,       # 训练轮数
        'batch_size': 2048,        # 批次大小
        'learning_rate': 0.001,  # 学习率
        'weight_decay': 1e-5,    # 权重衰减
        'train_ratio': 0.8,      # 训练集比例
    }
    
    print("\n" + "=" * 60)
    print("MLP梯度模型训练")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")
    
    # 训练
    model, history = train(
        dataset_path=dataset_path,
        model_save_path=model_path,
        **config
    )
    
    # 保存训练历史
    np.save(os.path.join(data_dir, "training_history.npy"), history)
    
    # 绘制训练曲线
    plot_training_history(history, plot_path)


if __name__ == "__main__":
    main()
