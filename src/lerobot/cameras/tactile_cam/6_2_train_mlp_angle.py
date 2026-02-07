"""
MLP模型训练脚本 (v2 - 与gs_sdk一致)

训练神经网络将 BGRXY 映射到梯度角度 gxyangles
输入: [B, G, R, X, Y] 归一化到 [0,1] (5维)
输出: [gx_angle, gy_angle] 弧度 (2维)

网络结构与 gs_sdk/gs_reconstruct.py 中的 BGRXYMLPNet 一致：
- 3层32维的ReLU MLP

使用方法:
1. 先运行 6_1_collect_mlp_dataset_v2.py 收集数据
2. 运行本脚本训练模型
3. 运行 6_3_test_mlp_v2.py 测试效果
"""

import os
import sys
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# 确保可以导入 lerobot 模块
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


class BGRXYMLPNet(nn.Module):
    """
    与gs_sdk一致的MLP网络结构
    
    输入: BGRXY (5维，归一化到[0,1])
    输出: gxyangles (2维，梯度角度，弧度)
    """
    def __init__(self):
        super(BGRXYMLPNet, self).__init__()
        input_size = 5
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class BGRXYDataset(Dataset):
    """BGRXY数据集 (与gs_sdk一致)"""
    
    def __init__(self, bgrxys, gxyangles):
        self.bgrxys = bgrxys
        self.gxyangles = gxyangles

    def __len__(self):
        return len(self.bgrxys)

    def __getitem__(self, index):
        bgrxy = torch.tensor(self.bgrxys[index], dtype=torch.float32)
        gxyangles = torch.tensor(self.gxyangles[index], dtype=torch.float32)
        return bgrxy, gxyangles


def evaluate(net, dataloader, device):
    """评估网络的MAE"""
    net.eval()
    losses = []
    with torch.no_grad():
        for bgrxys, gxyangles in dataloader:
            bgrxys = bgrxys.to(device)
            gxyangles = gxyangles.to(device)
            outputs = net(bgrxys)
            diffs = outputs - gxyangles
            losses.append(np.abs(diffs.cpu().numpy()))
    mae = np.mean(np.concatenate(losses))
    return mae


def train(dataset_path: str, model_save_dir: str,
          n_epochs: int = 200, batch_size: int = 1024,
          learning_rate: float = 0.002, 
          train_ratio: float = 0.8, device: str = None):
    """
    训练MLP模型 (与gs_sdk一致的训练流程)
    
    Args:
        dataset_path: 数据集路径 (.npz)
        model_save_dir: 模型保存目录
        n_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        train_ratio: 训练集比例
        device: 设备 ('cuda' or 'cpu')
    """
    # 设置设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] 使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 加载数据
    print(f"[INFO] 加载数据集: {dataset_path}")
    data = np.load(dataset_path)
    bgrxys = data['bgrxys']
    gxyangles = data['gxyangles']
    
    print(f"[INFO] 数据形状: bgrxys={bgrxys.shape}, gxyangles={gxyangles.shape}")
    print(f"[INFO] BGRXY范围: [{bgrxys.min():.3f}, {bgrxys.max():.3f}]")
    print(f"[INFO] gxyangles范围: [{gxyangles.min():.4f}, {gxyangles.max():.4f}] rad")
    
    # 划分训练集和测试集
    n_train = int(len(bgrxys) * train_ratio)
    perm = np.random.permutation(len(bgrxys))
    
    train_bgrxys = bgrxys[perm[:n_train]]
    train_gxyangles = gxyangles[perm[:n_train]]
    test_bgrxys = bgrxys[perm[n_train:]]
    test_gxyangles = gxyangles[perm[n_train:]]
    
    print(f"[INFO] 训练集: {len(train_bgrxys)}, 测试集: {len(test_bgrxys)}")
    
    # 创建数据加载器
    train_dataset = BGRXYDataset(train_bgrxys, train_gxyangles)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = BGRXYDataset(test_bgrxys, test_gxyangles)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    net = BGRXYMLPNet().to(device)
    print(f"[INFO] 模型结构:\n{net}")
    
    # 优化器和损失函数 (与gs_sdk一致使用L1Loss和StepLR)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 初始评估
    train_mae = evaluate(net, train_dataloader, device)
    test_mae = evaluate(net, test_dataloader, device)
    naive_mae = np.mean(np.abs(test_gxyangles - np.mean(train_gxyangles, axis=0)))
    
    print(f"\n[INFO] Naive MAE (预测为均值): {naive_mae:.4f} rad")
    print(f"[INFO] 初始 Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    
    # 训练记录
    history = {
        'train_maes': [train_mae], 
        'test_maes': [test_mae], 
        'naive_mae': naive_mae
    }
    best_test_mae = test_mae
    
    print(f"\n[INFO] 开始训练，共 {n_epochs} 轮")
    print("=" * 60)
    
    for epoch_idx in range(n_epochs):
        losses = []
        net.train()
        
        for bgrxys_batch, gxyangles_batch in train_dataloader:
            bgrxys_batch = bgrxys_batch.to(device)
            gxyangles_batch = gxyangles_batch.to(device)
            
            optimizer.zero_grad()
            outputs = net(bgrxys_batch)
            loss = criterion(outputs, gxyangles_batch)
            loss.backward()
            optimizer.step()
            
            diffs = outputs - gxyangles_batch
            losses.append(np.abs(diffs.cpu().detach().numpy()))
        
        net.eval()
        train_mae = np.mean(np.concatenate(losses))
        test_mae = evaluate(net, test_dataloader, device)
        
        history['train_maes'].append(train_mae)
        history['test_maes'].append(test_mae)
        
        # 保存最佳模型
        if test_mae < best_test_mae:
            best_test_mae = test_mae
            save_path = os.path.join(model_save_dir, "nnmodel.pth")
            torch.save(net.state_dict(), save_path)
        
        if (epoch_idx + 1) % 10 == 0 or epoch_idx == 0:
            print(f"Epoch {epoch_idx+1:3d}/{n_epochs} | "
                  f"Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")
        
        scheduler.step()
    
    print("=" * 60)
    print(f"[INFO] 训练完成！最佳测试MAE: {best_test_mae:.4f} rad")
    print(f"[INFO] 模型已保存: {os.path.join(model_save_dir, 'nnmodel.pth')}")
    
    # 保存训练曲线
    save_path = os.path.join(model_save_dir, "training_curve.png")
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(history['train_maes'])), history['train_maes'], 
             color='blue', label='Train MAE')
    plt.plot(np.arange(len(history['test_maes'])), history['test_maes'], 
             color='red', label='Test MAE')
    plt.axhline(y=history['naive_mae'], color='gray', linestyle='--', 
                label=f'Naive MAE: {history["naive_mae"]:.4f}')
    plt.xlabel('Epochs')
    plt.ylabel('MAE (rad)')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] 训练曲线已保存: {save_path}")
    
    return net, history


def main():
    """主函数"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 路径设置
    data_dir = os.path.join(current_dir, "data", "mlp_calibration_v2")
    dataset_path = os.path.join(data_dir, "mlp_dataset_v2.npz")
    model_save_dir = os.path.join(data_dir, "model")
    
    # 同时保存到 load 目录供测试使用
    load_dir = os.path.join(current_dir, "load")
    os.makedirs(load_dir, exist_ok=True)
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"[ERROR] 数据集不存在: {dataset_path}")
        print("[INFO] 请先运行 6_1_collect_mlp_dataset_v2.py 收集数据")
        return
    
    print("=" * 60)
    print("MLP梯度模型训练 (v2 - 与gs_sdk一致)")
    print("=" * 60)
    print("  n_epochs: 200")
    print("  batch_size: 1024")
    print("  learning_rate: 0.001")
    print("  scheduler: StepLR(step=10, gamma=0.5)")
    print("=" * 60)
    
    # 训练
    net, history = train(
        dataset_path=dataset_path,
        model_save_dir=model_save_dir,
        n_epochs=200,
        batch_size=1024,
        learning_rate=0.001,
        train_ratio=0.8,
        device=None
    )
    
    # 复制模型到 load 目录
    import shutil
    src_model = os.path.join(model_save_dir, "nnmodel.pth")
    dst_model = os.path.join(load_dir, "nnmodel_v2.pth")
    shutil.copy(src_model, dst_model)
    print(f"[INFO] 模型已复制到: {dst_model}")
    
    print("\n" + "=" * 60)
    print("完成！现在可以运行 6_3_test_mlp_v2.py 测试效果")
    print("=" * 60)


if __name__ == "__main__":
    main()
