"""
ST-GCN 摔倒检测模型训练

模型: 时空图卷积网络 (Spatial-Temporal Graph Convolutional Network)
输入: (B, T, V, C) — 批量 × 时间帧 × 17关键点 × 3坐标(x,y,conf)
输出: 2类 — fall / normal

数据集要求:
  data/fall_dataset/
    processed/
      X_train.npy  (N, T, V, C)  训练骨架
      y_train.npy  (N,)          0=normal, 1=fall
      X_val.npy    (M, T, V, C)  验证骨架
      y_val.npy    (M,)

用法:
  # 1. 准备数据 (用预处理脚本)
  python scripts/preprocess_fall.py

  # 2. 训练
  python scripts/training/train_fall_stgcn.py --epochs 100 --batch 32
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
# ST-GCN 模型
# ═══════════════════════════════════════════════════════════

class GraphConv(nn.Module):
    """图卷积层"""
    def __init__(self, in_channels, out_channels, num_nodes=17):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        # 可学习邻接矩阵
        self.A = nn.Parameter(torch.randn(3, num_nodes, num_nodes) * 0.02)
        self.num_nodes = num_nodes

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.shape
        A = self.A + self.A.transpose(-1, -2)  # 对称化
        A = torch.softmax(A, dim=-1)
        # 图卷积
        x_out = self.conv(x)  # (B, C_out, T, V)
        out = torch.zeros_like(x_out)
        for k in range(3):
            out = out + torch.einsum('nctv,vw->nctw', x_out, A[k])
        return out


class TCN(nn.Module):
    """时间卷积"""
    def __init__(self, channels, kernel_size=9, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(channels, channels, (kernel_size, 1),
                              stride=(stride, 1), padding=(pad, 0))

    def forward(self, x):
        return self.conv(x)


class STGCNBlock(nn.Module):
    """ST-GCN 基本块: GCN + TCN + residual"""
    def __init__(self, in_ch, out_ch, num_nodes=17, stride=1, residual=False):
        super().__init__()
        self.gcn = GraphConv(in_ch, out_ch, num_nodes)
        self.tcn = TCN(out_ch, kernel_size=9, stride=stride)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual
        if residual:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1))
            self.res_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        res = x
        out = self.gcn(x)
        out = self.tcn(out)
        out = self.bn(out)
        if self.residual:
            res = self.res_conv(res)
            res = self.res_bn(res)
            out = out + res
        return self.relu(out)


class STGCN(nn.Module):
    """ST-GCN 摔倒检测网络"""
    def __init__(self, in_channels=3, num_classes=2, num_nodes=17,
                 edge_importance=True):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        self.stgcn1 = STGCNBlock(in_channels, 64, num_nodes, residual=False)
        self.stgcn2 = STGCNBlock(64, 64, num_nodes, residual=True)
        self.stgcn3 = STGCNBlock(64, 64, num_nodes, residual=True)
        self.stgcn4 = STGCNBlock(64, 128, num_nodes, stride=2, residual=True)
        self.stgcn5 = STGCNBlock(128, 128, num_nodes, residual=True)
        self.stgcn6 = STGCNBlock(128, 128, num_nodes, residual=True)
        self.stgcn7 = STGCNBlock(128, 256, num_nodes, stride=2, residual=True)
        self.stgcn8 = STGCNBlock(256, 256, num_nodes, residual=True)
        self.stgcn9 = STGCNBlock(256, 256, num_nodes, residual=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, C, T, V)
        N, C, T, V = x.shape

        # BN
        x = x.permute(0, 2, 3, 1).reshape(N, T, V * C)
        x = self.data_bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.reshape(N, C, T, V)

        # ST-GCN blocks
        x = self.stgcn1(x)
        x = self.stgcn2(x)
        x = self.stgcn3(x)
        x = self.stgcn4(x)
        x = self.stgcn5(x)
        x = self.stgcn6(x)
        x = self.stgcn7(x)
        x = self.stgcn8(x)
        x = self.stgcn9(x)

        # Pool + classify
        x = self.pool(x)  # (B, 256, 1, 1)
        x = x.view(N, -1)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════

class FallDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)  # (N, C, T, V)
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X) * (4 if self.augment else 1)  # 4x 增强

    def __getitem__(self, idx):
        real_idx = idx % len(self.X) if self.augment else idx
        x = self.X[real_idx].clone()
        y = self.y[real_idx]

        if self.augment:
            aug_type = idx // len(self.X)  # 0=原始, 1-3=增强
            # x: (C, T, V) = (3, 60, 17)
            if aug_type == 1:
                # 空间旋转: 绕原点随机旋转 ±30°
                angle = (torch.rand(1).item() - 0.5) * 60 * 3.14159 / 180
                cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                x_xy = x[:2].clone()  # (2, T, V)
                x_xy_new = torch.stack([
                    x_xy[0] * cos_a - x_xy[1] * sin_a,
                    x_xy[0] * sin_a + x_xy[1] * cos_a
                ])
                x[:2] = x_xy_new
            elif aug_type == 2:
                T = x.shape[1]
                mask = torch.rand(T) > 0.1
                if mask.sum() > 5:
                    idx_valid = mask.nonzero().squeeze()
                    x = x[:, idx_valid]
                    C, _, V = x.shape
                    xt = x.permute(1,0,2).reshape(-1, C*V).T.unsqueeze(0)
                    xt = torch.nn.functional.interpolate(xt, size=60, mode='linear', align_corners=False).squeeze(0)
                    x = xt.reshape(C, V, 60).permute(0, 2, 1)
            elif aug_type == 3:
                # 关节点噪声
                noise = torch.randn_like(x) * 0.03
                noise[:, :, :] *= (x[2:3] > 0.1)  # 只在高置信度点加噪声
                x = x + noise
                # 镜像 (左右交换)
                if torch.rand(1) > 0.5:
                    swap_pairs = [(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
                    for a, b in swap_pairs:
                        tmp = x[:, :, a].clone()
                        x[:, :, a] = x[:, :, b]
                        x[:, :, b] = tmp
        return x, y


# ═══════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════

def train_model(args):
    root = get_project_root()
    data_dir = args.data or os.path.join(root, "data", "fall_dataset", "processed")
    output_dir = args.output or os.path.join(root, "models", "fall_detection")

    # 加载数据
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))

    # 转换为 (N, C, T, V) 格式
    if X_train.ndim == 4:  # (N, T, V, C) → (N, C, T, V)
        X_train = X_train.transpose(0, 3, 1, 2)
        X_val = X_val.transpose(0, 3, 1, 2)

    print(f"训练: {len(X_train)}, 验证: {len(X_val)}")
    print(f"正样本 (fall): {y_train.sum()}, 负样本 (normal): {len(y_train)-y_train.sum()}")

    train_ds = FallDataset(X_train, y_train, augment=True)
    val_ds = FallDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = STGCN(in_channels=X_train.shape[1], num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"Loss: {train_loss/len(train_loader):.4f}  "
              f"Acc: {acc:.4f}  {'  BEST!' if acc > best_acc else ''}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "stgcn_fall.pt"))

    print(f"\n最佳准确率: {best_acc:.4f}")

    # 导出 ONNX
    print("导出 ONNX...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "stgcn_fall.pt")))
    model.eval()
    dummy = torch.randn(1, X_train.shape[1], 60, 17).to(device)
    torch.onnx.export(model, dummy, os.path.join(output_dir, "stgcn_fall.onnx"),
                      input_names=["input"], output_names=["output"],
                      opset_version=12, dynamic_axes={"input": {2: "time"}})
    print(f"ONNX 已保存: {output_dir}/stgcn_fall.onnx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    root = get_project_root()
    processed_dir = os.path.join(root, "data", "fall_dataset", "processed")
    if not os.path.exists(os.path.join(processed_dir, "X_train.npy")):
        print("=" * 50)
        print("请先运行数据预处理: python scripts/preprocess_fall.py")
        print("=" * 50)
        print()
        print("数据集下载:")
        print("  UR Fall Detection: http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html")
        print("  或 Kaggle: https://www.kaggle.com/datasets/topcow/ur-fall-detection-dataset")
        print()
        print("下载后解压到 data/fall_dataset/raw/")
        return

    train_model(args)


if __name__ == "__main__":
    main()
