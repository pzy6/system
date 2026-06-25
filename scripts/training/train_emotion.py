"""
训练情绪识别模型 (EfficientNet-B0 + FER2013)

数据: FER2013 (Kaggle) — 35,887 张 48×48 灰度人脸, 7 类情绪
模型: EfficientNet-B0, 输入 224×224, 输出 7 类
导出: ONNX → models/emotion/efficient_emotion.onnx

用法:
  # 1. 下载数据
  #    https://www.kaggle.com/datasets/msambare/fer2013
  #    解压到 data/fer2013/ (含 train/ 和 test/ 子目录)

  # 2. 训练
  python scripts/training/train_emotion.py

  # 3. 导出 ONNX (自动完成)
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_model(num_classes=7):
    """创建 EfficientNet-B0 情绪分类模型"""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # 替换分类头
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


def train(args):
    import sys
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

    root = get_project_root()
    data_dir = args.data or os.path.join(root, "data", "fer2013")
    output_dir = args.output or os.path.join(root, "models", "emotion")

    if not os.path.isdir(data_dir):
        print(f"数据目录不存在: {data_dir}")
        print("请从 Kaggle 下载 FER2013: https://www.kaggle.com/datasets/msambare/fer2013")
        print("解压后目录结构: data/fer2013/train/ 和 data/fer2013/test/")
        return

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # EfficientNet 需 3 通道
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"), transform=val_transform
    )

    print(f"训练集: {len(train_dataset)} 张, 类别: {train_dataset.classes}")
    print(f"验证集: {len(val_dataset)} 张")

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = create_model(len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"Loss: {train_loss/len(train_loader):.4f}  "
              f"Acc: {acc:.4f}  {'  BEST!' if acc > best_acc else ''}", flush=True)

        if acc > best_acc:
            best_acc = acc
        # 每个 epoch 都保存（不仅最佳）
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "efficient_emotion.pt"))

    print(f"\n最佳准确率: {best_acc:.4f}")

    # 导出 ONNX
    print("导出 ONNX...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "efficient_emotion.pt")))
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model, dummy,
        os.path.join(output_dir, "efficient_emotion.onnx"),
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
    )
    print(f"ONNX 已保存: {output_dir}/efficient_emotion.onnx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
