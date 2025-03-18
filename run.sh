#!/bin/bash
# 按照原始ResNet论文参数设置训练多个模型

# 创建结果目录
mkdir -p ./results

echo "===== 开始训练 CIFAR-10 专用 ResNet 模型 ====="

# 训练CIFAR专用的ResNet20
echo "训练 ResNet20..."
python train.py --model ResNet20 --batch-size 128 --lr 0.1 --epochs 164

# 训练CIFAR专用的ResNet32
echo "训练 ResNet32..."
python train.py --model ResNet32 --batch-size 128 --lr 0.1 --epochs 164

# 训练CIFAR专用的ResNet44
echo "训练 ResNet44..."
python train.py --model ResNet44 --batch-size 128 --lr 0.1 --epochs 164

# 训练CIFAR专用的ResNet56
echo "训练 ResNet56..."
python train.py --model ResNet56 --batch-size 128 --lr 0.1 --epochs 164

# 训练CIFAR专用的ResNet110
echo "训练 ResNet110..."
python train.py --model ResNet110 --batch-size 128 --lr 0.1 --epochs 164

echo "===== 开始训练标准 ResNet 模型 ====="

# 训练标准ResNet18
echo "训练 ResNet18..."
python train.py --model ResNet18 --batch-size 128 --lr 0.1 --epochs 164

# 训练标准ResNet34
echo "训练 ResNet34..."
python train.py --model ResNet34 --batch-size 128 --lr 0.1 --epochs 164

# 训练标准ResNet50
echo "训练 ResNet50..."
python train.py --model ResNet50 --batch-size 128 --lr 0.1 --epochs 164

echo "===== 训练超深网络 ResNet1202（使用较小批量大小） ====="
python train.py --model ResNet1202 --batch-size 64 --lr 0.1 --epochs 164

echo "===== 所有模型训练完成 ====="
