# CIFAR-10 图像分类 ResNet 实现

本项目是对论文《Deep Residual Learning for Image Recognition》(He et al., 2016) 中提出的残差网络 (ResNet) 的复现，专注于 CIFAR-10 数据集上的图像分类任务。

详细的实现分析与实验结果请查看我的博客文章：[深度残差网络(ResNet)代码实现详解：PyTorch复现CIFAR-10图像分类](https://onef1shy.github.io/blog.html?post=ResNet-Code)

## 项目简介

残差网络 (ResNet) 是一种深度卷积神经网络架构，通过引入残差连接（跳跃连接）解决了深度神经网络训练中的梯度消失问题。本项目实现了多种深度的 ResNet 变体，包括标准 ResNet (ResNet18/34/50/101/152) 和专为 CIFAR-10 设计的变体 (ResNet20/32/44/56/110/1202)。

## 关于复现与最新技术

**重要说明**：本项目旨在精确复现2016年发表的原始ResNet论文中的方法和结果，而非展示当前最先进的图像分类技术。自ResNet发表以来，深度学习领域已经有了显著的进步。

## 项目结构

```
CIFAR10_ResNet/
├── resnet.py          # ResNet 模型定义
├── train.py           # 训练和评估脚本
├── run.sh             # 训练脚本示例
├── README.md          # 项目说明文档
├── data/              # 数据集存储目录 (自动下载)
└── results/           # 结果保存目录
    └── [ModelName]/   # 每个模型的结果目录
        ├── [ModelName]_best.pth    # 最佳模型权重
        ├── [ModelName]_results.csv # 详细训练结果
        ├── [ModelName]_results.txt # 训练结果摘要
        └── plots/                  # 可视化结果目录
            ├── confusion_matrix.png  # 混淆矩阵
            ├── loss_acc_curves.png   # 损失和准确率曲线
            ├── lr_curve.png          # 学习率曲线
            └── predictions.png       # 预测可视化
```

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- scikit-learn
- seaborn
- tqdm

可以通过以下命令安装依赖：

```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn tqdm
```

## 使用方法

### 基本训练

```bash
# 训练 ResNet18 (默认)
python train.py

# 训练 ResNet50
python train.py --model ResNet50

# 训练 CIFAR 专用的 ResNet20
python train.py --model ResNet20
```

### 自定义训练参数

```bash
# 自定义批量大小、学习率和训练轮数
python train.py --model ResNet50 --batch-size 64 --lr 0.01 --epochs 200

# 训练超深网络 ResNet1202 (建议减小批量大小)
python train.py --model ResNet1202 --batch-size 64
```

### 使用 run.sh 批量训练所有模型

项目提供了 `run.sh` 脚本用于按照原始 ResNet 论文的参数设置批量训练多个模型：

```bash
# 给予执行权限
chmod +x run.sh

# 运行脚本
./run.sh
```

该脚本会按顺序训练以下模型：
1. CIFAR-10 专用模型：ResNet20、ResNet32、ResNet44、ResNet56、ResNet110
2. 标准 ResNet 模型：ResNet18、ResNet34、ResNet50
3. 超深网络：ResNet1202（使用较小批量大小64）

所有模型均使用统一的训练参数：初始学习率0.1，训练164轮，在第82和123轮降低学习率。

## 模型性能概览

在CIFAR-10测试集上，不同模型的典型性能如下：

| 模型       | 参数量   | 论文准确率 | 复现准确率   |
| ---------- | -------- | ---------- | ------------ |
| ResNet20   | 0.27M    | 91.25%     | 91.61%       |
| ResNet32   | 0.47M    | 92.49%     | 92.44%       |
| ResNet44   | 0.66M    | 92.83%     | 92.84%       |
| ResNet56   | 0.86M    | 93.03%     | 93.05%       |
| ResNet110  | 1.73M    | 93.57%     | 92.71%       |
| ResNet1202 | 19.42M   | 92.07%     | 93.61%       |
| ResNet18   | 11.17M   | -          | 94.42%       |
| ResNet34   | 21.28M   | -          | 94.84%       |
| ResNet50   | 23.52M   | -          | 92.68%       |

*注：原论文没有在CIFAR-10上测试标准ResNet(18/34/50)，这些是本项目的额外实验。

更详细的性能分析和实验结果请参阅[博客文章](https://onef1shy.github.io/blog.html?post=ResNet-Code)。

## License

MIT License © [onefishy](https://github.com/onef1shy)

## ⭐ 支持项目

欢迎 Fork 和 Star ⭐，也欢迎提出建议和 PR～