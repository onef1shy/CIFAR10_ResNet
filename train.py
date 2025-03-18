import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
import csv

from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202

# Parse command line arguments
parser = argparse.ArgumentParser(description='CIFAR-10 Training with ResNet')
parser.add_argument('--model', type=str, default='ResNet18',
                    choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                             'ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202'],
                    help='Choose ResNet model to train')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate')
parser.add_argument('--epochs', type=int, default=164,
                    help='Total training epochs')
parser.add_argument('--results-dir', type=str,
                    default='./results', help='Directory to save results')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of worker threads for data loading')
args = parser.parse_args()


def main():
    # Get model name
    model_name = args.model

    # Create model-specific results directory
    model_results_dir = os.path.join(args.results_dir, model_name)
    model_plot_dir = os.path.join(model_results_dir, 'plots')

    # Create necessary directories
    os.makedirs(model_results_dir, exist_ok=True)
    os.makedirs(model_plot_dir, exist_ok=True)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CUDA optimization settings
    if torch.cuda.is_available():
        # Enable cudnn auto-tuner
        torch.backends.cudnn.benchmark = True
        # Use deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = False
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(
            f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")

    # Hyperparameters
    # 原始ResNet论文中，总共训练64k迭代，批量大小为128
    # 对于CIFAR-10数据集（50k训练样本），这相当于约164个epoch
    epochs = args.epochs  # 默认164个epoch
    BATCH_SIZE = args.batch_size  # 默认批量大小128
    LR = args.lr  # 默认初始学习率0.1

    # Data preprocessing
    transform_train = transforms.Compose([
        # 原始论文中的数据增强：4像素padding后随机裁剪到32x32
        transforms.RandomCrop(32, padding=4),
        # 随机水平翻转
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # 使用CIFAR-10的均值和标准差进行归一化
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # 测试集只需要归一化，不需要数据增强
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    print("Loading datasets...")
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                 transform=transform_train)  # Training dataset
    # Use multiple workers on Windows and enable pin_memory for faster GPU transfer
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)

    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_num = len(val_dataset)
    train_num = len(train_dataset)
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    print(f"Training set size: {train_num}, Validation set size: {val_num}")

    # CIFAR-10 labels
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Select model based on command line arguments
    model_dict = {
        'ResNet18': ResNet18,
        'ResNet34': ResNet34,
        'ResNet50': ResNet50,
        'ResNet101': ResNet101,
        'ResNet152': ResNet152,
        'ResNet20': ResNet20,
        'ResNet32': ResNet32,
        'ResNet44': ResNet44,
        'ResNet56': ResNet56,
        'ResNet110': ResNet110,
        'ResNet1202': ResNet1202
    }

    # Model definition
    print(f"Using {model_name} model for training...")
    model = model_dict[model_name]().to(device)

    # Define model filename
    model_filename = os.path.join(model_results_dir, f"{model_name}_best.pth")

    # Define results files
    results_csv_path = os.path.join(
        model_results_dir, f"{model_name}_results.csv")
    results_txt_path = os.path.join(
        model_results_dir, f"{model_name}_results.txt")

    # Define loss function and optimizer
    # Cross entropy loss for multi-class classification
    loss_function = nn.CrossEntropyLoss()
    # 原始ResNet论文中使用SGD优化器，动量为0.9，权重衰减为1e-4
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=1e-4)  # SGD with momentum and L2 regularization

    # For tracking training metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    lr_history = []

    # Learning rate scheduler based on model type
    if model_name in ['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56', 'ResNet110', 'ResNet1202']:
        # 原始ResNet论文中，学习率在32k和48k迭代时降低，总共训练64k迭代
        # 对于批量大小128，这相当于在第82和123个epoch降低学习率（总共164个epoch）
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[82, 123], gamma=0.1)
    else:
        # 对标准ResNet使用相同的学习率调度
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[82, 123], gamma=0.1)

    # Record best validation accuracy and loss
    best_acc = 0.0
    best_loss = float('inf')
    best_epoch = 0  # Track which epoch had the best performance

    # Create CSV file for logging results
    with open(results_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy',
                            'Val Accuracy', 'Learning Rate', 'Epoch Time(s)'])

    # Record training start time
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        # train
        print(f"------- Epoch {epoch + 1} training start -------")
        model.train()
        train_acc = 0.0
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"
            _, predict = torch.max(outputs, dim=1)
            train_acc += torch.eq(predict, labels).sum().item()

        train_loss = running_loss / train_steps
        train_accurate = train_acc / train_num

        # val
        model.eval()
        val_acc = 0.0
        val_running_loss = 0.0

        # For confusion matrix
        all_preds = []
        all_labels = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for step, val_data in enumerate(val_bar):
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(
                    device), val_labels.to(device)

                outputs = model(val_images)
                loss = loss_function(outputs, val_labels)

                val_running_loss += loss.item()

                _, predict = torch.max(outputs, dim=1)
                val_acc += torch.eq(predict, val_labels).sum().item()

                # Collect predictions and labels for confusion matrix
                all_preds.extend(predict.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())

        val_loss = val_running_loss / val_steps
        val_accurate = val_acc / val_num

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        # Record training metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_accurate)
        val_accs.append(val_accurate)
        lr_history.append(current_lr)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start

        print(f'Current learning rate: {current_lr}')
        print(f'[epoch {epoch + 1}] train_loss: {train_loss:.3f} val_loss:{val_loss:.3f} train_accuracy:{train_accurate:.3f} val_accuracy: {val_accurate:.3f} time: {epoch_time:.1f}s')

        # Log results to CSV file
        with open(results_csv_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                f"{train_accurate:.6f}", f"{val_accurate:.6f}",
                                f"{current_lr:.6f}", f"{epoch_time:.1f}"])

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Validation loss improved to {best_loss:.6f}")

        # If current model performs better in accuracy, save as best model and create visualizations
        if val_accurate > best_acc:
            best_acc = val_accurate
            best_epoch = epoch + 1
            best_train_acc = train_accurate
            best_train_loss = train_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_accurate,
                "val_loss": val_loss,
                "train_accuracy": train_accurate,
                "train_loss": train_loss,
                "epoch": epoch + 1
            }, model_filename)
            print(f"Found better model, saved to: {model_filename}")

            # When finding the best model, draw confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'{model_name} Confusion Matrix (Epoch {epoch+1})')
            confusion_matrix_path = os.path.join(
                model_plot_dir, f"confusion_matrix.png")
            plt.savefig(confusion_matrix_path)
            plt.close()
            print(f"Saved confusion matrix to: {confusion_matrix_path}")

            # Plot loss and accuracy curves
            plt.figure(figsize=(12, 5))

            # Plot loss curves
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(train_losses) + 1),
                     train_losses, label='Training Loss')
            plt.plot(range(1, len(val_losses) + 1),
                     val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{model_name} Loss Curves')
            plt.legend()
            plt.grid(True)

            # Plot accuracy curves
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(train_accs) + 1),
                     train_accs, label='Training Accuracy')
            plt.plot(range(1, len(val_accs) + 1),
                     val_accs, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'{model_name} Accuracy Curves')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            loss_acc_path = os.path.join(
                model_plot_dir, f"loss_acc_curves.png")
            plt.savefig(loss_acc_path)
            plt.close()
            print(f"Saved loss and accuracy curves to: {loss_acc_path}")

            # Plot learning rate curve
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(lr_history) + 1), lr_history)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title(f'{model_name} Learning Rate Curve')
            plt.grid(True)
            # Use log scale to better show learning rate changes
            plt.yscale('log')
            lr_path = os.path.join(model_plot_dir, f"lr_curve.png")
            plt.savefig(lr_path)
            plt.close()
            print(f"Saved learning rate curve to: {lr_path}")

            # Visualize some predictions
            visualize_predictions(model, val_loader, classes, device)
            pred_path = os.path.join(
                model_plot_dir, f"predictions.png")
            plt.savefig(pred_path)
            plt.close()
            print(f"Saved prediction visualization to: {pred_path}")

    print(f"Training complete. Best model saved to: {model_filename}")
    print(
        f"Best validation accuracy: {best_acc:.4f}, Best validation loss: {best_loss:.6f} (Epoch {best_epoch})")

    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # Save final summary to txt file
    with open(results_txt_path, 'w') as f:
        f.write(f"模型: {model_name}\n")
        f.write(f"训练参数:\n")
        f.write(f"  批量大小: {BATCH_SIZE}\n")
        f.write(f"  初始学习率: {LR}\n")
        f.write(f"  训练轮数: {epoch + 1}/{epochs}\n")
        f.write(f"\n")

        f.write(f"训练结果:\n")
        f.write(f"  最佳验证准确率: {best_acc:.6f} (轮次 {best_epoch})\n")
        f.write(f"  最佳验证损失: {best_loss:.6f}\n")
        f.write(f"  最佳训练准确率: {best_train_acc:.6f}\n")
        f.write(f"  最佳训练损失: {best_train_loss:.6f}\n\n")

        f.write(f"训练时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒\n")
        f.write(
            f"设备: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n\n")

        f.write(f"文件位置:\n")
        f.write(f"  最佳模型: {model_filename}\n")
        f.write(f"  CSV结果: {results_csv_path}\n")
        f.write(f"  可视化结果: {model_plot_dir}\n")

    print(f"Saved summary results to: {results_txt_path}")
    print(f"All results saved to: {model_results_dir}")


def visualize_predictions(model, data_loader, classes, device, num_images=10):
    """Visualize model predictions"""
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if images_so_far >= num_images:
                    return

                images_so_far += 1
                ax = plt.subplot(2, 5, images_so_far)
                ax.set_title(
                    f'Pred: {classes[preds[j]]}\nTrue: {classes[labels[j]]}')
                ax.axis('off')

                # Denormalize image
                mean = torch.tensor([0.4914, 0.4822, 0.4465])
                std = torch.tensor([0.2023, 0.1994, 0.2010])
                inp = inputs.cpu()[j].numpy().transpose((1, 2, 0))
                inp = std.numpy() * inp + mean.numpy()
                inp = np.clip(inp, 0, 1)

                ax.imshow(inp)

                if images_so_far == num_images:
                    break


if __name__ == "__main__":
    main()
