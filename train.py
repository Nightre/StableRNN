import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.jit as jit
from typing import List, Tuple
import matplotlib.pyplot as plt
import os

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
input_size = 28
sequence_length = 28
hidden_size = 256
num_classes = 10
batch_size = 128
learning_rate = 0.001

num_epochs_memory = 3  # 记忆训练轮数
num_epochs_classify = 3  # 分类训练轮数


class MemoryModule(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(MemoryModule, self).__init__()
        self.hidden_size = hidden_size
        self.input_state_size = hidden_size + input_size
        self.de_target_size = input_size + hidden_size

        # 记忆网络组件
        self.state_decoder = nn.Sequential(
            nn.Linear(self.input_state_size, 64),
            nn.GELU(),
            nn.Linear(64, hidden_size)
        )
        
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, self.de_target_size)
        )
    
    @jit.script_method
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        prev_h = torch.zeros_like(h_t)
        
        # 预分配输出张量
        all_encoder_outputs = torch.zeros(batch_size, seq_len, self.de_target_size, device=x.device)
        all_encoder_targets = torch.zeros(batch_size, seq_len, self.de_target_size, device=x.device)
        all_hidden_state = torch.zeros(batch_size, seq_len, self.hidden_size, device=x.device)
        
        # 使用并行计算优化
        for t in range(seq_len):
            x_t = x[:, t, :]
            prev_x = torch.zeros_like(x_t) if t == 0 else x[:, t-1, :]
            
            # 计算下一隐藏状态
            combined_input = torch.cat([x_t, h_t], dim=1)
            next_h = self.state_decoder(combined_input)
            
            # 保存目标和输出
            combined_target = torch.cat([x_t, prev_h], dim=1)
            combined_input = next_h
            
            all_encoder_targets[:, t] = combined_target
            all_encoder_outputs[:, t] = self.state_encoder(combined_input)
            all_hidden_state[:, t] = next_h
            
            # 更新状态 - 使用更高效的梯度控制
            prev_h = h_t.clone().detach()
            h_t = next_h.clone().detach()
        
        return all_encoder_outputs, all_encoder_targets, all_hidden_state
    
class ClassifyModule(jit.ScriptModule):
    def __init__(self, hidden_size, num_classes):
        super(ClassifyModule, self).__init__()
        self.units = 128
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, self.units),
            nn.GELU(),
            nn.Linear(self.units, self.units),
            nn.GELU(),
            nn.Linear(self.units, num_classes),
        )
    
    @jit.script_method
    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        return self.fc(h_t)
    
if __name__ == "__main__":

    # MNIST数据集
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=True, 
                                            transform=transforms.ToTensor(),
                                            download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                            transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False)

    # 初始化模型
    memory_model = MemoryModule(input_size, hidden_size).to(device)
    classify_model = ClassifyModule(hidden_size, num_classes).to(device)

    # 优化器分开配置
    memory_optimizer = torch.optim.Adam(memory_model.parameters(), lr=learning_rate)
    classify_optimizer = torch.optim.Adam(classify_model.parameters(), lr=learning_rate)

    # 损失函数
    reconstruction_loss = nn.SmoothL1Loss()
    ce_loss = nn.CrossEntropyLoss()

    # 训练记忆模块
    print("开始训练记忆模块...")
    for epoch in range(num_epochs_memory):
        memory_model.train()
        for i, (images, _) in enumerate(train_loader):  # 不需要标签
            images = images.squeeze(1).to(device)
            
            # 前向传播
            encoder_output, encoder_target, _ = memory_model(images)
            loss = reconstruction_loss(encoder_output, encoder_target)
            
            # 反向传播和优化
            memory_optimizer.zero_grad()
            loss.backward()
            memory_optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f'记忆训练 Epoch [{epoch+1}/{num_epochs_memory}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        print(f"[*] 记忆训练 Epoch {epoch+1} 完成")

    # 训练分类模块
    print("\n开始训练分类模块...")
    for epoch in range(num_epochs_classify):
        classify_model.train()
        memory_model.eval()  # 冻结记忆模块
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.squeeze(1).to(device)
            labels = labels.to(device)
            
            # 获取记忆模块的最终隐藏状态
            with torch.no_grad():
                _, encoder_target, all_hidden_state = memory_model(images)
                # 从目标中提取最终隐藏状态 (batch_size, hidden_size)
                h_t = all_hidden_state[:, -1, :]
            
            # 分类前向传播
            pred = classify_model(h_t)
            loss = ce_loss(pred, labels)
            
            # 反向传播和优化
            classify_optimizer.zero_grad()
            loss.backward()
            classify_optimizer.step()
            
            if (i+1) % 100 == 0:
                _, predicted = torch.max(pred, 1)
                accuracy = (predicted == labels).float().mean().item()
                print(f'分类训练 Epoch [{epoch+1}/{num_epochs_classify}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {accuracy*100:.2f}%')
        
        print(f"[*] 分类训练 Epoch {epoch+1} 完成")

    print('训练完成')

    # 测试模型
    memory_model.eval()
    classify_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.squeeze(1).to(device)
            labels = labels.to(device)
            
            # 获取记忆模块的最终隐藏状态
            _, encoder_target, m = memory_model(images)
            h_t = m[:, -1, :]
            
            # 分类
            outputs = classify_model(h_t)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'测试准确率: {100 * correct / total:.2f}%')

    # 从测试集中取前几个样本
    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images[:12].squeeze(1).to(device)  # 取前12张图片
    sample_labels = sample_labels[:12].to(device)

    # 获取预测结果
    with torch.no_grad():
        _, _, hidden_states = memory_model(sample_images)
        h_t = hidden_states[:, -1, :]  # 每个样本的最终 hidden state
        preds = classify_model(h_t)
        _, predicted_labels = torch.max(preds, 1)

    # 显示图像和预测
    plt.figure(figsize=(12, 4))
    for i in range(12):
        plt.subplot(2, 6, i + 1)
        plt.imshow(sample_images[i].cpu().numpy(), cmap='gray')
        plt.title(f'Pred: {predicted_labels[i].item()}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("visuals/sample_predictions.png")
    plt.show()

    print("已保存并展示 sample_predictions.png")

    # 保存模型
    torch.save({
        'memory': memory_model.state_dict(),
        'classify': classify_model.state_dict()
    }, 'separate_rnn_mnist.pth')