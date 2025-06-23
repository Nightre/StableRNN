import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
input_size = 28
sequence_length = 28
hidden_size = 256
num_layers = 1
num_classes = 10
batch_size = 128
num_epochs = 3
learning_rate = 0.001

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

# RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # 前向传播
        out, _ = self.rnn(x, h0)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 初始化模型
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.squeeze(1).to(device)  # 去除通道维度，形状为[batch_size, sequence_length, input_size]
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

print('训练完成')

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.squeeze(1).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'测试准确率: {100 * correct / total}%')

# 保存模型
torch.save(model.state_dict(), 'rnn_mnist.pth')