

import torch
import torch.nn as nn
import torch.optim as optim
from deepreplay.datasets.parabola import load_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. 加载数据
X, y = load_data()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 2. 定义 PyTorch 模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 2)
        nn.init.xavier_normal_(self.hidden.weight)
        self.output = nn.Linear(2, 1)
        nn.init.normal_(self.output.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden_out = self.sigmoid(self.hidden(x))
        out = self.sigmoid(self.output(hidden_out))
        return out, hidden_out

model = SimpleNN()

# 3. 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=5)

# 4. 训练并收集隐藏层输出和网格变换
epochs = 100
hidden_outputs = []
grid_transforms = []

# 创建网格（输入空间）
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
grid_x, grid_y = np.meshgrid(x_range, y_range)
grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

for epoch in range(epochs):
    optimizer.zero_grad()
    output, hidden_out = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    hidden_outputs.append(hidden_out.detach().numpy())
    _, grid_hidden = model(grid_tensor)
    grid_transforms.append(grid_hidden.detach().numpy())
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 5. 可视化特征空间（静态图）
fig, ax = plt.subplots(figsize=(6, 6))
epoch_to_plot = 50
hidden_out = hidden_outputs[epoch_to_plot - 1]
grid_transform = grid_transforms[epoch_to_plot - 1].reshape(20, 20, 2)

# 绘制散点
scatter = ax.scatter(hidden_out[:, 0], hidden_out[:, 1], c=y.numpy().ravel(), cmap='bwr', edgecolors='k')

# 绘制网格
for i in range(20):
    ax.plot(grid_transform[:, i, 0], grid_transform[:, i, 1], 'k-', lw=0.5)
    ax.plot(grid_transform[i, :, 0], grid_transform[i, :, 1], 'k-', lw=0.5)

ax.set_title(f"Feature Space with Grid at Epoch {epoch_to_plot}")
plt.savefig("feature_space_epoch50_grid_pytorch.png", dpi=120)

# 6. 生成动画
fig, ax = plt.subplots(figsize=(6, 6))

# 初始化散点和网格
hidden_out_init = hidden_outputs[0]
scatter = ax.scatter(hidden_out_init[:, 0], hidden_out_init[:, 1], c=y.numpy().ravel(), cmap='bwr', edgecolors='k')
grid_transform_init = grid_transforms[0].reshape(20, 20, 2)

grid_lines = []
for i in range(20):
    line_h, = ax.plot(grid_transform_init[:, i, 0], grid_transform_init[:, i, 1], 'k-', lw=0.5)
    line_v, = ax.plot(grid_transform_init[i, :, 0], grid_transform_init[i, :, 1], 'k-', lw=0.5)
    grid_lines.append(line_h)
    grid_lines.append(line_v)

ax.set_xlim(np.min([h[:, 0] for h in hidden_outputs]), np.max([h[:, 0] for h in hidden_outputs]))
ax.set_ylim(np.min([h[:, 1] for h in hidden_outputs]), np.max([h[:, 1] for h in hidden_outputs]))

def update(frame):
    # 更新散点
    hidden_out = hidden_outputs[frame]
    scatter.set_offsets(hidden_out)
    
    # 更新网格
    grid_transform = grid_transforms[frame].reshape(20, 20, 2)
    for i in range(20):
        grid_lines[i].set_data(grid_transform[:, i, 0], grid_transform[:, i, 1])
        grid_lines[i + 20].set_data(grid_transform[i, :, 0], grid_transform[i, :, 1])
    
    ax.set_title(f"Feature Space with Grid at Epoch {frame + 1}")
    return [scatter] + grid_lines

animation = FuncAnimation(fig, update, frames=range(epochs), interval=200, blit=True)
animation.save("feature_space_animation_grid_pytorch.mp4", dpi=120, writer='ffmpeg')

print("Visualization completed!")
