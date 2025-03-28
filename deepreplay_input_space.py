
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 设置随机种子以确保结果可重现
torch.manual_seed(41)

# 加载数据
data = np.loadtxt('datasets.txt')
X_loaded = data[:, :2]
y_loaded = data[:, 2:]
X = torch.tensor(X_loaded, dtype=torch.float32)
y = torch.tensor(y_loaded, dtype=torch.float32).view(-1, 1)

# 定义网格大小参数
GRID_SIZE = 10  # 用于可视化网格的粗网格大小
FINE_GRID_SIZE = 100  # 用于决策边界的细网格大小

# 定义 PyTorch 模型
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

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=3)

# 训练并收集数据
epochs = 200
hidden_outputs = []
grid_transforms = []
decision_outputs = []

# 创建网格（输入空间）
x_range = np.linspace(-1, 1, GRID_SIZE)
y_range = np.linspace(-1, 1, GRID_SIZE)
grid_x, grid_y = np.meshgrid(x_range, y_range)
grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

# 用于边界的高分辨率网格
fine_x_range = np.linspace(-1, 1, FINE_GRID_SIZE)
fine_y_range = np.linspace(-1, 1, FINE_GRID_SIZE)
fine_grid_x, fine_grid_y = np.meshgrid(fine_x_range, fine_y_range)
fine_grid_points = np.stack([fine_grid_x.ravel(), fine_grid_y.ravel()], axis=1)
fine_grid_tensor = torch.tensor(fine_grid_points, dtype=torch.float32)

for epoch in range(epochs):
    optimizer.zero_grad()
    output, hidden_out = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    hidden_outputs.append(hidden_out.detach().numpy())
    _, grid_hidden = model(grid_tensor)
    grid_transforms.append(grid_hidden.detach().numpy())
    fine_output, _ = model(fine_grid_tensor)
    decision_outputs.append(fine_output.detach().numpy().reshape(FINE_GRID_SIZE, FINE_GRID_SIZE))
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print(f"Decision output range at epoch {epochs}: {decision_outputs[-1].min()} - {decision_outputs[-1].max()}")

# 可视化输入空间的边界
fig, ax = plt.subplots(figsize=(6, 6))
epoch_to_plot = epoch
decision_out = decision_outputs[epoch_to_plot - 1]

# 绘制背景颜色（输入空间）
ax.contourf(fine_grid_x, fine_grid_y, decision_out, levels=[0, 0.5, 1], colors=['#00BFFF', '#DDA0DD'], alpha=0.3)
contour = ax.contour(fine_grid_x, fine_grid_y, decision_out, levels=[0.5], colors='k', linewidths=2)

# 绘制输入空间的网格
for i in range(GRID_SIZE):
    ax.plot(grid_x[:, i], grid_y[:, i], 'k-', lw=0.5)
    ax.plot(grid_x[i, :], grid_y[i, :], 'k-', lw=0.5)

# 绘制数据点
ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy().ravel(), cmap='bwr', edgecolors='none')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title(f"Input Space with Boundary at Epoch {epoch_to_plot}")
plt.savefig("input_space_epoch200_boundary_pytorch.png", dpi=120)

# 可视化特征空间
fig, ax = plt.subplots(figsize=(6, 6))
hidden_out = hidden_outputs[epoch_to_plot - 1]
grid_transform = grid_transforms[epoch_to_plot - 1].reshape(GRID_SIZE, GRID_SIZE, 2)

# 绘制散点
scatter = ax.scatter(hidden_out[:, 0], hidden_out[:, 1], c=y.numpy().ravel(), cmap='bwr', edgecolors='none')

# 绘制网格
for i in range(GRID_SIZE):
    ax.plot(grid_transform[:, i, 0], grid_transform[:, i, 1], 'k-', lw=0.5)
    ax.plot(grid_transform[i, :, 0], grid_transform[i, :, 1], 'k-', lw=0.5)

ax.set_title(f"Feature Space with Grid at Epoch {epoch_to_plot}")
plt.savefig("feature_space_epoch140_grid_pytorch.png", dpi=120)

# 生成动画（输入空间）
fig, ax = plt.subplots(figsize=(6, 6))

# 初始化
decision_out_init = decision_outputs[0]
contourf = ax.contourf(fine_grid_x, fine_grid_y, decision_out_init, levels=[0, 0.5, 1], colors=['#00BFFF', '#DDA0DD'], alpha=0.3)
contour = ax.contour(fine_grid_x, fine_grid_y, decision_out_init, levels=[0.5], colors='k', linewidths=2)
scatter = ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy().ravel(), cmap='bwr', edgecolors='none')

grid_lines = []
for i in range(GRID_SIZE):
    line_h, = ax.plot(grid_x[:, i], grid_y[:, i], 'k-', lw=0.5)
    line_v, = ax.plot(grid_x[i, :], grid_y[i, :], 'k-', lw=0.5)
    grid_lines.append(line_h)
    grid_lines.append(line_v)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel('X1')
ax.set_ylabel('X2')

def update(frame):
    global contourf, contour
    for coll in contourf.collections:
        coll.remove()
    for coll in contour.collections:
        coll.remove()

    decision_out = decision_outputs[frame]
    contourf = ax.contourf(fine_grid_x, fine_grid_y, decision_out, levels=[0, 0.5, 1], colors=['#00BFFF', '#DDA0DD'], alpha=0.3)
    contour = ax.contour(fine_grid_x, fine_grid_y, decision_out, levels=[0.5], colors='k', linewidths=2)

    ax.set_title(f"Input Space with Boundary at Epoch {frame + 1}")
    return [scatter] + grid_lines + contourf.collections + contour.collections

animation = FuncAnimation(fig, update, frames=range(epochs), interval=200, blit=False)
animation.save("input_space.gif", dpi=120, writer=PillowWriter(fps=24))

print("Visualization completed!")
