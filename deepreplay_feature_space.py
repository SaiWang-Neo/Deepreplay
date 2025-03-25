
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import CubicSpline

# 设置随机种子
torch.manual_seed(41)

# 加载数据
data = np.loadtxt('datasets.txt')
X_loaded = data[:, :2]
y_loaded = data[:, 2:]
X = torch.tensor(X_loaded, dtype=torch.float32)
y = torch.tensor(y_loaded, dtype=torch.float32).view(-1, 1)  # 确保 y 是 PyTorch 张量

# 定义网格大小参数
GRID_SIZE = 10
FINE_GRID_SIZE = 100
SPLINE_POINTS = 100

# 定义模型
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

# 训练
epochs = 200
hidden_outputs = []
grid_transforms = []
decision_outputs = []
fine_hidden_outputs = []

# 创建网格
x_range = np.linspace(-1, 1, GRID_SIZE)
y_range = np.linspace(-1, 1, GRID_SIZE)
grid_x, grid_y = np.meshgrid(x_range, y_range)
grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

# 细网格
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
    fine_output, fine_hidden = model(fine_grid_tensor)
    decision_outputs.append(fine_output.detach().numpy().reshape(FINE_GRID_SIZE, FINE_GRID_SIZE))
    fine_hidden_outputs.append(fine_hidden.detach().numpy().reshape(FINE_GRID_SIZE, FINE_GRID_SIZE, 2))
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 可视化特征空间（静态图）
fig, ax = plt.subplots(figsize=(6, 6))
epoch_to_plot = epochs - 1  # 使用最后一个epoch的结果
hidden_out = hidden_outputs[epoch_to_plot]
grid_transform = grid_transforms[epoch_to_plot].reshape(GRID_SIZE, GRID_SIZE, 2)
decision_out = decision_outputs[epoch_to_plot]
fine_hidden = fine_hidden_outputs[epoch_to_plot]

ax.contourf(fine_hidden[:, :, 0], fine_hidden[:, :, 1], decision_out, levels=[0, 0.5, 1], colors=['#00BFFF', '#DDA0DD'], alpha=0.3)
contour = ax.contour(fine_hidden[:, :, 0], fine_hidden[:, :, 1], decision_out, levels=[0.5], colors='k', linewidths=2)
scatter = ax.scatter(hidden_out[:, 0], hidden_out[:, 1], c=y_loaded.ravel(), cmap='bwr', edgecolors='none')

# 使用三次样条插值绘制平滑网格
t = np.linspace(0, 1, GRID_SIZE)
t_fine = np.linspace(0, 1, SPLINE_POINTS)

for i in range(GRID_SIZE):
    x = grid_transform[:, i, 0]
    y = grid_transform[:, i, 1]
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    ax.plot(cs_x(t_fine), cs_y(t_fine), 'k-', lw=0.5)

for i in range(GRID_SIZE):
    x = grid_transform[i, :, 0]
    y = grid_transform[i, :, 1]
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    ax.plot(cs_x(t_fine), cs_y(t_fine), 'k-', lw=0.5)

ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_xlabel('Hidden 1')
ax.set_ylabel('Hidden 2')
ax.set_title(f"Feature Space with Boundary at Epoch {epoch_to_plot + 1}")
plt.savefig("feature_space_epoch200_boundary_pytorch.png", dpi=120)

# 生成动画
fig, ax = plt.subplots(figsize=(6, 6))

# 初始化
hidden_out_init = hidden_outputs[0]
decision_out_init = decision_outputs[0]
fine_hidden_init = fine_hidden_outputs[0]
contourf = ax.contourf(fine_hidden_init[:, :, 0], fine_hidden_init[:, :, 1], decision_out_init, 
                      levels=[0, 0.5, 1], colors=['#00BFFF', '#DDA0DD'], alpha=0.3)
contour = ax.contour(fine_hidden_init[:, :, 0], fine_hidden_init[:, :, 1], decision_out_init, 
                    levels=[0.5], colors='k', linewidths=2)
scatter = ax.scatter(hidden_out_init[:, 0], hidden_out_init[:, 1], c=y_loaded.ravel(), 
                    cmap='bwr', edgecolors='none')

grid_transform_init = grid_transforms[0].reshape(GRID_SIZE, GRID_SIZE, 2)

grid_lines = []
for i in range(GRID_SIZE):
    cs_x_h = CubicSpline(t, grid_transform_init[:, i, 0])
    cs_y_h = CubicSpline(t, grid_transform_init[:, i, 1])
    line_h, = ax.plot(cs_x_h(t_fine), cs_y_h(t_fine), 'k-', lw=0.5)
    cs_x_v = CubicSpline(t, grid_transform_init[i, :, 0])
    cs_y_v = CubicSpline(t, grid_transform_init[i, :, 1])
    line_v, = ax.plot(cs_x_v(t_fine), cs_y_v(t_fine), 'k-', lw=0.5)
    grid_lines.append(line_h)
    grid_lines.append(line_v)

ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_xlabel('Hidden 1')
ax.set_ylabel('Hidden 2')

def update(frame):
    global contourf, contour
    # 清除旧的等高线
    for coll in contourf.collections:
        coll.remove()
    for coll in contour.collections:
        coll.remove()

    # 更新决策边界
    decision_out = decision_outputs[frame]
    fine_hidden = fine_hidden_outputs[frame]
    contourf = ax.contourf(fine_hidden[:, :, 0], fine_hidden[:, :, 1], decision_out, 
                          levels=[0, 0.5, 1], colors=['#00BFFF', '#DDA0DD'], alpha=0.3)
    contour = ax.contour(fine_hidden[:, :, 0], fine_hidden[:, :, 1], decision_out, 
                        levels=[0.5], colors='k', linewidths=2)

    # 更新散点位置
    hidden_out = hidden_outputs[frame]
    scatter.set_offsets(np.c_[hidden_out[:, 0], hidden_out[:, 1]])
    
    # 更新网格线
    grid_transform = grid_transforms[frame].reshape(GRID_SIZE, GRID_SIZE, 2)
    for i in range(GRID_SIZE):
        cs_x_h = CubicSpline(t, grid_transform[:, i, 0])
        cs_y_h = CubicSpline(t, grid_transform[:, i, 1])
        grid_lines[2*i].set_data(cs_x_h(t_fine), cs_y_h(t_fine))
        
        cs_x_v = CubicSpline(t, grid_transform[i, :, 0])
        cs_y_v = CubicSpline(t, grid_transform[i, :, 1])
        grid_lines[2*i+1].set_data(cs_x_v(t_fine), cs_y_v(t_fine))
    
    ax.set_title(f"Feature Space with Boundary at Epoch {frame + 1}")
    return [scatter] + grid_lines + contourf.collections + contour.collections

animation = FuncAnimation(fig, update, frames=range(epochs), interval=200, blit=False)
animation.save("feature_space.gif", dpi=120, writer=PillowWriter(fps=24))

print("Visualization completed!")
