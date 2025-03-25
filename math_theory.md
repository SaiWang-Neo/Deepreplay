# 神经网络实现逻辑与数学公式

以下是对基于 PyTorch 的简单两层神经网络实现逻辑的梳理，并用数学公式刻画整个过程。该网络用于二分类任务，并通过可视化展示训练过程中隐藏层的表示变化。

---

## 1. 数据加载与预处理
- **数据加载**：从文件 `datasets.txt` 中读取数据，假设每行数据包含两个输入特征和一个标签。
- **数据转换**：将读取的数据转换为 PyTorch 张量。
  - 输入特征矩阵：\( \mathbf{X} \in \mathbb{R}^{N \times 2} \)，其中 \( N \) 是样本数量。
  - 标签向量：\( \mathbf{y} \in \mathbb{R}^{N \times 1} \)。

---

## 2. 模型定义
- **神经网络结构**：一个两层全连接神经网络。
  - **隐藏层**：输入维度为 2，隐藏层神经元数量为 \( H \)（可调参数，例如 `HIDDEN_SIZE`）。
  - **输出层**：隐藏层输出映射到 1 维。
- **激活函数**：隐藏层和输出层均使用 Sigmoid 函数，定义为：
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]
- **权重初始化**：
  - 隐藏层权重 \( \mathbf{W}_h \) 使用 Xavier 正态初始化。
  - 输出层权重 \( \mathbf{W}_o \) 使用标准正态初始化。
- **数学表示**：
  - 隐藏层输出：
    \[
    \mathbf{h} = \sigma(\mathbf{W}_h \mathbf{x} + \mathbf{b}_h)
    \]
    其中 \( \mathbf{W}_h \in \mathbb{R}^{H \times 2} \)，\( \mathbf{b}_h \in \mathbb{R}^{H} \)。
  - 输出层输出：
    \[
    \hat{y} = \sigma对我来说(\mathbf{W}_o \mathbf{h} + \mathbf{b}_o)
    \]
    其中 \( \mathbf{W}_o \in \mathbb{R}^{1 \times H} \)，\( \mathbf{b}_o \in \mathbb{R} \)。

---

## 3. 损失函数与优化器
- **损失函数**：使用二元交叉熵损失（BCELoss），定义为：
  \[
  \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
  \]
- **优化器**：随机梯度下降（SGD），学习率设为 3。参数更新公式为：
  \[
  \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
  \]
  其中 \( \theta = \{\mathbf{W}_h, \mathbf{b}_h, \mathbf{W}_o, \mathbf{b}_o\} \)，\( \eta = 3 \)。

---

## 4. 训练过程
- **训练轮数**：共 200 轮。
- **每轮操作**：
  1. **前向传播**：计算隐藏层输出 \( \mathbf{h} \) 和模型输出 \( \hat{y} \)。
  2. **计算损失**：根据 \( \hat{y} \) 和 \( \mathbf{y} \) 计算 \( \mathcal{L} \)。
  3. **反向传播**：计算损失对模型参数的梯度。
  4. **参数更新**：使用优化器更新 \( \theta \)。

---

## 5. 网格生成
- **粗网格**：在输入空间 [-1, 1] × [-1, 1] 上生成 \( G \times G \) 的网格点（\( G = \) `GRID_SIZE`）。
  - 网格点坐标：
    \[
    \mathbf{G} = \{ (x_i, y_j) \mid x_i = -1 + \frac{2i}{G-1}, y_j = -1 + \frac{2j}{G-1}, i,j=0,\dots,G-1 \}
    \]
- **细网格**：在相同输入空间生成 \( F \times F \) 的网格点（\( F = \) `FINE_GRID_SIZE`）。
  - 网格点坐标：
    \[
    \mathbf{F} = \{ (x_k, y_l) \mid x_k = -1 + \frac{2k}{F-1}, y_l = -1 + \frac{2l}{F-1}, k,l=0,\dots,F-1 \}
    \]

---

## 6. 隐藏层映射
- **粗网格映射**：将粗网格点 \( \mathbf{g} \in \mathbf{G} \) 映射到隐藏空间：
  \[
  \mathbf{h}_g = \sigma(\mathbf{W}_h \mathbf{g} + \mathbf{b}_h)
  \]
- **细网格映射**：将细网格点 \( \mathbf{f} \in \mathbf{F} \) 映射到隐藏空间：
  \[
  \mathbf{h}_f = \sigma(\mathbf{W}_h \mathbf{f} + \mathbf{b}_h)
  \]
- **模型输出**：对细网格点计算模型预测：
  \[
  \hat{y}_f = \sigma(\mathbf{W}_o \mathbf{h}_f + \mathbf{b}_o)
  \]

---

## 7. 可视化
- **静态图**：
  - 使用细网格的隐藏层映射 \( \mathbf{h}_f \)（取前两个维度）和模型输出 \( \hat{y}_f \) 绘制决策边界（\( \hat{y}_f = 0.5 \)）。
  - 绘制训练数据在隐藏空间的散点图（基于 \( \mathbf{h} \)）。
  - 使用三次样条插值平滑连接粗网格映射 \( \mathbf{h}_g \) 的网格线。
- **动画**：
  - 动态展示每轮训练后隐藏空间的变化，包括：
    - 决策边界的移动。
    - 散点位置 \( \mathbf{h} \) 的更新。
    - 网格线 \( \mathbf{h}_g \) 的平滑变化。

---

## 8. 数学公式总结
- **模型**：
  \[
  \begin{align*}
  \mathbf{h} &= \sigma(\mathbf{W}_h \mathbf{x} + \mathbf{b}_h) \\
  \hat{y} &= \sigma(\mathbf{W}_o \mathbf{h} + \mathbf{b}_o)
  \end{align*}
  \]
- **损失函数**：
  \[
  \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
  \]
- **优化**：
  \[
  \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}, \quad \eta = 3
  \]
- **网格映射**：
  \[
  \mathbf{h}_g = \sigma(\mathbf{W}_h \mathbf{g} + \mathbf{b}_h), \quad \mathbf{g} \in \mathbf{G}
  \]
  \[
  \mathbf{h}_f = \sigma(\mathbf{W}_h \mathbf{f} + \mathbf{b}_h), \quad \mathbf{f} \in \mathbf{F}
  \]
- **决策边界**：在隐藏空间中，由 \( \hat{y}_f = 0.5 \) 确定，即：
  \[
  \mathbf{W}_o \mathbf{h}_f + \mathbf{b}_o = 0
  \]

---

## 9. 样条插值
- **三次样条插值**：用于平滑连接粗网格点在隐藏空间的映射 \( \mathbf{h}_g \)。
  - 对每条水平或垂直网格线，插值函数 \( s(t) \) 满足 \( s(t_i) = \mathbf{h}_i \)，且二阶导数连续。

---

## 10. 动画更新
- **每帧更新**：
  - 更新决策边界：基于当前 \( \mathbf{h}_f \) 和 \( \hat{y}_f \)。
  - 更新散点：基于当前 \( \mathbf{h} \)。
  - 更新网格线：基于当前 \( \mathbf{h}_g \)，通过样条插值生成平滑曲线。

