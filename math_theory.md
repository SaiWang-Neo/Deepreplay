\documentclass{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

\title{Implementation Logic and Mathematical Formulation of a Neural Network Visualization}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Implementation Logic and Mathematical Formulation}

\subsection*{1. Data Loading and Preprocessing}
\textbf{Data Loading and Preprocessing} \\
- \textit{Data Loading}: Data is read from the file \texttt{datasets.txt}, assuming each row contains two input features and one label. \\
- \textit{Data Conversion}: The loaded data is converted into PyTorch tensors. \\
\quad - Input feature matrix: \( \mathbf{X} \in \mathbb{R}^{N \times 2} \), where \( N \) is the number of samples. \\
\quad - Label vector: \( \mathbf{y} \in \mathbb{R}^{N \times 1} \).

\subsection*{2. Model Definition}
\textbf{Model Definition} \\
- \textit{Neural Network Structure}: A two-layer fully connected neural network. \\
\quad - \textit{Hidden Layer}: Input dimension is 2, with \( H \) hidden neurons (adjustable parameter, e.g., \texttt{HIDDEN\_SIZE}). \\
\quad - \textit{Output Layer}: Maps hidden layer output to 1 dimension. \\
- \textit{Activation Function}: Sigmoid function used for both layers, defined as: \\
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\] \\
- \textit{Weight Initialization}: \\
\quad - Hidden layer weights \( \mathbf{W}_h \) use Xavier normal initialization. \\
\quad - Output layer weights \( \mathbf{W}_o \) use standard normal initialization. \\
- \textit{Mathematical Representation}: \\
\quad - Hidden layer output: \\
\[
\mathbf{h} = \sigma(\mathbf{W}_h \mathbf{x} + \mathbf{b}_h)
\] \\
\quad where \( \mathbf{W}_h \in \mathbb{R}^{H \times 2} \), \( \mathbf{b}_h \in \mathbb{R}^{H} \). \\
\quad - Output layer output: \\
\[
\hat{y} = \sigma(\mathbf{W}_o \mathbf{h} + \mathbf{b}_o)
\] \\
\quad where \( \mathbf{W}_o \in \mathbb{R}^{1 \times H} \), \( \mathbf{b}_o \in \mathbb{R} \).

\subsection*{3. Loss Function and Optimizer}
\textbf{Loss Function and Optimizer} \\
- \textit{Loss Function}: Binary Cross-Entropy Loss (BCELoss), defined as: \\
\[
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
\] \\
- \textit{Optimizer}: Stochastic Gradient Descent (SGD) with learning rate 3. Parameter update rule: \\
\[
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
\] \\
where \( \theta = \{\mathbf{W}_h, \mathbf{b}_h, \mathbf{W}_o, \mathbf{b}_o\} \), \( \eta = 3 \).

\subsection*{4. Training Process}
\textbf{Training Process} \\
- \textit{Training Epochs}: 200 epochs. \\
- \textit{Per Epoch Operations}: \\
\quad 1. \textit{Forward Pass}: Compute hidden layer output \( \mathbf{h} \) and model output \( \hat{y} \). \\
\quad 2. \textit{Loss Computation}: Calculate \( \mathcal{L} \) based on \( \hat{y} \) and \( \mathbf{y} \). \\
\quad 3. \textit{Backward Pass}: Compute gradients of \( \mathcal{L} \) with respect to model parameters. \\
\quad 4. \textit{Parameter Update}: Update \( \theta \) using the optimizer.

\subsection*{5. Grid Generation}
\textbf{Grid Generation} \\
- \textit{Coarse Grid}: Generate a \( G \times G \) grid in the input space [-1, 1] \(\times\) [-1, 1] (\( G = \) \texttt{GRID\_SIZE}). \\
\quad - Grid point coordinates: \\
\[
\mathbf{G} = \{ (x_i, y_j) \mid x_i = -1 + \frac{2i}{G-1}, y_j = -1 + \frac{2j}{G-1}, i,j=0,\dots,G-1 \}
\] \\
- \textit{Fine Grid}: Generate a \( F \times F \) grid in the same input space (\( F = \) \texttt{FINE\_GRID\_SIZE}). \\
\quad - Grid point coordinates: \\
\[
\mathbf{F} = \{ (x_k, y_l) \mid x_k = -1 + \frac{2k}{F-1}, y_l = -1 + \frac{2l}{F-1}, k,l=0,\dots,F-1 \}
\]

\subsection*{6. Hidden Layer Mapping}
\textbf{Hidden Layer Mapping} \\
- \textit{Coarse Grid Mapping}: Map coarse grid points \( \mathbf{g} \in \mathbf{G} \) to the hidden space: \\
\[
\mathbf{h}_g = \sigma(\mathbf{W}_h \mathbf{g} + \mathbf{b}_h)
\] \\
- \textit{Fine Grid Mapping}: Map fine grid points \( \mathbf{f} \in \mathbf{F} \) to the hidden space: \\
\[
\mathbf{h}_f = \sigma(\mathbf{W}_h \mathbf{f} + \mathbf{b}_h)
\] \\
- \textit{Model Output}: Compute model predictions for fine grid points: \\
\[
\hat{y}_f = \sigma(\mathbf{W}_o \mathbf{h}_f + \mathbf{b}_o)
\]

\subsection*{7. Visualization}
\textbf{Visualization} \\
- \textit{Static Plot}: \\
\quad - Use fine grid hidden mappings \( \mathbf{h}_f \) (first two dimensions) and model output \( \hat{y}_f \) to plot the decision boundary (\( \hat{y}_f = 0.5 \)). \\
\quad - Plot training data points in the hidden space (based on \( \mathbf{h} \)). \\
\quad - Use cubic spline interpolation to smoothly connect coarse grid mappings \( \mathbf{h}_g \) as grid lines. \\
- \textit{Animation}: \\
\quad - Dynamically display changes in the hidden space across epochs, including: \\
\quad\quad - Movement of the decision boundary. \\
\quad\quad - Updates to scatter points \( \mathbf{h} \). \\
\quad\quad - Smooth transitions of grid lines \( \mathbf{h}_g \).

\subsection*{8. Mathematical Formula Summary}
\textbf{Mathematical Formula Summary} \\
- \textit{Model}: \\
\[
\begin{align*}
\mathbf{h} &= \sigma(\mathbf{W}_h \mathbf{x} + \mathbf{b}_h) \\
\hat{y} &= \sigma(\mathbf{W}_o \mathbf{h} + \mathbf{b}_o)
\end{align*}
\] \\
- \textit{Loss Function}: \\
\[
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
\] \\
- \textit{Optimization}: \\
\[
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}, \quad \eta = 3
\] \\
- \textit{Grid Mapping}: \\
\[
\mathbf{h}_g = \sigma(\mathbf{W}_h \mathbf{g} + \mathbf{b}_h), \quad \mathbf{g} \in \mathbf{G}
\] \\
\[
\mathbf{h}_f = \sigma(\mathbf{W}_h \mathbf{f} + \mathbf{b}_h), \quad \mathbf{f} \in \mathbf{F}
\] \\
- \textit{Decision Boundary}: In the hidden space, determined by \( \hat{y}_f = 0.5 \), i.e.: \\
\[
\mathbf{W}_o \mathbf{h}_f + \mathbf{b}_o = 0
\]

\subsection*{9. Spline Interpolation}
\textbf{Spline Interpolation} \\
- \textit{Cubic Spline Interpolation}: Used to smoothly connect coarse grid points in the hidden space \( \mathbf{h}_g \). \\
\quad - For each horizontal or vertical grid line, the interpolation function \( s(t) \) satisfies \( s(t_i) = \mathbf{h}_i \), with continuous second derivatives.

\subsection*{10. Animation Update}
\textbf{Animation Update} \\
- \textit{Per Frame Update}: \\
\quad - Update decision boundary: Based on current \( \mathbf{h}_f \) and \( \hat{y}_f \). \\
\quad - Update scatter points: Based on current \( \mathbf{h} \). \\
\quad - Update grid lines: Based on current \( \mathbf{h}_g \), generating smooth curves via spline interpolation.

\end{document}
