# Deep Unfolding Methods for Sparse Bayesian Learning in M/EEG Inverse Modeling

## 简介

脑源成像旨在从无创的EEG/MEG电磁记录中推断分布在大脑皮层上的潜在神经源活动。由于传感器数量有限以及测量噪声的存在，该问题本质上是一个高度病态的逆问题，其解决方案关键依赖于适当的先验建模和稳定的推理算法。

稀疏贝叶斯学习（SBL）及其代表性算法Champagne，基于II型贝叶斯学习框架，通过源方差的自适应估计实现可解释的稀疏重建。然而，此类方法通常依赖迭代优化过程，计算成本较高且收敛速度有限。相比之下，端到端的深度学习方法推理效率高，但往往存在物理一致性不足和泛化鲁棒性受限的问题。

本研究旨在引入数据驱动机制的同时整合物理信息，为此研究了一类用于脑源成像的物理信息混合学习方法。首先，通过展开经典Champagne算法的推理过程，提出了神经期望最大化（Neural EM）框架，该框架保留了稀疏贝叶斯推理的基本结构，同时引入可学习组件以增强更新规则的灵活性。此外，针对潜在的收敛效率问题，引入了自适应反演网络（Adaptive InversionNet）模型。该模型在统一的学习框架内整合了多种经典更新策略，并在推理过程中以数据驱动的方式自适应地调节它们的贡献。

通过在模拟EEG数据上的实验，验证了所提方法的可行性和有效性。结果表明，在物理约束下引入可学习的推理机制，为平衡计算效率、重建性能和模型可解释性提供了一种有前景的途径。本研究为高维病态逆问题的可学习贝叶斯推理方法设计提供了新的视角。

## 神经EM原理

神经EM通过将传统Champagne算法的EM迭代过程进行深度展开（unrolling），构建了一种可学习的物理信息神经网络框架。传统EM算法中的一次迭代被映射为网络中的一层，网络通过堆叠有限层数来逐步优化源活动估计。

### 数学形式

经典EM算法在第 $k$ 次迭代中的更新形式为：

**E-step（后验计算）：**

$$
\boldsymbol{x}(t)=\mathbf{\Gamma }\boldsymbol{L}^{\top}\mathbf{\Sigma }_{y}^{-1}\boldsymbol{y}(t)
$$

$$
\mathbf{\Sigma }_x=\mathbf{\Gamma }-\mathbf{\Gamma }\boldsymbol{L}^{\top}\mathbf{\Sigma }_{y}^{-1}\boldsymbol{L}\mathbf{\Gamma }
$$

**M-step（超参数更新）：**

$$
\gamma _{n}^{k+1}=[\mathbf{\Sigma }_{x}^{(k)}]_{n,n}+\frac{1}{T}\sum_{t=1}^T{x_{n}^{(k)}(t)^2}
$$

在神经EM中，E-step保持解析形式不变，不包含可训练参数。M-step被设计为可学习模块：

$$
\gamma _{n}^{(k+1)}=w_{1,n}^{+}\,[\mathbf{\Sigma }_{x}^{(k)}]_{n,n}+w_{2,n}^{+}\,\frac{1}{T}\sum_{t=1}^T{x_{n}^{(k)}}(t)^2
$$

其中 $w_{1,n}^{+}$ 与 $w_{2,n}^{+}$ 为可学习的非负权重。模型采用逐源独立加权更新形式，而非全连接层，以控制参数规模（$2N$ vs $2N^2$）并保持统计一致性。

### 半正定性约束

为保证源协方差矩阵 $\mathbf{\Gamma}$ 的半正定性，可学习权重通过Softplus进行非负映射：

$$
w_{i,n}^{+}=\mathrm{Softplus}(w_{i,n})
$$

该设计避免了对 $\gamma$ 本身直接施加非线性激活函数，从而保持EM更新中不活跃源、弱活跃源和强活跃源的多尺度分布特性不被破坏。

第一层权重初始化为 $\mathrm{Softplus}(w_{i,n})=1$，使网络在训练初期退化为经典EM更新。
<img src="figures/neural_em_architecture.png" alt="Neural EM 架构图" width="400">
![Neural EM 架构图](figures/Architecture%20of%20Neural%20EM.png)


### 训练策略

网络采用逐层训练方式。引入第 $k$ 层时分为两个阶段：Pass 1仅训练新增层，Pass 2联合微调前 $k$ 层。新层参数通过权重复制从第 $k-1$ 层初始化，即 $\mathbf{w}^{(k)}\gets \mathbf{w}^{(k-1)}$，以保证推断行为的连续性。
![逐层训练策略示意图](figures/Progressive%20Layer-wise%20Training%20Strategy.png)

## 自适应反演网络（Adaptive InversionNet）基本原理

在传统 Champagne/SBL 框架下，针对源方差超参数 $\gamma_n$，常见的三种更新形式分别为：

**EM 更新：**

$$
\gamma_{n}^{k+1}=[\mathbf{\Sigma}_{x}^{k}]_{n,n}+\frac{1}{T}\sum_{t=1}^T{\bigl( \bar{x}_{n}^{k}(t) \bigr) ^2}
$$

**MacKay 更新：**

$$
\gamma_{n}^{k+1}=\left[ \frac{1}{T}\sum_{t=1}^T{\bigl( \bar{x}_{n}^{k}(t) \bigr) ^2} \right] \left( \gamma_{n}^{k}\,\mathbf{L}_{n}^{\top}(\mathbf{\Sigma}_{y}^{k})^{-1}\mathbf{L}_n \right) ^{-1}
$$

**Convex Bounding 更新：**

$$
\gamma_{n}^{k+1}=\left[ \frac{1}{T}\sum_{t=1}^T{\bigl( \bar{x}_{n}^{k}(t) \bigr) ^2} \right] ^{\frac{1}{2}}\left( \mathbf{L}_{n}^{\top}(\mathbf{\Sigma}_{y}^{k})^{-1}\mathbf{L}_n \right) ^{-\frac{1}{2}}
$$

在自适应反演网络第 $k$ 层中的超参数更新（M-Step）被统一表示为：

$$
\gamma_{n}^{k+1}=w_{1,n}^{(k)}\,(\text{EM-term})+w_{2,n}^{(k)}\,(\text{MacKay-term})+w_{3,n}^{(k)}\,(\text{Convex Bounding-term})
$$

其中 $w_{1,n}^{(k)}$、$w_{2,n}^{(k)}$ 和 $w_{3,n}^{(k)}$ 为可学习的非负权重，用于调节三种更新策略在当前迭代中的相对贡献。通过该设计，网络能够在保持各更新规则解析形式与理论一致性的前提下，根据数据特性在不同推断路径之间进行动态权衡。

为保证超参数更新结果的物理合理性与数值稳定性，自适应反演网络对所有组合权重施加非负约束，并进一步通过归一化操作确保在每一个源位置 $n$ 上，不同更新策略的权重之和为 1。具体而言，设第 $k$ 层中对应第 $n$ 个源的三种更新权重为

$$
\tilde{w}_{1,n}^{(k)},\;\tilde{w}_{2,n}^{(k)},\;\tilde{w}_{3,n}^{(k)}
$$

则其实际参与更新的权重通过如下归一化形式给出：

$$
w_{i,n}^{(k)}=\frac{|\tilde{w}_{i,n}^{(k)}|}{\sum_{j=1}^3{|}\tilde{w}_{j,n}^{(k)}|},\qquad i\in \{1,2,3\}
$$

该约束确保 $w_{i,n}^{(k)}\ge 0$ 且

$$
\sum_{i=1}^3{w_{i,n}^{(k)}}=1
$$

从而使超参数更新始终保持在由 EM、MacKay 与 Convex Bounding 三种经典更新规则所张成的凸组合空间内。这一设计避免了不同更新分量在数值尺度上的不可控放大或相互抵消，并且能够自然地保证源协方差矩阵 $\mathbf{\Gamma}$ 始终为半正定矩阵，使更新结果始终处于合理的物理范围。

从推断角度看，上述归一化权重可被自然解释为在当前迭代层中，不同推断策略对第 $n$ 个源的"相对可信度"或"偏好程度"。

在初始化阶段，所有可学习权重被设置为相等值，即

$$
w_{1,n}^{(0)}=w_{2,n}^{(0)}=w_{3,n}^{(0)}=\tfrac{1}{3}
$$

使网络在训练初期等价于对 EM、MacKay 与 Convex Bounding 三种更新方式的均匀组合。该初始化策略避免了对任一特定推断策略的先验偏置，使模型在训练开始时保持中性且稳定的更新行为，并为后续通过数据驱动方式逐步学习不同更新策略的相对重要性提供了良好的起点。

随着训练的进行，网络能够根据不同源位置、不同迭代深度以及不同数据分布特性，自适应地调整各更新权重，从而在保持统计一致性与物理可解释性的同时，加速收敛并减少对深度展开层数的依赖。

这种设计使得网络在有限展开深度下即可实现更快的有效收敛，从而显著减少所需的网络层数和推理计算量，并在实践中能够超越单一推断算法独立迭代所能达到的重建性能。
![自适应反演网络单层结构图](figures/single%20layer%20of%20adaptive%20InversionNet.png)
