# PPO + 知识蒸馏 INT8 模型近似探索项目技术白皮书

## 1. 项目背景与核心目标

随着深度学习模型在边缘设备上的部署需求日益增长，INT8 量化已成为工业界的主流选择。然而，标准的 INT8 量化在某些对精度极度敏感的场景下仍可能带来不可接受的精度损失。此外，标准的 INT8 乘加运算（MAC）在某些超低功耗硬件上仍显得昂贵。

本项目提出了一种基于 **强化学习 (PPO)** 和 **知识蒸馏 (Knowledge Distillation)** 的自动化近似探索框架。

**核心目标**：
1.  **三段式近似 (Tri-segment Approximation)**：将复杂的 INT8 激活值映射简化为三段阶梯函数。
    *   $f(x) = 0, \text{if } 0 < x \le t_1$
    *   $f(x) = v_1, \text{if } t_1 < x \le t_2$
    *   $f(x) = v_2, \text{if } t_2 < x \le t_{max}$
    *   这种映射可以通过极低成本的比较器和查找表（LUT）实现，大幅降低硬件功耗。
2.  **自动化参数搜索**：利用 PPO 智能体自动寻找每一层最优的 $(t_1, v_1, t_2, v_2)$ 组合，替代人工调参。
3.  **精度保持**：通过知识蒸馏，强制 Student 模型（应用近似后）的特征分布与 Teacher 模型（标准 INT8）保持一致。

---

## 2. 强化学习 (RL) 基础与设计

本项目将神经网络的层级参数搜索建模为 **马尔可夫决策过程 (MDP)**。

### 2.1 强化学习基本概念映射
*   **智能体 (Agent)**：PPO 算法（Actor-Critic 网络）。
*   **环境 (Environment)**：待优化的卷积神经网络（Student 模型）。
*   **状态 (State)**：当前层的统计特征（均值、方差、深度等）。
*   **动作 (Action)**：当前层的近似参数 $(t_1, v_1, t_2, v_2)$。
*   **奖励 (Reward)**：近似后的模型精度与特征保真度。

### 2.2 PPO (Proximal Policy Optimization) 算法详解
PPO 是一种 **Policy Gradient** 方法，旨在解决传统策略梯度方法步长难以确定的问题。

*   **Actor-Critic 架构**：
    *   **Actor (策略网络)**：输入状态 $s$，输出动作概率分布 $\pi_\theta(a|s)$。本项目中，Actor 有 4 个独立的 Head，分别预测 $t_1, v_1, t_2, v_2$ 的概率分布（0-255 的离散空间）。
    *   **Critic (价值网络)**：输入状态 $s$，预测状态价值 $V_\phi(s)$，用于计算优势函数 $A(s,a)$。

*   **PPO 核心公式 (Clipped Surrogate Objective)**：
    $$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)] $$
    其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是概率比率。
    *   **作用**：`clip` 操作限制了策略更新的幅度，防止一次更新让策略偏离太远导致性能崩塌，保证了训练的稳定性。

*   **优势函数 (Advantage Estimation)**：
    本项目使用简单的蒙特卡洛估计或 GAE (Generalized Advantage Estimation) 来计算优势 $A_t = R_t - V(s_t)$，衡量当前动作比平均情况好多少。

### 2.3 状态空间设计 (`state_encoder_simple.py`)
为了让 RL Agent 具有泛化能力（即学会“看分布下菜碟”），我们不直接输入图像，而是输入层的**统计元数据**。
状态向量 $S \in \mathbb{R}^4$ 包含：
1.  **Layer Depth Ratio**: $l / L_{total}$，层在网络中的相对位置。浅层通常对边缘敏感，深层对语义敏感。
2.  **Activation Mean (Normalized)**: $\mu / 255$，激活值的平均强度。
3.  **Activation Std (Normalized)**: $\sigma / 128$，激活值的离散程度。
4.  **Percentile 90**: $P_{90} / 255$，激活值的上界分布，决定了 $t_{max}$ 的选取范围。

---

## 3. 知识蒸馏 (Knowledge Distillation) 机制

为了在极端的近似（如三段式映射）下保持精度，单纯的 Cross-Entropy Loss 往往不够，我们需要更强的监督信号。

### 3.1 Teacher-Student 架构
*   **Teacher**: 预训练好的 QAT (Quantization Aware Training) 模型，参数冻结。
*   **Student**: 结构相同，但在推理时，其激活值会被 `ActivationRecorder` 拦截并替换为近似值。

### 3.2 混合损失函数 (Hybrid Loss / Reward)
RL 的 Reward 函数由两部分组成：

$$ R = R_{acc} + R_{distill} $$

1.  **精度奖励 ($R_{acc}$)**：
    $$ R_{acc} = -\text{KL}(P_{student} || P_{teacher}) $$
    即 Student 输出的 Logits 分布与 Teacher 的 KL 散度的负值。这比单纯的 Accuracy 更平滑。

2.  **注意力转移损失 ($R_{distill}$ / Attention Transfer)**：
    参考 *Zagoruyko & Komodakis (ICLR 2017)* 的方法，强制 Student 关注与 Teacher 相同的空间区域。
    *   **注意力图计算**：对于特征图 $F \in \mathbb{R}^{C \times H \times W}$，计算空间注意力 $A = \sum_{c=1}^C |F_c|^2$。
    *   **损失计算**：
        $$ L_{AT} = \sum_{j \in \mathcal{I}} || \frac{A_S^j}{||A_S^j||_2} - \frac{A_T^j}{||A_T^j||_2} ||_2 $$
    *   **本项目实现**：在 `attention_transfer.py` 中，我们选取了关键层（如 `conv3_1`, `conv4_1`, `conv5_1`）进行对齐。

---

## 4. 项目文件结构与详细分析

### 4.1 核心训练逻辑
*   **`approx_train_ppo.py`**: PPO 训练的主入口。
    *   **`ActorCritic` 类**：定义了策略网络和价值网络。包含 4 个 Actor Head (`actor_t1`...`actor_v2`) 和 1 个 Critic Head。
    *   **`PPO` 类**：实现了 PPO 的 `update` 逻辑，计算 Surrogate Loss、Value Loss 和 Entropy Bonus（鼓励探索）。
    *   **`train_ppo` 函数**：
        1.  **Rollout**: 遍历网络每一层，Agent 根据状态输出动作，记录轨迹 $(s, a, r, logp)$。
        2.  **Evaluation**: 在一个 Batch 上运行 Student 模型，计算 Reward。
        3.  **Update**: 每隔一定步数（`update_timestep`），利用 Buffer 中的数据更新 Agent。
    *   **`evaluate_accuracy` 函数**：构建 `edits` 字典，将 Agent 生成的参数注入到 Student 模型中，进行实际推理。

### 4.2 模型手术与拦截
*   **`recorder.py`**: 实现对 PyTorch 模型的“非侵入式”修改。
    *   **`ActivationRecorder`**: 核心 Hook 类。它在 `forward` 过程中捕获中间层输出。
        *   `record(name, x)`: 存储激活值（用于 Teacher 生成 Ground Truth）。
        *   `maybe_edit(name, x)`: 检查是否有针对该层的 `edit` 函数（由 PPO 生成），如果有，则替换激活值（用于 Student）。
    *   **`EditOnlyRecorder`**: 极简版 Recorder，只修改不记录，用于加速训练和推理。

### 4.3 状态感知
*   **`state_encoder_simple.py`**:
    *   **`SimpleStateEncoder`**: 负责预计算所有层的统计特征。它接收全量数据集的直方图，计算出 4 维状态向量。这是连接“静态数据”与“动态决策”的桥梁。

### 4.4 辅助工具
*   **`attention_transfer.py`**:
    *   **`AttentionTransfer`**: 计算 Teacher 和 Student 特征图之间的注意力距离。包含 `compute_attention_map`（L2 范数压扁通道）和 `forward`（计算 MSE/L1 损失）。
*   **`approx_train.py`**:
    *   **`teacher_forward_with_scales`**: 一个特殊的 Forward 函数。它不仅计算 Logits，还顺便抓取了每一层的量化参数（Scale, Zero-point）。这是因为 Student 在模拟 INT8 近似时，必须知道当前层的量化参数才能正确解码/编码。

### 4.5 模型定义 (`model_*.py`)
*   **`model_vgg19_tap_quant.py` / `model_alexnet_tap_quant.py` / `model_nin_tap_quant.py` / `model_mobilenetv1_tap_quant.py`**:
    *   这些文件定义了支持 QAT 的模型结构。
    *   **关键差异**：与普通 torchvision 模型不同，它们在每一层的前后都手动插入了 `recorder` 的调用点。
    *   **`fuse_model`**: 定义了 Conv+BN+ReLU 的融合逻辑，这是 INT8 推理的标准优化。
    *   **新增模型支持**：
        *   **NiN (Network in Network)**: 经典的 1x1 卷积网络，CIFAR-10 精度约 88-90%。
        *   **MobileNetV1**: 轻量级深度可分离卷积网络，CIFAR-10 精度约 90-92%。
        *   **VGG16/19**: 经典深层网络，CIFAR-10 精度约 92-93%。
        *   **AlexNet**: 早期经典网络，CIFAR-10 精度约 80-85%。

---

## 5. 总结：项目工作流

1.  **预热 (Warmup)**：
    *   加载预训练的 QAT Teacher 模型。
    *   扫描数据集，收集每一层的激活值直方图 (`_collect_histograms`)。
    *   初始化 `StateEncoder`。

2.  **搜索 (Search - PPO Loop)**：
    *   **Observe**: Agent 观察第 $i$ 层的状态 $S_i$。
    *   **Act**: Agent 输出动作 $A_i = (t_1, v_1, t_2, v_2)$。
    *   **Apply**: `Recorder` 将 $A_i$ 转换为具体的映射函数，挂载到 Student 模型第 $i$ 层。
    *   **Reward**: 所有层决策完毕后，Student 推理一个 Batch，计算与 Teacher 的 KL 散度及 Attention Loss，反馈给 Agent。
    *   **Learn**: Agent 根据 Reward 更新策略，学会“什么样的层分布应该用什么样的近似参数”。

3.  **部署 (Deploy)**：
    *   导出 Reward 最高的 Top-N 组参数配置 (`result.json`)。
    *   在全量测试集上验证，选择满足精度约束的最佳配置。
    *   最终产出：一个可以直接部署的、带有三段式查找表（LUT）的轻量化 INT8 模型。
