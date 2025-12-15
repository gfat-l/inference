# LeNet-5 使用说明

## 📋 模型概述

**LeNet-5** 是最经典的卷积神经网络（1998年Yann LeCun提出），标志着深度学习在计算机视觉领域的成功应用。

### 网络特点
- ✅ **最简单的CNN结构**：纯顺序执行，无并行分支
- ✅ **参数量极少**：约60K参数（VGG16的1/250）
- ✅ **训练速度快**：CIFAR-10上10-20分钟即可完成
- ✅ **包含Conv和FC**：2个卷积层 + 2个FC层 = 4个量化层
- ✅ **教学价值高**：适合理解量化算法原理

### 网络结构
```
输入: (N, 3, 32, 32)
↓
Conv1 (3→6, 5×5) → ReLU → MaxPool
↓
Conv2 (6→16, 5×5) → ReLU → MaxPool
↓
Flatten (400)
↓
FC1 (400→120) → ReLU
↓
FC2 (120→84) → ReLU
↓
FC3 (84→10) [分类层，不量化]
↓
输出: (N, 10)
```

### 量化层
- `conv1`: 6通道卷积
- `conv2`: 16通道卷积
- `fc1`: 120神经元全连接
- `fc2`: 84神经元全连接

**注意：** fc3是分类层，不进行量化近似。

---

## 🚀 完整训练流程

### 1. 训练Float模型
```bash
python main.py --mode train_float --model lenet5 --epochs 30
```

**参数说明：**
- `--epochs 30`: LeNet-5较简单，30轮足够收敛
- 预期精度：75-80%（Float32）

**输出文件：**
- `outputs/lenet5_float_unfused.pth`

---

### 2. 训练QAT模型
```bash
python main.py --mode train_qat --model lenet5 --qat-epochs 10
```

**参数说明：**
- `--qat-epochs 10`: 量化感知训练10轮
- 自动从float模型初始化

**输出文件：**
- `outputs/lenet5_qat_preconvert.pth`

---

### 3. 运行PPO优化
```bash
python main.py --mode train_ppo_triseg --model lenet5 --episodes 1000 --result-file lenet5_ppo_result.json
```

**参数说明：**
- `--episodes 1000`: 由于只有4层，1000轮episode足够
- `--result-file`: 保存结果到指定文件

**输出文件：**
- `outputs/tri_ppo_int_codes/lenet5_ppo_result.json`

**特点：**
- LeNet-5只有4个量化层，训练速度快
- 注意力转移使用2层：`conv2`, `fc1`
- 每层动态计算tmax（p90+10）

---

### 4. 评估INT8精度
```bash
python main.py --mode eval_ppo_int8 --model lenet5 --config-path .\outputs\tri_ppo_int_codes\lenet5_ppo_result.json
```

**评估指标：**
- Teacher精度（INT8量化）
- Student精度（TAP近似）
- 精度下降

---

## 📊 预期性能

### 精度对比
| 阶段 | 预期精度 | 说明 |
|------|---------|------|
| Float32 | 75-80% | 基准精度 |
| QAT INT8 | 73-78% | 量化感知训练 |
| Teacher INT8 | 73-78% | 标准INT8量化 |
| Student TAP | 70-75% | TAP近似后 |

### 训练时间（单GPU）
| 阶段 | 预期时间 | 说明 |
|------|---------|------|
| Float训练 | 10-15分钟 | 30 epochs |
| QAT训练 | 5-8分钟 | 10 epochs |
| PPO优化 | 15-20分钟 | 1000 episodes, 4层 |

### 参数量对比
```
LeNet-5:  ~60K   (最少)
AlexNet:  ~3M    (50倍)
VGG16:    ~15M   (250倍)
VGG19:    ~20M   (330倍)
```

---

## 🔧 关键配置

### 层配置
```python
LENET5_LAYERS = ["conv1", "conv2", "fc1", "fc2"]  # 4个量化层
```

### 记录点格式
- **卷积层**: `block_output.conv1`, `block_output.conv2`
- **FC层**: `classifier.fc1.out`, `classifier.fc2.out`

### 注意力转移层
```python
attention_layers = ["conv2", "fc1"]  # 2层（网络较浅）
```

选择理由：
- `conv2`: 卷积特征的最后一层（16通道）
- `fc1`: FC特征的第一层（120神经元）

### tmax计算
每层独立计算：
```python
tmax = p90_code + 10
```
- `p90_code`: 去零后90%分位数
- 针对LeNet-5的小激活范围优化

---

## 🎯 使用场景

### 适合用于

1. **快速验证**
   - 算法原型测试
   - 快速迭代实验
   - 流程正确性验证

2. **教学演示**
   - CNN量化原理讲解
   - TAP近似算法演示
   - 对比不同量化方法

3. **基准测试**
   - 最简单网络的性能下限
   - 与复杂网络对比
   - 算法鲁棒性测试

### 不适合用于

1. **追求高精度**
   - LeNet-5本身精度有限（75-80%）
   - 不如VGG/AlexNet

2. **复杂任务**
   - 只有2个卷积层，特征提取能力弱
   - 不适合细粒度分类

3. **生产部署**
   - 精度较低，实用性有限
   - 建议使用VGG16/AlexNet

---

## 📈 与其他网络对比

| 特性 | LeNet-5 | AlexNet | VGG16 |
|------|---------|---------|-------|
| **量化层数** | 4 | 7 | 13 |
| **卷积层** | 2 | 5 | 13 |
| **FC层** | 2 | 2 | 0 |
| **参数量** | 60K | 3M | 15M |
| **Float精度** | 75-80% | 89% | 93% |
| **训练时间** | 10min | 1h | 3h |
| **PPO训练** | 20min | 40min | 2h |
| **适用场景** | 快速验证 | 平衡性能 | 高精度 |

---

## 💡 优化建议

### 1. 增加训练轮数
LeNet-5简单，可以多训练几轮：
```bash
python main.py --mode train_float --model lenet5 --epochs 50
```

### 2. 调整学习率
```bash
python main.py --mode train_float --model lenet5 --epochs 30 --lr 0.05
```

### 3. 增加PPO episodes
4层网络，可以更充分探索：
```bash
python main.py --mode train_ppo_triseg --model lenet5 --episodes 2000
```

### 4. 调整注意力权重
```bash
# 如果未来添加 --attention-weight 参数
python main.py --mode train_ppo_triseg --model lenet5 --attention-weight 50.0
```

---

## 🔍 调试技巧

### 1. 检查层输出形状
```python
python model_lenet5_tap_quant.py
```
输出每层的激活形状和记录点。

### 2. 查看直方图
训练时会收集每层的INT8直方图：
```
conv1: 256个码位的分布
conv2: 256个码位的分布
fc1: 256个码位的分布
fc2: 256个码位的分布
```

### 3. 监控tmax
每层的tmax应该合理：
- conv1, conv2: 通常20-40
- fc1, fc2: 通常30-60

### 4. 检查注意力损失
只有2个注意力层，损失应该较小。

---

## ⚠️ 常见问题

### Q1: LeNet-5精度太低怎么办？
**A:** LeNet-5设计简单，在CIFAR-10上精度本身有限（75-80%），这是正常的。如需更高精度，建议使用VGG16或AlexNet。

### Q2: PPO训练很快就收敛了？
**A:** 正常现象。LeNet-5只有4层，搜索空间小，1000个episodes足够。

### Q3: FC层的量化效果不好？
**A:** FC层参数多但通道少，量化难度较大。可以尝试：
- 增加calibration batches
- 调整attention weight
- 只量化conv层（fc层保持float）

### Q4: 如何只量化卷积层？
修改 `approx_train.py`:
```python
LENET5_LAYERS = ["conv1", "conv2"]  # 只量化2个卷积层
```

---

## 📚 相关文档

- **模型定义**: `model_lenet5_tap_quant.py`
- **训练脚本**: `train_qat.py`, `approx_train_ppo.py`
- **主程序**: `main.py`
- **工具函数**: `utils.py`, `recorder.py`

---

## 🎓 学习路径

### 新手路径
1. 先用LeNet-5理解整个流程
2. 观察4层的量化效果
3. 理解tmax, TAP近似的原理

### 进阶路径
1. 对比LeNet-5 vs AlexNet vs VGG16
2. 分析不同网络深度的影响
3. 研究FC层 vs Conv层的量化差异

### 研究路径
1. 改进LeNet-5结构（添加BN、Dropout）
2. 尝试不同的量化策略
3. 分析最小网络的量化下限

---

## ✅ 快速开始示例

### 完整流程（一键运行）
```bash
# 1. Float训练
python main.py --mode train_float --model lenet5 --epochs 30

# 2. QAT训练
python main.py --mode train_qat --model lenet5 --qat-epochs 10

# 3. PPO优化
python main.py --mode train_ppo_triseg --model lenet5 --episodes 1000

# 4. 评估
python main.py --mode eval_ppo_int8 --model lenet5 --config-path .\outputs\tri_ppo_int_codes\result.json
```

### 预期输出
```
[Float] Epoch 30 | Val Acc 77.23%
[QAT] Epoch 10 | Val Acc 75.48%
[PPO] Episode 1000 | Teacher 75.48% | Student 73.21% | Drop 2.27%
```

---

## 🎉 总结

LeNet-5是：
- ✅ **最简单的CNN**（4个量化层）
- ✅ **最快的训练**（20分钟完成全流程）
- ✅ **最好的教学案例**（理解量化原理）
- ✅ **最佳的验证工具**（快速测试算法）

虽然精度不如VGG/AlexNet，但对于：
- 算法开发
- 快速迭代
- 流程验证
- 教学演示

LeNet-5是完美的选择！
