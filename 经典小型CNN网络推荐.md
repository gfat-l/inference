# 经典小型CNN网络推荐

本文档列出适合集成到当前INT8量化框架的经典小型CNN网络。

---

## 📊 网络对比表

| 网络 | 层数 | 参数量 | CIFAR-10精度 | 计算量 | 特点 | 推荐指数 |
|------|------|--------|--------------|--------|------|---------|
| **VGG16** | 13卷积 | ~15M | ~93% | 高 | 简单堆叠 | ⭐⭐⭐⭐⭐ |
| **VGG19** | 16卷积 | ~20M | ~93% | 很高 | VGG16加深 | ⭐⭐⭐⭐ |
| **AlexNet** | 5卷积+2FC | ~3M | ~89% | 中 | 经典网络 | ⭐⭐⭐⭐⭐ |
| **LeNet-5** | 2卷积+3FC | ~60K | ~75% | 很低 | 最简单 | ⭐⭐⭐ |
| **SqueezeNet** | 26层(Fire模块) | ~1.2M | ~91% | 低 | 压缩网络 | ⭐⭐⭐⭐⭐ |
| **MobileNetV1** | 28层(深度可分离) | ~4M | ~92% | 很低 | 移动端优化 | ⭐⭐⭐⭐⭐ |
| **ResNet-18** | 18层(残差) | ~11M | ~95% | 中 | 残差连接 | ⭐⭐⭐⭐ |
| **ResNet-34** | 34层(残差) | ~21M | ~95% | 高 | ResNet-18加深 | ⭐⭐⭐ |

---

## 🎯 推荐网络详解

### 1. **LeNet-5** ⭐⭐⭐
**最简单的CNN，适合快速实验**

```
架构：
- Conv1: 32×32×1 → 28×28×6 (5×5 kernel)
- Pool1: 28×28×6 → 14×14×6
- Conv2: 14×14×6 → 10×10×16 (5×5 kernel)
- Pool2: 10×10×16 → 5×5×16
- FC1: 400 → 120
- FC2: 120 → 84
- FC3: 84 → 10

优点：
✓ 极简单，易于理解
✓ 参数量极少（~60K）
✓ 训练速度快
✓ 适合教学和快速原型

缺点：
✗ 精度较低（CIFAR-10约75%）
✗ 网络太浅

集成难度：⭐（最简单）
```

### 2. **SqueezeNet** ⭐⭐⭐⭐⭐
**强烈推荐！压缩网络的经典**

```
架构特点：
- Fire模块：squeeze层 + expand层
  - squeeze: 1×1卷积压缩通道
  - expand: 1×1和3×3卷积并行扩展
- 26层网络，但只有1.2M参数

Fire模块结构：
Input → Conv1×1(squeeze) → [Conv1×1 ∥ Conv3×3](expand) → Concat

网络层数：
- Conv1: 3→96
- Fire2-9: 8个Fire模块
- Conv10: 512→10
- AvgPool + Softmax

优点：
✓ 参数量小（1.2M，相当于AlexNet的1/50）
✓ 精度高（CIFAR-10可达91%+）
✓ 适合量化（大量1×1卷积）
✓ 适合部署

缺点：
✗ 实现稍复杂（Fire模块）
✗ 需要仔细调参

集成难度：⭐⭐⭐
推荐理由：参数少+精度高+适合量化
```

### 3. **MobileNetV1** ⭐⭐⭐⭐⭐
**强烈推荐！移动端优化网络**

```
架构特点：
- 深度可分离卷积（Depthwise Separable Convolution）
  - Depthwise Conv: 逐通道卷积
  - Pointwise Conv: 1×1卷积

标准卷积 vs 深度可分离：
- 标准: 3×3×C_in×C_out
- DW+PW: 3×3×C_in + 1×1×C_in×C_out
- 计算量降低约9倍

网络结构（CIFAR-10版本）：
- Conv1: 3×3×3×32
- DW+PW×13: 13个深度可分离模块
- AvgPool + FC

优点：
✓ 计算量极低
✓ 参数量适中（~4M）
✓ 精度优秀（CIFAR-10 ~92%）
✓ 设计理念先进

缺点：
✗ 深度可分离卷积实现复杂
✗ 需要特殊处理量化

集成难度：⭐⭐⭐⭐
推荐理由：效率极高+设计先进
```

### 4. **ResNet-18** ⭐⭐⭐⭐
**残差网络，精度最高**

```
架构特点：
- 残差连接（Skip Connection）
- BasicBlock: Conv-BN-ReLU-Conv-BN + Shortcut
- 4个stage，每个stage多个block

网络结构：
- Conv1: 7×7×3×64
- Stage1: 2×BasicBlock(64)
- Stage2: 2×BasicBlock(128)
- Stage3: 2×BasicBlock(256)
- Stage4: 2×BasicBlock(512)
- AvgPool + FC

优点：
✓ 精度极高（CIFAR-10 ~95%）
✓ 训练稳定（残差连接）
✓ 应用广泛

缺点：
✗ 参数量较大（~11M）
✗ 计算量较高
✗ 残差连接增加量化难度

集成难度：⭐⭐⭐⭐⭐
推荐理由：精度高+架构先进
```

---

## 🔧 集成实施建议

### 优先级排序

**第1批（已完成）：**
- ✅ VGG16 - 已实现
- ✅ VGG19 - 已实现  
- ✅ AlexNet - 已实现

**第2批（强烈推荐）：**
1. **SqueezeNet** - 最优性价比
2. **MobileNetV1** - 移动端标杆
3. **LeNet-5** - 快速测试

**第3批（高级）：**
1. **ResNet-18** - 最高精度
2. **ResNet-34** - 更深的ResNet

### 实施步骤（以SqueezeNet为例）

#### 1. 创建模型文件
```bash
# 创建 model_squeezenet_tap_quant.py
- 定义Fire模块
- 实现SqueezeNetTapQuant类
- 添加记录点（block_output.fire{i}）
- 实现融合逻辑
```

#### 2. 修改配置文件
```python
# approx_train.py
SQUEEZENET_LAYERS = [
    "conv1",
    "fire2", "fire3", "fire4", "fire5",
    "fire6", "fire7", "fire8", "fire9",
    "conv10"
]

def _build_teacher_int8_squeezenet(args):
    # 构建INT8 teacher
    ...
```

#### 3. 集成训练流程
```python
# approx_train_ppo.py
elif model_type == "squeezenet":
    teacher = _build_teacher_int8_squeezenet(args)
    student = _build_student_fq_squeezenet(args).to(device)
    CONV_LAYERS_USED = SQUEEZENET_LAYERS
    attention_layers_default = ['fire4', 'fire6', 'fire8']
```

#### 4. 更新命令行参数
```python
# main.py
choices=["vgg16", "vgg19", "alexnet", "squeezenet"]
```

---

## 📐 网络层数对比

```
LeNet-5:     2 conv + 3 FC = 5 层
AlexNet:     5 conv + 2 FC = 7 层
VGG16:       13 conv = 13 层
VGG19:       16 conv = 16 层
SqueezeNet:  1 conv + 8 Fire + 1 conv = 10 主要层
MobileNetV1: 1 conv + 13 DW-PW = 14 主要层
ResNet-18:   1 conv + 8 Blocks×2 = 17 层
```

---

## 💡 选择建议

### 根据目标选择

**追求精度：**
1. ResNet-18 (95%)
2. VGG16/VGG19 (93%)
3. MobileNetV1 (92%)

**追求效率：**
1. MobileNetV1（计算量最低）
2. SqueezeNet（参数量最少）
3. LeNet-5（最简单）

**追求平衡：**
1. SqueezeNet（强烈推荐）
2. MobileNetV1
3. AlexNet

**快速实验：**
1. LeNet-5（几分钟训练）
2. AlexNet（半小时训练）
3. SqueezeNet（1小时训练）

### 根据研究方向选择

**量化研究：**
- SqueezeNet（1×1卷积多，适合量化）
- MobileNetV1（深度可分离，挑战量化）

**网络压缩：**
- SqueezeNet（压缩设计）
- MobileNetV1（效率优化）

**精度优先：**
- ResNet-18（残差网络）
- VGG19（深度网络）

---

## 🎓 学习曲线

```
难度等级：

⭐ LeNet-5
   - 最简单
   - 1小时掌握
   - 适合入门

⭐⭐ AlexNet / VGG16 / VGG19
   - 简单堆叠
   - 2小时掌握
   - 已完成集成

⭐⭐⭐ SqueezeNet
   - Fire模块
   - 4小时掌握
   - 值得学习

⭐⭐⭐⭐ MobileNetV1
   - 深度可分离
   - 1天掌握
   - 设计巧妙

⭐⭐⭐⭐⭐ ResNet-18
   - 残差连接
   - 2天掌握
   - 架构复杂
```

---

## 📚 参考资源

### 论文
- **LeNet-5**: "Gradient-Based Learning Applied to Document Recognition" (1998)
- **AlexNet**: "ImageNet Classification with Deep CNNs" (2012)
- **VGG**: "Very Deep Convolutional Networks" (2014)
- **SqueezeNet**: "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters" (2016)
- **MobileNetV1**: "MobileNets: Efficient CNNs for Mobile Vision" (2017)
- **ResNet**: "Deep Residual Learning for Image Recognition" (2015)

### 代码参考
```python
# PyTorch官方实现
torchvision.models.squeezenet1_1
torchvision.models.mobilenet_v2
torchvision.models.resnet18

# 可以参考但需要适配CIFAR-10
```

---

## ✅ 下一步行动

### 推荐集成顺序

1. **SqueezeNet**（第一优先）
   - 理由：参数少+精度高+适合量化
   - 预期工作量：1天
   - 预期效果：优秀

2. **LeNet-5**（快速验证）
   - 理由：最简单，快速测试流程
   - 预期工作量：2小时
   - 预期效果：基础验证

3. **MobileNetV1**（挑战性）
   - 理由：移动端标准，设计先进
   - 预期工作量：2天
   - 预期效果：优秀

4. **ResNet-18**（高级功能）
   - 理由：最高精度，残差连接
   - 预期工作量：3天
   - 预期效果：最佳精度

---

## 📊 预期性能对比

| 网络 | 训练时间 | INT8精度 | 压缩率 | 适用场景 |
|------|---------|----------|--------|---------|
| LeNet-5 | 10min | ~70% | 极高 | 教学/快速测试 |
| AlexNet | 1h | ~87% | 高 | 基准对比 |
| VGG16 | 3h | ~91% | 中 | 标准测试 |
| VGG19 | 4h | ~91% | 中 | 深度对比 |
| SqueezeNet | 1.5h | ~89% | 极高 | **推荐** |
| MobileNetV1 | 2h | ~90% | 极高 | 移动部署 |
| ResNet-18 | 3h | ~93% | 中高 | 精度要求高 |

**注：精度为使用当前PPO框架的预期值**

---

## 🚀 快速开始

当前已支持的网络：
```bash
# VGG16
python main.py --mode train_ppo_triseg --model vgg16

# VGG19（新增）
python main.py --mode train_ppo_triseg --model vgg19

# AlexNet
python main.py --mode train_ppo_triseg --model alexnet
```

建议下一个集成：
```bash
# SqueezeNet（待实现）
python main.py --mode train_ppo_triseg --model squeezenet
```
