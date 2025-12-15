# AlexNet 集成使用说明

## ✅ 已完成的修改

### 1. 文件修改清单

- **`approx_train.py`**
  - 添加 `ALEXNET_CONV_LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5"]`
  - 添加 `_build_teacher_int8_alexnet(args)` 函数
  - 添加 `_build_student_fq_alexnet(args)` 函数

- **`approx_train_ppo.py`**
  - 导入 AlexNet 相关类和函数
  - 在 `train_ppo()` 函数中添加模型选择逻辑
  - 自动根据模型类型使用对应的层列表和注意力层配置
  - 保存结果时包含模型类型信息

- **`train_qat.py`**
  - 导入 `AlexNetTapQuant`
  - `train_float()` 支持模型选择
  - `train_qat()` 支持模型选择
  - 自动保存到对应的文件名（`alexnet_*.pth` 或 `vgg16_*.pth`）

- **`main.py`**
  - 添加 `--model` 参数，支持 `vgg16` 和 `alexnet` 选择

### 2. 模型差异对比

| 特性 | VGG16 | AlexNet |
|------|-------|---------|
| **总层数** | 13层（卷积） | 7层（5卷积+2FC） |
| **卷积层** | 13层 | 5层 |
| **FC层** | 不近似 | 2层（fc1, fc2） |
| **层命名** | `conv1_1`, `conv1_2`, ... `conv5_3` | `conv1`...`conv5`, `fc1`, `fc2` |
| **注意力层** | `conv3_1`, `conv4_1`, `conv5_1` | `conv3`, `conv4`, `fc1` |
| **权重文件** | `vgg16_qat_preconvert.pth` | `alexnet_qat_preconvert.pth` |
| **记录点格式** | `block_output.conv1_1` | `block_output.conv1`, `classifier.fc1.out` |

---

## 🚀 完整使用流程

### 步骤1: 训练 AlexNet 浮点模型

```bash
python main.py --mode train_float --model alexnet \
  --epochs 20 --lr 0.01 --batch-size 128
```

**预期输出：**
- 训练 20 个 epoch
- 保存到 `outputs/alexnet_float_unfused.pth`

---

### 步骤2: 训练 AlexNet QAT 模型

```bash
python main.py --mode train_qat --model alexnet \
  --qat-epochs 10 --qat-lr 0.001 --batch-size 128
```

**预期输出：**
- 加载浮点模型
- 融合 Conv+BN+ReLU
- QAT 训练 10 个 epoch
- 保存到 `outputs/alexnet_qat_preconvert.pth`

---

### 步骤3: 运行 PPO 训练

```bash
python main.py --mode train_ppo_triseg --model alexnet \
  --backend fbgemm --lr 0.0003 --episodes 10000 \
  --batch-size 32 --calib-batches 16 \
  --eval-every 10000 --result-file alexnet_exp1.json
```

**关键参数说明：**
- `--model alexnet`: 使用 AlexNet 架构
- `--episodes 10000`: PPO 训练回合数
- `--calib-batches 16`: 用于直方图收集的批次数
- `--result-file`: 保存结果的文件名

**自动配置：**
- ✅ 使用 7 层（5个卷积层 + 2个FC层）
- ✅ 卷积层：`conv1`, `conv2`, `conv3`, `conv4`, `conv5`
- ✅ FC层：`fc1`, `fc2`（也进行近似处理）
- ✅ 注意力层：`conv3`, `conv4`, `fc1`（混合conv和fc层）
- ✅ tmax 基于每层 90% 分位数 + 10
- ✅ 4D 状态编码器

**训练过程：**
1. 收集激活值直方图（16 batches，包括FC层）
2. 计算每层 tmax（p90 + 10，卷积层和FC层分别计算）
3. 初始化 4D 状态编码器（7层状态）
4. 初始化 3 层注意力转移模块（conv3, conv4, fc1）
5. PPO 训练 10000 episodes
6. 跟踪 Top-30 配置
7. 训练结束后批量评估
8. 选择最佳配置（优先满足 2% 约束）

**FC层处理说明：**
- FC层的激活值也进行INT8近似
- 使用与卷积层相同的三段近似算法
- FC层的记录点格式：`classifier.fc1.out`, `classifier.fc2.out`
- fc3不进行近似（直接输出logits）

**预期输出：**
- 保存到 `outputs/tri_ppo_int_codes/alexnet_exp1.json`
- 包含每层的 INT8 码值参数
- 包含评估指标（Teacher/Student 准确率、精度下降）

---

## 📊 对比实验示例

### 同时训练两个模型

**VGG16:**
```bash
python main.py --mode train_ppo_triseg --model vgg16 \
  --backend fbgemm --episodes 10000 --batch-size 32 \
  --result-file vgg16_exp1.json
```

**AlexNet:**
```bash
python main.py --mode train_ppo_triseg --model alexnet \
  --backend fbgemm --episodes 10000 --batch-size 32 \
  --result-file alexnet_exp1.json
```

### 预期差异

| 指标 | VGG16 | AlexNet |
|------|-------|---------|
| **训练时间/episode** | ~长 | ~中等（7层） |
| **状态空间** | 13层 × 4D | 7层 × 4D (5conv+2fc) |
| **动作空间** | 13层 × 4码 | 7层 × 4码 |
| **tmax范围** | 各层不同 | 各层不同（FC层通常更小） |
| **FC层近似** | 无 | fc1, fc2都近似 |

---

## 🔍 验证和调试

### 检查模型是否正确加载

```bash
# 查看保存的结果文件
cat outputs/tri_ppo_int_codes/alexnet_exp1.json
```

**预期 JSON 结构：**
```json
{
  "layers": {
    "conv1": {"t1_code": 7, "v1_code": 14, "t2_code": 21, "v2_code": 28, "tmax_code": 35},
    "conv2": {...},
    "conv3": {...},
    "conv4": {...},
    "conv5": {...},
    "fc1": {"t1_code": 8, "v1_code": 16, "t2_code": 24, "v2_code": 32, "tmax_code": 40},
    "fc2": {...}
  },
  "backend": "fbgemm",
  "model": "alexnet",
  "selection": "constrained-best (drop=1.5% <= 2.0%)",
  "metrics": {
    "acc_teacher": 92.5,
    "acc_student": 91.0,
    "acc_drop": 1.5
  }
}
```

### 常见问题排查

**问题1: 找不到权重文件**
```
FileNotFoundError: outputs/alexnet_qat_preconvert.pth
```
**解决：** 先运行步骤1和步骤2训练模型

**问题2: 直方图键名不匹配**
```
KeyError: 'block_output.conv1'
```
**解决：** 已自动处理，确认使用最新代码

**问题3: 注意力层不存在**
```
AttributeError: 'AlexNetTapQuant' object has no attribute 'conv3_1'
```
**解决：** 已修复，AlexNet 使用 `conv2`, `conv3`, `conv4`

---

## 📈 性能监控

### 训练日志解读

```
[PPO] Using AlexNet with 7 layers (5 conv + 2 fc)
[PPO] Attention Transfer: 3 layers, weight=100.0
  [init] conv1: t1=7, v1=14, t2=21, v2=28, tmax=35 (p90+10)
  [init] conv2: t1=9, v1=18, t2=27, v2=36, tmax=45 (p90+10)
  ...
  [init] fc1: t1=8, v1=16, t2=24, v2=32, tmax=40 (p90+10)
  [init] fc2: t1=6, v1=12, t2=18, v2=24, tmax=30 (p90+10)
[Episode 10] reward=-2.3456, kd_loss=1.2345, attn_loss=0.0123, top1_reward=-2.1000
  conv1: t1=7, v1=14, t2=21, v2=28, tmax=35 (p90+10)
```

**关键指标：**
- `reward`: 越接近 0 越好（负的 KD loss + 注意力 loss）
- `kd_loss`: KL 散度，越小越好
- `attn_loss`: 注意力对齐损失，越小越好
- `top1_reward`: Top-30 中最好的 reward

---

## 🎯 高级配置

### 调整注意力权重

目前代码中 `attention_weight = 100.0`，如需调整：

**编辑 `approx_train_ppo.py` 第 419 行：**
```python
attention_weight = 100.0  # 可改为 50.0, 200.0 等
```

### 调整 Top-K 配置数量

**编辑 `approx_train_ppo.py` 第 464 行：**
```python
max_top_configs = 30  # 可改为 50, 100 等
```

### 修改约束阈值

**命令行参数：**
```bash
--max-acc-drop 3.0  # 允许最大精度下降 3%
```

---

## 📝 代码架构说明

### 关键函数调用链

```
main.py
  └─ train_ppo(args)  [approx_train_ppo.py]
       ├─ 模型选择逻辑
       │   ├─ if model_type == "alexnet":
       │   │    ├─ _build_teacher_int8_alexnet()
       │   │    ├─ _build_student_fq_alexnet()
       │   │    └─ CONV_LAYERS_USED = ALEXNET_CONV_LAYERS
       │   └─ else:  # vgg16
       │        ├─ _build_teacher_int8()
       │        └─ CONV_LAYERS_USED = CONV_LAYERS
       │
       ├─ 收集直方图（自动处理不同模型）
       ├─ 计算 tmax（基于 p90 + 10）
       ├─ 初始化 SimpleStateEncoder(conv_layers=CONV_LAYERS_USED)
       ├─ 初始化 AttentionTransfer(layers=attention_layers_default)
       └─ PPO 训练循环
            └─ 使用 CONV_LAYERS_USED 统一处理
```

### 自动化适配机制

所有使用 `CONV_LAYERS` 的地方都已替换为 `CONV_LAYERS_USED`，包括：
- ✅ 直方图收集
- ✅ 状态编码器初始化
- ✅ 训练循环（采样动作）
- ✅ Reward 计算
- ✅ Top-30 配置跟踪
- ✅ 评估和保存

---

## 🔧 扩展到其他模型

如需添加新模型（如 ResNet），参考以下步骤：

1. **创建模型文件** `model_resnet_tap_quant.py`
2. **定义层列表** `RESNET_CONV_LAYERS = [...]`
3. **添加构建函数** `_build_teacher_int8_resnet()`, `_build_student_fq_resnet()`
4. **在 `approx_train_ppo.py` 中添加分支**：
   ```python
   elif model_type == "resnet":
       teacher = _build_teacher_int8_resnet(args)
       student = _build_student_fq_resnet(args).to(device)
       CONV_LAYERS_USED = RESNET_CONV_LAYERS
       attention_layers_default = [...]
   ```
5. **更新 `main.py` 的 choices**：
   ```python
   choices=["vgg16", "alexnet", "resnet"]
   ```

---

## ✅ 检查清单

训练 AlexNet 前确认：

- [ ] 已完成浮点模型训练（`alexnet_float_unfused.pth` 存在）
- [ ] 已完成 QAT 训练（`alexnet_qat_preconvert.pth` 存在）
- [ ] CUDA 可用（`torch.cuda.is_available() == True`）
- [ ] 数据集已下载（`data/cifar-10-batches-py/` 存在）
- [ ] 输出目录存在（`outputs/` 目录）

训练完成后验证：

- [ ] 结果文件包含 7 层配置（conv1-conv5 + fc1-fc2）
- [ ] 结果文件包含 `"model": "alexnet"`
- [ ] 精度下降在合理范围内（< 2%）
- [ ] Top-30 配置已评估
- [ ] 日志显示正确的层数（7 layers: 5 conv + 2 fc）
- [ ] FC层的tmax和参数合理（通常比卷积层小）

---

## 📚 参考资料

- **VGG16 层结构**: 13 个卷积层，3×3 卷积核
- **AlexNet 层结构**: 5 个卷积层，变化的卷积核尺寸
- **PPO 算法**: Proximal Policy Optimization
- **注意力转移**: Attention Transfer for feature alignment
- **INT8 量化**: Post-training quantization with approximation

---

## 🆘 技术支持

如遇到问题：

1. 检查终端错误信息
2. 查看生成的日志文件
3. 验证权重文件是否存在
4. 确认 CUDA 设备可用
5. 对比 VGG16 和 AlexNet 的训练日志差异

**常用调试命令：**
```bash
# 检查权重文件
ls -lh outputs/*.pth

# 查看最近的结果
cat outputs/tri_ppo_int_codes/alexnet_*.json

# 测试模型加载
python -c "from model_alexnet_tap_quant import AlexNetTapQuant; m = AlexNetTapQuant(); print('OK')"
```
