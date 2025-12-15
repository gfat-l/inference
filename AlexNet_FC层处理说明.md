# AlexNet FC层近似处理实现说明

## 📋 修改概述

为AlexNet添加了FC层（fc1, fc2）的INT8近似处理，使得总共有7层需要近似（5个卷积层 + 2个FC层）。

---

## 🔧 关键修改点

### 1. **层列表更新** (`approx_train.py`)

```python
# 原来：只有5个卷积层
ALEXNET_CONV_LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5"]

# 现在：5个卷积层 + 2个FC层
ALEXNET_LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]
ALEXNET_CONV_LAYERS = ALEXNET_LAYERS  # 兼容性别名
```

### 2. **注意力层配置** (`approx_train_ppo.py`)

```python
# 混合选择卷积层和FC层
attention_layers_default = ['conv3', 'conv4', 'fc1']
```

**选择理由：**
- `conv3`: 中层卷积特征
- `conv4`: 高层卷积特征  
- `fc1`: 全连接特征（关键）

### 3. **记录点键名区分**

不同类型的层使用不同的键名格式：

| 层类型 | 记录点键名格式 | 示例 |
|--------|---------------|------|
| 卷积层 | `block_output.{layer_name}` | `block_output.conv1` |
| FC层 | `classifier.{layer_name}.out` | `classifier.fc1.out` |

### 4. **直方图收集** (`approx_train_ppo.py`)

```python
if model_type == "alexnet":
    hists = {}
    for lyr in CONV_LAYERS_USED:
        if lyr.startswith('conv'):
            hists[f"block_output.{lyr}"] = torch.zeros(256, dtype=torch.long)
        elif lyr.startswith('fc'):
            hists[f"classifier.{lyr}.out"] = torch.zeros(256, dtype=torch.long)
    
    # 收集时也要区分
    for lyr in CONV_LAYERS_USED:
        if lyr.startswith('conv'):
            k = f"block_output.{lyr}"
        elif lyr.startswith('fc'):
            k = f"classifier.{lyr}.out"
        
        if k in rec.storage:
            v = rec.storage[k].flatten()
            hists[k] += torch.bincount(v, minlength=256).to(torch.long)
```

### 5. **tmax计算** (`approx_train_ppo.py`)

```python
for layer_name in CONV_LAYERS_USED:
    # 根据层类型选择正确的键名
    if layer_name.startswith('conv'):
        hist_key = f"block_output.{layer_name}"
    elif layer_name.startswith('fc'):
        hist_key = f"classifier.{layer_name}.out"
    
    # 计算p90 + 10
    p90_code = _q_nonzero_from_hist(hist, 0.9)
    tmax_code = p90_code + 10
```

### 6. **近似编辑点** (`approx_train_ppo.py`)

训练循环和评估函数中都需要正确设置edit键：

```python
# Build edits
edits = {}
for layer_name, action_codes in layer_actions.items():
    s, z = scales.get(layer_name, (1.0, 0.0))
    tmax_code = layer_tmax_codes[layer_name]
    
    # 选择正确的键名
    if layer_name.startswith('conv'):
        key = f"block_output.{layer_name}"
    elif layer_name.startswith('fc'):
        key = f"classifier.{layer_name}.out"
    
    def make_edit(codes, s_, z_, tmax_):
        def _fn(x):
            return TriApproxINT8_PPO(x, s_, z_, codes, tmax_)
        return _fn
    
    edits[key] = make_edit(action_codes.to(device), s, z, tmax_code)
```

### 7. **Scale收集增强** (`approx_train.py`)

```python
@torch.no_grad()
def teacher_forward_with_scales(teacher, images_cpu):
    scales: Dict[str, Tuple[float,float]] = {}
    def make_sniffer(name):
        def _fn(x):
            scales[name] = (float(x.q_scale()), float(x.q_zero_point()))
            return x
        return _fn
    
    edits = {}
    # VGG16的卷积层
    for k in CONV_LAYERS:
        edits[f"block_output.{k}"] = make_sniffer(k)
    
    # AlexNet的FC层
    if hasattr(teacher, 'fc1'):  # 检测是否为AlexNet
        for fc_name in ['fc1', 'fc2']:
            edits[f"classifier.{fc_name}.out"] = make_sniffer(fc_name)
    
    rec = ActivationRecorder(store_cpu=False, edits=edits)
    logits = teacher(images_cpu, recorder=rec)
    return logits, scales
```

### 8. **Scale缓存兜底** (`approx_train_ppo.py`)

```python
# For AlexNet FC layers, ensure we have scale information
if model_type == "alexnet":
    for lyr in ['fc1', 'fc2']:
        if lyr not in scale_cache:
            scale_cache[lyr] = (1.0, 0.0)  # 默认scale
```

---

## 🎯 工作流程

### 完整的数据流

```
1. 训练开始
   └─ 模型选择: model_type = "alexnet"
   └─ 层列表: CONV_LAYERS_USED = ['conv1'...'conv5', 'fc1', 'fc2']

2. 直方图收集 (calib_batches=16)
   ├─ 卷积层: block_output.conv1 → 256维直方图
   ├─ 卷积层: block_output.conv2 → 256维直方图
   ├─ ...
   ├─ FC层: classifier.fc1.out → 256维直方图
   └─ FC层: classifier.fc2.out → 256维直方图

3. tmax计算
   ├─ 每层: 取p90分位数 + 10
   ├─ conv1: tmax = 35 (例如)
   ├─ fc1: tmax = 28 (FC层通常更小)
   └─ fc2: tmax = 25

4. 状态编码器初始化
   └─ 7层 × 4D状态向量
       ├─ [layer_depth_ratio, act_mean, act_std, p90]
       └─ conv1: [0.0, 0.45, 0.23, 0.85]
           fc1: [0.83, 0.38, 0.19, 0.72]

5. PPO训练循环
   ├─ Episode开始
   ├─ 采样batch
   ├─ Teacher前向 (收集scales)
   │   ├─ conv1 scale: (0.125, 128)
   │   ├─ fc1 scale: (0.098, 130)
   │   └─ ...
   ├─ 对每层采样动作
   │   ├─ conv1: state → policy → [t1=7, v1=14, t2=21, v2=28]
   │   ├─ fc1: state → policy → [t1=6, v1=12, t2=18, v2=24]
   │   └─ ...
   ├─ 构建edits字典
   │   ├─ "block_output.conv1" → TriApproxINT8_PPO(...)
   │   ├─ "classifier.fc1.out" → TriApproxINT8_PPO(...)
   │   └─ ...
   ├─ Student前向 (应用近似)
   ├─ 计算KD loss
   ├─ 计算Attention loss (conv3, conv4, fc1)
   ├─ 计算reward
   └─ 更新PPO

6. 后处理
   ├─ Top-30配置跟踪
   ├─ 批量评估
   └─ 保存结果 (包含7层配置)
```

---

## ⚠️ 注意事项

### 1. **键名必须精确匹配**

❌ 错误：
```python
# FC层错误使用卷积层格式
key = f"block_output.fc1"  # 找不到！
```

✅ 正确：
```python
# FC层使用正确格式
key = f"classifier.fc1.out"  # OK
```

### 2. **所有处理点都要更新**

需要修改的位置（共6处）：
- ✅ 直方图收集
- ✅ tmax计算
- ✅ edits构建（训练循环）
- ✅ edits构建（评估函数）
- ✅ scale收集增强
- ✅ scale缓存兜底

### 3. **FC层特性**

| 特性 | 卷积层 | FC层 |
|------|--------|------|
| **输入形状** | [B, C, H, W] | [B, Features] |
| **激活值范围** | 较大 | 通常较小 |
| **tmax期望** | 30-50 | 20-35 |
| **近似难度** | 中等 | 较容易 |

### 4. **fc3不近似**

```python
# fc3直接输出logits，不进行近似
# 原因：
# 1. 输出层对精度敏感
# 2. 近似可能导致分类边界偏移
# 3. 不是ReLU激活，近似效果差
```

---

## 🧪 测试验证

### 检查点清单

```bash
# 1. 检查层数
# 日志应显示: "Using AlexNet with 7 layers (5 conv + 2 fc)"

# 2. 检查初始化输出
# 应该看到7层的tmax初始化:
#   [init] conv1: t1=7, v1=14, t2=21, v2=28, tmax=35 (p90+10)
#   ...
#   [init] fc1: t1=6, v1=12, t2=18, v2=24, tmax=30 (p90+10)
#   [init] fc2: t1=5, v1=10, t2=15, v2=20, tmax=25 (p90+10)

# 3. 检查结果JSON
# 应包含7个层的配置
cat outputs/tri_ppo_int_codes/alexnet_exp1.json | grep -c "tmax_code"
# 期望输出: 7

# 4. 检查注意力层
# 日志应显示: "Attention Transfer: 3 layers"
# 层名应为: conv3, conv4, fc1

# 5. 验证FC层scale
# 添加调试输出检查scale_cache是否包含fc1和fc2
```

### 常见问题

**问题1: KeyError: 'classifier.fc1.out'**
```
原因: 直方图收集时键名不匹配
解决: 确认使用正确的键名格式
```

**问题2: FC层tmax异常大/小**
```
原因: FC层激活值分布可能与卷积层不同
解决: 这是正常的，FC层通常范围更集中
```

**问题3: scale_cache中没有fc1/fc2**
```
原因: teacher_forward_with_scales未收集FC层scale
解决: 已添加兜底逻辑，使用默认(1.0, 0.0)
```

---

## 📊 预期效果

### 性能对比

| 配置 | 层数 | Episode时间 | 精度影响 |
|------|------|------------|---------|
| **只近似卷积层** | 5层 | ~快 | 小 |
| **近似卷积+FC** | 7层 | ~中等 | 稍大 |

### FC层近似的意义

1. **更全面的量化**
   - 覆盖整个推理路径
   - FC层也有大量计算

2. **精度权衡**
   - FC层激活值通常更稳定
   - 近似影响相对可控

3. **探索空间扩大**
   - 状态空间: 5×4D → 7×4D
   - 动作空间: 5×4码 → 7×4码
   - 搜索难度增加，但更细粒度

---

## 🚀 实际使用

完整命令不变，自动处理FC层：

```bash
python main.py --mode train_ppo_triseg --model alexnet \
  --backend fbgemm --lr 0.0003 --episodes 10000 \
  --batch-size 32 --calib-batches 16 \
  --result-file alexnet_with_fc.json
```

输出将自动包含7层配置！
