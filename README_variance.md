# 方差分布分析功能说明

## 新增功能

已为项目添加卷积核权重和激活值的方差分布统计功能。

## 使用方法

### 1. 运行方差分析

```bash
python main.py --mode analyze_variance --backend fbgemm
```

### 2. 输出文件

分析完成后，会在 `outputs/variance_analysis/` 目录下生成两个JSON文件：

- **`weight_variance.json`** - 卷积核权重方差统计
  - 每层的均值、方差、标准差
  - 权重的最小值和最大值
  - 按输出通道的方差统计

- **`activation_variance.json`** - 激活值方差统计
  - 每层激活的均值、方差、标准差（基于10个batch的平均）
  - 激活值的形状信息

### 3. 输出格式示例

#### weight_variance.json
```json
{
  "conv1_1.conv": {
    "mean": 0.0023,
    "var": 0.0156,
    "std": 0.1249,
    "shape": [64, 3, 3, 3],
    "min": -0.5234,
    "max": 0.4987,
    "per_channel_var_mean": 0.0145,
    "per_channel_var_std": 0.0023
  }
}
```

#### activation_variance.json
```json
{
  "block_output.conv1_1": {
    "mean_avg": 0.3456,
    "var_avg": 0.2345,
    "std_avg": 0.4842,
    "shape": [128, 64, 32, 32],
    "num_samples": 10
  }
}
```

## 代码修改说明

### 1. `recorder.py` 更新
- `ActivationRecorder` 新增 `record_variance` 参数
- 新增 `_compute_variance()` 方法计算激活值方差
- 新增 `WeightVarianceRecorder` 类用于记录权重方差

### 2. `activation_tools.py` 更新
- 新增 `analyze_variance_distribution()` 函数
- 同时分析权重和激活值的方差分布
- 生成汇总报告并保存JSON文件

### 3. `main.py` 更新
- 新增 `analyze_variance` 模式
- 集成方差分析功能

## 技术细节

### 权重方差统计
- 遍历所有 `Conv2d` 层
- 计算整体统计量和按通道统计量
- 支持量化模型的权重分析

### 激活值方差统计
- 使用10个batch的数据求平均，避免单batch偏差
- 支持量化激活值（自动反量化为浮点值统计）
- 记录每层的形状信息便于分析

## 应用场景

1. **量化敏感层识别** - 找出方差较大的层，这些层对量化更敏感
2. **近似计算策略优化** - 根据方差分布设计更合理的激活值近似规则
3. **模型压缩分析** - 评估不同层的信息量分布
4. **调试量化问题** - 对比量化前后的方差变化
