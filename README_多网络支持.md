# 多网络架构支持 - 快速开始

本项目已扩展支持多种网络架构的INT8量化和近似探索。

## 支持的网络

- ✅ **VGG16** (原始实现)
- ✅ **AlexNet** (新增，带BN层)
- 🚧 **VGG11** (需要实现)
- 🚧 **VGG13** (需要实现)
- 🚧 **VGG19** (需要实现)

## 快速开始

### 1. 测试AlexNet实现

```bash
# 测试AlexNet模型
python test_models.py --model alexnet

# 测试所有模型
python test_models.py
```

### 2. 使用AlexNet训练

```bash
# 训练浮点模型
python main.py float --model alexnet --epochs 20 --lr 0.01

# QAT训练
python main.py qat --model alexnet --qat-epochs 10

# 导出INT8模型
python main.py export-int8 --model alexnet

# PPO近似探索
python main.py ppo --model alexnet --episodes 500

# 评估INT8模型
python main.py eval-int8 --model alexnet
```

### 3. 添加新网络

详细步骤请参考：[如何添加新网络架构.md](./如何添加新网络架构.md)

简要步骤：

1. **创建模型文件**
   ```bash
   # 复制AlexNet模板
   cp model_alexnet_tap_quant.py model_yournet_tap_quant.py
   ```

2. **修改网络结构**
   - 更新 `__init__` 中的层定义
   - 更新 `forward` 中的前向传播逻辑
   - 更新辅助函数 `get_yournet_layer_names()` 等

3. **注册到配置系统**
   
   在 `model_configs.py` 中添加：
   ```python
   class YourNetConfig(ModelConfig):
       def __init__(self, num_classes: int = 10):
           from model_yournet_tap_quant import YourNetTapQuant
           super().__init__("YourNet", YourNetTapQuant, num_classes)
       
       def get_layer_names(self):
           # 返回层名列表
           return [...]
       
       def get_tap_points(self):
           # 返回tap点映射
           return {...}
   
   # 注册
   MODEL_REGISTRY["yournet"] = YourNetConfig
   ```

4. **测试实现**
   ```bash
   python test_models.py --model yournet
   ```

## 项目结构

```
规范代码ppo_优化/
├── model_vgg16_tap_quant.py       # VGG16原始实现
├── model_alexnet_tap_quant.py     # AlexNet新实现
├── model_configs.py               # 统一的模型配置系统
├── test_models.py                 # 模型测试工具
├── 如何添加新网络架构.md          # 详细添加指南
├── main.py                        # 主程序入口
├── approx_train_ppo.py           # PPO训练逻辑
├── train_qat.py                  # QAT训练逻辑
├── inference_int8.py             # INT8推理
├── recorder.py                   # 激活记录器
└── ...
```

## 核心概念

### Tap点命名规范

所有网络必须遵循统一的tap点命名：

**卷积层**：
- 融合前：`features.{layer}.conv_out`, `features.{layer}.bn_out`, `features.{layer}.relu_out`
- 融合后：`block_output.{layer}`

**全连接层**：
- 融合前：`classifier.{layer}.linear_out`, `classifier.{layer}.relu_out`
- 融合后：`classifier.{layer}.out`

**入口点**：
- `block_input.{layer}`

### 模型要求

每个模型类必须实现：

1. `__init__(self, num_classes=10)` - 初始化
2. `fuse_model(self)` - 融合conv+bn+relu
3. `forward(self, x, recorder=None)` - 支持记录的前向传播
4. `self.quant` 和 `self.dequant` - 量化桩

## 常用命令

### 训练工作流

```bash
# 完整工作流（以AlexNet为例）
python main.py float --model alexnet --epochs 20
python main.py qat --model alexnet --qat-epochs 10
python main.py export-int8 --model alexnet
python main.py ppo --model alexnet --episodes 500 --result-file alexnet_ppo.json
python main.py eval-int8 --model alexnet
```

### 对比不同网络

```bash
# 训练VGG16
python main.py float --model vgg16 --out ./outputs/vgg16
python main.py qat --model vgg16 --out ./outputs/vgg16

# 训练AlexNet
python main.py float --model alexnet --out ./outputs/alexnet
python main.py qat --model alexnet --out ./outputs/alexnet

# 对比结果
# 查看 ./outputs/vgg16 和 ./outputs/alexnet 目录
```

## 调试技巧

### 检查模型结构

```python
from model_configs import get_model_config

config = get_model_config("alexnet")
model = config.create_model()
print(model)
```

### 检查tap点

```python
from recorder import ActivationRecorder
from model_configs import get_model_config
import torch

config = get_model_config("alexnet")
model = config.create_model()
model.eval()

rec = ActivationRecorder()
x = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    y = model(x, recorder=rec)

# 查看所有记录的tap点
for key, value in rec.acts.items():
    print(f"{key}: {value.shape}")
```

### 验证量化

```python
from torch.ao.quantization import prepare_qat, convert, get_default_qat_qconfig

# 准备QAT
model.train()
model.fuse_model()
model.qconfig = get_default_qat_qconfig('fbgemm')
prepare_qat(model, inplace=True)

# 训练几步...

# 转换为INT8
model.eval()
convert(model, inplace=True)

# 测试
x = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    y = model(x)
print(f"INT8输出: {y.shape}")
```

## 常见问题

### Q: 如何知道tap点名称是否正确？

A: 运行 `python test_models.py --model yournet`，测试会验证所有tap点。

### Q: 融合后模型输出不一致怎么办？

A: 检查融合配置，确保模块名称匹配：
```python
# 确保Sequential的子模块命名正确
nn.Sequential(OrderedDict([
    ("conv", ...),  # 名称必须是 "conv"
    ("bn", ...),    # 名称必须是 "bn"
    ("relu", ...),  # 名称必须是 "relu"
]))
```

### Q: 如何添加残差连接？

A: 参考文档 [如何添加新网络架构.md](./如何添加新网络架构.md) 的"常见问题"部分。

## 更多信息

- **详细添加指南**: [如何添加新网络架构.md](./如何添加新网络架构.md)
- **原始功能说明**: [项目功能详细分析.md](./项目功能详细分析.md)
- **AlexNet实现**: [model_alexnet_tap_quant.py](./model_alexnet_tap_quant.py)
- **配置系统**: [model_configs.py](./model_configs.py)

## 贡献

添加新网络后，请：

1. 运行 `python test_models.py --model yournet` 确保测试通过
2. 更新本文档的"支持的网络"部分
3. 在 `model_configs.py` 中注册新模型

---

**提示**: 如果遇到问题，先运行测试脚本诊断：
```bash
python test_models.py --model yournet
```
