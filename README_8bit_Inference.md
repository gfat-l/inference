# AlexNet INT8 C++ 推理实施说明

本文档详细说明了如何从 PyTorch QAT 模型导出数据，并在 C++ 环境中实现 8-bit 推理。

## 1. 目录结构

```
cpp_file/
  alexnet_8bit/
    Alexnet.h          # 推理核心头文件
    Alexnet.cpp        # 推理核心实现（Conv2d, MaxPool, FC, Requantize）
    main.cpp           # 主程序（加载模型、读取输入、推理、对比结果）
    Makefile           # (可选) 编译脚本
  alexnet_8bit_export/ # [生成] 存放导出的模型参数和测试数据
    model_config.txt   # 模型结构与量化参数配置
    *_w_int8.txt       # 各层 int8 权重 (文本格式，每行一个数值)
    *_w_scales.txt     # 各层 per-channel 权重缩放因子 (文本格式)
    *_bias_int32.txt   # 各层 int32 偏置 (文本格式)
    input_float.txt    # 测试用输入数据 (文本格式)
    output_logits_ref.txt # Python 端生成的参考输出 (文本格式)
```

## 2. 实施步骤

### 第一步：导出模型与生成测试数据

使用提供的 Python 脚本 `export_alexnet_int8_for_cpp.py` 和 `gen_test_data.py`。

1.  **导出模型参数**：
    该脚本会加载训练好的 QAT 模型 (`outputs/alexnet_qat_preconvert.pth`)，提取每一层的权重、偏置、Scale、ZeroPoint，并保存为 C++ 可读取的文本格式 (`.txt`)。
    ```bash
    python export_alexnet_int8_for_cpp.py --ckpt outputs/alexnet_qat_preconvert.pth --out cpp_file/alexnet_8bit_export
    ```

2.  **生成测试数据**：
    该脚本会从 CIFAR-10 测试集中读取一张图片，进行预处理（归一化），保存为 `input_float.txt`。同时会运行 Python 端的量化推理，保存结果 `output_logits_ref.txt` 用于验证 C++ 精度。
    ```bash
    python gen_test_data.py --ckpt outputs/alexnet_qat_preconvert.pth --out_dir cpp_file/alexnet_8bit_export
    ```

### 第二步：编译 C++ 推理程序

进入项目根目录，使用 C++ 编译器进行编译。

**使用 g++:**
```bash
g++ cpp_file/alexnet_8bit/Alexnet.cpp cpp_file/alexnet_8bit/main.cpp -o alexnet_int8 -O3
```

**使用 Visual Studio:**
创建一个新的空项目，将 `Alexnet.h`, `Alexnet.cpp`, `main.cpp` 添加到项目中。

### 第三步：运行推理

直接运行生成的可执行文件（程序默认会在 `cpp_file/alexnet_8bit_export` 路径下寻找模型文件）。

```bash
./alexnet_int8
```

程序将输出：
1.  加载各层参数的日志。
2.  加载 `input_float.txt` 的信息。
3.  各类别的预测概率。
4.  与 Python 参考结果的误差对比（Diff）。

## 3. 关键技术原理

### 3.1 输入图像处理与量化
*   **训练时预处理**：
    *   Resize/Crop 到 32x32。
    *   ToTensor: 像素值从 [0, 255] 变为 [0.0, 1.0]。
    *   Normalize: `(x - mean) / std`。CIFAR-10 的 mean=(0.4914, ...), std=(0.2470, ...)。
*   **C++ 端量化**：
    C++ 程序接收的是经过上述 Normalize 后的 **Float** 数据。在进入第一层卷积之前，程序会根据第一层的输入量化参数 (`in_scale`, `in_zp`) 将其量化为 INT8 (uint8 存储)：
    $$ q = \text{clamp}(\text{round}(x / S_{in}) + Z_{in}, 0, 255) $$
    这一步在 `Alexnet.cpp` 的 `forward` 函数开头自动完成。

### 3.2 卷积与全连接 (Conv & FC)
*   **计算核心**：
    $$ \text{acc} = \sum (q_{in} - Z_{in}) \times (q_{w} - Z_{w}) + b_{quant} $$
    *   $q_{in}$: 输入激活值 (uint8, 0-255)。
    *   $Z_{in}$: 输入零点。
    *   $q_{w}$: 权重 (int8, -128-127)。
    *   $Z_{w}$: 权重零点 (通常为0，对称量化)。
    *   $b_{quant}$: 预量化的偏置 (int32)。
    *   $\text{acc}$: 32位累加器，防止溢出。

### 3.3 Per-Channel 重量化 (Requantization)
由于权重是 Per-Channel 量化的，每个输出通道 $oc$ 有独立的 $S_{w}[oc]$。
*   **Effective Scale**:
    $$ M[oc] = \frac{S_{in} \times S_{w}[oc]}{S_{out}} $$
*   **输出计算**:
    $$ q_{out} = \text{clamp}(\text{round}(\text{acc} \times M[oc]) + Z_{out}, 0, 255) $$
    这一步将 int32 累加值映射回下一层所需的 uint8 范围。

### 3.4 算子融合 (Fusion)
*   **Conv + Bias**: 偏置直接加在 int32 累加器上。
*   **Conv + ReLU**: 通过 Requantize 时的 `clamp(0, 255)` 隐式实现。因为 ReLU 的输出非负，且量化参数 $Z_{out}$ 通常对应实数 0，截断负值即等效于 ReLU。

## 4. 常见问题
*   **精度对不齐**：
    *   检查 `input_float.txt` 是否与 Python 端完全一致。
    *   检查 Bias 是否正确加载（融合后的 Conv 必须有 Bias）。
    *   检查 Rounding 策略（C++ `std::round` vs Python `torch.round`）。通常误差在 1e-5 级别是正常的。
*   **文件路径错误**：
    *   请确保 `main.cpp` 中的 `model_dir` 变量指向正确的导出文件夹路径。

## 5. 文件存储格式说明

所有导出的数据文件（权重、偏置、输入、输出）均采用 **纯文本格式 (.txt)** 存储，以便于跨平台读取和调试。

*   **格式**：每个数值占一行（或以空白字符分隔）。
*   **读取方式**：使用标准 C++ 流操作符 `>>` 即可连续读取。
*   **示例**：
    ```text
    -12
    5
    0
    127
    ...
    ```

