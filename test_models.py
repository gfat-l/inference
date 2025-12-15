"""
测试新模型的正确性
用于验证新添加的网络架构是否正确实现
"""

import torch
import torch.nn as nn
from torch.ao.quantization import prepare_qat, convert, get_default_qat_qconfig

from recorder import ActivationRecorder
from model_configs import get_model_config, list_available_models


def test_model_creation(model_name: str):
    """测试模型创建"""
    print(f"\n{'='*60}")
    print(f"测试 {model_name.upper()} - 模型创建")
    print('='*60)
    
    try:
        config = get_model_config(model_name)
        model = config.create_model()
        print(f"✓ 模型创建成功")
        print(f"  模型名称: {config.name}")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        return model, config
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return None, None


def test_forward_pass(model: nn.Module, model_name: str):
    """测试前向传播"""
    print(f"\n测试 {model_name.upper()} - 前向传播")
    print('-'*60)
    
    try:
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (2, 10), f"输出shape错误: {y.shape}"
        print(f"✓ 前向传播成功")
        print(f"  输入shape: {x.shape}")
        print(f"  输出shape: {y.shape}")
        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False


def test_recorder(model: nn.Module, config, model_name: str):
    """测试记录器"""
    print(f"\n测试 {model_name.upper()} - 激活记录器")
    print('-'*60)
    
    try:
        model.eval()
        rec = ActivationRecorder()
        x = torch.randn(1, 3, 32, 32)
        
        with torch.no_grad():
            y = model(x, recorder=rec)
        
        recorded_keys = list(rec.acts.keys())
        print(f"✓ 记录器工作正常")
        print(f"  记录的tap点数量: {len(recorded_keys)}")
        print(f"  前5个tap点:")
        for key in recorded_keys[:5]:
            print(f"    - {key}: {rec.acts[key].shape}")
        
        # 验证tap点
        expected_tap_points = config.get_tap_points()
        layer_names = config.get_layer_names()
        
        missing_taps = []
        for layer_name in layer_names:
            tap_name = expected_tap_points.get(layer_name)
            if tap_name and tap_name not in recorded_keys:
                missing_taps.append(tap_name)
        
        if missing_taps:
            print(f"  ⚠ 缺少的tap点: {missing_taps}")
        else:
            print(f"  ✓ 所有关键tap点都已记录")
        
        return len(missing_taps) == 0
    except Exception as e:
        print(f"✗ 记录器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion(model: nn.Module, model_name: str):
    """测试模型融合"""
    print(f"\n测试 {model_name.upper()} - 模型融合")
    print('-'*60)
    
    try:
        model.eval()
        
        # 融合前
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            y_before = model(x)
        
        # 融合
        model.fuse_model()
        
        # 融合后
        with torch.no_grad():
            y_after = model(x)
        
        # 检查输出是否一致
        diff = torch.abs(y_before - y_after).max().item()
        
        print(f"✓ 模型融合成功")
        print(f"  融合前后输出最大差异: {diff:.6f}")
        
        if diff > 1e-5:
            print(f"  ⚠ 差异较大，可能存在融合问题")
            return False
        else:
            print(f"  ✓ 融合正确")
            return True
            
    except Exception as e:
        print(f"✗ 模型融合失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantization(model: nn.Module, model_name: str):
    """测试量化"""
    print(f"\n测试 {model_name.upper()} - 量化准备")
    print('-'*60)
    
    try:
        model.train()
        model.fuse_model()
        
        # QAT配置
        qconfig = get_default_qat_qconfig('fbgemm')
        model.qconfig = qconfig
        
        # 准备QAT
        prepare_qat(model, inplace=True)
        
        # 测试前向传播
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        
        # 测试反向传播
        loss = y.sum()
        loss.backward()
        
        print(f"✓ QAT准备成功")
        print(f"  前向传播: {y.shape}")
        print(f"  反向传播: 正常")
        
        # 转换为INT8
        model.eval()
        convert(model, inplace=True)
        
        with torch.no_grad():
            y_int8 = model(x)
        
        print(f"✓ INT8转换成功")
        print(f"  INT8输出: {y_int8.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 量化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_layer_names(config, model_name: str):
    """测试层名称配置"""
    print(f"\n测试 {model_name.upper()} - 层名称配置")
    print('-'*60)
    
    try:
        layer_names = config.get_layer_names()
        tap_points = config.get_tap_points()
        
        print(f"✓ 配置读取成功")
        print(f"  可探索层数量: {len(layer_names)}")
        print(f"  层名称: {layer_names}")
        print(f"\n  Tap点映射:")
        for layer_name, tap_name in tap_points.items():
            print(f"    {layer_name:15s} -> {tap_name}")
        
        return True
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False


def run_all_tests(model_name: str):
    """运行所有测试"""
    print(f"\n{'#'*60}")
    print(f"# 开始测试: {model_name.upper()}")
    print(f"{'#'*60}")
    
    # 测试1: 模型创建
    model, config = test_model_creation(model_name)
    if model is None:
        return False
    
    # 测试2: 层名称配置
    test_layer_names(config, model_name)
    
    # 测试3: 前向传播
    if not test_forward_pass(model, model_name):
        return False
    
    # 测试4: 记录器
    if not test_recorder(model, config, model_name):
        return False
    
    # 测试5: 模型融合
    # 重新创建模型（因为融合会修改模型）
    model = config.create_model()
    if not test_fusion(model, model_name):
        return False
    
    # 测试6: 量化
    model = config.create_model()
    if not test_quantization(model, model_name):
        return False
    
    print(f"\n{'='*60}")
    print(f"✓ {model_name.upper()} 所有测试通过！")
    print(f"{'='*60}")
    
    return True


def main():
    """主函数"""
    print("="*60)
    print("模型测试工具")
    print("="*60)
    
    available_models = list_available_models()
    print(f"\n可用模型: {', '.join(available_models)}")
    
    # 测试所有模型
    results = {}
    for model_name in available_models:
        try:
            success = run_all_tests(model_name)
            results[model_name] = success
        except Exception as e:
            print(f"\n✗ {model_name.upper()} 测试出错: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = False
    
    # 汇总结果
    print(f"\n\n{'='*60}")
    print("测试结果汇总")
    print('='*60)
    for model_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{model_name:15s}: {status}")
    
    # 统计
    passed = sum(1 for s in results.values() if s)
    total = len(results)
    print(f"\n总计: {passed}/{total} 个模型测试通过")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试模型实现")
    parser.add_argument("--model", type=str, default=None,
                       help="指定要测试的模型名称（不指定则测试所有）")
    args = parser.parse_args()
    
    if args.model:
        run_all_tests(args.model)
    else:
        main()
