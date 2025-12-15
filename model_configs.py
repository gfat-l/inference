"""
模型配置文件
统一管理不同网络架构的配置信息
"""

from typing import Dict, List, Type
import torch.nn as nn


class ModelConfig:
    """
    模型配置基类
    """
    def __init__(
        self,
        name: str,
        model_class: Type[nn.Module],
        num_classes: int = 10,
    ):
        self.name = name
        self.model_class = model_class
        self.num_classes = num_classes
    
    def get_layer_names(self) -> List[str]:
        """返回所有需要近似探索的层名称"""
        raise NotImplementedError
    
    def get_tap_points(self) -> Dict[str, str]:
        """返回所有tap点（用于记录和编辑的关键位置）"""
        raise NotImplementedError
    
    def create_model(self) -> nn.Module:
        """创建模型实例"""
        return self.model_class(num_classes=self.num_classes)


class VGG16Config(ModelConfig):
    """VGG16配置"""
    def __init__(self, num_classes: int = 10):
        from model_vgg16_tap_quant import VGG16TapQuant
        super().__init__("VGG16", VGG16TapQuant, num_classes)
    
    def get_layer_names(self) -> List[str]:
        """VGG16的13个卷积层"""
        layers = []
        # block1: conv1_1, conv1_2
        layers.extend([f"conv1_{i}" for i in range(1, 3)])
        # block2: conv2_1, conv2_2
        layers.extend([f"conv2_{i}" for i in range(1, 3)])
        # block3: conv3_1, conv3_2, conv3_3
        layers.extend([f"conv3_{i}" for i in range(1, 4)])
        # block4: conv4_1, conv4_2, conv4_3
        layers.extend([f"conv4_{i}" for i in range(1, 4)])
        # block5: conv5_1, conv5_2, conv5_3
        layers.extend([f"conv5_{i}" for i in range(1, 4)])
        return layers
    
    def get_tap_points(self) -> Dict[str, str]:
        """VGG16的tap点"""
        tap_points = {}
        for layer_name in self.get_layer_names():
            tap_points[layer_name] = f"block_output.{layer_name}"
        return tap_points


class VGG11Config(ModelConfig):
    """VGG11配置"""
    def __init__(self, num_classes: int = 10):
        # 需要创建对应的model_vgg11_tap_quant.py
        try:
            from model_vgg11_tap_quant import VGG11TapQuant
            super().__init__("VGG11", VGG11TapQuant, num_classes)
        except ImportError:
            raise ImportError("请先创建 model_vgg11_tap_quant.py 文件")
    
    def get_layer_names(self) -> List[str]:
        """VGG11的8个卷积层"""
        # VGG11: 1-1-2-2-2 结构
        return [
            "conv1_1",
            "conv2_1",
            "conv3_1", "conv3_2",
            "conv4_1", "conv4_2",
            "conv5_1", "conv5_2",
        ]
    
    def get_tap_points(self) -> Dict[str, str]:
        tap_points = {}
        for layer_name in self.get_layer_names():
            tap_points[layer_name] = f"block_output.{layer_name}"
        return tap_points


class VGG13Config(ModelConfig):
    """VGG13配置"""
    def __init__(self, num_classes: int = 10):
        try:
            from model_vgg13_tap_quant import VGG13TapQuant
            super().__init__("VGG13", VGG13TapQuant, num_classes)
        except ImportError:
            raise ImportError("请先创建 model_vgg13_tap_quant.py 文件")
    
    def get_layer_names(self) -> List[str]:
        """VGG13的10个卷积层"""
        # VGG13: 2-2-2-2-2 结构
        layers = []
        for block in range(1, 6):
            for i in range(1, 3):
                layers.append(f"conv{block}_{i}")
        return layers
    
    def get_tap_points(self) -> Dict[str, str]:
        tap_points = {}
        for layer_name in self.get_layer_names():
            tap_points[layer_name] = f"block_output.{layer_name}"
        return tap_points


class VGG19Config(ModelConfig):
    """VGG19配置"""
    def __init__(self, num_classes: int = 10):
        try:
            from model_vgg19_tap_quant import VGG19TapQuant
            super().__init__("VGG19", VGG19TapQuant, num_classes)
        except ImportError:
            raise ImportError("请先创建 model_vgg19_tap_quant.py 文件")
    
    def get_layer_names(self) -> List[str]:
        """VGG19的16个卷积层"""
        # VGG19: 2-2-4-4-4 结构
        layers = []
        layers.extend([f"conv1_{i}" for i in range(1, 3)])  # 2层
        layers.extend([f"conv2_{i}" for i in range(1, 3)])  # 2层
        layers.extend([f"conv3_{i}" for i in range(1, 5)])  # 4层
        layers.extend([f"conv4_{i}" for i in range(1, 5)])  # 4层
        layers.extend([f"conv5_{i}" for i in range(1, 5)])  # 4层
        return layers
    
    def get_tap_points(self) -> Dict[str, str]:
        tap_points = {}
        for layer_name in self.get_layer_names():
            tap_points[layer_name] = f"block_output.{layer_name}"
        return tap_points


class AlexNetConfig(ModelConfig):
    """AlexNet配置"""
    def __init__(self, num_classes: int = 10):
        from model_alexnet_tap_quant import AlexNetTapQuant
        super().__init__("AlexNet", AlexNetTapQuant, num_classes)
    
    def get_layer_names(self) -> List[str]:
        """AlexNet的5个卷积层 + 2个全连接层"""
        conv_layers = [f"conv{i}" for i in range(1, 6)]  # conv1-conv5
        fc_layers = ["fc1", "fc2"]  # fc1, fc2 (不包括最后的fc3)
        return conv_layers + fc_layers
    
    def get_tap_points(self) -> Dict[str, str]:
        """AlexNet的tap点"""
        tap_points = {}
        # 卷积层
        for i in range(1, 6):
            tap_points[f"conv{i}"] = f"block_output.conv{i}"
        # 全连接层
        tap_points["fc1"] = "classifier.fc1.out"
        tap_points["fc2"] = "classifier.fc2.out"
        tap_points["fc3"] = "classifier.fc3.out"
        return tap_points


# ========== 模型注册表 ==========

MODEL_REGISTRY: Dict[str, Type[ModelConfig]] = {
    "vgg16": VGG16Config,
    "vgg11": VGG11Config,
    "vgg13": VGG13Config,
    "vgg19": VGG19Config,
    "alexnet": AlexNetConfig,
}


def get_model_config(model_name: str, num_classes: int = 10) -> ModelConfig:
    """
    根据模型名称获取配置
    
    Args:
        model_name: 模型名称 (vgg16, vgg11, vgg13, vgg19, alexnet)
        num_classes: 分类数量
    
    Returns:
        ModelConfig实例
    
    Example:
        >>> config = get_model_config("alexnet")
        >>> model = config.create_model()
        >>> layer_names = config.get_layer_names()
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"未知模型: {model_name}. 可用模型: {available}")
    
    config_class = MODEL_REGISTRY[model_name]
    return config_class(num_classes=num_classes)


def list_available_models() -> List[str]:
    """列出所有可用的模型"""
    return list(MODEL_REGISTRY.keys())


if __name__ == "__main__":
    print("可用模型列表：")
    for model_name in list_available_models():
        try:
            config = get_model_config(model_name)
            print(f"\n{config.name}:")
            print(f"  - 层数: {len(config.get_layer_names())}")
            print(f"  - 层名称: {config.get_layer_names()}")
        except ImportError as e:
            print(f"\n{model_name}: {e}")
