import json
import argparse
import sys

def print_config_table(config_path):
    # 读取配置文件
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 获取layers字典
    layers_config = config.get("layers", {})

    # 获取所有层名，直接使用文件中的顺序
    layers = list(layers_config.keys())

    # 打印表头
    print("Layer\t\tT1\tT2\tV1\tV2\tTMAX")
    print("-" * 50)

    # 打印每一行
    for layer in layers:
        params = layers_config[layer]
        t1 = params.get('t1_code', 'N/A')
        t2 = params.get('t2_code', 'N/A')
        v1 = params.get('v1_code', 'N/A')
        v2 = params.get('v2_code', 'N/A')
        tmax = params.get('tmax_code', 'N/A')
        print(f"{layer}\t\t{t1}\t{t2}\t{v1}\t{v2}\t{tmax}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print approximation config as a table.")
    parser.add_argument("--config", required=True, help="Path to the config JSON file.")
    args = parser.parse_args()

    print_config_table(args.config)