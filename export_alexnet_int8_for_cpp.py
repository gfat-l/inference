import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

from model_alexnet_tap_quant import AlexNetTapQuant


def build_int8_model(ckpt_path: str, backend: str) -> torch.nn.Module:
    """Load QAT checkpoint -> fuse -> prepare_qat -> load weights -> convert to quantized model."""
    torch.backends.quantized.engine = backend

    model = AlexNetTapQuant(num_classes=10)
    model.eval()
    model.fuse_model()

    # Prepare QAT graph to match saved checkpoint structure
    model.qconfig = get_default_qat_qconfig(backend)
    model.train()
    prepare_qat(model, inplace=True)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)

    model.eval()
    qmodel = convert(model, inplace=False)
    qmodel.eval()
    return qmodel


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _save_tensor_txt(t: torch.Tensor, path: str) -> None:
    arr = t.detach().cpu().numpy().flatten()
    with open(path, "w") as f:
        for val in arr:
            f.write(f"{val}\n")


def export_alexnet(qmodel: torch.nn.Module, out_dir: str) -> None:
    """Export per-layer int8 weights, bias (int32), and quant params for C++ inference."""
    _ensure_dir(out_dir)

    layer_order = [
        ("conv1", qmodel.conv1),
        ("conv2", qmodel.conv2),
        ("conv3", qmodel.conv3),
        ("conv4", qmodel.conv4),
        ("conv5", qmodel.conv5),
        ("fc1", qmodel.fc1),
        ("fc2", qmodel.fc2),
        ("fc3", qmodel.fc3),
    ]

    # Capture activation scales/zps by running a dummy forward and hooking inputs/outputs
    act_scales: Dict[str, Dict[str, float]] = {}

    def make_pre(name: str):
        def hook(module, inputs):
            x = inputs[0]
            act_scales.setdefault(name, {})["in_scale"] = float(x.q_scale())
            act_scales[name]["in_zp"] = int(x.q_zero_point())
        return hook

    def make_post(name: str):
        def hook(module, inputs, output):
            act_scales.setdefault(name, {})["out_scale"] = float(output.q_scale())
            act_scales[name]["out_zp"] = int(output.q_zero_point())
        return hook

    hooks = []
    for name, module in layer_order:
        hooks.append(module.register_forward_pre_hook(make_pre(name)))
        hooks.append(module.register_forward_hook(make_post(name)))

    # Forward a dummy batch to populate activation stats (scales are fixed after convert)
    dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        _ = qmodel(dummy)

    for h in hooks:
        h.remove()

    meta: List[Dict] = []

    for name, module in layer_order:
        # Handle Sequential containers (due to fusion)
        if isinstance(module, torch.nn.Sequential):
            # Assume the first module holds the weights (conv or linear)
            # After fusion, conv/linear is at index 0
            quantized_module = module[0]
        else:
            quantized_module = module

        weight_q = quantized_module.weight()
        weight_int8 = weight_q.int_repr().to(torch.int8)

        # Weight quant params (per-channel for conv/linear with default qconfig)
        if weight_q.is_quantized and weight_q.qscheme() in (
            torch.per_channel_symmetric,
            torch.per_channel_affine,
        ):
            w_scales = weight_q.q_per_channel_scales().detach().cpu().tolist()
            w_zps = weight_q.q_per_channel_zero_points().detach().cpu().tolist()
            w_axis = weight_q.q_per_channel_axis()
        else:
            w_scales = [float(weight_q.q_scale())]
            w_zps = [int(weight_q.q_zero_point())]
            w_axis = 0

        in_scale = act_scales[name]["in_scale"]
        in_zp = act_scales[name]["in_zp"]
        out_scale = act_scales[name]["out_scale"]
        out_zp = act_scales[name]["out_zp"]

        # Bias: stored as float in quantized modules; convert to int32 using s_bias = s_in * s_w
        bias_fp = quantized_module.bias().detach().cpu() if quantized_module.bias() is not None else None
        if bias_fp is not None:
            s_w = torch.tensor(w_scales, dtype=torch.float32).view(-1)
            s_in = torch.tensor(in_scale, dtype=torch.float32)
            bias_int32 = torch.round(bias_fp.view(-1) / (s_in * s_w)).to(torch.int32)
            bias_path = os.path.join(out_dir, f"{name}_bias_int32.txt")
            _save_tensor_txt(bias_int32, bias_path)
        else:
            bias_path = None

        weight_path = os.path.join(out_dir, f"{name}_w_int8.txt")
        _save_tensor_txt(weight_int8, weight_path)

        meta.append(
            {
                "name": name,
                "op_type": quantized_module.__class__.__name__,
                "weight_path": os.path.basename(weight_path),
                "bias_path": os.path.basename(bias_path) if bias_path else None,
                "weight_shape": list(weight_int8.shape),
                "weight_scales": w_scales,
                "weight_zero_points": w_zps,
                "weight_qaxis": w_axis,
                "in_scale": in_scale,
                "in_zero_point": in_zp,
                "out_scale": out_scale,
                "out_zero_point": out_zp,
                # Conv/linear specific attrs
                "stride": getattr(quantized_module, "stride", None),
                "padding": getattr(quantized_module, "padding", None),
                "dilation": getattr(quantized_module, "dilation", None),
                "groups": getattr(quantized_module, "groups", None),
            }
        )

    # Save simple text config for easy C++ parsing
    config_path = os.path.join(out_dir, "model_config.txt")
    with open(config_path, "w") as f:
        f.write(f"{len(layer_order)}\n")  # Number of layers
        for meta_item in meta:
            # Format: Name OpType InCh OutCh K S P G InScale InZP OutScale OutZP
            # For FC, K=0, S=0, P=0. Shape is [Out, In]
            name = meta_item["name"]
            op_type = meta_item["op_type"]
            
            if "Conv" in op_type:
                out_ch, in_ch, k, _ = meta_item["weight_shape"]
                s = meta_item["stride"][0]
                p = meta_item["padding"][0]
                g = meta_item["groups"]
            elif "Linear" in op_type:
                out_ch, in_ch = meta_item["weight_shape"]
                k, s, p, g = 0, 0, 0, 1
            
            f.write(f"{name} {op_type} {in_ch} {out_ch} {k} {s} {p} {g} "
                    f"{meta_item['in_scale']:.12f} {meta_item['in_zero_point']} "
                    f"{meta_item['out_scale']:.12f} {meta_item['out_zero_point']}\n")

            # Save per-channel scales as binary
            w_scales_tensor = torch.tensor(meta_item["weight_scales"], dtype=torch.float32)
            _save_tensor_txt(w_scales_tensor, os.path.join(out_dir, f"{name}_w_scales.txt"))

    print(f"[Saved] config -> {config_path}")
    print(f"[Saved] weights/bias/scales to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AlexNet QAT int8 params for C++")
    parser.add_argument("--ckpt", type=str, default="outputs/alexnet_qat_preconvert.pth")
    parser.add_argument("--out", type=str, default="cpp_file/alexnet_8bit_export")
    parser.add_argument("--backend", type=str, default="fbgemm", choices=["fbgemm", "qnnpack"])
    args = parser.parse_args()

    qmodel = build_int8_model(args.ckpt, args.backend)
    export_alexnet(qmodel, args.out)


if __name__ == "__main__":
    main()
