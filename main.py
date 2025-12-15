# main.py
import argparse
from approx_train import train_supervised
from approx_train_ppo import train_ppo
from train_qat import train_float, train_qat, eval_float
from inference_int8 import export_int8, eval_int8
from activation_tools import (
    dump_acts_float,
    demo_edit_float,
    dump_acts_int8,
    eval_int8_modified,
    eval_int8_with_ppo_config,
    analyze_variance_distribution,
    collect_activation_percentiles,
    analyze_layer_sensitivity,
)

def build_parser():
    p = argparse.ArgumentParser("VGG16 QAT CIFAR10 (modularized)")
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--out", type=str, default="./outputs")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--backend", type=str, default="fbgemm")
    
    # 模型架构选择
    p.add_argument("--model", type=str, default="vgg16", 
                   choices=["vgg16", "vgg19", "alexnet", "lenet5", "nin", "mobilenetv1"],
                   help="Model architecture to use")

    # float 训练
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.01)

    # QAT
    p.add_argument("--qat-epochs", type=int, default=10)
    p.add_argument("--qat-lr", type=float, default=1e-3)
    p.add_argument("--warmup-float-epochs", type=int, default=2)
    p.add_argument("--acc-drop-budget", type=float, default=2.0)
    p.add_argument("--calib-batches", type=int, default=8)
    p.add_argument("--kd-T", dest="kd_T", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.1)
    # PPO specific
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--update-timestep", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=50)
    # tmax calculation mode
    p.add_argument("--tmax-mode", type=str, default="percentile", 
                   choices=["fixed", "percentile"],
                   help="tmax calculation mode: 'fixed' uses fixed value, 'percentile' uses percentile + offset")
    p.add_argument("--tmax-fixed", type=int, default=32,
                   help="Fixed tmax value (used when --tmax-mode=fixed)")
    p.add_argument("--tmax-percentile", type=float, default=90,
                   help="Percentile for tmax calculation (used when --tmax-mode=percentile)")
    p.add_argument("--tmax-offset", type=int, default=10,
                   help="Offset added to percentile value (used when --tmax-mode=percentile)")
    # Coverage Analysis specific
    p.add_argument("--coverage-ignore-zeros", action="store_true", 
                   help="If set, coverage analysis will only consider non-zero values (ignore 0s in denominator).")
    # Eval-only (PPO config) optional path
    p.add_argument("--config-path", type=str, default=None, help="Path to PPO result.json; default to outputs/tri_ppo_int_codes/result.json")
    # Optional: expose max-acc-drop for PPO training
    p.add_argument("--max-acc-drop", type=float, default=2.0, help="Max allowed accuracy drop (%) for constrained-best selection in PPO")
    # Result file output control
    p.add_argument("--result-file", type=str, default="result.json", help="Output result filename (saved in outputs/tri_ppo_int_codes/)")
    # p.add_argument("--epochs", type=int, default=3)
    # p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "train_float",
            "eval_float",
            "train_qat",
            "export_int8",
            "eval_int8",
            "dump_acts_float",
            "demo_edit_float",
            "dump_acts_int8",
            "eval_int8_modified",
            "analyze_variance",
            "collect_percentiles",
            "train_supervised_triseg",
            "train_ppo_triseg",
            "eval_ppo_int8",
            "analyze_sensitivity",
            "analyze_coverage",
        ],
    )
    return p

def main():
    args = build_parser().parse_args()

    if args.mode == "train_float":
        train_float(args)
    elif args.mode == "eval_float":
        eval_float(args)
    elif args.mode == "train_qat":
        train_qat(args)
    elif args.mode == "export_int8":
        export_int8(args)
    elif args.mode == "eval_int8":
        eval_int8(args)
    elif args.mode == "dump_acts_float":
        dump_acts_float(args)
    elif args.mode == "demo_edit_float":
        demo_edit_float(args)
    elif args.mode == "dump_acts_int8":
        dump_acts_int8(args)
    elif args.mode == "eval_int8_modified":
        eval_int8_modified(args)
    elif args.mode == "analyze_variance":
        analyze_variance_distribution(args)
    elif args.mode == "collect_percentiles":
        collect_activation_percentiles(args)
    elif args.mode == "analyze_sensitivity":
        analyze_layer_sensitivity(args)
    elif args.mode == "train_supervised_triseg":
        train_supervised(args)
    elif args.mode == "train_ppo_triseg":
        train_ppo(args)
    elif args.mode == "eval_ppo_int8":
        eval_int8_with_ppo_config(args, config_path=getattr(args, "config_path", None))
    elif args.mode == "analyze_coverage":
        from activation_tools import analyze_coverage_distribution
        analyze_coverage_distribution(args, config_path=getattr(args, "config_path", None))

if __name__ == "__main__":
    main()
