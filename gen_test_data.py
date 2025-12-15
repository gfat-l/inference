import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from model_alexnet_tap_quant import AlexNetTapQuant
from torch.ao.quantization import prepare_qat, get_default_qat_qconfig, convert

# CIFAR-10 Mean and Std
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def gen_data(args):
    # 1. Prepare Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    # Download/Load one image from CIFAR10 test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    img, label = testset[0] # Get first image
    
    # Save normalized float input
    # Shape: [3, 32, 32]
    input_float = img.numpy()
    np.savetxt(os.path.join(args.out_dir, "input_float.txt"), input_float.flatten(), fmt='%.8f')
    print(f"[Saved] input_float.txt (Label: {label})")

    # 2. Load Model to get Quantization Params
    # We need the input scale/zp from the converted model to generate reference int8 input
    # or just to verify.
    
    # Rebuild model structure
    model = AlexNetTapQuant(num_classes=10)
    model.eval()
    model.fuse_model()
    
    # Prepare QAT
    torch.backends.quantized.engine = args.backend
    model.qconfig = get_default_qat_qconfig(args.backend)
    model.train()
    prepare_qat(model, inplace=True)
    
    # Load weights
    if os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state)
    else:
        print(f"[Warning] Checkpoint {args.ckpt} not found. Using random weights for demo.")

    model.eval()
    qmodel = convert(model, inplace=False)
    
    # 3. Run Inference (Reference)
    with torch.no_grad():
        # Add batch dimension
        input_tensor = img.unsqueeze(0) 
        output = qmodel(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        
    # Save output logits
    np.savetxt(os.path.join(args.out_dir, "output_logits_ref.txt"), output.numpy().flatten(), fmt='%.8f')
    print(f"[Saved] output_logits_ref.txt")
    
    print("\nReference Output Probabilities:")
    for i, p in enumerate(probs[0]):
        print(f"Class {i}: {p:.4f}")
        
    # 4. Generate Reference INT8 Input (Optional, for debugging C++ quantization)
    # The C++ code does this internally: q = (x / scale) + zp
    # We can check what the first layer expects.
    # The input to conv1 comes from self.quant, so we use qmodel.quant's params
    input_scale = qmodel.quant.scale
    input_zp = qmodel.quant.zero_point
    print(f"\nModel Input Scale: {input_scale}, ZP: {input_zp}")
    
    input_int8 = torch.quantize_per_tensor(input_tensor, float(input_scale), int(input_zp), torch.quint8)
    # Extract int_repr
    input_int8_val = input_int8.int_repr().numpy().astype(np.uint8)
    np.savetxt(os.path.join(args.out_dir, "input_int8_ref.txt"), input_int8_val.flatten(), fmt='%d')
    print(f"[Saved] input_int8_ref.txt (for debug comparison)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="cpp_file/alexnet_8bit_export")
    parser.add_argument("--ckpt", type=str, default="outputs/alexnet_qat_preconvert.pth")
    parser.add_argument("--backend", type=str, default="fbgemm")
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    gen_data(args)
