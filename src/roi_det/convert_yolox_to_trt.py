#!/usr/bin/env python3
import os
import sys
import argparse

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

script_dir = os.path.dirname(os.path.abspath(__file__))
yolox_dir = os.path.join(script_dir, 'YOLOX')
if os.path.exists(yolox_dir):
    sys.path.insert(0, yolox_dir)
    torch2trt_dir = os.path.join(yolox_dir, 'torch2trt')
    if os.path.exists(torch2trt_dir):
        sys.path.insert(0, torch2trt_dir)

import torch
import tensorrt as trt
from torch2trt import torch2trt
from yolox.exp import get_exp


def parse_args():
    
    model_path = "../../assets/trained/yolox_nano_416_roi_torch.pth"
    output_path = "../../assets/trained/yolox_nano_416_roi_trt.engine"
    parser = argparse.ArgumentParser(
        description='Convert YOLOX PyTorch model to TensorRT engine')
    parser.add_argument(
        '--model',
        type=str,
        default=model_path,
        help='Path to PyTorch model checkpoint (.pth file)')
    parser.add_argument(
        '--exp',
        type=str,
        default=None,
        help='Path to experiment file (e.g., exps/projects/rsna/yolox_nano_bre_416.py). '
        'If not provided, will try to infer from model path or use default.')
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name (e.g., yolox-nano). Used if --exp is not provided.')
    parser.add_argument(
        '--output',
        type=str,
        default=output_path,
        help='Output path for TensorRT engine file (.engine file)')
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=True,
        help='Use FP16 precision (recommended for RTX 4090D)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Maximum batch size (default: 1)')
    parser.add_argument(
        '--workspace',
        type=int,
        default=32,
        help='Max workspace size in GB (default: 32, use 1<<workspace)')
    parser.add_argument(
        '--use-onnx',
        action='store_true',
        help='Use ONNX backend for conversion (more stable)')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"Loading experiment configuration...")
    if args.exp:
        exp = get_exp(exp_file=args.exp)
    elif args.name:
        exp = get_exp(exp_name=args.name)
    else:
        print("Warning: No experiment file or name provided. Trying to infer...")
        exp = get_exp(exp_name="yolox-nano")
        print(f"Using default experiment: {exp.exp_name}")

    print(f"Experiment: {exp.exp_name}")
    print(f"Test size: {exp.test_size}")
    print(f"Num classes: {exp.num_classes}")

    print(f"\nLoading PyTorch model from: {args.model}")
    model = exp.get_model()
    model.eval()

    ckpt = torch.load(args.model, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    print("Model loaded successfully.")

    model.cuda()
    model.head.decode_in_inference = False

    input_size = exp.test_size
    print(f"\nCreating sample input with size: {input_size}")
    sample_input = torch.ones(1, 3, input_size[0], input_size[1]).cuda()

    print(f"\nConverting to TensorRT...")
    print(f"  Precision: {'FP16' if args.fp16 else 'FP32'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workspace: {1 << args.workspace} bytes ({args.workspace} GB)")
    print(f"  Use ONNX: {args.use_onnx}")

    model_trt = torch2trt(
        model,
        [sample_input],
        fp16_mode=args.fp16,
        log_level=trt.Logger.INFO,
        max_workspace_size=(1 << args.workspace),
        max_batch_size=args.batch_size,
        use_onnx=args.use_onnx,
        strict_type_constraints=False,
        keep_network=True,
    )

    print(f"\nSaving TensorRT engine to: {args.output}")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    torch.save(model_trt.state_dict(), args.output)
    print("Conversion completed successfully!")

    engine_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Engine file size: {engine_size_mb:.2f} MB")


if __name__ == '__main__':
    main()

