#!/usr/bin/env python3
"""
ONNX Export Script for Robot Dog TD3 Model
Converts trained .pth models to .onnx format for Intel NCS2 inference

Author: Matthew Thomas Beck
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from pathlib import Path

# Define the exact Actor class from training (standalone to avoid import issues)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.layer_1(state))
        a = F.relu(self.layer_2(a))
        a = torch.tanh(self.layer_3(a)) * self.max_action
        return a

def export_model_to_onnx(
    pth_path: str,
    onnx_path: str = None,
    opset_version: int = 11,
    verbose: bool = True
):
    """
    Export a trained TD3 Actor model to ONNX format.
    
    Args:
        pth_path: Path to the trained .pth model file
        onnx_path: Output path for .onnx file (auto-generated if None)
        opset_version: ONNX opset version (11 recommended for NCS2)
        verbose: Whether to print detailed export information
    
    Returns:
        str: Path to the exported ONNX file
    """
    
    # Configuration from training code
    STATE_DIM = 21  # 12 joints + 8 commands + 1 intensity
    ACTION_DIM = 24  # 12 mid + 12 target angles
    MAX_ACTION = 1.0
    
    if verbose:
        print(f"🤖 Robot Dog TD3 ONNX Export")
        print(f"   📁 Input model: {pth_path}")
        print(f"   🎯 State dimension: {STATE_DIM}")
        print(f"   🎯 Action dimension: {ACTION_DIM}")
        print(f"   🎯 Max action: {MAX_ACTION}")
        print(f"   🔧 ONNX opset: {opset_version}")
    
    # Validate input file
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model file not found: {pth_path}")
    
    # Auto-generate output path if not provided
    if onnx_path is None:
        pth_stem = Path(pth_path).stem
        onnx_path = str(Path(pth_path).parent / f"{pth_stem}.onnx")
    
    if verbose:
        print(f"   📤 Output ONNX: {onnx_path}")
    
    # Set device (CPU for export)
    device = torch.device("cpu")
    
    # Create Actor model with exact architecture from training
    if verbose:
        print(f"   🏗️  Creating Actor model...")
    
    model = Actor(STATE_DIM, ACTION_DIM, MAX_ACTION).to(device)
    
    # Load trained weights
    if verbose:
        print(f"   📥 Loading trained weights...")
    
    try:
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
        
        # Check if this is a full TD3 checkpoint or just Actor weights
        if 'actor_state_dict' in checkpoint:
            # Full TD3 checkpoint
            model.load_state_dict(checkpoint['actor_state_dict'])
            if verbose:
                print(f"   ✅ Loaded Actor weights from TD3 checkpoint")
                print(f"   📊 Checkpoint info:")
                print(f"      - Episode: {checkpoint.get('episode_counter', 'N/A')}")
                print(f"      - Total steps: {checkpoint.get('total_steps', 'N/A')}")
                print(f"      - Episode reward: {checkpoint.get('episode_reward', 'N/A')}")
        else:
            # Direct Actor weights
            model.load_state_dict(checkpoint)
            if verbose:
                print(f"   ✅ Loaded Actor weights directly")
                
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input with correct shape
    if verbose:
        print(f"   🔧 Creating dummy input...")
    
    dummy_input = torch.randn(1, STATE_DIM, device=device)
    
    # Validate model forward pass
    if verbose:
        print(f"   🧪 Testing model forward pass...")
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
            expected_shape = (1, ACTION_DIM)
            if output.shape != expected_shape:
                raise ValueError(f"Model output shape {output.shape} doesn't match expected {expected_shape}")
            
            # Check output range (should be [-MAX_ACTION, MAX_ACTION] due to tanh)
            min_val = output.min().item()
            max_val = output.max().item()
            if verbose:
                print(f"   ✅ Forward pass successful")
                print(f"   📊 Output range: [{min_val:.3f}, {max_val:.3f}]")
                print(f"   📊 Expected range: [{-MAX_ACTION:.3f}, {MAX_ACTION:.3f}]")
                
    except Exception as e:
        raise RuntimeError(f"Model forward pass failed: {e}")
    
    # Export to ONNX
    if verbose:
        print(f"   🚀 Exporting to ONNX...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["state"],
            output_names=["action"],
            dynamic_axes={
                "state": {0: "batch_size"},
                "action": {0: "batch_size"}
            },
            verbose=verbose
        )
        
        if verbose:
            print(f"   ✅ ONNX export successful!")
            
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")
    
    # Validate exported file
    if verbose:
        print(f"   🔍 Validating exported file...")
    
    if not os.path.exists(onnx_path):
        raise RuntimeError(f"ONNX file was not created: {onnx_path}")
    
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
    if verbose:
        print(f"   📊 ONNX file size: {file_size:.1f} MB")
    
    # Test ONNX model loading (optional validation)
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        if verbose:
            print(f"   ✅ ONNX model validation passed")
    except ImportError:
        if verbose:
            print(f"   ⚠️  ONNX validation skipped (onnx package not available)")
    except Exception as e:
        if verbose:
            print(f"   ⚠️  ONNX validation warning: {e}")
    
    if verbose:
        print(f"\n🎉 Export completed successfully!")
        print(f"   📁 ONNX file: {onnx_path}")
        print(f"   🔧 Ready for Intel NCS2 / OpenVINO deployment")
    
    return onnx_path

def batch_export_models(
    models_dir: str,
    output_dir: str = None,
    opset_version: int = 11,
    verbose: bool = True
):
    """
    Batch export all .pth models in a directory to ONNX format.
    
    Args:
        models_dir: Directory containing .pth model files
        output_dir: Output directory for ONNX files (uses models_dir if None)
        opset_version: ONNX opset version
        verbose: Whether to print detailed information
    """
    
    if output_dir is None:
        output_dir = models_dir
    
    # Find all .pth files
    pth_files = list(Path(models_dir).glob("*.pth"))
    
    if not pth_files:
        print(f"❌ No .pth files found in {models_dir}")
        return
    
    print(f"🔍 Found {len(pth_files)} .pth files to export")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    successful_exports = 0
    failed_exports = 0
    
    for pth_file in pth_files:
        try:
            print(f"\n📁 Processing: {pth_file.name}")
            
            # Generate ONNX output path
            onnx_name = f"{pth_file.stem}.onnx"
            onnx_path = os.path.join(output_dir, onnx_name)
            
            # Export the model
            export_model_to_onnx(
                str(pth_file),
                onnx_path,
                opset_version,
                verbose=False  # Less verbose for batch processing
            )
            
            successful_exports += 1
            print(f"   ✅ Exported: {onnx_name}")
            
        except Exception as e:
            failed_exports += 1
            print(f"   ❌ Failed: {e}")
    
    # Summary
    print(f"\n📊 Batch Export Summary:")
    print(f"   ✅ Successful: {successful_exports}")
    print(f"   ❌ Failed: {failed_exports}")
    print(f"   📁 Output directory: {output_dir}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Export Robot Dog TD3 models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export single model
  python export_model.py model/td3_steps_300000_episode_464_reward_376.97.pth
  
  # Export with custom output path
  python export_model.py model/td3_steps_300000_episode_464_reward_376.97.pth -o model.onnx
  
  # Batch export all models in directory
  python export_model.py -b model/
  
  # Export with custom ONNX opset
  python export_model.py model.pth --opset 10
        """
    )
    
    parser.add_argument(
        "model_path",
        nargs="?",
        help="Path to .pth model file (required for single export)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output path for ONNX file (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "-b", "--batch",
        help="Batch export all .pth files in specified directory"
    )
    
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (default: 11, recommended for NCS2)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            # Batch export mode
            if not os.path.isdir(args.batch):
                print(f"❌ Directory not found: {args.batch}")
                return 1
            
            batch_export_models(args.batch, verbose=args.verbose)
            
        elif args.model_path:
            # Single export mode
            export_model_to_onnx(
                args.model_path,
                args.output,
                args.opset,
                args.verbose
            )
            
        else:
            parser.print_help()
            return 1
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
