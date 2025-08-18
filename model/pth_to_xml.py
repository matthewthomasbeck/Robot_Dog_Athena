#!/usr/bin/env python3
"""
Direct PyTorch to OpenVINO IR Converter for Robot Dog TD3 Model
Converts trained .pth models directly to OpenVINO IR format for Intel NCS2

Author: Matthew Thomas Beck
Date: 2024
"""

import torch
import torch.onnx
import os
import sys

# Add the parent directory to path to import training modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Actor class directly from the training module
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = torch.nn.Linear(state_dim, 400)
        self.layer_2 = torch.nn.Linear(400, 300)
        self.layer_3 = torch.nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.nn.functional.relu(self.layer_1(state))
        a = torch.nn.functional.relu(self.layer_2(a))
        a = torch.tanh(self.layer_3(a)) * self.max_action
        return a

def convert_pth_to_openvino():
    """Convert PyTorch model directly to OpenVINO IR format with static dimensions"""
    
    print("🚀 Direct PyTorch to OpenVINO IR Converter (Fixed)")
    print("=" * 55)
    
    # Model configuration - FIXED DIMENSIONS
    STATE_DIM = 19  # Fixed: 12 joints + 6 commands + 1 intensity
    ACTION_DIM = 24  # Fixed: 4 legs × 2 angles × 3 joints = 24
    MAX_ACTION = 1.0
    
    # Paths
    pth_path = "/Users/matthewthomasbeck/Downloads/blind_rl_model.pth"
    xml_path = "/Users/matthewthomasbeck/Downloads/blind_rl_model.xml"
    
    print(f"📁 Input model: {pth_path}")
    print(f"📤 Output model: {xml_path}")
    print(f"🎯 State dimension: {STATE_DIM} (FIXED)")
    print(f"🎯 Action dimension: {ACTION_DIM} (FIXED)")
    
    # Check if input file exists
    if not os.path.exists(pth_path):
        print(f"❌ Model file not found: {pth_path}")
        return False
    
    try:
        # Load checkpoint
        print(f"📥 Loading PyTorch checkpoint...")
        checkpoint = torch.load(pth_path, map_location='cpu')
        
        # Check checkpoint structure
        if isinstance(checkpoint, dict):
            print(f"✅ Checkpoint keys: {list(checkpoint.keys())}")
            if 'actor_state_dict' in checkpoint:
                print(f"✅ Found actor_state_dict with {len(checkpoint['actor_state_dict'])} layers")
            else:
                print(f"⚠️  No actor_state_dict found, assuming direct weights")
        else:
            print(f"✅ Direct weights checkpoint")
        
        # Create Actor model with FIXED dimensions
        print(f"🏗️  Creating Actor model with FIXED dimensions...")
        model = Actor(STATE_DIM, ACTION_DIM, MAX_ACTION)
        
        # Load weights
        if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['actor_state_dict'])
            print(f"✅ Loaded actor weights from checkpoint")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded weights directly")
        
        # Set to evaluation mode
        model.eval()
        
        # Create dummy input with FIXED shape
        print(f"🔧 Creating dummy input with FIXED shape...")
        dummy_input = torch.randn(1, STATE_DIM)  # Fixed: (1, 21)
        
        # Test forward pass
        print(f"🧪 Testing forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ Forward pass successful: {output.shape}")
            print(f"📊 Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
            # Verify output shape is correct
            if output.shape != (1, ACTION_DIM):
                print(f"❌ Output shape mismatch: expected (1, {ACTION_DIM}), got {output.shape}")
                return False
        
        # Convert to OpenVINO IR with EXPLICIT static shapes
        print(f"🔄 Converting to OpenVINO IR format with static shapes...")
        try:
            import openvino.runtime as ov
            from openvino.runtime import Core
            
            # Method 1: Use ONNX as intermediate with EXPLICIT shapes
            print(f"   🔍 Using ONNX intermediate with EXPLICIT shapes...")
            
            # Create temporary ONNX file with EXPLICIT input/output shapes
            temp_onnx = "/tmp/temp_model.onnx"
            
            # Export with EXPLICIT dynamic axes to force static shapes
            torch.onnx.export(
                model,
                dummy_input,
                temp_onnx,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["state"],
                output_names=["action"],
                dynamic_axes={
                    'state': {0: 'batch_size'},  # Only batch dimension can vary
                    'action': {0: 'batch_size'}   # Only batch dimension can vary
                }
            )
            
            print(f"   ✅ ONNX export successful with explicit shapes")
            
            # CRITICAL: Freeze the input shape to prevent dynamic dimensions
            print(f"   🔒 Freezing input shape to prevent dynamic dimensions...")
            
            # Use Model Optimizer (MO) to convert ONNX to OpenVINO IR with static shapes
            try:
                from openvino.tools import mo
                print(f"   🔧 Using Model Optimizer for conversion...")
                
                # Convert ONNX to OpenVINO IR with explicit input shapes
                ov_model = mo.convert_model(
                    temp_onnx,
                    input_shape=[1, STATE_DIM],  # Explicit static input shape
                    output_shape=[1, ACTION_DIM]  # Explicit static output shape
                )
                print(f"   ✅ Model Optimizer conversion successful with static shapes")
                
            except ImportError:
                # Fallback: try to use the core conversion
                print(f"   🔧 Model Optimizer not available, using core conversion...")
                core = Core()
                ov_model = core.read_model(temp_onnx)
                
                # Try to reshape the model to static dimensions
                try:
                    # Get the model and reshape it
                    ov_model.reshape({0: [1, STATE_DIM]})
                    print(f"   ✅ Model reshaped to static input shape")
                except Exception as reshape_error:
                    print(f"   ⚠️  Could not reshape model: {reshape_error}")
                    print(f"   💡 Continuing with original model...")
            
            print(f"   ✅ Model shapes processed")
            
            # Clean up temp file
            if os.path.exists(temp_onnx):
                os.remove(temp_onnx)
            
            print(f"✅ PyTorch to OpenVINO conversion successful with static shapes")
            
            # Save as OpenVINO IR format (.xml + .bin)
            ov.save_model(ov_model, xml_path)
            print(f"✅ OpenVINO IR model saved: {xml_path}")
            
            # Verify the saved model has static shapes
            print(f"🔍 Verifying saved model has static shapes...")
            saved_model = core.read_model(xml_path)
            
            try:
                input_shape = saved_model.input(0).shape
                output_shape = saved_model.output(0).shape
                print(f"   📊 Input shape: {input_shape} (should be [1, {STATE_DIM}])")
                print(f"   📊 Output shape: {output_shape} (should be [1, {ACTION_DIM}])")
                
                # Check if shapes are static by looking for dynamic symbols
                input_is_static = all(dim != -1 and '?' not in str(dim) for dim in input_shape)
                output_is_static = all(dim != -1 and '?' not in str(dim) for dim in output_shape)
                
                if input_is_static and output_is_static:
                    print(f"   ✅ Both input and output shapes are static!")
                else:
                    print(f"   ❌ Shapes are still dynamic - conversion failed")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Could not verify shapes: {e}")
                return False
            
            # Check if both files were created
            bin_path = xml_path.replace('.xml', '.bin')
            if os.path.exists(xml_path) and os.path.exists(bin_path):
                xml_size = os.path.getsize(xml_path) / 1024  # KB
                bin_size = os.path.getsize(bin_path) / (1024 * 1024)  # MB
                print(f"✅ OpenVINO IR files created successfully:")
                print(f"   📄 XML file: {xml_path} ({xml_size:.1f} KB)")
                print(f"   📦 BIN file: {bin_path} ({bin_size:.1f} MB)")
            else:
                print(f"⚠️  Some OpenVINO IR files missing")
                return False
                
        except ImportError:
            print(f"❌ OpenVINO not available")
            return False
        except Exception as e:
            print(f"❌ OpenVINO conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n🎉 Direct PyTorch to OpenVINO IR conversion completed!")
        print(f"📁 OpenVINO IR files: {xml_path} + {bin_path}")
        print(f"🔧 Model now has STATIC shapes - ready for Intel NCS2 deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_pth_to_openvino()
    sys.exit(0 if success else 1) 
