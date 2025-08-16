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
    """Convert PyTorch model directly to OpenVINO IR format"""
    
    print("🚀 Direct PyTorch to OpenVINO IR Converter")
    print("=" * 50)
    
    # Model configuration
    STATE_DIM = 21  # 12 joints + 8 commands + 1 intensity
    ACTION_DIM = 24  # 12 mid + 12 target angles
    MAX_ACTION = 1.0
    
    # Paths
    pth_path = "/Users/matthewthomasbeck/Downloads/blind_rl_model.pth"
    xml_path = "/Users/matthewthomasbeck/Downloads/blind_rl_model.xml"
    
    print(f"📁 Input model: {pth_path}")
    print(f"📤 Output model: {xml_path}")
    print(f"🎯 State dimension: {STATE_DIM}")
    print(f"🎯 Action dimension: {ACTION_DIM}")
    
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
        
        # Create Actor model
        print(f"🏗️  Creating Actor model...")
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
        
        # Create dummy input
        print(f"🔧 Creating dummy input...")
        dummy_input = torch.randn(1, STATE_DIM)
        
        # Test forward pass
        print(f"🧪 Testing forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ Forward pass successful: {output.shape}")
            print(f"📊 Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Convert directly to OpenVINO IR
        print(f"🔄 Converting to OpenVINO IR format...")
        try:
            import openvino.runtime as ov
            from openvino.runtime import Core
            
            # Try different methods for PyTorch to OpenVINO conversion
            print(f"   🔍 Attempting PyTorch to OpenVINO conversion...")
            
            # Method 1: Try using openvino.tools (newer versions)
            try:
                from openvino.tools import mo
                ov_model = mo.convert_model(model, example_input=dummy_input)
                print(f"   ✅ Used openvino.tools.mo.convert_model")
            except ImportError:
                # Method 2: Try using openvino.runtime (older versions)
                try:
                    ov_model = ov.convert_model(model, example_input=dummy_input)
                    print(f"   ✅ Used openvino.runtime.convert_model")
                except AttributeError:
                    # Method 3: Try using openvino directly
                    try:
                        import openvino as ov_main
                        ov_model = ov_main.convert_model(model, example_input=dummy_input)
                        print(f"   ✅ Used openvino.convert_model")
                    except AttributeError:
                        # Method 4: Use the legacy approach with ONNX as intermediate
                        print(f"   ⚠️  Direct conversion not available, using ONNX intermediate...")
                        
                        # Create temporary ONNX file
                        temp_onnx = "/tmp/temp_model.onnx"
                        torch.onnx.export(
                            model,
                            dummy_input,
                            temp_onnx,
                            export_params=True,
                            opset_version=11,
                            do_constant_folding=True,
                            input_names=["state"],
                            output_names=["action"]
                        )
                        
                        # Convert ONNX to OpenVINO
                        core = Core()
                        ov_model = core.read_model(temp_onnx)
                        
                        # Clean up temp file
                        if os.path.exists(temp_onnx):
                            os.remove(temp_onnx)
                        
                        print(f"   ✅ Used ONNX intermediate conversion")
            
            print(f"✅ PyTorch to OpenVINO conversion successful")
            
            # Save as OpenVINO IR format (.xml + .bin)
            ov.save_model(ov_model, xml_path)
            print(f"✅ OpenVINO IR model saved: {xml_path}")
            
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
        print(f"🔧 Ready for Intel NCS2 deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_pth_to_openvino()
    sys.exit(0 if success else 1) 
