#!/usr/bin/env python3
"""
Quick Export Script for Robot Dog TD3 Model
Exports trained .pth models to ONNX and OpenVINO formats for Intel NCS2

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
# We need to import it this way to avoid the movement module dependency
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        import torch.nn as nn
        import torch.nn.functional as F
        
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        import torch.nn.functional as F
        a = F.relu(self.layer_1(state))
        a = F.relu(self.layer_2(a))
        a = torch.tanh(self.layer_3(a)) * self.max_action
        return a

def quick_export():
    """Quick export of the trained model to ONNX and OpenVINO"""
    
    print("🚀 Quick Export for Robot Dog TD3 Model")
    print("=" * 50)
    
    # Model configuration
    STATE_DIM = 21  # 12 joints + 8 commands + 1 intensity
    ACTION_DIM = 24  # 12 mid + 12 target angles
    MAX_ACTION = 1.0
    
    # Paths
    pth_path = "/home/matthewthomasbeck/Projects/Robot_Dog/model/blind_rl_model.pth"
    onnx_path = "/home/matthewthomasbeck/Projects/Robot_Dog/model/blind_rl_model_new.onnx"
    
    print(f"📁 Input model: {pth_path}")
    print(f"🎯 State dimension: {STATE_DIM}")
    print(f"🎯 Action dimension: {ACTION_DIM}")
    
    # Check if input file exists
    if not os.path.exists(pth_path):
        print(f"❌ Model file not found: {pth_path}")
        return False
    
    try:
        # Load checkpoint
        print(f"📥 Loading checkpoint...")
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
        
        # Export to ONNX
        print(f"🚀 Exporting to ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,  # Good for NCS2
            do_constant_folding=True,
            input_names=["state"],
            output_names=["action"],
            dynamic_axes={
                "state": {0: "batch_size"},
                "action": {0: "batch_size"}
            }
        )
        
        # Verify ONNX file
        if os.path.exists(onnx_path):
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"✅ ONNX export successful! File size: {file_size:.1f} MB")
            
            # Try to validate ONNX
            try:
                import onnx
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print(f"✅ ONNX validation passed")
            except ImportError:
                print(f"⚠️  ONNX validation skipped (onnx package not available)")
            except Exception as e:
                print(f"⚠️  ONNX validation warning: {e}")
        else:
            print(f"❌ ONNX file was not created")
            return False
        
        # Try OpenVINO conversion
        print(f"🔄 Attempting OpenVINO conversion...")
        try:
            from openvino.runtime import Core
            import openvino.runtime as ov
            
            # Check OpenVINO version
            try:
                import openvino
                print(f"   📊 OpenVINO version: {openvino.__version__}")
            except:
                print(f"   📊 OpenVINO version: Unknown")
            
            # Method 1: ONNX → OpenVINO
            print(f"   📥 Converting ONNX to OpenVINO...")
            core = Core()
            ov_model = core.read_model(onnx_path)
            
            # Save as OpenVINO format using the correct method
            ov_path = onnx_path.replace('.onnx', '_openvino.xml')
            
            # Try different methods for saving OpenVINO models
            try:
                # Method 1: Use ov.save_model (newer versions)
                ov.save_model(ov_model, ov_path)
                print(f"   ✅ Used ov.save_model method")
            except AttributeError:
                try:
                    # Method 2: Use core.save_model (older versions)
                    core.save_model(ov_model, ov_path)
                    print(f"   ✅ Used core.save_model method")
                except AttributeError:
                    # Method 3: Use serialize method
                    ov_model.serialize(ov_path)
                    print(f"   ✅ Used serialize method")
            
            if os.path.exists(ov_path):
                print(f"✅ OpenVINO conversion successful: {ov_path}")
                # Also check if .bin file was created
                bin_path = ov_path.replace('.xml', '.bin')
                if os.path.exists(bin_path):
                    bin_size = os.path.getsize(bin_path) / (1024 * 1024)
                    print(f"✅ OpenVINO binary file created: {bin_path} ({bin_size:.1f} MB)")
                else:
                    print(f"⚠️  OpenVINO binary file not found")
            else:
                print(f"⚠️  OpenVINO conversion failed")
                
        except ImportError:
            print(f"⚠️  OpenVINO not available, skipping conversion")
        except Exception as e:
            print(f"⚠️  OpenVINO conversion failed: {e}")
            print(f"   💡 ONNX file is still valid and can be used directly with OpenVINO")
        
        print(f"\n🎉 Export completed successfully!")
        print(f"📁 ONNX file: {onnx_path}")
        print(f"🔧 Ready for Intel NCS2 / OpenVINO deployment")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_export()
    sys.exit(0 if success else 1) 
