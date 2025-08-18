#!/usr/bin/env python3
"""
Complete Pipeline Test for Robot Dog TD3 Model
Tests the entire pipeline: PyTorch -> OpenVINO IR -> Inference

Author: Matthew Thomas Beck
Date: 2024
"""

import torch
import numpy as np
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

def test_complete_pipeline():
    """Test the complete pipeline from PyTorch to inference"""
    
    print("🧪 Complete Pipeline Test for Robot Dog TD3 Model")
    print("=" * 60)
    
    # Model configuration - MUST MATCH TRAINING EXACTLY
    STATE_DIM = 19  # 12 joints + 6 commands + 1 intensity
    ACTION_DIM = 24  # 4 legs × 2 angles × 3 joints = 24
    MAX_ACTION = 1.0
    
    # Paths
    pth_path = "/Users/matthewthomasbeck/Downloads/blind_rl_model.pth"
    xml_path = "/Users/matthewthomasbeck/Downloads/blind_rl_model.xml"
    
    print(f"📁 PyTorch model: {pth_path}")
    print(f"📁 OpenVINO IR model: {xml_path}")
    print(f"🎯 State dimension: {STATE_DIM}")
    print(f"🎯 Action dimension: {ACTION_DIM}")
    
    # Step 1: Test PyTorch model
    print(f"\n🔍 Step 1: Testing PyTorch model...")
    if not test_pytorch_model(pth_path, STATE_DIM, ACTION_DIM):
        print(f"❌ PyTorch model test failed")
        return False
    
    # Step 2: Test OpenVINO IR model
    print(f"\n🔍 Step 2: Testing OpenVINO IR model...")
    if not test_openvino_model(xml_path):
        print(f"❌ OpenVINO IR model test failed")
        return False
    
    # Step 3: Test inference pipeline
    print(f"\n🔍 Step 3: Testing inference pipeline...")
    if not test_inference_pipeline(xml_path):
        print(f"❌ Inference pipeline test failed")
        return False
    
    print(f"\n🎉 Complete pipeline test completed successfully!")
    print(f"✅ PyTorch model: Working")
    print(f"✅ OpenVINO IR model: Working")
    print(f"✅ Inference pipeline: Working")
    print(f"🔧 Ready for deployment!")
    
    return True

def test_pytorch_model(pth_path, state_dim, action_dim):
    """Test the PyTorch model directly"""
    
    if not os.path.exists(pth_path):
        print(f"❌ PyTorch model not found: {pth_path}")
        return False
    
    try:
        # Load checkpoint
        checkpoint = torch.load(pth_path, map_location='cpu')
        
        # Create Actor model
        MAX_ACTION = 1.0  # Add the missing constant
        model = Actor(state_dim, action_dim, MAX_ACTION)
        
        # Load weights
        if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['actor_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Set to evaluation mode
        model.eval()
        
        # Test with realistic input (matching training data format)
        print(f"   🧪 Testing with realistic input...")
        
        # Create realistic state vector: [12 joint angles, 6 commands, 1 intensity]
        # Joint angles: normalized to [-1, 1] range
        joint_angles = np.random.uniform(-1.0, 1.0, 12).astype(np.float32)
        
        # Commands: 6D one-hot encoding
        commands = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 'w' command
        
        # Intensity: normalized to [-1.0, 1.0] range
        intensity = np.array([0.0], dtype=np.float32)  # Intensity 5 (neutral)
        
        # Combine into state vector
        state = np.concatenate([joint_angles, commands, intensity])
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        
        print(f"   📊 Input state shape: {state_tensor.shape}")
        print(f"   📊 Input state range: [{state_tensor.min().item():.3f}, {state_tensor.max().item():.3f}]")
        
        # Run inference
        with torch.no_grad():
            output = model(state_tensor)
            print(f"   ✅ Output shape: {output.shape}")
            print(f"   📊 Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
            # Verify output shape
            if output.shape != (1, action_dim):
                print(f"   ❌ Output shape mismatch: expected (1, {action_dim}), got {output.shape}")
                return False
        
        print(f"   ✅ PyTorch model test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ PyTorch model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_openvino_model(xml_path):
    """Test the OpenVINO IR model"""
    
    if not os.path.exists(xml_path):
        print(f"❌ OpenVINO IR model not found: {xml_path}")
        return False
    
    try:
        # Try to load with OpenVINO
        from openvino.runtime import Core
        
        print(f"   📥 Loading OpenVINO IR model...")
        core = Core()
        
        # Load the IR model
        model = core.read_model(xml_path)
        print(f"   ✅ IR model loaded successfully")
        
        # Get input and output info
        input_info = model.input(0)
        output_info = model.output(0)
        
        print(f"   📊 Input name: {input_info.get_any_name()}")
        print(f"   📊 Output name: {output_info.get_any_name()}")
        
        # Check if shapes are static by looking for dynamic symbols
        try:
            input_shape = input_info.shape
            output_shape = output_info.shape
            print(f"   📊 Input shape: {input_shape}")
            print(f"   📊 Output shape: {output_shape}")
            
            # Verify shapes are static by checking for dynamic symbols
            input_is_static = all(dim != -1 and '?' not in str(dim) for dim in input_shape)
            output_is_static = all(dim != -1 and '?' not in str(dim) for dim in output_shape)
            
            if input_is_static and output_is_static:
                print(f"   ✅ Both input and output shapes are static!")
            else:
                print(f"   ❌ Shapes are still dynamic - conversion failed")
                return False
                
            # Verify dimensions
            if input_shape != (1, 19):
                print(f"   ❌ Input shape mismatch: expected (1, 19), got {input_shape}")
                return False
                
            if output_shape != (1, 24):
                print(f"   ❌ Output shape mismatch: expected (1, 24), got {output_shape}")
                return False
                
        except Exception as e:
            print(f"   ❌ Could not verify shapes: {e}")
            return False
        
        print(f"   ✅ OpenVINO IR model test passed")
        return True
        
    except ImportError:
        print(f"   ❌ OpenVINO not available")
        return False
    except Exception as e:
        print(f"   ❌ OpenVINO IR model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_pipeline(xml_path):
    """Test the complete inference pipeline"""
    
    try:
        # Try to load with OpenVINO
        from openvino.runtime import Core
        
        print(f"   📥 Loading OpenVINO IR model for inference...")
        core = Core()
        model = core.read_model(xml_path)
        
        # Compile for CPU (for testing)
        compiled_model = core.compile_model(model, "CPU")
        print(f"   ✅ Model compiled for CPU")
        
        # Test with realistic input (matching training data format)
        print(f"   🧪 Testing inference with realistic input...")
        
        # Create realistic state vector: [12 joint angles, 6 commands, 1 intensity]
        # Joint angles: normalized to [-1, 1] range
        joint_angles = np.random.uniform(-1.0, 1.0, 12).astype(np.float32)
        
        # Commands: 6D one-hot encoding
        commands = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 'w' command
        
        # Intensity: normalized to [-1.0, 1.0] range
        intensity = np.array([0.0], dtype=np.float32)  # Intensity 5 (neutral)
        
        # Combine into state vector
        input_vec = np.concatenate([joint_angles, commands, intensity])
        input_vec = input_vec.reshape(1, -1)
        
        print(f"   📊 Input shape: {input_vec.shape}")
        print(f"   📊 Input range: [{input_vec.min():.3f}, {input_vec.max():.3f}]")
        
        # Run inference
        result = compiled_model([input_vec])
        print(f"   ✅ Inference successful!")
        
        # Just print the raw result to see what we're getting
        print(f"   📊 Raw result type: {type(result)}")
        print(f"   📊 Raw result: {result}")
        
        # Try to access the result in different ways
        if isinstance(result, dict):
            print(f"   📊 Result is dict with keys: {list(result.keys())}")
            for key, value in result.items():
                print(f"   📊 Key '{key}': type={type(value)}, value={value}")
                if hasattr(value, 'numpy'):
                    print(f"   📊 Key '{key}' numpy: {value.numpy()}")
        elif isinstance(result, list):
            print(f"   📊 Result is list with {len(result)} elements")
            for i, item in enumerate(result):
                print(f"   📊 Element {i}: type={type(item)}, value={item}")
        else:
            print(f"   📊 Result is direct: type={type(result)}")
            print(f"   📊 Result value: {result}")
        
        print(f"   ✅ Inference pipeline test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Inference pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    sys.exit(0 if success else 1) 