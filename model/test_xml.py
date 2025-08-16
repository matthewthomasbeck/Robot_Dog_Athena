#!/usr/bin/env python3
"""
Test OpenVINO IR Models for Robot Dog
Tests OpenVINO IR models (.xml + .bin) for NCS2 compatibility

Author: Matthew Thomas Beck
Date: 2024
"""

import numpy as np
import os

def test_openvino_ir_model():
    """Test the OpenVINO IR model for NCS2 compatibility"""
    
    print("🧪 Testing OpenVINO IR Model for NCS2")
    print("=" * 50)
    
    # Test the OpenVINO IR model
    xml_path = "/Users/matthewthomasbeck/Downloads/blind_rl_model_openvino.xml"
    bin_path = xml_path.replace('.xml', '.bin')
    
    print(f"📁 XML file: {xml_path}")
    print(f"📁 BIN file: {bin_path}")
    
    # Check if both files exist
    if not os.path.exists(xml_path):
        print(f"❌ XML file not found: {xml_path}")
        return False
        
    if not os.path.exists(bin_path):
        print(f"❌ BIN file not found: {bin_path}")
        return False
    
    try:
        # Try to load with OpenVINO
        from openvino.runtime import Core
        
        print(f"📥 Loading OpenVINO IR model...")
        core = Core()
        
        # Check available devices
        devices = core.available_devices
        print(f"🔍 Available devices: {devices}")
        
        # Load the IR model
        model = core.read_model(xml_path)
        print(f"✅ IR model loaded successfully")
        
        # Get input and output info
        input_info = model.input(0)
        output_info = model.output(0)
        
        try:
            input_shape = input_info.shape
            print(f"   📊 Input shape: {input_shape}")
        except:
            print(f"   📊 Input shape: Dynamic or unknown")
        
        try:
            output_shape = output_info.shape
            print(f"   📊 Output shape: {output_shape}")
        except:
            print(f"   📊 Output shape: Dynamic or unknown")
        
        # Try CPU compilation first (for testing)
        print(f"🔄 Testing CPU compilation...")
        try:
            compiled_model = core.compile_model(model, "CPU")
            print(f"✅ Model compiled for CPU")
            
            # Test with dummy input
            dummy_input = np.random.randn(1, 21).astype(np.float32)
            print(f"🧪 Testing with dummy input: {dummy_input.shape}")
            
            # Run inference
            result = compiled_model([dummy_input])
            print(f"✅ CPU inference successful!")
            print(f"   📊 Output shape: {result.shape}")
            print(f"   📊 Output range: [{result.min():.3f}, {result.max():.3f}]")
            
            # Test with realistic input (all zeros)
            zero_input = np.zeros((1, 21), dtype=np.float32)
            zero_result = compiled_model([zero_input])
            print(f"✅ Zero input test successful!")
            print(f"   📊 Zero input output range: [{zero_result.min():.3f}, {zero_result.max():.3f}]")
            
        except Exception as e:
            print(f"⚠️  CPU compilation failed: {e}")
            print(f"   💡 This might be expected on ARM devices")
        
        # Test with NCS2 compilation (the main goal)
        print(f"🔄 Testing NCS2 compilation...")
        try:
            ncs2_model = core.compile_model(model, "MYRIAD")
            print(f"✅ Model compiled for NCS2 (MYRIAD)")
            
            # Test NCS2 inference
            ncs2_result = ncs2_model([dummy_input])
            print(f"✅ NCS2 inference successful!")
            print(f"   📊 NCS2 output shape: {ncs2_result.shape}")
            print(f"   📊 NCS2 output range: [{ncs2_result.min():.3f}, {ncs2_result.max():.3f}]")
            
            # Test with realistic input on NCS2
            ncs2_zero_result = ncs2_model([zero_input])
            print(f"✅ NCS2 zero input test successful!")
            print(f"   📊 NCS2 zero input output range: [{ncs2_zero_result.min():.3f}, {ncs2_zero_result.max():.3f}]")
            
        except Exception as e:
            print(f"❌ NCS2 compilation failed: {e}")
            print(f"   💡 This is critical - NCS2 is the target device")
            return False
        
        print(f"\n🎉 OpenVINO IR model test completed successfully!")
        print(f"📁 Model files: {xml_path} + {bin_path}")
        print(f"🔧 Ready for NCS2 deployment!")
        
        return True
        
    except ImportError:
        print(f"❌ OpenVINO not available")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to test the OpenVINO IR model"""
    
    print("🚀 Robot Dog OpenVINO IR Model Tester")
    print("=" * 40)
    
    # Test the OpenVINO IR model
    if not test_openvino_ir_model():
        print(f"❌ OpenVINO IR model testing failed")
        return False
    
    print(f"\n🎉 All tests completed successfully!")
    print(f"   💡 Your model is ready for NCS2 deployment!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 