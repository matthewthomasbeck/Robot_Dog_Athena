#!/usr/bin/env python3
"""
Test OpenVINO IR Models for Robot Dog on Intel Hardware
Tests OpenVINO IR models (.xml + .bin) on Intel chips to diagnose dynamic dimension issues

Author: Matthew Thomas Beck
Date: 2024
"""

import numpy as np
import os
import platform

def test_openvino_ir_model_intel():
    """Test the OpenVINO IR model on Intel hardware"""
    
    print("🧪 Testing OpenVINO IR Model on Intel Hardware")
    print("=" * 55)
    
    # Check system architecture
    print(f"🖥️  System: {platform.system()}")
    print(f"🏗️  Architecture: {platform.machine()}")
    print(f"💻 Processor: {platform.processor()}")
    
    # Test the OpenVINO IR model
    xml_path = "/Users/matthewthomasbeck/Downloads/blind_rl_model.xml"
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
        
        print(f"📊 Input name: {input_info.get_any_name()}")
        print(f"📊 Output name: {output_info.get_any_name()}")
        
        try:
            input_shape = input_info.shape
            print(f"📊 Input shape: {input_shape}")
            print(f"📊 Input type: {input_info.get_element_type()}")
        except Exception as e:
            print(f"📊 Input shape: Dynamic or unknown - {e}")
        
        try:
            output_shape = output_info.shape
            print(f"📊 Output shape: {output_shape}")
            print(f"📊 Output type: {output_info.get_element_type()}")
        except Exception as e:
            print(f"📊 Output shape: Dynamic or unknown - {e}")
        
        # Try CPU compilation on Intel
        print(f"🔄 Testing Intel CPU compilation...")
        try:
            compiled_model = core.compile_model(model, "CPU")
            print(f"✅ Model compiled for Intel CPU")
            
            # Test with dummy input - try to infer shape from model
            try:
                # Get the actual input shape if possible
                if hasattr(input_info, 'get_shape') and input_info.get_shape().is_static:
                    input_shape = input_info.get_shape()
                    print(f"📊 Using static input shape: {input_shape}")
                else:
                    # Try common shapes for RL models
                    possible_shapes = [(1, 21), (1, 20), (1, 22), (1, 24), (1, 18)]
                    input_shape = None
                    
                    for shape in possible_shapes:
                        try:
                            dummy_input = np.random.randn(*shape).astype(np.float32)
                            # Test if this shape works
                            test_result = compiled_model([dummy_input])
                            input_shape = shape
                            print(f"📊 Discovered working input shape: {input_shape}")
                            break
                        except:
                            continue
                    
                    if input_shape is None:
                        print(f"⚠️  Could not determine working input shape, using default (1, 21)")
                        input_shape = (1, 21)
                
                dummy_input = np.random.randn(*input_shape).astype(np.float32)
                print(f"🧪 Testing with input: {dummy_input.shape}")
                
                # Run inference
                result = compiled_model([dummy_input])
                print(f"✅ Intel CPU inference successful!")
                print(f"   📊 Output shape: {result.shape}")
                print(f"   📊 Output range: [{result.min():.3f}, {result.max():.3f}]")
                print(f"   📊 Output sample: {result.flatten()[:5]}")
                
                # Test with realistic input (all zeros)
                zero_input = np.zeros(input_shape, dtype=np.float32)
                zero_result = compiled_model([zero_input])
                print(f"✅ Zero input test successful!")
                print(f"   📊 Zero input output range: [{zero_result.min():.3f}, {zero_result.max():.3f}]")
                print(f"   📊 Zero input output sample: {zero_result.flatten()[:5]}")
                
                # Test with ones input
                ones_input = np.ones(input_shape, dtype=np.float32)
                ones_result = compiled_model([ones_input])
                print(f"✅ Ones input test successful!")
                print(f"   📊 Ones input output range: [{ones_result.min():.3f}, {ones_result.max():.3f}]")
                print(f"   📊 Ones input output sample: {ones_result.flatten()[:5]}")
                
            except Exception as e:
                print(f"❌ Input shape discovery failed: {e}")
                return False
            
        except Exception as e:
            print(f"❌ Intel CPU compilation failed: {e}")
            print(f"   💡 This indicates a model conversion issue")
            return False
        
        # Try to get more model details
        print(f"\n🔍 Model Analysis:")
        try:
            # Try to get layer information
            layers = model.get_ops()
            print(f"📊 Number of layers: {len(layers)}")
            
            # Show first few layers
            for i, layer in enumerate(layers[:5]):
                print(f"   Layer {i}: {layer.get_any_name()} - {layer.get_type_name()}")
            
            if len(layers) > 5:
                print(f"   ... and {len(layers) - 5} more layers")
                
        except Exception as e:
            print(f"📊 Layer info unavailable: {e}")
        
        print(f"\n🎉 Intel CPU test completed successfully!")
        print(f"📁 Model files: {xml_path} + {bin_path}")
        print(f"🔧 Model works on Intel CPU - issue is ARM-specific")
        
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
    """Main function to test the OpenVINO IR model on Intel hardware"""
    
    print("🚀 Robot Dog OpenVINO IR Model Intel Tester")
    print("=" * 45)
    
    # Test the OpenVINO IR model on Intel
    if not test_openvino_ir_model_intel():
        print(f"❌ Intel CPU testing failed")
        print(f"   💡 This indicates a model conversion issue")
        return False
    
    print(f"\n🎉 Intel CPU test completed successfully!")
    print(f"   💡 Your model works on Intel - the issue is ARM-specific")
    print(f"   💡 You'll need to rearchitect your model to avoid dynamic dimensions")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 