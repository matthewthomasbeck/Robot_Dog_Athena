#!/usr/bin/env python3
"""
Model testing and verification module for Robot Dog TD3 training.
Run this to check if your saved models are actually saving meaningful data.

Usage:
    ./python.sh ~/Projects/Robot_Dog/test_models.py
"""

import os
import torch


def verify_saved_model(filepath):
    """
    Verify that a saved model actually contains the expected TD3 data.
    This will help confirm that models are being saved properly.
    """
    if not os.path.exists(filepath):
        print(f"❌ Model file not found: {filepath}")
        return False
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(filepath, map_location='cpu')
        
        print(f"✅ Model verification for: {filepath}")
        print(f"   File size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
        print(f"   Keys found: {list(checkpoint.keys())}")
        
        # Check for required TD3 components
        required_keys = [
            'actor_state_dict',
            'critic_1_state_dict', 
            'critic_2_state_dict',
            'actor_target_state_dict',
            'critic_1_target_state_dict',
            'critic_2_target_state_dict',
            'actor_optimizer_state_dict',
            'critic_1_optimizer_state_dict',
            'critic_2_optimizer_state_dict'
        ]
        
        missing_keys = []
        for key in required_keys:
            if key in checkpoint:
                # Check if the state dict has actual parameters
                state_dict = checkpoint[key]
                if isinstance(state_dict, dict) and len(state_dict) > 0:
                    print(f"   ✅ {key}: {len(state_dict)} layers")
                else:
                    print(f"   ⚠️  {key}: Empty or invalid")
                    missing_keys.append(key)
            else:
                print(f"   ❌ {key}: Missing")
                missing_keys.append(key)
        
        # Check training metadata
        if 'episode_counter' in checkpoint:
            print(f"   📊 Episode: {checkpoint['episode_counter']}")
        if 'total_steps' in checkpoint:
            print(f"   📊 Total Steps: {checkpoint['total_steps']}")
        if 'episode_reward' in checkpoint:
            print(f"   📊 Episode Reward: {checkpoint['episode_reward']:.4f}")
        
        if missing_keys:
            print(f"❌ Model is INCOMPLETE - missing {len(missing_keys)} components")
            return False
        else:
            print(f"✅ Model is COMPLETE and contains all TD3 components")
            return True
            
    except Exception as e:
        print(f"❌ Failed to verify model: {e}")
        return False


def test_model_saving():
    """
    Test function to verify that model saving is working correctly.
    Run this to check if your saved models are valid.
    """
    models_dir = "/home/matthewthomasbeck/Projects/Robot_Dog/model"
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    # Find all TD3 model files
    td3_files = [f for f in os.listdir(models_dir) if f.startswith('td3_episode_') and f.endswith('.pth')]
    
    if not td3_files:
        print("❌ No TD3 model files found in models directory")
        return
    
    print(f"🔍 Found {len(td3_files)} TD3 model files:")
    print("=" * 50)
    
    for filename in sorted(td3_files):
        filepath = os.path.join(models_dir, filename)
        print(f"\n📁 Testing: {filename}")
        print("-" * 30)
        verify_saved_model(filepath)
    
    print("\n" + "=" * 50)
    print("🎯 Model verification complete!")


def test_individual_model(episode_number):
    """
    Test a specific episode model by number.
    
    Args:
        episode_number (int): The episode number to test (e.g., 10, 20, 30)
    """
    models_dir = "/home/matthewthomasbeck/Projects/Robot_Dog/model"
    filename = f"td3_episode_{episode_number}_reward_0.00.pth"
    filepath = os.path.join(models_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"❌ Model file not found: {filepath}")
        return False
    
    print(f"🔍 Testing individual model: {filename}")
    print("=" * 50)
    return verify_saved_model(filepath)


def list_all_models():
    """
    List all available TD3 models with their details.
    """
    models_dir = "/home/matthewthomasbeck/Projects/Robot_Dog/model"
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return
    
    # Find all TD3 model files
    td3_files = [f for f in os.listdir(models_dir) if f.startswith('td3_episode_') and f.endswith('.pth')]
    
    if not td3_files:
        print("❌ No TD3 model files found in models directory")
        return
    
    print(f"📁 Available TD3 Models ({len(td3_files)} total):")
    print("=" * 50)
    
    for filename in sorted(td3_files):
        filepath = os.path.join(models_dir, filename)
        file_size = os.path.getsize(filepath) / (1024*1024)
        print(f"   📄 {filename} ({file_size:.1f} MB)")
    
    print("=" * 50)


if __name__ == "__main__":
    print("🔍 Robot Dog TD3 Model Testing Suite")
    print("=" * 50)
    
    try:
        # Test all models
        test_model_saving()
        
        print("\n" + "=" * 50)
        print("📋 Additional Testing Options:")
        print("   - test_individual_model(10)  # Test episode 10")
        print("   - test_individual_model(20)  # Test episode 20")
        print("   - list_all_models()          # List all available models")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
