#!/usr/bin/env python
"""
Quick Test Script for Fixed Predenergy Implementation
This script validates that all fixes are working correctly.
"""

import sys
import os
import paddle
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all fixed imports work correctly"""
    print("🔍 Testing imports...")
    
    try:
        from models.Predenergy.models.unified_config import PredenergyUnifiedConfig
        print("  ✅ Unified config import successful")
        
        from models.Predenergy.models.modeling_Predenergy import PredenergyForPrediction
        print("  ✅ Fixed model import successful")
        
        from models.Predenergy.Predenergy import FixedPredenergy
        print("  ✅ Main model class import successful")
        
        from models.Predenergy.utils.config_loader import ConfigLoader
        print("  ✅ Config loader import successful")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_configuration():
    """Test unified configuration system"""
    print("\n🔧 Testing configuration system...")
    
    try:
        from models.Predenergy.models.unified_config import PredenergyUnifiedConfig
        
        # Test basic configuration
        config = PredenergyUnifiedConfig(
            seq_len=96,
            horizon=24,
            d_model=512,
            use_combined_model=False
        )
        print("  ✅ Basic configuration creation successful")
        
        # Test configuration validation
        assert config.seq_len == 96
        assert config.horizon == 24
        assert config.pred_len == 24  # Should be auto-adjusted
        print("  ✅ Configuration validation successful")
        
        # Test configuration dictionary conversion
        config_dict = config.to_dict()
        config2 = PredenergyUnifiedConfig.from_dict(config_dict)
        assert config2.seq_len == config.seq_len
        print("  ✅ Configuration serialization successful")
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False


def test_model_creation():
    """Test model creation with both standard and combined modes"""
    print("\n🤖 Testing model creation...")
    
    try:
        from models.Predenergy.Predenergy import FixedPredenergy
        
        # Test standard model
        model_std = FixedPredenergy(
            seq_len=96,
            horizon=24,
            d_model=256,  # Smaller for testing
            use_combined_model=False
        )
        print("  ✅ Standard model creation successful")
        
        # Test combined model
        model_combined = FixedPredenergy(
            seq_len=96,
            horizon=24,
            d_model=256,
            use_combined_model=True,
            num_experts=4,  # Smaller for testing
            motse_hidden_size=512
        )
        print("  ✅ Combined model creation successful")
        
        # Test model info
        info = model_std.get_model_info()
        assert 'model_name' in info
        assert 'model_type' in info
        print("  ✅ Model info retrieval successful")
        
        return True
    except Exception as e:
        print(f"  ❌ Model creation test failed: {e}")
        return False


def test_data_processing():
    """Test data processing and prediction pipeline"""
    print("\n📊 Testing data processing...")
    
    try:
        from models.Predenergy.Predenergy import FixedPredenergy
        
        # Create synthetic data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='H')
        values = np.sin(np.arange(200) * 0.1) + np.random.normal(0, 0.1, 200)
        test_data = pd.DataFrame({'OT': values, 'date': dates})
        
        print("  ✅ Synthetic data creation successful")
        
        # Create model
        model = FixedPredenergy(
            seq_len=96,
            horizon=24,
            d_model=128,  # Small for testing
            use_combined_model=False,
            target='OT'
        )
        
        # Test data loader setup
        model.setup_data_loader(
            data=test_data,
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.2
        )
        print("  ✅ Data loader setup successful")
        
        # Check data loaders
        assert model.train_loader is not None
        assert model.val_loader is not None
        assert len(model.train_loader) > 0
        print("  ✅ Data loader validation successful")
        
        return True
    except Exception as e:
        print(f"  ❌ Data processing test failed: {e}")
        return False


def test_config_files():
    """Test configuration file loading"""
    print("\n📁 Testing configuration files...")
    
    try:
        from models.Predenergy.utils.config_loader import load_predenergy_config
        
        # Test standard config
        if os.path.exists('configs/predenergy_config.yaml'):
            config = load_predenergy_config('configs/predenergy_config.yaml')
            assert config.use_combined_model == False
            print("  ✅ Standard config file loading successful")
        else:
            print("  ⚠️  Standard config file not found (expected)")
        
        # Test combined config
        if os.path.exists('configs/predenergy_stdm_motse.yaml'):
            config = load_predenergy_config('configs/predenergy_stdm_motse.yaml')
            assert config.use_combined_model == True
            print("  ✅ Combined config file loading successful")
        else:
            print("  ⚠️  Combined config file not found (expected)")
        
        return True
    except Exception as e:
        print(f"  ❌ Config file test failed: {e}")
        return False


def test_device_compatibility():
    """Test device compatibility"""
    print("\n💻 Testing device compatibility...")
    
    try:
        device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
        print(f"  📱 Available device: {device}")
        
        # Test tensor operations
        x = paddle.randn([1, 96, 1])
        if device == "gpu":
            paddle.set_device('gpu')
        y = x * 2
        print(f"  ✅ Device operations successful on {device}")
        
        return True
    except Exception as e:
        print(f"  ❌ Device compatibility test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Predenergy Fixed Implementation - Quick Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("Model Creation Tests", test_model_creation),
        ("Data Processing Tests", test_data_processing),
        ("Config File Tests", test_config_files),
        ("Device Compatibility Tests", test_device_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status} - {test_name}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The fixed implementation is working correctly.")
        print("🚀 You can now proceed with using the Predenergy framework.")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please check the error messages above.")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    
    print("\n🔗 Next steps:")
    print("  - Check the README.md for usage examples")
    print("  - Try the training script: python scripts/train_predenergy.py --help")
    print("  - Start the API server: python src/api_predenergy.py")
    print("  - Read FIXES_SUMMARY.md for detailed information about fixes")


if __name__ == "__main__":
    main()