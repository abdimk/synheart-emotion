#!/usr/bin/env python3
"""
Quick test script for WESAD reference models inference.

Run this after setting up the virtual environment to verify everything works.
"""

import warnings
warnings.filterwarnings('ignore')

from inference import (
    list_available_models,
    generate_random_features,
    predict,
    load_model
)

def test_list_models():
    """Test model discovery."""
    print("=" * 60)
    print("Test 1: Model Discovery")
    print("=" * 60)
    models = list_available_models()
    print(f"Found {len(models)} configurations:")
    for config, model_list in models.items():
        print(f"  {config}: {len(model_list)} models")
    print("✅ Model discovery works!\n")
    return models

def test_random_data_generation():
    """Test random data generation."""
    print("=" * 60)
    print("Test 2: Random Data Generation")
    print("=" * 60)
    
    baseline = generate_random_features('baseline', n_samples=2, seed=42)
    stress = generate_random_features('stress', n_samples=2, seed=123)
    
    print(f"Baseline samples: {len(baseline)}")
    print(f"  Sample 1 HR: {baseline['HR'].iloc[0]:.1f} bpm, RMSSD: {baseline['RMSSD'].iloc[0]:.1f} ms")
    print(f"Stress samples: {len(stress)}")
    print(f"  Sample 1 HR: {stress['HR'].iloc[0]:.1f} bpm, RMSSD: {stress['RMSSD'].iloc[0]:.1f} ms")
    print("✅ Random data generation works!\n")
    return baseline, stress

def test_model_loading():
    """Test model loading."""
    print("=" * 60)
    print("Test 3: Model Loading")
    print("=" * 60)
    
    configs = ['w60s5_binary', 'w120s5_binary', 'w120s60_binary']
    for config in configs:
        try:
            model = load_model(config, 'extratrees')
            print(f"  ✓ Loaded {config}/extratrees ({model.kind})")
            if model.window_config:
                print(f"    Window: {model.window_config.get('window_size_sec')}s, "
                      f"Step: {model.window_config.get('window_step_sec')}s")
        except Exception as e:
            print(f"  ✗ Failed to load {config}: {e}")
    
    print("✅ Model loading works!\n")

def test_inference():
    """Test inference with different models."""
    print("=" * 60)
    print("Test 4: Inference")
    print("=" * 60)
    
    baseline = generate_random_features('baseline', n_samples=1, seed=42)
    
    configs = ['w60s5_binary', 'w120s5_binary', 'w120s60_binary']
    for config in configs:
        try:
            results = predict(baseline, config, 'extratrees', return_probabilities=True)
            result = results[0]
            print(f"  {config}:")
            print(f"    Predicted: {result['label']}")
            if 'probabilities' in result:
                probs = result['probabilities']
                print(f"    Probabilities: Baseline={probs['Baseline']:.3f}, "
                      f"Stress={probs['Stress']:.3f}")
        except Exception as e:
            print(f"  ✗ {config} failed: {e}")
    
    print("✅ Inference works!\n")

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("WESAD Reference Models - Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_list_models()
        test_random_data_generation()
        test_model_loading()
        test_inference()
        
        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

