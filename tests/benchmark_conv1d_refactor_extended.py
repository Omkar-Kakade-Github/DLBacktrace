import timeit
import numpy as np
import torch
import sys
import os

# Add project root to path for robust imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_DLBacktrace = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root_DLBacktrace)

try:
    from main.pytorch_backtrace.logic.utils.prop import calculate_wt_conv_unit_1d_v2 as calculate_wt_conv_unit_1d_original
    from main.pytorch_backtrace.logic.utils.layer_refactors.Conv_1d_layer import calculate_weighted_convolution_relevance_1d as calculate_wt_conv_unit_1d_refactored
except ImportError as e:
    print(f"Error importing functions: {e}")
    print(f"Ensure the script is run from a location where DLBacktrace package is accessible,")
    print(f"or the DLBacktrace project is in PYTHONPATH.")
    print(f"Current sys.path includes: {project_root_DLBacktrace}")
    sys.exit(1)

# Define some activation functions for testing non-monotonic case
def np_tanh_for_test(x):
    z = np.tanh(x)
    return z.astype(np.float32)

def np_swish_for_test(x, beta=0.75): # Simplified swish
    return x * (1.0 / (1.0 + np.exp(-beta * x)))

def run_benchmark(desc, kernel_size, in_channels, out_channels, patch_size, act_config, num_repeats=10):
    print(f"\n--- Benchmarking: {desc} ---")
    print(f"Parameters: Kernel_Size={kernel_size}, In_Channels={in_channels}, Out_Channels={out_channels}, Patch_Size={patch_size}, Activation: {act_config['type']}")

    # Generate Test Data with consistent seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Create patch of input shape [patch_size, in_channels]
    patch = np.random.rand(patch_size, in_channels).astype(np.float32)
    
    # Create relevance values shape [out_channels]
    wts = np.random.rand(out_channels).astype(np.float32)
    
    # Create weights shape [kernel_size, in_channels, out_channels]
    w = np.random.randn(kernel_size, in_channels, out_channels).astype(np.float32)
    
    # Create bias shape [out_channels]
    b = np.random.randn(out_channels).astype(np.float32)

    original_result, refactored_result = None, None
    original_time, refactored_time = float('inf'), float('inf')

    # Run Original Function
    try:
        # Initial run for correctness check
        original_result = calculate_wt_conv_unit_1d_original(patch.copy(), wts.copy(), w.copy(), b.copy(), act_config.copy())
        
        # Timing
        timer_original = timeit.Timer(lambda: calculate_wt_conv_unit_1d_original(patch.copy(), wts.copy(), w.copy(), b.copy(), act_config.copy()))
        times_original = timer_original.repeat(repeat=3, number=num_repeats)
        original_time = min(times_original) / num_repeats
        print(f"Original function best average time: {original_time:.6f} s")
    except Exception as e:
        print(f"Error running original function: {e}")
        original_result = None

    # Run Refactored Function
    try:
        # Initial run for correctness check
        refactored_result = calculate_wt_conv_unit_1d_refactored(patch.copy(), wts.copy(), w.copy(), b.copy(), act_config.copy())
        
        # Timing
        timer_refactored = timeit.Timer(lambda: calculate_wt_conv_unit_1d_refactored(patch.copy(), wts.copy(), w.copy(), b.copy(), act_config.copy()))
        times_refactored = timer_refactored.repeat(repeat=3, number=num_repeats)
        refactored_time = min(times_refactored) / num_repeats
        print(f"Refactored function best average time: {refactored_time:.6f} s")
    except Exception as e:
        print(f"Error running refactored function: {e}")
        refactored_result = None

    # Compare Accuracy
    if original_result is not None and refactored_result is not None:
        if np.allclose(original_result, refactored_result, atol=1e-6, rtol=1e-5):
            print("Accuracy: Outputs MATCH")
        else:
            print("Accuracy: Outputs DO NOT MATCH")
            diff = np.abs(original_result - refactored_result)
            print(f"  Max absolute difference: {np.max(diff):.2e}")
            print(f"  Mean absolute difference: {np.mean(diff):.2e}")
            # Only compute relative difference if there are significant absolute differences
            if np.max(diff) > 1e-5:
                rel_diff = diff / (np.abs(original_result) + 1e-9)
                print(f"  Max relative difference: {np.max(rel_diff):.2e}")
                
                # Print some sample values at maximum difference location
                idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"  Sample at max diff idx {idx}: Original={original_result[idx]}, Refactored={refactored_result[idx]}")
    else:
        print("Accuracy: Cannot compare due to errors in execution.")

    # Compare Performance
    if original_time != float('inf') and refactored_time != float('inf') and refactored_time > 1e-9:
        speedup = original_time / refactored_time
        print(f"Speedup (Refactored vs Original): {speedup:.2f}x")
    else:
        print("Performance: Could not calculate speedup due to errors or zero/inf time.")
    
    return original_time, refactored_time

if __name__ == "__main__":
    # Define test parameters with progressively larger dimensions
    test_configs = [
        # Medium dimensions
        {"kernel_size": 5, "in_channels": 64, "out_channels": 128, "patch_size": 5}, 
        # Large dimensions
        {"kernel_size": 7, "in_channels": 128, "out_channels": 256, "patch_size": 7},
        # Extra large dimensions 
        {"kernel_size": 9, "in_channels": 256, "out_channels": 512, "patch_size": 9},
    ]
    
    # Monitor time improvements as dimensions increase
    print("\n===== EXTENDED BENCHMARKS FOR DIFFERENT DIMENSIONS =====\n")
    print("Testing with Monotonic ReLU-like activation:")
    
    act_mono_relu_like = {
        "type": "mono",
        "range": {"l": 0.0, "u": None},
        "func": None
    }
    
    results = []
    
    for i, config in enumerate(test_configs):
        desc = f"Monotonic ReLU-like (Test {i+1})"
        orig_time, refact_time = run_benchmark(
            desc, 
            config["kernel_size"],
            config["in_channels"], 
            config["out_channels"],
            config["patch_size"],
            act_mono_relu_like,
            num_repeats=5 if i < len(test_configs) - 1 else 3  # Fewer repeats for largest test
        )
        results.append((config, orig_time, refact_time))
        
    print("\n===== RESULTS SUMMARY =====")
    print("\nDimensions vs Speedup:")
    print(f"{'Kernel':<8} {'InCh':<8} {'OutCh':<8} {'Patch':<8} {'Original (s)':<15} {'Refactored (s)':<15} {'Speedup':<10}")
    print("-" * 72)
    
    for config, orig_time, refact_time in results:
        speedup = orig_time / refact_time if refact_time > 0 else float('inf')
        print(f"{config['kernel_size']:<8} {config['in_channels']:<8} {config['out_channels']:<8} {config['patch_size']:<8} "
              f"{orig_time:.8f}     {refact_time:.8f}     {speedup:.2f}x")
        
    print("\nNon-monotonic activation test with large dimensions:")
    act_non_mono_tanh = {
        "type": "non_mono",
        "range": {"l": None, "u": None},
        "func": np_tanh_for_test
    }
    
    # Test one large configuration with non-monotonic activation
    large_config = test_configs[1]  # Use the second largest configuration
    run_benchmark(
        "Non-Monotonic Tanh (Large)", 
        large_config["kernel_size"],
        large_config["in_channels"], 
        large_config["out_channels"],
        large_config["patch_size"],
        act_non_mono_tanh,
        num_repeats=5
    )
    
    print("\n--- Benchmarking complete ---")
    print("Note: Speedups can vary based on system load and specific data.") 
