import timeit
import numpy as np
import torch
import sys
import os

# Add project root to path for robust imports
# Assuming script is in DLBacktrace/tests/ and project root is DLBacktrace/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_DLBacktrace = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root_DLBacktrace)

try:
    from dl_backtrace.pytorch_backtrace.backtrace.utils.prop import calculate_wt_fc as calculate_wt_fc_original
    from dl_backtrace.pytorch_backtrace.backtrace.utils.layer_refactors.Linear_layer import calculate_wt_fc as calculate_wt_fc_refactored
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

def run_benchmark(desc, in_features, out_features, act_config, num_repeats=20):
    print(f"\n--- Benchmarking: {desc} ---")
    print(f"Parameters: In={in_features}, Out={out_features}, Activation: {act_config['type']}")

    # 1. Generate Test Data
    # Ensure reproducibility for a given run_benchmark call
    np.random.seed(42)
    torch.manual_seed(42)

    wts = np.random.rand(out_features).astype(np.float32)
    inp = np.random.rand(in_features).astype(np.float32)
    
    w = torch.randn(out_features, in_features, dtype=torch.float32)
    b = torch.randn(out_features, dtype=torch.float32)

    original_result, refactored_result = None, None
    original_time, refactored_time = float('inf'), float('inf')

    # 2. Run Original Function
    try:
        # Initial run for correctness check
        original_result = calculate_wt_fc_original(wts.copy(), inp.copy(), w.clone(), b.clone(), act_config.copy())
        
        # Timing
        timer_original = timeit.Timer(lambda: calculate_wt_fc_original(wts.copy(), inp.copy(), w.clone(), b.clone(), act_config.copy()))
        times_original = timer_original.repeat(repeat=5, number=num_repeats) # 5 sets of num_repeats runs
        original_time = min(times_original) / num_repeats # Average time of the best set
        print(f"Original function best average time: {original_time:.6f} s")
    except Exception as e:
        print(f"Error running original function: {e}")
        original_result = None # Ensure it's None if error

    # 3. Run Refactored Function
    try:
        # Initial run for correctness check
        refactored_result = calculate_wt_fc_refactored(wts.copy(), inp.copy(), w.clone(), b.clone(), act_config.copy())
        
        # Timing
        timer_refactored = timeit.Timer(lambda: calculate_wt_fc_refactored(wts.copy(), inp.copy(), w.clone(), b.clone(), act_config.copy()))
        times_refactored = timer_refactored.repeat(repeat=5, number=num_repeats)
        refactored_time = min(times_refactored) / num_repeats
        print(f"Refactored function best average time: {refactored_time:.6f} s")
    except Exception as e:
        print(f"Error running refactored function: {e}")
        refactored_result = None # Ensure it's None if error

    # 4. Compare Accuracy
    if original_result is not None and refactored_result is not None:
        # Adjust tolerances as needed. LRP can have small float differences.
        # Values from typical LRP tests are often around atol=1e-6, rtol=1e-5
        if np.allclose(original_result, refactored_result, atol=1e-6, rtol=1e-5):
            print("Accuracy: Outputs MATCH")
        else:
            print("Accuracy: Outputs DO NOT MATCH")
            diff = np.abs(original_result - refactored_result)
            print(f"  Max absolute difference: {np.max(diff):.2e}")
            print(f"  Mean absolute difference: {np.mean(diff):.2e}")
            # Find where they differ most significantly for debugging if needed
            # rel_diff = diff / (np.abs(original_result) + 1e-9) # Relative diff
            # print(f"  Max relative difference: {np.max(rel_diff):.2e}")

    elif original_result is None or refactored_result is None:
        print("Accuracy: Cannot compare due to errors in execution.")

    # 5. Compare Performance
    if original_time != float('inf') and refactored_time != float('inf') and refactored_time > 1e-9 : # Avoid division by zero if time is tiny
        speedup = original_time / refactored_time
        print(f"Speedup (Refactored vs Original): {speedup:.2f}x")
    else:
        print("Performance: Could not calculate speedup due to errors or zero/inf time.")

if __name__ == "__main__":
    # Test Case 1: Monotonic Activation (Simulating ReLU with lower bound 0)
    act_mono_relu_like = {
        "type": "mono",
        "range": {"l": 0.0, "u": None},
        "func": None
    }
    run_benchmark("Monotonic ReLU-like (Small)", in_features=128, out_features=64, act_config=act_mono_relu_like)
    run_benchmark("Monotonic ReLU-like (Medium)", in_features=512, out_features=256, act_config=act_mono_relu_like)


    # Test Case 2: Monotonic Activation (Bounded both sides, e.g. clipped linear)
    act_mono_bounded = {
        "type": "mono",
        "range": {"l": -0.5, "u": 0.5},
        "func": None
    }
    run_benchmark("Monotonic Bounded (Small)", in_features=128, out_features=64, act_config=act_mono_bounded)

    # Test Case 3: Non-Monotonic Activation (Tanh)
    # For non-monotonic, original prop.py expects act["func"] to be one of its defined np_* functions or similar.
    # The refactored Linear_layer.py uses np.vectorize(act["func"]), which is flexible.
    act_non_mono_tanh = {
        "type": "non_mono",
        "range": {"l": None, "u": None}, # Tanh is naturally bounded, explicit range check is optional
        "func": np_tanh_for_test 
    }
    run_benchmark("Non-Monotonic Tanh (Small)", in_features=128, out_features=64, act_config=act_non_mono_tanh)
    run_benchmark("Non-Monotonic Tanh (Medium)", in_features=512, out_features=256, act_config=act_non_mono_tanh)
    
    # Test Case 4: Non-Monotonic Activation (Swish like) with explicit range checks
    act_non_mono_swish_ranged = {
        "type": "non_mono",
        "range": {"l": -0.2, "u": 2.0}, # Explicit ranges for testing that logic path
        "func": np_swish_for_test 
    }
    run_benchmark("Non-Monotonic Swish-like with range (Small)", in_features=128, out_features=64, act_config=act_non_mono_swish_ranged)

    print("\n--- Benchmarking complete ---")
    print("Note: Speedups can vary based on system load and specific data.")
