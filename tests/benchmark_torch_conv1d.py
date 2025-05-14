import timeit
import numpy as np
import torch
import sys
import os

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

try:
    from main.pytorch_backtrace.logic.utils.prop import calculate_wt_conv_unit_1d_v2 as calculate_wt_conv_unit_1d_original
    from main.pytorch_backtrace.logic.utils.layer_refactors.Conv_1d_layer import calculate_weighted_convolution_relevance_1d as calculate_wt_conv_unit_1d_numpy
    from main.pytorch_backtrace.logic.utils.layer_refactors.torch_conv_1d_layer import calculate_weighted_convolution_relevance_1d as calculate_wt_conv_unit_1d_torch
except ImportError as e:
    print(f"Error importing functions: {e}")
    print(f"Current sys.path includes: {project_root}")
    sys.exit(1)

# Define activation functions for testing
def np_tanh_for_test(x):
    return np.tanh(x).astype(np.float32)

def torch_tanh_for_test(x):
    return torch.tanh(x)

def np_swish_for_test(x, beta=0.75):
    return x * (1.0 / (1.0 + np.exp(-beta * x)))

def torch_swish_for_test(x, beta=0.75):
    return x * (1.0 / (1.0 + torch.exp(-beta * x)))

def run_benchmark(desc, kernel_size, in_channels, out_channels, patch_size, act_config_np, act_config_torch, num_repeats=10):
    print(f"\n--- Benchmarking: {desc} ---")
    print(f"Parameters: Kernel_Size={kernel_size}, In_Channels={in_channels}, Out_Channels={out_channels}, Patch_Size={patch_size}")
    
    # Generate reproducible test data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # NumPy data
    np_patch = np.random.rand(patch_size, in_channels).astype(np.float32)
    np_relevance = np.random.rand(out_channels).astype(np.float32)
    np_weights = np.random.randn(kernel_size, in_channels, out_channels).astype(np.float32)
    np_bias = np.random.randn(out_channels).astype(np.float32)
    
    # PyTorch data (same values, different format)
    torch_patch = torch.tensor(np_patch, dtype=torch.float32).cuda()
    torch_relevance = torch.tensor(np_relevance, dtype=torch.float32).cuda()
    torch_weights = torch.tensor(np_weights, dtype=torch.float32).cuda()
    torch_bias = torch.tensor(np_bias, dtype=torch.float32).cuda()
    
    results = {}
    times = {}
    
    # Run original NumPy function
    try:
        results['original'] = calculate_wt_conv_unit_1d_original(
            np_patch.copy(), np_relevance.copy(), np_weights.copy(), np_bias.copy(), act_config_np.copy())
        
        timer = timeit.Timer(lambda: calculate_wt_conv_unit_1d_original(
            np_patch.copy(), np_relevance.copy(), np_weights.copy(), np_bias.copy(), act_config_np.copy()))
        times['original'] = min(timer.repeat(repeat=3, number=num_repeats)) / num_repeats
        print(f"Original NumPy function time: {times['original']:.6f} s")
    except Exception as e:
        print(f"Error in original function: {e}")
        results['original'] = None
    
    # Run refactored NumPy function
    try:
        results['numpy'] = calculate_wt_conv_unit_1d_numpy(
            np_patch.copy(), np_relevance.copy(), np_weights.copy(), np_bias.copy(), act_config_np.copy())
        
        timer = timeit.Timer(lambda: calculate_wt_conv_unit_1d_numpy(
            np_patch.copy(), np_relevance.copy(), np_weights.copy(), np_bias.copy(), act_config_np.copy()))
        times['numpy'] = min(timer.repeat(repeat=3, number=num_repeats)) / num_repeats
        print(f"Refactored NumPy function time: {times['numpy']:.6f} s")
    except Exception as e:
        print(f"Error in refactored NumPy function: {e}")
        results['numpy'] = None
    
    # Run PyTorch function
    try:
        # First run for compilation & warmup
        for _ in range(5):
            _ = calculate_wt_conv_unit_1d_torch(
                torch_patch.clone(), torch_relevance.clone(), torch_weights.clone(), torch_bias.clone(), act_config_torch.copy())
        
        # Correctness check
        results['torch'] = calculate_wt_conv_unit_1d_torch(
            torch_patch.clone(), torch_relevance.clone(), torch_weights.clone(), torch_bias.clone(), act_config_torch.copy())
        results['torch'] = results['torch'].cpu().numpy()  # Convert to NumPy for comparison
        
        # Timing with GPU synchronization
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        times_ms = []
        for _ in range(5):  # 5 sets of measurements
            batch_times = []
            for _ in range(num_repeats):
                start.record()
                _ = calculate_wt_conv_unit_1d_torch(
                    torch_patch.clone(), torch_relevance.clone(), torch_weights.clone(), torch_bias.clone(), act_config_torch.copy())
                end.record()
                torch.cuda.synchronize()
                batch_times.append(start.elapsed_time(end) / 1000)  # convert ms to s
            times_ms.append(min(batch_times))
        
        times['torch'] = min(times_ms)
        print(f"PyTorch CUDA function time: {times['torch']:.6f} s")
    except Exception as e:
        print(f"Error in PyTorch function: {e}")
        results['torch'] = None
    
    # Compare accuracy
    if all(x is not None for x in results.values()):
        print("\nAccuracy Comparisons:")
        
        # Original vs NumPy refactor
        if np.allclose(results['original'], results['numpy'], atol=1e-6, rtol=1e-5):
            print("NumPy Original vs NumPy Refactor: MATCH")
        else:
            diff = np.abs(results['original'] - results['numpy'])
            print(f"NumPy Original vs NumPy Refactor: DIFFER (max diff: {np.max(diff):.2e})")
        
        # Original vs PyTorch
        if np.allclose(results['original'], results['torch'], atol=1e-5, rtol=1e-4):
            print("NumPy Original vs PyTorch: MATCH")
        else:
            diff = np.abs(results['original'] - results['torch'])
            print(f"NumPy Original vs PyTorch: DIFFER (max diff: {np.max(diff):.2e}, mean diff: {np.mean(diff):.2e})")
            if np.max(diff) > 1e-3:  # Only show detailed analysis for significant differences
                idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"  Max diff at {idx}: Original={results['original'][idx]}, PyTorch={results['torch'][idx]}")
        
        # NumPy refactor vs PyTorch
        if np.allclose(results['numpy'], results['torch'], atol=1e-5, rtol=1e-4):
            print("NumPy Refactor vs PyTorch: MATCH")
        else:
            diff = np.abs(results['numpy'] - results['torch'])
            print(f"NumPy Refactor vs PyTorch: DIFFER (max diff: {np.max(diff):.2e}, mean diff: {np.mean(diff):.2e})")
    else:
        print("\nCannot compare accuracy due to execution errors.")
    
    # Compare performance
    if all(x is not None for x in times.values()):
        print("\nPerformance Comparisons:")
        if times['numpy'] > 0:
            print(f"NumPy Refactor speedup vs Original: {times['original'] / times['numpy']:.2f}x")
        if times['torch'] > 0:
            print(f"PyTorch CUDA speedup vs Original: {times['original'] / times['torch']:.2f}x")
            print(f"PyTorch CUDA speedup vs NumPy Refactor: {times['numpy'] / times['torch']:.2f}x")
    else:
        print("\nCannot compare performance due to execution errors.")
    
    return times, results

if __name__ == "__main__":
    print("=== PyTorch Conv1D LRP Benchmarking ===")
    
    # Check for CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. PyTorch tests will run on CPU.")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test configurations
    test_configs = [
        {
            "desc": "Small Conv1D, Monotonic ReLU-like",
            "kernel_size": 3,
            "in_channels": 16,
            "out_channels": 32,
            "patch_size": 3,
            "act_config_np": {
                "type": "mono",
                "range": {"l": 0.0, "u": None},
                "func": None
            },
            "act_config_torch": {
                "type": "mono",
                "range": {"l": 0.0, "u": None},
                "func": None
            },
        },
        {
            "desc": "Medium Conv1D, Monotonic ReLU-like",
            "kernel_size": 5,
            "in_channels": 64,
            "out_channels": 128,
            "patch_size": 5,
            "act_config_np": {
                "type": "mono",
                "range": {"l": 0.0, "u": None},
                "func": None
            },
            "act_config_torch": {
                "type": "mono",
                "range": {"l": 0.0, "u": None},
                "func": None
            },
        },
        {
            "desc": "Small Conv1D, Non-Monotonic (Tanh)",
            "kernel_size": 3,
            "in_channels": 16,
            "out_channels": 32,
            "patch_size": 3,
            "act_config_np": {
                "type": "non_mono",
                "range": {"l": None, "u": None},
                "func": np_tanh_for_test
            },
            "act_config_torch": {
                "type": "non_mono",
                "range": {"l": None, "u": None},
                "func": torch_tanh_for_test
            },
        },
        {
            "desc": "Large Conv1D, Monotonic ReLU-like",
            "kernel_size": 7,
            "in_channels": 128,
            "out_channels": 256,
            "patch_size": 7,
            "act_config_np": {
                "type": "mono",
                "range": {"l": 0.0, "u": None},
                "func": None
            },
            "act_config_torch": {
                "type": "mono",
                "range": {"l": 0.0, "u": None},
                "func": None
            },
            "num_repeats": 5  # Fewer repeats for large test
        }
    ]
    
    results_summary = []
    
    for config in test_configs:
        num_repeats = config.pop("num_repeats", 10)
        times, _ = run_benchmark(**config, num_repeats=num_repeats)
        results_summary.append({
            "desc": config["desc"],
            "times": times
        })
    
    print("\n=== SUMMARY ===")
    print(f"{'Test Case':<35} {'Original':<12} {'NumPy Refactor':<15} {'PyTorch':<12} {'PyTorch Speedup':<15}")
    print("-" * 80)
    
    for result in results_summary:
        times = result["times"]
        if all(k in times for k in ['original', 'numpy', 'torch']):
            speedup = times['numpy'] / times['torch'] if times['torch'] > 0 else float('inf')
            print(f"{result['desc']:<35} {times['original']:.6f} s  {times['numpy']:.6f} s  {times['torch']:.6f} s  {speedup:.2f}x")
        else:
            missing = [k for k in ['original', 'numpy', 'torch'] if k not in times]
            print(f"{result['desc']:<35} Missing data for: {', '.join(missing)}")
    
    print("\nNote: PyTorch speedup is relative to NumPy refactored version")
    print("      Small timing differences may be due to system load variations") 
