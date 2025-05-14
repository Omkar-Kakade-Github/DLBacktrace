import torch
import numpy as np
import timeit
import functools
import os
import sys

# Add project root to path for robust imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_DLBacktrace = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root_DLBacktrace)

try:
    from main.pytorch_backtrace.logic.utils.prop import calculate_wt_conv_unit_1d_v2 as calculate_wt_conv_unit_1d_original
    from main.pytorch_backtrace.logic.utils.layer_refactors.Conv_1d_layer import calculate_weighted_convolution_relevance_1d as calculate_wt_conv_unit_1d_numpy
    from main.pytorch_backtrace.logic.utils.layer_refactors.torch_conv_1d_layer import calculate_weighted_convolution_relevance_1d as calculate_wt_conv_unit_1d_torch
    
    # Import NumPy-based activations from prop.py
    from main.pytorch_backtrace.logic.utils.prop import (
        np_swish, np_wave, np_absolute, np_tanh, np_sigmoid
    )
except ImportError as e:
    print(f"Error importing functions: {e}")
    print(f"Current sys.path includes: {project_root_DLBacktrace}")
    sys.exit(1)

# ---------------------------------------------------------------------------- #
# PyTorch-native Activation Functions
# ---------------------------------------------------------------------------- #

def torch_sigmoid_custom(x):
    return torch.sigmoid(x)

def torch_tanh_custom(x):
    return torch.tanh(x)

def torch_swish(x, beta=0.75):
    beta_t = torch.tensor(beta, device=x.device, dtype=x.dtype)
    return x * torch.sigmoid(beta_t * x)

def torch_wave(x, alpha=1.0):
    alpha_t = torch.tensor(alpha, device=x.device, dtype=x.dtype)
    exp_one_t = torch.exp(torch.tensor(1.0, device=x.device, dtype=x.dtype))
    return (alpha_t * x * exp_one_t) / (torch.exp(-x) + torch.exp(x))

def torch_absolute(x, alpha=1.0):
    alpha_t = torch.tensor(alpha, device=x.device, dtype=x.dtype)
    return alpha_t * x * torch.tanh(x)

# ---------------------------------------------------------------------------- #
# Benchmark Setup
# ---------------------------------------------------------------------------- #

def get_act_configs():
    """Returns a list of activation configurations for benchmarking."""
    configs = [
        {"name": "Sigmoid", "np_func": np_sigmoid, "torch_func": torch_sigmoid_custom},
        {"name": "Tanh", "np_func": np_tanh, "torch_func": torch_tanh_custom},
        {"name": "Swish (beta=0.75)", "np_func": functools.partial(np_swish, beta=0.75), "torch_func": functools.partial(torch_swish, beta=0.75)},
        {"name": "Wave (alpha=1.0)", "np_func": functools.partial(np_wave, alpha=1.0), "torch_func": functools.partial(torch_wave, alpha=1.0)},
        {"name": "Absolute (alpha=1.0)", "np_func": functools.partial(np_absolute, alpha=1.0), "torch_func": functools.partial(torch_absolute, alpha=1.0)},
    ]
    return configs

def run_benchmark_conv1d(desc, kernel_size, in_channels, out_channels, patch_size, 
                        np_activation_func, torch_activation_func, 
                        num_timer_repeats=5, num_runs_per_repeat=10):
    print(f"\n--- Benchmarking: {desc} ---")
    print(f"Parameters: Kernel={kernel_size}, InChannels={in_channels}, OutChannels={out_channels}, Patch={patch_size}")

    # 1. Generate Test Data (consistent for both versions)
    np.random.seed(42)
    torch.manual_seed(42)

    # NumPy data
    np_patch = np.random.rand(patch_size, in_channels).astype(np.float32)
    np_relevance = np.random.rand(out_channels).astype(np.float32)
    np_weights = np.random.randn(kernel_size, in_channels, out_channels).astype(np.float32)
    np_bias = np.random.randn(out_channels).astype(np.float32)
    
    # PyTorch data (same values, different format)
    torch_patch = torch.from_numpy(np_patch).clone()
    torch_relevance = torch.from_numpy(np_relevance).clone()
    torch_weights = torch.from_numpy(np_weights).clone()
    torch_bias = torch.from_numpy(np_bias).clone()
    
    act_dict_np = {"type": "non_mono", "func": np_activation_func, "range": {"l": None, "u": None}}
    act_dict_torch = {"type": "non_mono", "func": torch_activation_func, "range": {"l": None, "u": None}}

    numpy_result, numpy_refactored_result, cuda_result_np = None, None, None
    numpy_orig_time, numpy_refactored_time, cuda_time = float('inf'), float('inf'), float('inf')

    # 2. Run Original NumPy Version
    try:
        numpy_result = calculate_wt_conv_unit_1d_original(
            np_patch.copy(), np_relevance.copy(), np_weights.copy(), np_bias.copy(), act_dict_np.copy())
        
        timer_numpy = timeit.Timer(lambda: calculate_wt_conv_unit_1d_original(
            np_patch.copy(), np_relevance.copy(), np_weights.copy(), np_bias.copy(), act_dict_np.copy()))
        times_numpy = timer_numpy.repeat(repeat=num_timer_repeats, number=num_runs_per_repeat)
        numpy_orig_time = min(times_numpy) / num_runs_per_repeat
        print(f"  Original NumPy best average:   {numpy_orig_time:.6f} s")
    except Exception as e:
        print(f"  Error running original NumPy version: {e}")
        numpy_result = None

    # 3. Run Refactored NumPy Version
    try:
        numpy_refactored_result = calculate_wt_conv_unit_1d_numpy(
            np_patch.copy(), np_relevance.copy(), np_weights.copy(), np_bias.copy(), act_dict_np.copy())
        
        timer_numpy_refactored = timeit.Timer(lambda: calculate_wt_conv_unit_1d_numpy(
            np_patch.copy(), np_relevance.copy(), np_weights.copy(), np_bias.copy(), act_dict_np.copy()))
        times_numpy_refactored = timer_numpy_refactored.repeat(repeat=num_timer_repeats, number=num_runs_per_repeat)
        numpy_refactored_time = min(times_numpy_refactored) / num_runs_per_repeat
        print(f"  Refactored NumPy best average: {numpy_refactored_time:.6f} s")
    except Exception as e:
        print(f"  Error running refactored NumPy version: {e}")
        numpy_refactored_result = None

    # 4. Run PyTorch CUDA Version
    if torch.cuda.is_available():
        try:
            device_cuda = torch.device("cuda")
            torch_patch_cuda = torch_patch.to(device_cuda)
            torch_relevance_cuda = torch_relevance.to(device_cuda)
            torch_weights_cuda = torch_weights.to(device_cuda)
            torch_bias_cuda = torch_bias.to(device_cuda)

            # Initial runs for warm-up and compilation
            for _ in range(5):
                _ = calculate_wt_conv_unit_1d_torch(
                    torch_patch_cuda.clone(), torch_relevance_cuda.clone(), 
                    torch_weights_cuda.clone(), torch_bias_cuda.clone(), 
                    act_dict_torch.copy())
                torch.cuda.synchronize()

            # First run for correctness check
            cuda_result = calculate_wt_conv_unit_1d_torch(
                torch_patch_cuda.clone(), torch_relevance_cuda.clone(),
                torch_weights_cuda.clone(), torch_bias_cuda.clone(),
                act_dict_torch.copy())
            cuda_result_np = cuda_result.cpu().numpy()

            # Accurate CUDA timing with synchronization
            def cuda_op_to_time():
                _ = calculate_wt_conv_unit_1d_torch(
                    torch_patch_cuda.clone(), torch_relevance_cuda.clone(),
                    torch_weights_cuda.clone(), torch_bias_cuda.clone(),
                    act_dict_torch.copy())
                torch.cuda.synchronize()  # Crucial for accurate GPU timing

            timer_cuda = timeit.Timer(cuda_op_to_time)
            times_cuda = timer_cuda.repeat(repeat=num_timer_repeats, number=num_runs_per_repeat)
            cuda_time = min(times_cuda) / num_runs_per_repeat
            print(f"  PyTorch CUDA best average:    {cuda_time:.6f} s")

        except Exception as e:
            print(f"  Error running PyTorch CUDA version: {e}")
            cuda_result_np = None
    else:
        print("  PyTorch CUDA: Not available")

    # 5. Compare Accuracy
    print("\n  Accuracy Comparisons:")
    
    if numpy_result is not None and numpy_refactored_result is not None:
        if np.allclose(numpy_result, numpy_refactored_result, atol=1e-6, rtol=1e-5):
            print("    Original vs Refactored NumPy: MATCH")
        else:
            diff = np.abs(numpy_result - numpy_refactored_result)
            print(f"    Original vs Refactored NumPy: DIFFER (max diff: {np.max(diff):.2e})")
    
    if numpy_result is not None and cuda_result_np is not None:
        if np.allclose(numpy_result, cuda_result_np, atol=1e-5, rtol=1e-4):
            print("    Original NumPy vs PyTorch:    MATCH")
        else:
            diff = np.abs(numpy_result - cuda_result_np)
            print(f"    Original NumPy vs PyTorch:    DIFFER (max diff: {np.max(diff):.2e}, mean: {np.mean(diff):.2e})")
            
    if numpy_refactored_result is not None and cuda_result_np is not None:
        if np.allclose(numpy_refactored_result, cuda_result_np, atol=1e-5, rtol=1e-4):
            print("    Refactored NumPy vs PyTorch:  MATCH")
        else:
            diff = np.abs(numpy_refactored_result - cuda_result_np)
            print(f"    Refactored NumPy vs PyTorch:  DIFFER (max diff: {np.max(diff):.2e}, mean: {np.mean(diff):.2e})")

    # 6. Compare Performance
    print("\n  Performance Comparisons:")
    if numpy_refactored_time != float('inf') and numpy_orig_time != float('inf'):
        speedup = numpy_orig_time / numpy_refactored_time
        print(f"    NumPy Refactor vs Original:     {speedup:.2f}x")
    
    if cuda_time != float('inf'):
        if numpy_orig_time != float('inf'):
            speedup = numpy_orig_time / cuda_time
            print(f"    CUDA vs Original NumPy:        {speedup:.2f}x")
        
        if numpy_refactored_time != float('inf'):
            speedup = numpy_refactored_time / cuda_time
            print(f"    CUDA vs Refactored NumPy:      {speedup:.2f}x")
    
    return {
        'original': numpy_orig_time,
        'refactored': numpy_refactored_time,
        'cuda': cuda_time
    }

if __name__ == "__main__":
    print("Starting DL_Backtrace Conv1D Layer Benchmark - LARGE CONFIG ONLY")
    print("=".ljust(80, "="))
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("WARNING: CUDA not available. PyTorch tests will run on CPU.")
    
    activation_configurations = get_act_configs()
    
    # Define the large configuration only
    large_config = {
        "name": "Large", 
        "kernel_size": 9, 
        "in_channels": 256, 
        "out_channels": 512, 
        "patch_size": 9
    }
    
    results = []
    
    # Run monotonic test (ReLU-like) first
    print("\n" + (" Testing Monotonic Activation (ReLU-like) - Large Config ").center(80, "-"))
    
    desc = f"Monotonic ReLU-like - Large Conv1D"
    
    # Create activation configs specific to monotonic case
    act_dict_np = {"type": "mono", "range": {"l": 0.0, "u": None}, "func": None}
    act_dict_torch = {"type": "mono", "range": {"l": 0.0, "u": None}, "func": None}
    
    # Increase the number of runs to get more stable results
    num_timer_repeats = 5
    num_runs = 10
    
    np_patch = np.random.rand(large_config['patch_size'], large_config['in_channels']).astype(np.float32)
    np_relevance = np.random.rand(large_config['out_channels']).astype(np.float32)
    np_weights = np.random.randn(large_config['kernel_size'], large_config['in_channels'], large_config['out_channels']).astype(np.float32)
    np_bias = np.random.randn(large_config['out_channels']).astype(np.float32)
    
    mono_times = {}
    
    try:
        # Original NumPy
        timer = timeit.Timer(lambda: calculate_wt_conv_unit_1d_original(
            np_patch.copy(), np_relevance.copy(), np_weights.copy(), np_bias.copy(), act_dict_np.copy()))
        times = timer.repeat(repeat=num_timer_repeats, number=num_runs)
        mono_times['original'] = min(times) / num_runs
        
        # Refactored NumPy
        timer = timeit.Timer(lambda: calculate_wt_conv_unit_1d_numpy(
            np_patch.copy(), np_relevance.copy(), np_weights.copy(), np_bias.copy(), act_dict_np.copy()))
        times = timer.repeat(repeat=num_timer_repeats, number=num_runs)
        mono_times['refactored'] = min(times) / num_runs
        
        # PyTorch CUDA (if available)
        if torch.cuda.is_available():
            torch_patch = torch.from_numpy(np_patch).cuda()
            torch_relevance = torch.from_numpy(np_relevance).cuda()
            torch_weights = torch.from_numpy(np_weights).cuda()
            torch_bias = torch.from_numpy(np_bias).cuda()
            
            # Warmup
            for _ in range(10):  # More warmup iterations for stable CUDA timing
                _ = calculate_wt_conv_unit_1d_torch(
                    torch_patch.clone(), torch_relevance.clone(),
                    torch_weights.clone(), torch_bias.clone(),
                    act_dict_torch.copy())
                torch.cuda.synchronize()
            
            def cuda_op():
                _ = calculate_wt_conv_unit_1d_torch(
                    torch_patch.clone(), torch_relevance.clone(),
                    torch_weights.clone(), torch_bias.clone(),
                    act_dict_torch.copy())
                torch.cuda.synchronize()
            
            timer = timeit.Timer(cuda_op)
            times = timer.repeat(repeat=num_timer_repeats, number=num_runs)
            mono_times['cuda'] = min(times) / num_runs
        
        print(f"\n--- {desc} ---")
        print(f"  Original NumPy:    {mono_times.get('original', float('inf')):.6f} s")
        print(f"  Refactored NumPy:  {mono_times.get('refactored', float('inf')):.6f} s")
        print(f"  PyTorch CUDA:      {mono_times.get('cuda', float('inf')):.6f} s")
        
        if all(k in mono_times for k in ['original', 'refactored', 'cuda']):
            print("\n  Speedup:")
            print(f"    NumPy Refactor vs Original:  {mono_times['original'] / mono_times['refactored']:.2f}x")
            print(f"    CUDA vs Original NumPy:      {mono_times['original'] / mono_times['cuda']:.2f}x")
            print(f"    CUDA vs Refactored NumPy:    {mono_times['refactored'] / mono_times['cuda']:.2f}x")
        
        results.append({
            'desc': f"Monotonic - Large",
            'times': mono_times
        })
        
    except Exception as e:
        print(f"\n--- {desc} ---")
        print(f"  Error: {e}")
    
    # Run non-monotonic tests with various activations
    print("\n" + (" Testing Non-Monotonic Activations - Large Config ").center(80, "-"))
    
    for act_conf in activation_configurations:
        desc = f"{act_conf['name']} - Large Conv1D"
        times = run_benchmark_conv1d(
            desc=desc,
            kernel_size=large_config["kernel_size"],
            in_channels=large_config["in_channels"],
            out_channels=large_config["out_channels"],
            patch_size=large_config["patch_size"],
            np_activation_func=act_conf["np_func"],
            torch_activation_func=act_conf["torch_func"],
            num_timer_repeats=5,
            num_runs_per_repeat=10
        )
        
        results.append({
            'desc': desc,
            'times': times
        })
    
    # Print summary table
    print("\n=== SUMMARY (LARGE CONFIG) ===")
    print(f"{'Test Case':<30} {'Original':<12} {'Refactored':<12} {'PyTorch CUDA':<12} {'CUDA Speedup':<15}")
    print("-" * 80)
    
    for result in results:
        times = result["times"]
        if all(k in times for k in ['original', 'refactored', 'cuda']):
            speedup = times['refactored'] / times['cuda'] if times['cuda'] > 0 else float('inf')
            print(f"{result['desc']:<30} {times['original']:.6f} s  {times['refactored']:.6f} s  {times['cuda']:.6f} s  {speedup:.2f}x")
        else:
            missing = [k for k in ['original', 'refactored', 'cuda'] if k not in times]
            print(f"{result['desc']:<30} Missing data for: {', '.join(missing)}")
    
    print("\nNote: CUDA speedup is relative to refactored NumPy version")
    print("      Larger input sizes show the most significant GPU performance advantage") 
