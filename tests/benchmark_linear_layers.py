import torch
import numpy as np
import timeit # Changed from time to timeit
import functools
import os
import sys

# Add project root to path for robust imports
# Assuming script is in DLBacktrace/tests/ and project root is DLBacktrace/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_DLBacktrace = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root_DLBacktrace)

from main.pytorch_backtrace.logic.utils.layer_refactors.Linear_layer import calculate_wt_fc as calculate_wt_fc_numpy
from main.pytorch_backtrace.logic.utils.layer_refactors.Linear_layer_torch_version import calculate_wt_fc_cuda
# Import NumPy-based activations from prop.py
from main.pytorch_backtrace.logic.utils.prop import (
    np_swish, np_wave, np_pulse, np_absolute, np_hard_sigmoid, np_sigmoid, np_tanh
)

# ---------------------------------------------------------------------------- #
# PyTorch-native Activation Functions
# ---------------------------------------------------------------------------- #

def torch_sigmoid_custom(x): # Renamed to avoid clash if torch.sigmoid is used directly
    return torch.sigmoid(x)

def torch_tanh_custom(x): # Renamed to avoid clash if torch.tanh is used directly
    return torch.tanh(x)

def torch_swish(x, beta=0.75):
    beta_t = torch.tensor(beta, device=x.device, dtype=x.dtype)
    return x * torch.sigmoid(beta_t * x)

def torch_wave(x, alpha=1.0):
    alpha_t = torch.tensor(alpha, device=x.device, dtype=x.dtype)
    # Ensure exp_one_t has the same dtype as x to avoid potential dtype mismatches
    exp_one_t = torch.exp(torch.tensor(1.0, device=x.device, dtype=x.dtype))
    return (alpha_t * x * exp_one_t) / (torch.exp(-x) + torch.exp(x))

def torch_pulse(x, alpha=1.0):
    alpha_t = torch.tensor(alpha, device=x.device, dtype=x.dtype)
    return alpha_t * (1 - torch.tanh(x).pow(2))

def torch_absolute(x, alpha=1.0):
    alpha_t = torch.tensor(alpha, device=x.device, dtype=x.dtype)
    return alpha_t * x * torch.tanh(x)

def torch_hard_sigmoid(x):
    return torch.clamp(0.2 * x + 0.5, 0, 1)

# ---------------------------------------------------------------------------- #
# Benchmark Setup
# ---------------------------------------------------------------------------- #

def get_act_configs():
    """Returns a list of activation configurations for benchmarking."""
    configs = [
        {"name": "Sigmoid", "np_func": np_sigmoid, "torch_func": torch_sigmoid_custom},
        {"name": "Tanh", "np_func": np_tanh, "torch_func": torch_tanh_custom},
        {"name": "HardSigmoid", "np_func": np_hard_sigmoid, "torch_func": torch_hard_sigmoid},
        {"name": "Swish (beta=0.75)", "np_func": functools.partial(np_swish, beta=0.75), "torch_func": functools.partial(torch_swish, beta=0.75)},
        {"name": "Swish (beta=1.0)", "np_func": functools.partial(np_swish, beta=1.0), "torch_func": functools.partial(torch_swish, beta=1.0)},
        {"name": "Wave (alpha=1.0)", "np_func": functools.partial(np_wave, alpha=1.0), "torch_func": functools.partial(torch_wave, alpha=1.0)},
        {"name": "Wave (alpha=0.5)", "np_func": functools.partial(np_wave, alpha=0.5), "torch_func": functools.partial(torch_wave, alpha=0.5)},
        {"name": "Pulse (alpha=1.0)", "np_func": functools.partial(np_pulse, alpha=1.0), "torch_func": functools.partial(torch_pulse, alpha=1.0)},
        {"name": "Absolute (alpha=1.0)", "np_func": functools.partial(np_absolute, alpha=1.0), "torch_func": functools.partial(torch_absolute, alpha=1.0)},
    ]
    return configs

def run_benchmark_numpy_vs_cuda(desc, in_features, out_features, np_activation_func, torch_activation_func, num_timer_repeats=5, num_runs_per_repeat=10):
    print(f"\n--- Benchmarking: {desc} ---")
    print(f"Parameters: In={in_features}, Out={out_features}, Activation: {np_activation_func.__name__ if hasattr(np_activation_func, '__name__') else np_activation_func.func.__name__}")

    # 1. Generate Test Data (consistent for both versions)
    np.random.seed(42)
    torch.manual_seed(42)

    # Data for NumPy version
    inp_np = np.random.rand(in_features).astype(np.float32)
    w_np_orig = np.random.rand(out_features, in_features).astype(np.float32)
    b_np_orig = np.random.rand(out_features).astype(np.float32)
    wts_np = np.random.rand(out_features).astype(np.float32)
    
    # PyTorch tensors (w, b) needed for the NumPy version's calculate_wt_fc_numpy function signature
    w_torch_param_for_numpy_ver = torch.from_numpy(w_np_orig)
    b_torch_param_for_numpy_ver = torch.from_numpy(b_np_orig)

    # Data for PyTorch CUDA version
    inp_torch_cuda = torch.from_numpy(inp_np)
    w_torch_cuda = torch.from_numpy(w_np_orig)
    b_torch_cuda = torch.from_numpy(b_np_orig)
    wts_torch_cuda = torch.from_numpy(wts_np)
    
    act_dict_np = {"type": "non_mono", "func": np_activation_func, "range": {"l": None, "u": None}}
    act_dict_torch = {"type": "non_mono", "func": torch_activation_func, "range": {"l": None, "u": None}}

    numpy_result, cuda_result_np = None, None
    numpy_time_avg, cuda_time_avg = float('inf'), float('inf')

    # 2. Run NumPy Version (CPU)
    try:
        numpy_result = calculate_wt_fc_numpy(wts_np.copy(), inp_np.copy(), w_torch_param_for_numpy_ver.clone(), b_torch_param_for_numpy_ver.clone(), act_dict_np.copy())
        timer_numpy = timeit.Timer(lambda: calculate_wt_fc_numpy(wts_np.copy(), inp_np.copy(), w_torch_param_for_numpy_ver.clone(), b_torch_param_for_numpy_ver.clone(), act_dict_np.copy()))
        times_numpy = timer_numpy.repeat(repeat=num_timer_repeats, number=num_runs_per_repeat)
        numpy_time_avg = min(times_numpy) / num_runs_per_repeat
        print(f"  NumPy (CPU) best average:     {numpy_time_avg:.6f} s")
    except Exception as e:
        print(f"  Error running NumPy version: {e}")

    # 3. Run PyTorch CUDA Version
    if torch.cuda.is_available():
        try:
            device_cuda = torch.device("cuda")
            inp_c = inp_torch_cuda.to(device_cuda)
            w_c = w_torch_cuda.to(device_cuda)
            b_c = b_torch_cuda.to(device_cuda)
            wts_c = wts_torch_cuda.to(device_cuda)

            # Initial run for correctness check & warm-up
            cuda_result_torch = calculate_wt_fc_cuda(wts_c.clone(), inp_c.clone(), w_c.clone(), b_c.clone(), act_dict_torch.copy(), device=device_cuda)
            cuda_result_np = cuda_result_torch.cpu().numpy()
            torch.cuda.synchronize() # Warm-up sync

            def cuda_op_to_time():
                calculate_wt_fc_cuda(wts_c.clone(), inp_c.clone(), w_c.clone(), b_c.clone(), act_dict_torch.copy(), device=device_cuda)
                torch.cuda.synchronize() # Crucial for accurate GPU timing with timeit

            timer_cuda = timeit.Timer(cuda_op_to_time)
            times_cuda = timer_cuda.repeat(repeat=num_timer_repeats, number=num_runs_per_repeat)
            cuda_time_avg = min(times_cuda) / num_runs_per_repeat
            print(f"  PyTorch (CUDA) best average:  {cuda_time_avg:.6f} s")
        except Exception as e:
            print(f"  Error running PyTorch CUDA version: {e}")
            cuda_result_np = None
    else:
        print(f"  PyTorch (CUDA): Not available")

    # 4. Compare Accuracy
    if numpy_result is not None and cuda_result_np is not None:
        if np.allclose(numpy_result, cuda_result_np, atol=1e-5, rtol=1e-4): # Adjusted tolerances slightly
            print(f"  Accuracy: Outputs MATCH (atol=1e-5, rtol=1e-4)")
        else:
            print(f"  Accuracy: Outputs DO NOT MATCH")
            diff = np.abs(numpy_result - cuda_result_np)
            print(f"    Max absolute difference: {np.max(diff):.2e}")
            print(f"    Mean absolute difference: {np.mean(diff):.2e}")
    elif numpy_result is None or cuda_result_np is None:
        print(f"  Accuracy: Cannot compare due to errors in execution.")

    # 5. Compare Performance
    if numpy_time_avg != float('inf') and cuda_time_avg != float('inf') and cuda_time_avg > 1e-9:
        speedup = numpy_time_avg / cuda_time_avg
        print(f"  Speedup (CUDA vs NumPy CPU): {speedup:.2f}x")
    else:
        print(f"  Performance: Could not calculate speedup.")

if __name__ == "__main__":
    print("Starting DL_Backtrace Linear Layer Benchmark (NumPy CPU vs PyTorch CUDA)")
    print("=".ljust(80, '='))

    activation_configurations = get_act_configs()
    
    # Define a set of feature sizes to test
    feature_sets = [
        {"name": "Small", "in": 128, "out": 64},
        {"name": "Medium", "in": 512, "out": 256},
        {"name": "Large", "in": 1024, "out": 512},
        # {"name": "XLarge", "in": 2048, "out": 1024} # Can add larger sizes
    ]

    num_timer_repeats_main = 5
    num_runs_per_repeat_main = 10

    for features in feature_sets:
        print(f"\n{(' Testing Feature Set: ' + features['name'] + ' (In: ' + str(features['in']) + ', Out: ' + str(features['out']) + ') ').center(80, '-')}")
        for act_conf in activation_configurations:
            desc = f"{act_conf['name']} - {features['name']} Features"
            run_benchmark_numpy_vs_cuda(
                desc=desc,
                in_features=features["in"],
                out_features=features["out"],
                np_activation_func=act_conf["np_func"],
                torch_activation_func=act_conf["torch_func"],
                num_timer_repeats=num_timer_repeats_main,
                num_runs_per_repeat=num_runs_per_repeat_main
            )
    
    print("\n" + "=".ljust(80, '='))
    print("Benchmark complete.")
    print("Note: Ensure no other heavy processes are running for stable results.")
    print("Consider factors like GPU model, system load, and specific data for variations.")
