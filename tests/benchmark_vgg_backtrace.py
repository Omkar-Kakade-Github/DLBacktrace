import torch
import torchvision
import torchvision.transforms as transforms
import time # For benchmarking
import os
import sys
import numpy as np # Import NumPy

# Add project root to path for robust imports
# Assuming script is in DLBacktrace/tests/ and project root is DLBacktrace/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_DLBacktrace = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root_DLBacktrace)

# Assuming the project structure allows these imports
# Adjust paths if DLBacktrace is not directly in PYTHONPATH
from main.pytorch_backtrace.logic.backtrace import Backtrace
from main.pytorch_backtrace.models.simple_vgg import VGG19, load_pretrained_vgg

def get_cifar10_data(batch_size=4, image_size=224):
    """Loads CIFAR10 dataset and returns a DataLoader."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Standard normalization
    ])

    # Using the test set for benchmarking is common
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    print(f"CIFAR10 dataset loaded. Number of images in test set: {len(testset)}")
    return testloader

def compare_relevance_results(original_results, refactored_results, atol=1e-5, rtol=1e-4):
    """Compares relevance dictionaries from original and refactored LRP handlers."""
    if not isinstance(original_results, dict) or not isinstance(refactored_results, dict):
        print("Error: One or both results are not dictionaries.")
        return

    all_layers_match_numerically = True
    mismatched_layers_count = 0

    # Iterate through layers present in original results
    # This assumes original_results is the baseline for layer structure
    print("\n{:<20} | {:<10} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12}".format(
        "Layer", "Match", "MAE", "Min_Orig", "Max_Orig", "Min_Refac", "Max_Refac", "Max_Abs_Diff"))
    print("-" * 120)

    for layer_name, orig_relevance in original_results.items():
        if layer_name not in refactored_results:
            print(f"Layer '{layer_name}': Not found in refactored results.")
            all_layers_match_numerically = False
            mismatched_layers_count += 1
            continue

        refac_relevance = refactored_results[layer_name]

        if not isinstance(orig_relevance, np.ndarray) or not isinstance(refac_relevance, np.ndarray):
            print(f"Layer '{layer_name}': Relevance data is not a NumPy array for one or both results.")
            all_layers_match_numerically = False
            mismatched_layers_count += 1
            continue
        
        if orig_relevance.shape != refac_relevance.shape:
            print(f"Layer '{layer_name}': Shape mismatch. Original: {orig_relevance.shape}, Refactored: {refac_relevance.shape}")
            all_layers_match_numerically = False
            mismatched_layers_count += 1
            continue

        # Perform numerical comparison
        is_close = np.allclose(orig_relevance, refac_relevance, atol=atol, rtol=rtol)
        abs_diff = np.abs(orig_relevance - refac_relevance)
        mae = np.mean(abs_diff)
        
        min_orig = np.min(orig_relevance)
        max_orig = np.max(orig_relevance)
        mean_orig = np.mean(orig_relevance)
        std_orig = np.std(orig_relevance)

        min_refac = np.min(refac_relevance)
        max_refac = np.max(refac_relevance)
        mean_refac = np.mean(refac_relevance)
        std_refac = np.std(refac_relevance)

        min_abs_diff = np.min(abs_diff)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff) # Same as MAE
        std_abs_diff = np.std(abs_diff)

        match_status = "Match" if is_close else "MISMATCH"
        print(f"{layer_name:<20} | {match_status:<10} | {mae:<12.4e} | {min_orig:<12.4e} | {max_orig:<12.4e} | {min_refac:<12.4e} | {max_refac:<12.4e} | {max_abs_diff:<12.4e}")
        
        if not is_close:
            print(f"{'':<20} | {'Details:':<10} | {'Mean AbsDiff:':<12} {mean_abs_diff:<12.4e} | {'Std AbsDiff:':<12} {std_abs_diff:<12.4e}")
            print(f"{'':<20} | {'Original:':<10} | {'Mean:':<12} {mean_orig:<12.4e} | {'Std:':<12} {std_orig:<12.4e}")
            print(f"{'':<20} | {'Refactored:':<10} | {'Mean:':<12} {mean_refac:<12.4e} | {'Std:':<12} {std_refac:<12.4e}")
            all_layers_match_numerically = False
            mismatched_layers_count += 1
    
    # Check for any layers in refactored_results that were not in original_results
    extra_layers_count = 0
    for layer_name in refactored_results:
        if layer_name not in original_results:
            print(f"Layer '{layer_name}': Found in refactored results but not in original (Extra Layer).")
            extra_layers_count +=1
            all_layers_match_numerically = False # Technically a mismatch in structure
    
    if extra_layers_count > 0:
        print(f"\nFound {extra_layers_count} extra layer(s) in refactored results.")

    print("\n--- Comparison Summary ---")
    if all_layers_match_numerically and extra_layers_count == 0:
        print("All layers match numerically within the specified tolerance (atol={}, rtol={}).".format(atol, rtol))
    else:
        print("Numerical differences or structural mismatches found.")
        if mismatched_layers_count > 0:
            print(f"Number of layers with numerical mismatches or missing in refactored: {mismatched_layers_count}")
        if extra_layers_count > 0:
             print(f"Number of extra layers found in refactored results: {extra_layers_count}")

def run_benchmark():
    """Runs the backtrace benchmarking."""
    print("Starting benchmark...")

    # --- 1. Data Loading ---
    # Using a small batch size for individual backtrace relevance calculation
    # For VGG19, typically batch_size of 1 is used for explainability if memory is a concern for activations
    # Or a slightly larger one if just benchmarking throughput for the backtrace step itself.
    # Let's use 1 for now to focus on per-sample processing.
    batch_size_for_backtrace = 1
    cifar_loader = get_cifar10_data(batch_size=batch_size_for_backtrace, image_size=224)

    # --- 2. Model Initialization ---
    num_classes_cifar = 10
    print(f"Loading VGG19 model for {num_classes_cifar} classes (CIFAR10)...")
    # Load VGG19, try with pretrained weights adapted for 10 classes
    # The load_pretrained_vgg function handles re-initializing the final layer.
    model = load_pretrained_vgg(num_classes=num_classes_cifar)
    model.eval() # Set model to evaluation mode

    # If GPU is available, move model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # --- 3. Backtrace Object Initialization ---
    # The BacktracePyTorch class itself does not seem to have a model_type for VGG
    # It might default or infer. We should verify its __init__ logic if issues arise.
    # For now, assuming the default init path for general PyTorch models is suitable.
    print("Initializing BacktracePyTorch object...")
    # activation_dict might be needed if custom activations are used or specific handling is required.
    # For standard VGG with ReLUs, the automatic detection or default might be okay.
    # Let's pass an empty dict for now, as per some examples.
    backtrace_obj = Backtrace(model=model, activation_dict={})
    print("BacktracePyTorch object initialized.")

    # --- 4. Benchmarking ---
    # Get a single batch of data
    try:
        inputs, labels = next(iter(cifar_loader))
        inputs = inputs.to(device)
        labels = labels.to(device)
    except StopIteration:
        print("Error: CIFAR10 DataLoader is empty. Check dataset path or download.")
        return
    
    print(f"\nProcessing one batch of size {inputs.shape[0]}...")

    # Get model outputs and all layer activations
    # The `predict` method in BacktracePyTorch gets all layer activations.
    # It internally calls model(inputs) via self.all_out_model(inputs).
    print("Getting model predictions and layer activations...")
    all_layer_activations = backtrace_obj.predict(inputs) 
    # print("Activations obtained for layers:", list(all_layer_activations.keys()))

    # Relevance propagation parameters (can be adjusted)
    multiplier = 100.0
    scaler = 0
    max_unit = 0 # Set to >0 for normalization, 0 for raw scores
    thresholding = 0.5
    task = "classification" # Or "binary-classification" if appropriate

    # Run with original handlers
    print("\nRunning with ORIGINAL handlers...")
    start_time_orig = time.time()
    relevance_original = backtrace_obj.eval(
        all_out=all_layer_activations,
        mode="default", # proportional_eval
        multiplier=multiplier,
        scaler=scaler,
        max_unit=max_unit,
        thresholding=thresholding,
        task=task,
        use_refactored_handlers=False
    )
    end_time_orig = time.time()
    time_original = end_time_orig - start_time_orig
    print(f"Time taken with ORIGINAL handlers: {time_original:.4f} seconds")
    # print("Relevance keys (original):", list(relevance_original.keys()))


    # Run with refactored handlers
    print("\nRunning with REFACTORED handlers...")
    start_time_refac = time.time()
    relevance_refactored = backtrace_obj.eval(
        all_out=all_layer_activations,
        mode="default", # proportional_eval
        multiplier=multiplier,
        scaler=scaler,
        max_unit=max_unit,
        thresholding=thresholding,
        task=task,
        use_refactored_handlers=True
    )
    end_time_refac = time.time()
    time_refactored = end_time_refac - start_time_refac
    print(f"Time taken with REFACTORED handlers: {time_refactored:.4f} seconds")
    # print("Relevance keys (refactored):", list(relevance_refactored.keys()))

    # --- 5. Comparison ---
    print("\n--- Comparing Relevance Results ---")
    compare_relevance_results(relevance_original, relevance_refactored)

    print("\nBenchmark finished.")
    if time_original > 0: # Avoid division by zero
        speedup = time_original / time_refactored if time_refactored > 0 else float('inf')
        print(f"Speedup with refactored handlers: {speedup:.2f}x")

if __name__ == '__main__':
    run_benchmark() 
 