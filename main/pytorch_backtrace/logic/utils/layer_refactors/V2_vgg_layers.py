import math
import torch
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional, Sequence, Any, Callable, Dict

def calculate_padding_pytorch(
    kernel_size: Tuple[int, int],
    inp_shape_WHC: Tuple[int, int, int],
    padding_mode_str: Union[str, int, Tuple[int, int]],
    strides_WH: Tuple[int, int],
    const_val: float = 0.0  # Retained for signature compatibility, though unused
) -> Tuple[Tuple[int, int, int], List[List[int]]]:
    """
    Calculates padding for a 3D tensor (W,H,C) based on conventions similar to those
    found in libraries like PyTorch. Optimized for clarity and potential compilation.

    Args:
        kernel_size: Tuple (kW, kH) representing kernel width and height.
                     Both kW and kH must be positive integers.
        inp_shape_WHC: Tuple (W, H, C) representing the input tensor's width, height,
                       and channel dimensions. W, H, C must be positive integers.
        padding_mode_str: Defines the padding scheme. Can be:
                          - 'valid': No padding is applied.
                          - 'same': Padding is calculated such that the output spatial
                            dimensions are ceil(input_dim / stride_dim).
                          - int: A single integer value `p`, used as symmetric padding
                            for both width and height (p_W = p, p_H = p). Must be non-negative.
                          - Tuple[int, int]: A tuple `(pad_W, pad_H)` for symmetric
                            padding on width and height respectively. Values must be non-negative.
        strides_WH: Tuple (sW, sH) representing strides for width and height.
                    Both sW and sH must be positive integers.
        const_val: Value intended for padding if padding were applied by this function.
                   Note: this function only calculates padding amounts and the
                   resulting shape; it does not apply padding. This parameter is
                   retained for signature compatibility.

    Returns:
        Tuple: (padded_input_shape_WHC, final_paddings_list)
          - padded_input_shape_WHC: Tuple (new_W, new_H, C_original) representing
            the dimensions of the input if it were padded.
          - final_paddings_list: A list of lists in the format
            `[[pad_W_before, pad_W_after], [pad_H_before, pad_H_after], [0,0]]`,
            detailing the padding amounts for width, height, and channel dimensions.
            The channel dimension is never padded.

    Raises:
        ValueError: If kernel_size or strides_WH elements are not positive integers.
        ValueError: If padding_mode_str is an int or tuple with negative values,
                    or if tuple is not of length 2.
        ValueError: If padding_mode_str is an unrecognized string.
    """
    # --- Input Validation (preserved from original) ---
    if not (isinstance(inp_shape_WHC, tuple) and len(inp_shape_WHC) == 3 and
            all(isinstance(dim, int) and dim > 0 for dim in inp_shape_WHC)):
        raise ValueError("inp_shape_WHC must be a tuple of 3 positive integers (W, H, C).")

    in_W, in_H, in_C = inp_shape_WHC

    if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2 and
            isinstance(kernel_size[0], int) and kernel_size[0] > 0 and
            isinstance(kernel_size[1], int) and kernel_size[1] > 0):
        raise ValueError("kernel_size must be a tuple of 2 positive integers (kW, kH).")
    k_W, k_H = kernel_size

    if not (isinstance(strides_WH, tuple) and len(strides_WH) == 2 and
            isinstance(strides_WH[0], int) and strides_WH[0] > 0 and
            isinstance(strides_WH[1], int) and strides_WH[1] > 0):
        raise ValueError("strides_WH must be a tuple of 2 positive integers (sW, sH).")
    s_W, s_H = strides_WH

    # --- Padding Calculation (logic preserved, 'same' mode arithmetic optimized) ---
    if padding_mode_str == 'valid':
        pad_W_before, pad_W_after = 0, 0
        pad_H_before, pad_H_after = 0, 0
    elif padding_mode_str == 'same':
        # Optimized integer arithmetic for ceil(A/B) with positive A, B: (A + B - 1) // B
        # This replaces math.ceil(float(A)/float(B)) and avoids float conversions.
        out_W = (in_W + s_W - 1) // s_W
        out_H = (in_H + s_H - 1) // s_H

        # Calculation of total padding. All operations are integer.
        # The result of max(0, <integer_expression>) is an integer.
        # Original int() cast on max() result is now redundant and removed.
        pad_W_total = max(0, (out_W - 1) * s_W + k_W - in_W)
        pad_H_total = max(0, (out_H - 1) * s_H + k_H - in_H)

        pad_W_before = pad_W_total // 2
        pad_W_after = pad_W_total - pad_W_before
        pad_H_before = pad_H_total // 2
        pad_H_after = pad_H_total - pad_H_before
    elif isinstance(padding_mode_str, int):
        if padding_mode_str < 0:
            raise ValueError("Integer padding value must be non-negative.")
        # Symmetric padding based on the single integer value
        pad_W_before, pad_W_after = padding_mode_str, padding_mode_str
        pad_H_before, pad_H_after = padding_mode_str, padding_mode_str
    elif isinstance(padding_mode_str, tuple):
        if len(padding_mode_str) != 2:
            raise ValueError(f"Tuple padding_mode_str must be (pad_W, pad_H) containing two integers, got {padding_mode_str}")
        pad_W_explicit, pad_H_explicit = padding_mode_str
        
        if not (isinstance(pad_W_explicit, int) and pad_W_explicit >= 0 and \
                isinstance(pad_H_explicit, int) and pad_H_explicit >= 0):
            raise ValueError(f"Padding values in tuple (pad_W, pad_H) must be non-negative integers, got {(pad_W_explicit, pad_H_explicit)}")
        
        # Symmetric padding based on tuple values
        pad_W_before, pad_W_after = pad_W_explicit, pad_W_explicit
        pad_H_before, pad_H_after = pad_H_explicit, pad_H_explicit
    else:
        raise ValueError(f"Unknown padding mode: {padding_mode_str}. Must be 'valid', 'same', an int, or a tuple (pad_W, pad_H).")

    # --- Formatting Output (preserved from original) ---
    final_paddings_list = [
        [pad_W_before, pad_W_after],
        [pad_H_before, pad_H_after],
        [0, 0]  # No padding for channels dimension
    ]
    
    padded_shape_W = in_W + pad_W_before + pad_W_after
    padded_shape_H = in_H + pad_H_before + pad_H_after
    
    # Original used int() casts here. These are technically redundant if all inputs
    # and intermediate padding values are integers (which they are with the optimization).
    # Kept for strict adherence to original output format details if there was a subtle reason,
    # though their removal would be safe.
    padded_input_shape_WHC = (int(padded_shape_W), int(padded_shape_H), in_C)

    return padded_input_shape_WHC, final_paddings_list

# For potential performance improvement when called frequently in a PyTorch context:
# import torch # if not already imported
# compiled_calculate_padding = torch.compile(calculate_padding_pytorch_optimized, mode="reduce-overhead")
# Or, for potentially more autotuning (but check numerical results carefully, though simple for this func):
# compiled_calculate_padding = torch.compile(calculate_padding_pytorch_optimized, mode="max-autotune-no-cudagraphs")

def calculate_wt_max_unit_pytorch(
    patch_batch: torch.Tensor, # Shape: (B, H, W, C)
    wts_batch: torch.Tensor,   # Shape: (B, C)
    pool_size_wh: Tuple[int, int] # Kernel (Width, Height) - for signature consistency, unused by max_unit
) -> torch.Tensor: # Shape: (B, H, W, C), dtype=torch.float64
    """
    Batched version of calculate_wt_max_unit_pytorch.
    Preserves float64 promotions and NaN/inf propagation.
    `pool_size_wh` is (KernelWidth, KernelHeight), passed for signature consistency.
    """
    B, H, W, C_dim = patch_batch.shape

    max_per_channel_batch = torch.amax(patch_batch, dim=(1, 2), keepdim=True) # (B, 1, 1, C)
    is_max_location_batch = (patch_batch == max_per_channel_batch) # (B,H,W,C), bool

    count_max_locations_f32_batch = torch.sum(is_max_location_batch, dim=(1, 2), keepdim=True) # (B,1,1,C), float32
    
    inverse_counts_f32_batch = 1.0 / count_max_locations_f32_batch # (B,1,1,C), float64

    normalized_max_mask_f32_batch = is_max_location_batch * inverse_counts_f32_batch # (B,H,W,C), float64

    weights_reshaped_f32_batch = wts_batch.view(B, 1, 1, C_dim) # (B,1,1,C), float64
    
    output_tensor_batch = normalized_max_mask_f32_batch * weights_reshaped_f32_batch # (B,H,W,C), float64
    return output_tensor_batch

def calculate_wt_maxpool_pytorch(
    wts: torch.Tensor,
    inp: torch.Tensor,
    pool_size_tpl: Union[int, Tuple[int, int]],
    padding_mode_str_or_int: Union[str, int, Tuple[int, int]],
    stride_int: Union[int, Tuple[int, int]]
) -> torch.Tensor:
    """ PyTorch port of relevance propagation for MaxPool2D.
    Args:
        wts: Relevance from output. Expected (N,C,OutH,OutW) or (C,OutH,OutW).
        inp: Input activations. Expected (N,C,InH,InW) or (C,InH,InW).
        pool_size_tpl: Kernel size. PyTorch style (kH, kW) or int.
        padding_mode_str_or_int: Padding mode string ('same', 'valid'), 
                                 int for symmetric W/H padding,
                                 or tuple (pad_W, pad_H) for symmetric W/H padding.
        stride_int: Stride. PyTorch style (sH, sW) or int.
    Returns:
        Propagated relevance to input. Shape matches input `inp` (after squeeze/unsqueeze).
    """
     # --- Device and Initial Input Handling ---
    device = inp.device
    wts_on_device = wts.to(device)

    original_wts_ndim = wts_on_device.ndim
    
    # Standardize inputs to (N,C,H,W) for PyTorch operations
    current_inp = inp
    current_wts = wts_on_device
    if current_inp.ndim == 3: 
        current_inp = current_inp.unsqueeze(0)
    if current_wts.ndim == 3:
        current_wts = current_wts.unsqueeze(0)
    
    N, C, InH, InW = current_inp.shape
    # L (number of patches) will be OutH * OutW. These are derived by F.unfold.
    # We can also get OutH, OutW from current_wts to calculate L for reshaping wts.
    _N_wts, _C_wts, OutH_wts, OutW_wts = current_wts.shape
    L = OutH_wts * OutW_wts

    # --- Parse Kernel and Stride (to PyTorch convention: kH, kW, sH, sW) ---
    if isinstance(pool_size_tpl, int):
        kH, kW = pool_size_tpl, pool_size_tpl # Kernel Height, Kernel Width
    else: # tuple (kH_pytorch, kW_pytorch)
        kH, kW = pool_size_tpl[0], pool_size_tpl[1]
        
    if isinstance(stride_int, int):
        sH, sW = stride_int, stride_int # Stride Height, Stride Width
    else: # tuple (sH_pytorch, sW_pytorch)
        sH, sW = stride_int[0], stride_int[1]

    # --- Padding Calculation ---
    # `calculate_padding_pytorch` expects kernel (width,height) and input_shape (W,H,C)
    inp_sample_whc_shape = (InW, InH, C)
    kernel_wh_for_calc_pad = (kW, kH) # (KernelWidth, KernelHeight)
    stride_wh_for_calc_pad = (sW, sH) # (StrideWidth, StrideHeight)
    
    _padded_shape_whc, np_paddings_list = calculate_padding_pytorch(
        kernel_size=kernel_wh_for_calc_pad,
        inp_shape_WHC=inp_sample_whc_shape,
        padding_mode_str=padding_mode_str_or_int, 
        strides_WH=stride_wh_for_calc_pad
    )
    pad_W_before, pad_W_after = np_paddings_list[0]
    pad_H_before, pad_H_after = np_paddings_list[1]
    
    # F.pad for (N,C,H,W) needs (pad_left, pad_right, pad_top, pad_bottom)
    torch_pad_dims = (pad_W_before, pad_W_after, pad_H_before, pad_H_after)
    
    try:
        input_padded_bchw = F.pad(current_inp, torch_pad_dims, 'constant', value=float('-inf'))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"CUDA out of memory during F.pad in calculate_wt_maxpool_pytorch.")
            print(f"  Input shape: {current_inp.shape}")
            print(f"  Padding dimensions: {torch_pad_dims}")
        raise e
    # Padded shape: (N, C, InH_pad, InW_pad)
    _N_pad, _C_pad, InH_pad, InW_pad = input_padded_bchw.shape

    # --- Unfold input to get patches ---
    # F.unfold kernel_size=(kH,kW), stride=(sH,sW) (PyTorch H,W convention)
    try:
        patches_unfolded = F.unfold(input_padded_bchw, kernel_size=(kH, kW), stride=(sH, sW))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"CUDA out of memory during F.unfold in calculate_wt_maxpool_pytorch.")
            print(f"  Padded input shape: {input_padded_bchw.shape}")
            print(f"  Kernel size: {(kH, kW)}, Stride: {(sH, sW)}")
        raise e
    # patches_unfolded shape: (N, C * kH * kW, L)
    num_blocks_L = patches_unfolded.shape[2]

    # Reshape patches for _calculate_wt_max_unit_pytorch
    patches = patches_unfolded.view(N, C, kH * kW, num_blocks_L).permute(0, 3, 2, 1).reshape(N * num_blocks_L, kH, kW, C)

    # 1. Reshape to (N, C, kH, kW, L)
    patches_temp_NCkHkWL = patches_unfolded.view(N, C, kH, kW, num_blocks_L)
    # 2. Permute to (N, L, kH, kW, C) - to make N*L the batch dim
    patches_permuted_NLkHkWC = patches_temp_NCkHkWL.permute(0, 4, 2, 3, 1)
    # 3. Reshape to (N*L, kH, kW, C)
    patches_for_batch_fn = patches_permuted_NLkHkWC.reshape(N * num_blocks_L, kH, kW, C)
    
    # --- Prepare wts for batched call: (N*L, C) ---
    # current_wts is (N, C, OutH_wts, OutW_wts)
    wts_permuted = current_wts.permute(0, 2, 3, 1) # (N, OutH_wts, OutW_wts, C)
    wts_for_batch_fn = wts_permuted.reshape(N * num_blocks_L, C)

    # --- Call batched max_unit calculation ---
    # Original loop passed (k_W, k_H) i.e. (KernelWidth, KernelHeight) to calculate_wt_max_unit
    pool_size_arg_for_batched_fn = (kW, kH) # (KernelWidth, KernelHeight)

    try:
        propagated_relevance_patches_batched = calculate_wt_max_unit_pytorch(
            patch_batch=patches_for_batch_fn, 
            wts_batch=wts_for_batch_fn,
            pool_size_wh=pool_size_arg_for_batched_fn
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"CUDA out of memory during call to calculate_wt_max_unit_pytorch from calculate_wt_maxpool_pytorch.")
            print(f"  Patches shape for batch_fn: {patches_for_batch_fn.shape}")
            print(f"  Weights shape for batch_fn: {wts_for_batch_fn.shape}")
        # We might also want to catch OOM inside calculate_wt_max_unit_pytorch itself.
        # For now, catching it at the call site.
        raise e
    
    # propagated_relevance_patches_batched shape: (N*L, kH, kW, C)

    del patches_unfolded, patches, patches_permuted_NLkHkWC, patches_for_batch_fn
    del wts_permuted, wts_for_batch_fn
    if 'cuda' in device.type: torch.cuda.empty_cache()
    
    # --- Fold updates back ---
    # Reshape updates_batch from (N*L, kH, kW, C) to (N, C*kH*kW, L) for F.fold
    updates_reshaped = propagated_relevance_patches_batched.reshape(N, num_blocks_L, kH, kW, C)
    updates_permuted_for_fold = updates_reshaped.permute(0, 4, 2, 3, 1) # (N, C, kH, kW, L)
    updates_for_fold = updates_permuted_for_fold.reshape(N, C * kH * kW, num_blocks_L)

    # F.fold output_size is (H_pad, W_pad)
    output_padded_bchw = F.fold(
        updates_for_fold, 
        output_size=(InH_pad, InW_pad), 
        kernel_size=(kH, kW), 
        stride=(sH, sW)
    )
    # output_padded_bchw shape: (N, C, InH_pad, InW_pad)

    del propagated_relevance_patches_batched, updates_reshaped, updates_permuted_for_fold, updates_for_fold
    if 'cuda' in device.type: torch.cuda.empty_cache()

    # --- Unpad ---
    unpadded_bchw = output_padded_bchw[
        :, :, 
        pad_H_before : InH_pad - pad_H_after,
        pad_W_before : InW_pad - pad_W_after
    ] # Shape: (N, C, InH, InW)

    # --- Final Shape Adjustment based on original wts_ndim ---
    final_relevance = unpadded_bchw
    if original_wts_ndim == 3:
        final_relevance = final_relevance.squeeze(0) # (C, InH, InW)
    
    return final_relevance

# For potential performance improvement by reducing Python overhead:
# import torch
# compiled_calculate_wt_maxpool = torch.compile(calculate_wt_maxpool_pytorch_optimized, mode="reduce-overhead")

def calculate_wt_conv_unit_pytorch(
    patch_hwi: torch.Tensor,    # Input patch, shape (kH, kW, C_in)
    wts_o: torch.Tensor,        # Relevance from output layer, shape (C_out,)
    w_oihw: torch.Tensor,       # Kernel weights, shape (C_out, C_in, kH, kW)
    b_o: torch.Tensor,          # Bias, shape (C_out,)
    act: Dict[str, Any]         # Activation configuration dictionary
) -> torch.Tensor:
    """
    Optimized PyTorch port of calculate_wt_conv_unit. Calculates relevance contribution for a conv unit.
    Maintains numerical equivalence with the original, including epsilon stabilization and
    positive/negative pathway handling.

    Args:
        patch_hwi: Input patch tensor with shape (kH, kW, C_in). Assumed on target device.
        wts_o: Relevance from the output channels, shape (C_out,). Assumed on target device.
        w_oihw: Kernel weights tensor, shape (C_out, C_in, kH, kW). Assumed on target device.
        b_o: Bias tensor, shape (C_out,). Assumed on target device.
        act: Activation configuration dictionary.

    Returns:
        torch.Tensor: Relevance propagated to the patch, shape (kH, kW, C_in).
    """
    device = patch_hwi.device
    target_dtype = patch_hwi.dtype
    
    num_patches = patch_hwi.shape[0]
    C_out = w_oihw.shape[0]

    # Ensure all tensors are on the same device and have the target_dtype.
    # .to() is a no-op if already correct.
    kernel_oihw = w_oihw.to(device=device, dtype=target_dtype)
    bias_o = b_o.to(device=device, dtype=target_dtype)
    rel_wts_o = wts_o.to(device=device, dtype=target_dtype) # Renamed for clarity

    # --- Bias processing (positive and absolute negative parts) ---
    # Preserves original logic.
    bias_pos_o = torch.relu(bias_o)         # Shape (C_out,)
    bias_neg_o = torch.relu(-bias_o)        # Shape (C_out,), abs of negative part
    
    # --- Prepare patch for multiplication with kernel ---
    # patch_hwi (kH, kW, C_in) -> permute to (C_in, kH, kW) -> unsqueeze to (1, C_in, kH, kW)
    # This matches PyTorch's NCHW convention for the "batch" of one patch.
    patch_1ihw = patch_hwi.permute(0, 3, 1, 2)

    # --- Element-wise product of kernel and patch (broadcasted) ---
    # kernel_oihw (C_out, C_in, kH, kW) * patch_1ihw (num_patches, C_in, kH, kW)
    # Expected result: (num_patches, C_out, C_in, kH, kW)
    # These are the individual W_ijk * X_k terms before summing over input channels/spatial.
    try:
        # kernel_oihw needs to be (1, C_out, C_in, kH, kW)
        # patch_1ihw needs to be (num_patches, 1, C_in, kH, kW)
        individual_contrib_oihw = kernel_oihw.unsqueeze(0) * patch_1ihw.unsqueeze(1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"CUDA out of memory during element-wise product in calculate_wt_conv_unit_pytorch.")
            print(f"  Kernel shape (after unsqueeze): {kernel_oihw.unsqueeze(0).shape}")
            print(f"  Patch shape (after unsqueeze): {patch_1ihw.unsqueeze(1).shape}")
        # Re-raise the original error if it's not OOM or to propagate OOM after printing
        raise e

    # --- Positive and negative contributions from individual_contrib_oihw ---
    # p_ind_contrib_oihw gets max(0, individual_contrib_oihw)
    p_ind_contrib_oihw = torch.relu(individual_contrib_oihw)
    # n_ind_contrib_oihw gets min(0, individual_contrib_oihw)
    n_ind_contrib_oihw = torch.min(individual_contrib_oihw, torch.tensor(0.0, device=device, dtype=target_dtype))

    # --- Sums of contributions per output channel (sum over C_in, kH, kW dimensions) ---
    # p_sum_o is sum of all positive (W_ijk * X_k) terms for each output neuron j.
    p_sum_o = torch.sum(p_ind_contrib_oihw, dim=(2, 3, 4)) # Shape (C_out,)
    # n_sum_o is sum of absolute values of all negative (W_ijk * X_k) terms for each output neuron j.
    # Original `* -1.0` makes the sum of negative terms positive.
    n_sum_o = torch.sum(n_ind_contrib_oihw, dim=(2, 3, 4)) * -1.0 # Shape (C_out,)

    # --- Saturation logic ---
    # actual_z_o_pre_bias_o is the sum of all (W_ijk * X_k) for each output neuron j, before bias.
    actual_z_o_pre_bias_o = torch.sum(individual_contrib_oihw, dim=(2, 3, 4)) # Shape (C_out,)
    # total_output_activation_estimate_o is Z_j + B_j
    total_output_activation_estimate_o = actual_z_o_pre_bias_o + bias_o.unsqueeze(0) # Shape (C_out,)

    # Initialize saturation flags (True means "pass"/not saturated initially)
    p_pass_o = torch.ones_like(p_sum_o, dtype=torch.bool, device=device)
    n_pass_o = torch.ones_like(n_sum_o, dtype=torch.bool, device=device)

    act_type = act.get("type")
    if act_type == 'mono':
        act_range = act.get("range", {})
        lower_bound = act_range.get("l")
        upper_bound = act_range.get("u")
        
        # If lower_bound exists and is a number, apply saturation for positive contributions.
        if lower_bound is not None:
            try:
                # Ensure bound is float for comparison.
                # Original code used float(lower_bound) implying it could be string or other type.
                lb_val = float(lower_bound) 
                p_pass_o = (total_output_activation_estimate_o >= lb_val)
            except (ValueError, TypeError):
                # Handle case where lower_bound is not convertible to float, or missing.
                # Original code did not explicitly handle this, so we maintain no change if unconvertible.
                pass 
                
        # If upper_bound exists and is a number, apply saturation for negative contributions.
        if upper_bound is not None:
            try:
                ub_val = float(upper_bound)
                n_pass_o = (total_output_activation_estimate_o <= ub_val)
            except (ValueError, TypeError):
                pass
    # Non-mono saturation logic is explicitly skipped, matching original.

    # Convert boolean saturation flags to float for multiplication, using target_dtype.
    p_pass_o_float = p_pass_o.to(target_dtype)
    n_pass_o_float = n_pass_o.to(target_dtype)

    # --- Denominator calculation (stabilized) ---
    # Sum of terms contributing to the LRP denominator for each output neuron.
    # These are the positive parts of (activations + bias) and negative parts of (activations + bias),
    # gated by saturation flags.
    # (p_sum_o * p_pass_o_float) = sum positive activations, if not saturated low
    # (bias_pos_o * p_pass_o_float) = positive bias, if not saturated low
    # (n_sum_o * n_pass_o_float) = sum abs negative activations, if not saturated high
    # (bias_neg_o * n_pass_o_float) = abs negative bias, if not saturated high
    denominator_terms_sum_o = (p_sum_o * p_pass_o_float) + \
                              (bias_pos_o.unsqueeze(0) * p_pass_o_float) + \
                              (n_sum_o * n_pass_o_float) + \
                              (bias_neg_o.unsqueeze(0) * n_pass_o_float)

    # Epsilon for stabilization. Ensure its dtype matches denominator_terms_sum_o.
    epsilon_val = 1e-9 # Original epsilon value

    denominator_o = denominator_terms_sum_o + torch.tensor(epsilon_val, device=device, dtype=target_dtype)
    
    # Handle cases where denominator_o might be zero even after adding epsilon (if epsilon is too small or target_dtype is low-precision).
    # To ensure division by non-zero, replace true zeros with epsilon itself.
    # This is a common stabilization for LRP to prevent division by zero and NaNs.
    # This step ensures we don't divide by a value smaller (in magnitude) than epsilon.
    is_zero_denom = (denominator_o == 0.0)
    denominator_o_safe = torch.where(is_zero_denom, 
                                     torch.tensor(epsilon_val, device=device, dtype=target_dtype), 
                                     denominator_o)

    # --- Calculate relevance fraction ---
    # rel_fraction_o = R_j / (Z_j_stabilized)
    rel_fraction_o = rel_wts_o / denominator_o_safe # Shape (C_out,)

    # Reshape relevance fraction components for broadcasting with (C_out, C_in, kH, kW) tensors.
    # Factors become (C_out, 1, 1, 1).
    # This is (R_j / Z_j_stab) * s_p_j, where s_p_j is the positive saturation pass flag.
    rel_frac_p_contrib_factor = (rel_fraction_o * p_pass_o_float).view(num_patches, C_out, 1, 1, 1)
    # This is (R_j / Z_j_stab) * s_n_j, where s_n_j is the negative saturation pass flag.
    rel_frac_n_contrib_factor = (rel_fraction_o * n_pass_o_float).view(num_patches, C_out, 1, 1, 1)

    # --- Distribute relevance back to individual contributions ---
    # Relevance from positive contributions: (W_ijk * X_k)_pos * (R_j / Z_j_stab) * s_p_j
    wt_mat_pos_path_oihw = p_ind_contrib_oihw * rel_frac_p_contrib_factor
    # Relevance from negative contributions: (W_ijk * X_k)_neg * (R_j / Z_j_stab) * s_n_j
    # n_ind_contrib_oihw is already negative-valued (or zero).
    wt_mat_neg_path_oihw = n_ind_contrib_oihw * rel_frac_n_contrib_factor
    
    # Total relevance propagated through each W_ijk * X_k term.
    wt_mat_oihw = wt_mat_pos_path_oihw + wt_mat_neg_path_oihw # Shape (C_out, C_in, kH, kW)

    # --- Sum relevance contributions over output channels (dim 0) to get relevance for each input X_k ---
    # R_k = sum_j (relevance propagated through W_ijk * X_k)
    relevance_for_patch_ihw = torch.sum(wt_mat_oihw, dim=1) # Shape (C_in, kH, kW)
    
    # Transpose to final shape (kH, kW, C_in) to match input patch_hwi format.
    relevance_for_patch_hwi = relevance_for_patch_ihw.permute(0, 2, 3, 1)
    
    return relevance_for_patch_hwi

# For potential performance improvement by reducing Python overhead:
# import torch
# compiled_calculate_wt_conv_unit = torch.compile(calculate_wt_conv_unit_pytorch_optimized, mode="reduce-overhead")

def calculate_wt_conv_pytorch(
    wts: torch.Tensor,          # Output relevance (N,Cout,Hout,Wout) or (Cout,Hout,Wout)
    inp: torch.Tensor,          # Input activations (N,Cin,Hin,Win) or (Cin,Hin,Win)
    kernel_weights: torch.Tensor, # Kernel (Cout, Cin/groups, kH, kW)
    bias: torch.Tensor,         # Bias (Cout,)
    padding_config: Union[str, int, Tuple[int, int]],
    strides_config: Tuple[int, int], # PyTorch style (sH, sW)
    act: Dict[str, Any]
) -> torch.Tensor:
    """ 
    Optimized LRP for Conv2D using unfold/fold and chunking (assumes groups=1).
    """
    device = inp.device
    original_inp_ndim = inp.ndim

    # --- Standardize inputs to NCHW and ensure on device ---
    current_inp = inp.to(device)
    if current_inp.ndim == 3: current_inp = current_inp.unsqueeze(0)
    
    current_wts = wts.to(device)
    if current_wts.ndim == 3: current_wts = current_wts.unsqueeze(0)
    
    N, Cin_main, InH, InW = current_inp.shape
    _N_wts, Cout_main, OutH, OutW = current_wts.shape # Renamed OutH_wts, OutW_wts

    # Kernel and bias will be passed to batched unit, which handles their device/dtype
    # For shape extraction:
    _Cout_k, Cin_k_actual, kH, kW = kernel_weights.shape

    # Basic check for groups=1 assumption (more robust checks might be needed for general groups)
    if Cin_main != Cin_k_actual:
        # This could be a depthwise conv if Cin_k_actual == 1 and Cout_k == Cin_main
        is_depthwise_compatible = (Cin_k_actual == 1 and _Cout_k == Cin_main)
        if not is_depthwise_compatible:
            raise ValueError(f"Input channels Cin_main ({Cin_main}) and kernel Cin_k_actual ({Cin_k_actual}) "
                             f"mismatch for current groups=1 or depthwise assumption. Kernel shape: {kernel_weights.shape}")
    
    sH, sW = strides_config[0], strides_config[1]

    # --- Padding Calculation (Input: WHC, Kernel: WH) ---
    inp_sample_whc_shape = (InW, InH, Cin_main)
    kernel_wh_for_calc_pad = (kW, kH)      
    stride_wh_for_calc_pad = (sW, sH)      
    
    _padded_shape_whc, np_paddings_list = calculate_padding_pytorch(
        kernel_size=kernel_wh_for_calc_pad,
        inp_shape_WHC=inp_sample_whc_shape,
        padding_mode_str=padding_config,
        strides_WH=stride_wh_for_calc_pad
        # const_val=0.0 # calculate_padding_pytorch has default 0.0
    )
    pad_W_before, pad_W_after = np_paddings_list[0]
    pad_H_before, pad_H_after = np_paddings_list[1]
    
    torch_pad_dims = (pad_W_before, pad_W_after, pad_H_before, pad_H_after)
    # Padding value for conv LRP is typically 0.0 (unlike -inf for maxpool)
    input_padded_bchw = F.pad(current_inp, torch_pad_dims, 'constant', value=0.0)
    _N_pad, _Cin_pad, PaddedH, PaddedW = input_padded_bchw.shape # Renamed InH_pad, InW_pad

    # --- Extract Patches using F.unfold ---
    # Output: (N, Cin_main * kH * kW, NumPatches)
    patches_unfolded = F.unfold(
        input_padded_bchw,
        kernel_size=(kH, kW),
        stride=(sH, sW),
        padding=(0,0) # Padding was manually applied
    )
    NumPatches = patches_unfolded.shape[2]

    if NumPatches != OutH * OutW:
        raise ValueError(f"Mismatch in number of patches: F.unfold got {NumPatches}, "
                         f"expected from output relevance shape {OutH*OutW}. "
                         f"InH={InH},InW={InW}, kH={kH},kW={kW}, sH={sH},sW={sW}, "
                         f"padH={pad_H_before}+{pad_H_after}, padW={pad_W_before}+{pad_W_after}, "
                         f"PaddedH={PaddedH}, PaddedW={PaddedW}, OutH={OutH}, OutW={OutW}")
    if NumPatches == 0:
        # If no patches, output relevance should match input shape but be zero.
        # The final output shape is current_inp.shape if original_inp_ndim matched current_inp.ndim before unsqueeze
        # or current_inp.squeeze(0).shape if original_inp_ndim was 3.
        final_zero_shape = inp.shape # Use original inp's shape
        return torch.zeros(final_zero_shape, dtype=torch.float64, device=device) # Return float64 as expected

    # Reshape patches for the batched unit function:
    # (N, Cin*KH*KW, NP) -> (N, Cin, KH, KW, NP) -> permute to (N, NP, KH, KW, Cin) -> reshape (N*NP, KH, KW, Cin)
    all_patches_temp_N_Cin_KH_KW_NP = patches_unfolded.view(N, Cin_main, kH, kW, NumPatches)
    all_patches_permuted_N_NP_KH_KW_Cin = all_patches_temp_N_Cin_KH_KW_NP.permute(0, 4, 2, 3, 1)
    all_patches_for_batched_unit = all_patches_permuted_N_NP_KH_KW_Cin.reshape(N * NumPatches, kH, kW, Cin_main)

    # Reshape output relevance (wts) for the batched unit function:
    # current_wts: (N, Cout, OutH, OutW) -> permute to (N, OutH, OutW, Cout) -> reshape (N*NP, Cout)
    output_relevance_permuted_N_OH_OW_Cout = current_wts.permute(0, 2, 3, 1)
    output_relevance_for_batched_unit = output_relevance_permuted_N_OH_OW_Cout.reshape(N * NumPatches, Cout_main)

    # --- Chunked Processing ---
    # Initialize accumulator for all patch relevances. It must be float64.
    accumulated_updates_NP_khw_inc = torch.zeros_like(all_patches_for_batched_unit, dtype=torch.float64, device=device)

    chunk_size = 32  # Increased from 32, adjust based on typical NumPatches and GPU memory
    if NumPatches < chunk_size * 2: # If less than 2 chunks, might as well do it in one
        chunk_size = NumPatches

    for i in range(0, N * NumPatches, chunk_size):
        patches_chunk = all_patches_for_batched_unit[i : i + chunk_size]
        relevance_chunk = output_relevance_for_batched_unit[i : i + chunk_size]
        
        chunk_updates_khw_inc = calculate_wt_conv_unit_pytorch(
            patch_hwi=patches_chunk,
            wts_o=relevance_chunk,
            w_oihw=kernel_weights,
            b_o=bias,
            act=act
        )
        accumulated_updates_NP_khw_inc[i : i + chunk_size] = chunk_updates_khw_inc
        
        # Optional: Aggressive cleanup for very memory sensitive cases during debug
        del patches_chunk, relevance_chunk, chunk_updates_khw_inc
        if device.type == 'cuda': torch.cuda.empty_cache()

    # `accumulated_updates_NP_khw_inc` is (N*NP, KH, KW, Cin), dtype float64
    # Prepare for F.fold: (N*NP, KH, KW, Cin) -> (N, NP, KH, KW, Cin)
    #                      -> permute (N, Cin, KH, KW, NP) -> reshape (N, Cin*KH*KW, NP)
    updates_temp_N_NP_KH_KW_Cin = accumulated_updates_NP_khw_inc.view(N, NumPatches, kH, kW, Cin_main)
    updates_permuted_N_Cin_KH_KW_NP = updates_temp_N_NP_KH_KW_Cin.permute(0, 4, 2, 3, 1)
    updates_for_fold = updates_permuted_N_Cin_KH_KW_NP.reshape(N, Cin_main * kH * kW, NumPatches)

    # --- Accumulate updates using F.fold ---
    output_canvas_b_cin_ph_pw = F.fold(
        updates_for_fold,
        output_size=(PaddedH, PaddedW), 
        kernel_size=(kH, kW),
        stride=(sH, sW),
        padding=(0,0) # F.fold's padding is for the output canvas, not input to unfold
    ) # Output dtype will be float64

    # --- Unpad: Crop the output canvas to original input dimensions ---
    # output_canvas is (N, Cin, PaddedH, PaddedW)
    final_output_b_cin_h_w = output_canvas_b_cin_ph_pw[
        :, :,  # All batches and channels
        pad_H_before : PaddedH - pad_H_after,
        pad_W_before : PaddedW - pad_W_after
    ]
    
    # Adjust final shape based on original input dimensionality
    if original_inp_ndim == 3:
        final_output_b_cin_h_w = final_output_b_cin_h_w.squeeze(0) # (Cin,Hin,Win)
    
    return final_output_b_cin_h_w

# For potential performance improvement:
# import torch
# compiled_calculate_wt_conv = torch.compile(calculate_wt_conv_pytorch_optimized, mode="reduce-overhead")

def calculate_wt_fc_pytorch(wts, inp, w, b, act, device=None):
    """
    Optimized calculation of relevance propagation for a linear layer using PyTorch.
    
    Parameters:
    -----------
    wts : Union[numpy.ndarray, torch.Tensor]
        Weights for relevance calculation. Expected shape (O,) or (1,O).
    inp : Union[numpy.ndarray, torch.Tensor]
        Input values. Expected shape (I,) or (1,I).
    w : Union[numpy.ndarray, torch.Tensor]
        Weight tensor of the layer. Expected shape (O,I).
    b : Union[numpy.ndarray, torch.Tensor]
        Bias tensor of the layer. Expected shape (O,).
    act : dict
        Activation function details.
        
    Returns:
    --------
    torch.Tensor
        Weighted matrix for relevance propagation, shape (I,).
    """
    # Determine device and dtype from model parameters (w) if available, else default
    if isinstance(w, torch.Tensor):
        target_device = w.device
        target_dtype = w.dtype
    elif isinstance(wts, torch.Tensor): # Fallback to wts if w is not tensor
        target_device = wts.device
        target_dtype = wts.dtype
    else: # All inputs might be numpy arrays
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_dtype = torch.float32 # Default PyTorch float type

    # Convert all inputs to torch.Tensor on the target_device with target_dtype
    if not isinstance(wts, torch.Tensor):
        wts = torch.tensor(wts, dtype=target_dtype, device=target_device)
    else:
        wts = wts.to(device=target_device, dtype=target_dtype)
        
    if not isinstance(inp, torch.Tensor):
        inp = torch.tensor(inp, dtype=target_dtype, device=target_device)
    else:
        inp = inp.to(device=target_device, dtype=target_dtype)
    
    if not isinstance(w, torch.Tensor):
        w_pt = torch.tensor(w, dtype=target_dtype, device=target_device)
    else:
        w_pt = w.to(device=target_device, dtype=target_dtype) # Use w_pt to avoid modifying input 'w' if it's a tensor passed by reference
    
    if not isinstance(b, torch.Tensor):
        b_pt = torch.tensor(b, dtype=target_dtype, device=target_device)
    else:
        b_pt = b.to(device=target_device, dtype=target_dtype)

    # Squeeze wts if it's (1,O) to (O,)
    if wts.ndim == 2 and wts.shape[0] == 1:
        wts = wts.squeeze(0)
    
    # Squeeze inp if it's (1,I) to (I,)
    if inp.ndim == 2 and inp.shape[0] == 1:
        inp = inp.squeeze(0) # Now shape (I,)
     
    # Calculate contribution matrix: mul_mat[j,i] = w[j,i] * inp[i]
    # w is (O,I), inp is (I,). inp.unsqueeze(0) makes it (1,I) for broadcasting.
    mul_mat = w * inp.unsqueeze(0) # Shape (O,I)

    # Pre-compute masks (used later for wt_mat construction and conditional logic)
    pos_mask_full = mul_mat > 0 # Shape (O,I)
    neg_mask_full = mul_mat < 0 # Shape (O,I)
     
    # Calculate sums of positive and negative contributions (more efficient using relu)
    # p_sums = sum_i max(0, mul_mat_ji) for each j
    # n_sums = sum_i max(0, -mul_mat_ji) for each j (sum of absolute values of negative parts)
    p_sums = torch.sum(torch.relu(mul_mat), dim=1) # Shape (O,)
    n_sums = torch.sum(torch.relu(-mul_mat), dim=1) # Shape (O,)
     
    # Split bias into positive and negative parts (using relu)
    p_bias = torch.relu(b) # Shape (O,)
    n_bias = torch.relu(-b) # Shape (O,)
     
    # Total sums (intermediate, used for activation decisions)
    # Preserving exact calculation order for floating point equivalence: ((A+B)-C)-D
    t_sums = p_sums + p_bias - n_sums - n_bias # Shape (O,)
     
    # Handle activation constraints
    act_type = act.get("type")
    act_range = act.get("range", {}) # Default to empty dict if "range" is missing

    if act_type == "mono":
        # Preserving original conditional logic for applying thresholds
        if "l" in act_range and act_range["l"]: # Checks key existence and truthiness of value
            l_threshold = act_range["l"]
            p_sums = torch.where(t_sums < l_threshold, torch.zeros_like(p_sums), p_sums)
            
        if "u" in act_range and act_range["u"]: # Checks key existence and truthiness of value
            u_threshold = act_range["u"]
            n_sums = torch.where(t_sums > u_threshold, torch.zeros_like(n_sums), n_sums)
            
    elif act_type == "non_mono":
        func = act.get("func")
        if func is None:
             # Original code didn't explicitly raise, but this is a critical missing piece.
             # Depending on strictness, either raise or ensure downstream handles it.
             # For now, assume func is present if type is non_mono, like original implies.
            pass

        if func is not None: # Proceed if func is actually available
            both_positive = (p_sums > 0) & (n_sums > 0) # boolean mask (O,)
            if torch.any(both_positive):
                indices = torch.where(both_positive)[0] # 1D tensor of integer indices
                
                if indices.numel() > 0: # Ensure indices is not empty before proceeding
                    t_sums_at_indices = t_sums[indices]
                    p_sums_arg_at_indices = p_sums[indices] + p_bias[indices]
                    n_sums_arg_at_indices = -1 * (n_sums[indices] + n_bias[indices])

                    # This loop with .item() is preserved for numerical equivalence with arbitrary Python func.
                    # Results are created on the correct device and dtype.
                    applied_t_acts_list = [func(x.item()) for x in t_sums_at_indices]
                    applied_p_acts_list = [func(x.item()) for x in p_sums_arg_at_indices]
                    applied_n_acts_list = [func(x.item()) for x in n_sums_arg_at_indices]

                    applied_t_acts = torch.tensor(applied_t_acts_list, device=target_device, dtype=target_dtype)
                    applied_p_acts = torch.tensor(applied_p_acts_list, device=target_device, dtype=target_dtype)
                    applied_n_acts = torch.tensor(applied_n_acts_list, device=target_device, dtype=target_dtype)
                    
                    p_sums_val_at_indices = p_sums[indices].clone() 
                    n_sums_val_at_indices = n_sums[indices].clone()

                    p_sums_val_at_indices[applied_t_acts == applied_n_acts] = 0
                    n_sums_val_at_indices[applied_t_acts == applied_p_acts] = 0
                    
                    # In-place update using advanced indexing
                    p_sums[indices] = p_sums_val_at_indices
                    n_sums[indices] = n_sums_val_at_indices
            
            # Apply range constraints AFTER func processing, as in original
            if "l" in act_range and act_range["l"]:
                l_threshold = act_range["l"]
                p_sums = torch.where(t_sums < l_threshold, torch.zeros_like(p_sums), p_sums)
                
            if "u" in act_range and act_range["u"]:
                u_threshold = act_range["u"]
                n_sums = torch.where(t_sums > u_threshold, torch.zeros_like(n_sums), n_sums)
    
    # Denominators for aggregation weights
    # Preserving exact calculation order: (((A+B)+C)+D)
    denominators = p_sums + n_sums + p_bias + n_bias # Shape (O,)
    # Stabilization: if denominator is 0, use 1 instead. Preserves original logic.
    safe_denominators = torch.where(denominators == 0, torch.ones_like(denominators), denominators)
    
    # Calculate aggregation weights
    p_sums_plus_bias = p_sums + p_bias # Pre-calculate for re-use
    n_sums_plus_bias = n_sums + n_bias # Pre-calculate for re-use

    # p_agg_wts - Preserving original factor calculation and stabilization
    p_factor1 = p_sums_plus_bias / safe_denominators
    p_factor2_val = p_sums / p_sums_plus_bias 
    p_factor2_val = torch.nan_to_num(p_factor2_val, nan=0.0) # Handles 0/0 -> 0
    p_factor2 = torch.where(p_sums_plus_bias != 0, p_factor2_val, torch.zeros_like(p_sums))
    
    mask_p_positive = p_sums > 0
    p_agg_wts = torch.where(mask_p_positive, p_factor1 * p_factor2, torch.zeros_like(p_sums))
     
    # n_agg_wts - Preserving original factor calculation and stabilization
    n_factor1 = n_sums_plus_bias / safe_denominators
    n_factor2_val = n_sums / n_sums_plus_bias
    n_factor2_val = torch.nan_to_num(n_factor2_val, nan=0.0) # Handles 0/0 -> 0
    n_factor2 = torch.where(n_sums_plus_bias != 0, n_factor2_val, torch.zeros_like(n_sums))

    mask_n_positive = n_sums > 0
    n_agg_wts = torch.where(mask_n_positive, n_factor1 * n_factor2, torch.zeros_like(n_sums))
     
    # Safe divisors for p_sums and n_sums (used in the final relevance distribution step)
    # Preserves original stabilization logic.
    p_sums_safe = torch.where(p_sums == 0, torch.ones_like(p_sums), p_sums)
    n_sums_safe = torch.where(n_sums == 0, torch.ones_like(n_sums), n_sums)
    
    # Pre-compute coefficients for relevance distribution
    p_coeffs = wts * p_agg_wts 
    n_coeffs = wts * n_agg_wts * -1.0 # Original logic including -1.0 factor
     
    # Vectorized calculation of wt_mat (replaces the loop over O)
    p_coeffs_exp = p_coeffs.unsqueeze(1)      # (O,1)
    n_coeffs_exp = n_coeffs.unsqueeze(1)      # (O,1)
    p_sums_safe_exp = p_sums_safe.unsqueeze(1) # (O,1)
    n_sums_safe_exp = n_sums_safe.unsqueeze(1) # (O,1)
    
    # Calculate positive contribution part: (mul_mat / sum_pos_contribs_safe) * relevance_coeffs_pos
    # This is applied only where mul_mat was originally positive.
    # If p_sums was 0, p_coeffs is 0, so term_p_calc effectively becomes 0 where relevant.
    term_p_calc = (mul_mat / p_sums_safe_exp) * p_coeffs_exp
    
    # Calculate negative contribution part: (mul_mat / sum_neg_contribs_abs_safe) * relevance_coeffs_neg
    # This is applied only where mul_mat was originally negative.
    # If n_sums was 0, n_coeffs is 0, so term_n_calc effectively becomes 0 where relevant.
    term_n_calc = (mul_mat / n_sums_safe_exp) * n_coeffs_exp

    # Combine contributions using masks.
    # wt_mat_pos_contrib will have non-zero values from term_p_calc where pos_mask_full is true, else zero.
    wt_mat_pos_contrib = torch.where(pos_mask_full, term_p_calc, torch.zeros_like(mul_mat))
    # wt_mat_neg_contrib will have non-zero values from term_n_calc where neg_mask_full is true, else zero.
    wt_mat_neg_contrib = torch.where(neg_mask_full, term_n_calc, torch.zeros_like(mul_mat))
    
    # Since pos_mask_full and neg_mask_full are disjoint for non-zero mul_mat,
    # adding these gives the final wt_mat correctly.
    wt_mat = wt_mat_pos_contrib + wt_mat_neg_contrib # Shape (O,I)
     
    # Sum relevance contributions for each input neuron
    return torch.sum(wt_mat, dim=0) # Result (I,)

# To use with torch.compile for potential further (modest) speedup, especially for the non_mono loop:
# compiled_calculate_wt_fc_pytorch_optimized = torch.compile(calculate_wt_fc_pytorch_optimized, mode="reduce-overhead")
# Or, for potentially more autotuning (but check numerical results carefully):
# compiled_calculate_wt_fc_pytorch_optimized = torch.compile(calculate_wt_fc_pytorch_optimized, mode="max-autotune-no-cudagraphs")

def calculate_wt_rshp_pytorch(
    wts: torch.Tensor,
    inp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reshapes an input PyTorch tensor 'wts' to match the shape of another PyTorch tensor 'inp'.
    This is a PyTorch port of the original NumPy function `calculate_wt_rshp`.

    The content of `inp` is not used, only its `shape`.
    The total number of elements in `wts` must be compatible
    with the total number of elements defined by `inp.shape`.
    The output tensor will be on the same device as the input `wts` tensor.

    Args:
        wts (torch.Tensor): The PyTorch tensor whose elements will be reshaped.
        inp (Optional[torch.Tensor], optional): A PyTorch tensor whose shape
            will be used as the target shape for reshaping `wts`.
            This argument is required (cannot be None in practice for reshaping).
            Defaults to None, matching the original NumPy function's signature,
            but will raise an error if None is passed.

    Returns:
        torch.Tensor: A new tensor (often a view) with the same data as
                      `wts` but with the shape of `inp`, on the same device
                      as `wts`.

    Raises:
        TypeError: If `wts` is not a PyTorch tensor, or if `inp` is provided
                   but is not a PyTorch tensor.
        ValueError: If `inp` is None, as target shape cannot be determined.
        RuntimeError: If reshaping is not possible due to incompatible sizes
                      (PyTorch typically raises RuntimeError for this).
    """
    if not isinstance(wts, torch.Tensor):
        raise TypeError(f"Input 'wts' must be a PyTorch tensor, got {type(wts)}")

    if inp is None:
        raise ValueError(
            "Input 'inp' cannot be None, as it is required to provide the target shape for reshaping."
        )
    
    if not isinstance(inp, torch.Tensor):
        raise TypeError(
            f"Input 'inp' must be a PyTorch tensor to provide the target shape, got {type(inp)}"
        )

    target_shape: torch.Size = inp.shape # Get the shape from the 'inp' tensor
    
    try:
        # Using .reshape() is a direct translation of np.reshape(wts, inp.shape)
        # Using .reshape_as(inp) is also a PyTorch-idiomatic way:
        # reshaped_tensor = wts.reshape_as(inp)
        reshaped_tensor = wts.reshape(target_shape)
    except RuntimeError as e:
        # PyTorch's reshape errors are usually quite descriptive.
        # Add more context for clarity.
        original_numel = wts.numel()
        target_numel = inp.numel() # Calculate numel from inp tensor
        raise RuntimeError(
            f"Cannot reshape tensor 'wts' of size {original_numel} (shape {wts.shape}) "
            f"into target shape {target_shape} (from 'inp', target numel {target_numel}). Original PyTorch error: {e}"
        ) from e
        
    return reshaped_tensor
