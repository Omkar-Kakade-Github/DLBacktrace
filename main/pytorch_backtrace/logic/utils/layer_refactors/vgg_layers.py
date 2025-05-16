import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional, Sequence, Any, Callable, Dict

def calculate_padding_pytorch_cuda(
    input_tensor: torch.Tensor,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    padding_mode: Union[str, Sequence[int], Sequence[Sequence[int]]],
    constant_value: float = 0.0,
    spatial_dims_indices: Optional[Sequence[int]] = None
) -> Tuple[torch.Tensor, List[List[int]]]:
    """Calculates and optionally applies padding to a PyTorch tensor.

    This function computes padding amounts based on the input tensor shape,
    kernel shape, strides, and padding mode. It supports 'valid', 'same',
    symmetric tuple-defined padding, and explicit padding per dimension.
    Padding is primarily calculated for two spatial dimensions, whose indices
    can be specified by `spatial_dims_indices`. Assumes tensor is on the
    target device (e.g., CUDA).

    Args:
        input_tensor (torch.Tensor): The input N-dimensional tensor.
        kernel_shape (Sequence[int]): A 2-element sequence `(kernel_height, kernel_width)`
            defining the kernel dimensions.
        strides (Sequence[int]): A 2-element sequence `(stride_height, stride_width)`
            defining the strides for the spatial dimensions.
        padding_mode (Union[str, Sequence[int], Sequence[Sequence[int]]]):
            The padding mode. Can be:
            - 'valid': No padding is applied.
            - 'same': Padding is calculated to ensure the output spatial dimensions
              (when performing an operation like convolution) are `ceil(input_dim / stride)`.
              Padding is applied to dimensions specified by `spatial_dims_indices`.
              Other dimensions receive no padding from this mode.
            - Sequence[int] of length 2 (e.g., `(pad_h_sym, pad_w_sym)`):
              Interpreted as symmetric padding values for the spatial dimensions.
              `pad_h_sym` is applied to both before/after of the height dimension,
              and `pad_w_sym` for the width dimension. Spatial dimensions are
              defined by `spatial_dims_indices`. Other dimensions get no padding.
            - Sequence[Sequence[int]] (e.g., `[[pt, pb], [pl, pr], ...]`):
              Explicit padding specification for each dimension of `input_tensor`.
              Length must match `input_tensor.ndim`. `spatial_dims_indices` is ignored.
              All padding values must be non-negative integers.
        constant_value (float, optional): The value used for padding. Defaults to 0.0.
        spatial_dims_indices (Optional[Sequence[int]], optional): A 2-element sequence
            `(dim_idx_h, dim_idx_w)` indicating which dimensions of `input_tensor`
            are considered height and width for 'same' or symmetric tuple padding.
            If None, defaults to `(0, 1)`. Negative indices are allowed.
            Ignored if `padding_mode` is an explicit list of lists.

    Returns:
        Tuple[torch.Tensor, List[List[int]]]:
            - The (potentially) padded PyTorch tensor, on the same device as input.
              If no padding is applied, the original tensor is returned.
            - A list of lists, where each inner list `[pad_before, pad_after]`
              specifies the integer padding applied to the corresponding dimension.

    Raises:
        ValueError: If `kernel_shape` or `strides` are not 2-element sequences of positive integers.
        ValueError: If `spatial_dims_indices` are invalid, out of bounds, or refer
                    to the same dimension.
        ValueError: If `padding_mode` is an explicit list of lists (`Sequence[Sequence[int]]`)
                    but its length does not match `input_tensor.ndim`, or if it
                    contains invalid padding pairs or negative values.
        ValueError: If `padding_mode` is a symmetric tuple (`Sequence[int]`) but
                    contains negative padding values.
    """
    if not (hasattr(kernel_shape, '__len__') and len(kernel_shape) == 2 and all(isinstance(k, int) for k in kernel_shape)):
        raise ValueError(f"kernel_shape must be a 2-element sequence of integers, got {kernel_shape}")
    if not (hasattr(strides, '__len__') and len(strides) == 2 and all(isinstance(s, int) and s > 0 for s in strides)):
        raise ValueError(f"strides must be a 2-element sequence of positive integers, got {strides}")

    if spatial_dims_indices is None:
        spatial_dims_indices = (0, 1)
    if not (hasattr(spatial_dims_indices, '__len__') and len(spatial_dims_indices) == 2 and all(isinstance(idx, int) for idx in spatial_dims_indices)):
        raise ValueError(f"spatial_dims_indices must be a 2-element sequence of integers, got {spatial_dims_indices}")
        
    dim_h_idx = spatial_dims_indices[0] if spatial_dims_indices[0] >= 0 else input_tensor.ndim + spatial_dims_indices[0]
    dim_w_idx = spatial_dims_indices[1] if spatial_dims_indices[1] >= 0 else input_tensor.ndim + spatial_dims_indices[1]

    if not (0 <= dim_h_idx < input_tensor.ndim and \
            0 <= dim_w_idx < input_tensor.ndim):
        raise ValueError(
            f"spatial_dims_indices ({spatial_dims_indices[0]}, {spatial_dims_indices[1]}) "
            f"are out of bounds for input_tensor with {input_tensor.ndim} dimensions."
        )
    if dim_h_idx == dim_w_idx:
        raise ValueError(
            f"spatial_dims_indices ({spatial_dims_indices[0]}, {spatial_dims_indices[1]}) "
            f"must refer to two different dimensions."
        )


    input_h = input_tensor.shape[dim_h_idx]
    input_w = input_tensor.shape[dim_w_idx]
    kernel_h, kernel_w = kernel_shape[0], kernel_shape[1]
    stride_h, stride_w = strides[0], strides[1]

    padding_amounts_list_logical = [[0, 0] for _ in range(input_tensor.ndim)]

    if isinstance(padding_mode, str):
        padding_mode_lower = padding_mode.lower()
        if padding_mode_lower == 'valid':
            pass # No padding
        elif padding_mode_lower == 'same':
            output_h = (input_h + stride_h - 1) // stride_h
            total_pad_h = max(0, (output_h - 1) * stride_h + kernel_h - input_h)
            padding_amounts_list_logical[dim_h_idx] = [total_pad_h // 2, total_pad_h - (total_pad_h // 2)]

            output_w = (input_w + stride_w - 1) // stride_w
            total_pad_w = max(0, (output_w - 1) * stride_w + kernel_w - input_w)
            padding_amounts_list_logical[dim_w_idx] = [total_pad_w // 2, total_pad_w - (total_pad_w // 2)]
        else:
            # Unknown string mode, treated as 'valid' (no padding)
            pass
    elif isinstance(padding_mode, Sequence):
        if not padding_mode: # Empty sequence
            pass # Treated as 'valid'
        elif isinstance(padding_mode[0], int): # Potential Sequence[int]
            if len(padding_mode) == 2 and all(isinstance(p, int) for p in padding_mode):
                pad_h_sym, pad_w_sym = padding_mode[0], padding_mode[1]
                if pad_h_sym < 0 or pad_w_sym < 0:
                    raise ValueError("Symmetric padding values in padding_mode tuple must be non-negative.")
                padding_amounts_list_logical[dim_h_idx] = [pad_h_sym, pad_h_sym]
                padding_amounts_list_logical[dim_w_idx] = [pad_w_sym, pad_w_sym]
            # else: malformed Sequence[int], treated as 'valid'
        elif isinstance(padding_mode[0], Sequence): # Potential Sequence[Sequence[int]]
            if len(padding_mode) != input_tensor.ndim:
                raise ValueError(
                    f"Explicit padding_mode (Sequence[Sequence[int]]) must have length "
                    f"equal to input_tensor.ndim ({input_tensor.ndim}), got {len(padding_mode)}."
                )
            temp_padding_list = []
            for i, p_pair in enumerate(padding_mode):
                if not (isinstance(p_pair, Sequence) and len(p_pair) == 2 and
                        all(isinstance(val, int) and val >= 0 for val in p_pair)):
                    raise ValueError(
                        f"Invalid padding pair at index {i}: {p_pair}. "
                        "Each pair must be [pad_before, pad_after] with non-negative integers."
                    )
                temp_padding_list.append([p_pair[0], p_pair[1]])
            padding_amounts_list_logical = temp_padding_list
        # else: Sequence of unknown element types, treated as 'valid'
    # else: Not str or Sequence, treated as 'valid'

    is_any_padding_needed = any(p_val != 0 for dim_pads in padding_amounts_list_logical for p_val in dim_pads)

    if not is_any_padding_needed:
        return input_tensor, padding_amounts_list_logical
    
    # Convert padding_amounts_list_logical to PyTorch's F.pad format:
    # (pad_last_dim_start, pad_last_dim_end, pad_2nd_last_dim_start, ...)
    pytorch_pad_arg = []
    for i in range(input_tensor.ndim):
        dim_idx_in_logical_list = input_tensor.ndim - 1 - i
        pad_pair = padding_amounts_list_logical[dim_idx_in_logical_list]
        pytorch_pad_arg.extend([pad_pair[0], pad_pair[1]])
    
    padded_tensor = F.pad(input_tensor, tuple(pytorch_pad_arg), mode='constant', value=constant_value)
    return padded_tensor, padding_amounts_list_logical

def calculate_wt_max_unit_pytorch_cuda(
    patch_tensor: torch.Tensor,
    weights_tensor: torch.Tensor,
    pool_size: Any # Unused in the current implementation
) -> torch.Tensor:
    """
    Calculates weighted contributions from units that achieve the maximum value
    within each channel of a 3D patch, optimized for PyTorch CUDA execution.

    The process involves:
    1. Identifying the maximum value for each channel across the spatial dimensions
       of the input `patch_tensor` (assumed shape: Height x Width x Channels).
    2. Creating a mask where locations holding these channel-wise maximum values
       are marked.
    3. Normalizing this mask such that for each channel, the sum of marks at
       maximum locations equals 1.0 (i.e., if N locations share the max,
       each gets 1/N).
    4. Multiplying the normalized mask by channel-specific `weights_tensor`.

    Args:
        patch_tensor (torch.Tensor): A 3D PyTorch tensor on a CUDA device,
            typically representing spatial data with channels (e.g., HxWxC).
            Expected dtype is a floating-point type (e.g., torch.float32).
        weights_tensor (torch.Tensor): A 1D PyTorch tensor on the same CUDA device
            as `patch_tensor`, containing weights for each channel.
            Its length must match the number of channels in `patch_tensor`.
            Expected dtype should be compatible with `patch_tensor`.
        pool_size (Any): This argument is present in the original function signature
            but is not used in the current implementation of this function.

    Returns:
        torch.Tensor: A tensor of the same shape and dtype as `patch_tensor`,
            residing on the same CUDA device. Locations corresponding to
            channel-wise maximums are set to their normalized share of the
            respective channel's weight, and all other locations are zero.
            If a channel in `patch_tensor` has spatial dimensions of size 0,
            or if all values in a channel are NaN, that channel in the output
            will be all zeros.

    Raises:
        ValueError: If `patch_tensor` is not a 3D tensor.
        ValueError: If `weights_tensor` is not a 1D tensor.
        ValueError: If the number of channels in `patch_tensor` (patch_tensor.shape[2])
                    does not match the length of `weights_tensor` (weights_tensor.shape[0]).
        ValueError: If spatial dimensions of `patch_tensor` (patch_tensor.shape[0] or
                    patch_tensor.shape[1]) are zero.
    """
    if not isinstance(patch_tensor, torch.Tensor) or patch_tensor.ndim != 3:
        raise ValueError(f"Input 'patch_tensor' must be a 3D PyTorch tensor, got shape {patch_tensor.shape if isinstance(patch_tensor, torch.Tensor) else type(patch_tensor)}")
    if not isinstance(weights_tensor, torch.Tensor) or weights_tensor.ndim != 1:
        raise ValueError(f"Input 'weights_tensor' must be a 1D PyTorch tensor, got shape {weights_tensor.shape if isinstance(weights_tensor, torch.Tensor) else type(weights_tensor)}")

    num_channels = patch_tensor.shape[2]
    if weights_tensor.shape[0] != num_channels:
        raise ValueError(f"Number of channels in 'patch_tensor' ({num_channels}) must match "
                         f"the length of 'weights_tensor' ({weights_tensor.shape[0]}).")

    if patch_tensor.shape[0] == 0 or patch_tensor.shape[1] == 0:
        # If spatial dimensions are zero, max cannot be found, result is zeros.
        return torch.zeros_like(patch_tensor, device=patch_tensor.device, dtype=patch_tensor.dtype)

    # Ensure weights_tensor is on the same device and dtype as patch_tensor
    # This is good practice for utility functions.
    # If they are already correct, .to() is a no-op.
    device = patch_tensor.device
    dtype = patch_tensor.dtype
    weights_tensor = weights_tensor.to(device=device, dtype=dtype)

    # 1. Find the maximum value for each channel across spatial dimensions (H, W).
    # `torch.amax` is equivalent to `torch.max` when reducing over multiple dims.
    # `max_per_channel` will have shape (1, 1, C) for broadcasting.
    # Note: torch.max propagates NaNs like np.max.
    max_per_channel = torch.amax(patch_tensor, dim=(0, 1), keepdim=True)

    # 2. Create a boolean mask where patch values equal their channel's maximum.
    # If max_per_channel[c] is NaN, (patch_tensor == NaN) is always False.
    is_max_location = (patch_tensor == max_per_channel)

    # Convert boolean mask to float.
    max_locations_mask = is_max_location.to(dtype=dtype)

    # 3. Normalize the mask.
    # Count number of maximum locations per channel. Shape: (1, 1, C).
    count_max_locations_per_channel = torch.sum(max_locations_mask, dim=(0, 1), keepdim=True)

    # Calculate normalization factor (1.0 / count).
    # Handle division by zero: if count is 0, norm factor becomes 0.
    # `torch.where(condition, x, y)`
    zeros_for_inverse = torch.tensor(0.0, device=device, dtype=dtype)
    inverse_counts = torch.where(
        count_max_locations_per_channel > 0,
        1.0 / count_max_locations_per_channel,
        zeros_for_inverse
    )
    
    # Apply normalization: each max location gets value 1.0 / N_c.
    normalized_max_mask = max_locations_mask * inverse_counts # Broadcasting (H,W,C) * (1,1,C)

    # 4. Multiply by channel-specific weights.
    # Reshape weights to (1, 1, C) for broadcasting.
    weights_reshaped = weights_tensor.reshape(1, 1, -1)
    output_tensor = normalized_max_mask * weights_reshaped # Broadcasting (H,W,C) * (1,1,C)

    return output_tensor

def calculate_wt_maxpool_pytorch_cuda(
    input_tensor_hwc: torch.Tensor,
    weights_tensor_ohwc: torch.Tensor,
    pool_size_hw: Tuple[int, int],
    strides_hw: Tuple[int, int],
    padding_mode_or_values: Union[str, Tuple[int, int], List[List[int]]],
    padding_constant_value: float = -np.inf
) -> torch.Tensor:
    """
    Performs a custom weighted max-pooling operation on a PyTorch tensor (CUDA).

    Assumes input_tensor_hwc is (Height, Width, Channels) and
    weights_tensor_ohwc is (OutHeight, OutWidth, Channels), both on CUDA.
    Output dimensions from pooling must match weights_tensor_ohwc dimensions.

    Args:
        input_tensor_hwc (torch.Tensor): Input tensor (InH, InW, C) on CUDA.
        weights_tensor_ohwc (torch.Tensor): Weights tensor (OutH, OutW, C) on CUDA.
        pool_size_hw (Tuple[int, int]): (PoolH, PoolW).
        strides_hw (Tuple[int, int]): (StrideH, StrideW).
        padding_mode_or_values: Padding mode string ('same', 'valid'),
            or symmetric padding tuple (pad_h_sym, pad_w_sym),
            or explicit padding List[[pt,pb],[pl,pr],[0,0]].
        padding_constant_value (float, optional): Value for padding. Defaults to -np.inf.

    Returns:
        torch.Tensor: Result tensor of shape (InH, InW, C) on CUDA.
        
    Raises:
        ValueError: If dimensions or device properties are inconsistent.
    """
    if input_tensor_hwc.ndim != 3:
        raise ValueError(f"input_tensor_hwc must be 3D (H,W,C), got {input_tensor_hwc.ndim}D.")
    if weights_tensor_ohwc.ndim != 3:
        raise ValueError(f"weights_tensor_ohwc must be 3D (OutH,OutW,C), got {weights_tensor_ohwc.ndim}D.")
    if input_tensor_hwc.shape[2] != weights_tensor_ohwc.shape[2]:
        raise ValueError(f"Channel mismatch: input has {input_tensor_hwc.shape[2]}, weights have {weights_tensor_ohwc.shape[2]}.")
    if input_tensor_hwc.device != weights_tensor_ohwc.device:
        raise ValueError("input_tensor_hwc and weights_tensor_ohwc must be on the same device.")
    if not input_tensor_hwc.is_cuda:
        # This function is optimized for CUDA; could add a CPU path or raise error
        # For now, let's proceed, F.unfold/fold work on CPU too but goal is CUDA.
        pass


    device = input_tensor_hwc.device
    dtype = input_tensor_hwc.dtype

    InH, InW, C = input_tensor_hwc.shape
    PoolH, PoolW = pool_size_hw
    StrideH, StrideW = strides_hw

    # 1. Pad the input (HWC)
    _padding_arg = padding_mode_or_values # Same logic as NumPy for arg validation could be here
    input_padded_hwc, actual_paddings_list = calculate_padding_pytorch_cuda(
        input_tensor=input_tensor_hwc,
        kernel_shape=pool_size_hw,
        strides=strides_hw,
        padding_mode=_padding_arg,
        constant_value=padding_constant_value,
        spatial_dims_indices=(0,1)
    )
    PaddedH, PaddedW, _ = input_padded_hwc.shape

    # For F.unfold, input needs to be BCHW (Batch, Channels, Height, Width)
    input_padded_bchw = input_padded_hwc.permute(2, 0, 1).unsqueeze(0) # C,H,W -> 1,C,H,W

    # 2. Extract patches using F.unfold
    # F.unfold output: (B, C * PoolH * PoolW, NumPatches)
    # NumPatches = OutH_eff * OutW_eff
    patches_unfolded = F.unfold(
        input_padded_bchw,
        kernel_size=pool_size_hw,
        stride=strides_hw,
        padding=(0,0) # Padding already applied manually
    )
    
    _, _, NumPatches = patches_unfolded.shape
    
    # Calculate effective output dimensions from NumPatches if needed
    # OutH_eff = (PaddedH - PoolH) // StrideH + 1
    # OutW_eff = (PaddedW - PoolW) // StrideW + 1
    # if NumPatches != OutH_eff * OutW_eff : error
    # For simplicity, assume F.unfold gives correct NumPatches matching OutH_eff * OutW_eff

    Expected_OutH, Expected_OutW, _ = weights_tensor_ohwc.shape
    # If OutH_eff*OutW_eff from unfold doesn't match product of Expected_OutH*Expected_OutW, it's an issue.
    # This check is a bit indirect with F.unfold's L. A direct check based on calculated OutH/W is better.
    _OutH_calc = (PaddedH - PoolH) // StrideH + 1
    _OutW_calc = (PaddedW - PoolW) // StrideW + 1
    if _OutH_calc * _OutW_calc != NumPatches:
         raise ValueError("Mismatch in number of patches from F.unfold and calculated.")
    if _OutH_calc != Expected_OutH or _OutW_calc != Expected_OutW:
        raise ValueError(
             f"Effective output dimensions ({_OutH_calc}, {_OutW_calc}) from pooling "
             f"do not match weights_tensor_ohwc dimensions ({Expected_OutH}, {Expected_OutW})."
        )
    if NumPatches == 0: # No patches extracted
        return torch.zeros_like(input_tensor_hwc)


    # Reshape patches_unfolded: (B, C, PoolH, PoolW, NumPatches) -> (NumPatches, PoolH, PoolW, C)
    all_patches_reshaped = patches_unfolded.view(1, C, PoolH, PoolW, NumPatches)
    all_patches_reshaped = all_patches_reshaped.permute(0, 4, 2, 3, 1).squeeze(0) # NP, PH, PW, C

    # Reshape weights: (OutH*OutW, C) -> (NumPatches, C)
    all_weights_reshaped = weights_tensor_ohwc.reshape(NumPatches, C)

    # 3. Apply calculate_wt_max_unit logic vectorized
    max_per_patch_channel, _ = torch.max(all_patches_reshaped.flatten(1, 2), dim=1, keepdim=True) # Max over PoolH*PoolW
    max_per_patch_channel = max_per_patch_channel.view(NumPatches, 1, 1, C) # Reshape to NP,1,1,C
    
    is_max_location = (all_patches_reshaped == max_per_patch_channel)
    max_locations_mask = is_max_location.to(dtype=dtype)
    
    count_max_locs = torch.sum(max_locations_mask, dim=(1, 2), keepdim=True)
    
    zeros_for_inverse = torch.tensor(0.0, device=device, dtype=dtype)
    inverse_counts = torch.where(
        count_max_locs > 1e-9, # Avoid division by zero
        1.0 / count_max_locs,
        zeros_for_inverse
    )
    
    normalized_max_mask = max_locations_mask * inverse_counts
    
    # updates_all has shape (NumPatches, PoolH, PoolW, C)
    updates_all = normalized_max_mask * all_weights_reshaped.view(NumPatches, 1, 1, C)

    # 4. Accumulate updates using F.fold
    # Prepare updates_all for F.fold: (B, C * PoolH * PoolW, NumPatches)
    updates_for_fold = updates_all.permute(3, 1, 2, 0).reshape(C * PoolH * PoolW, NumPatches) # C,PH,PW,NP -> C*PH*PW, NP
    updates_for_fold = updates_for_fold.unsqueeze(0) # Add Batch dim: 1, C*PH*PW, NP

    output_canvas_bchw = F.fold(
        updates_for_fold,
        output_size=(PaddedH, PaddedW),
        kernel_size=pool_size_hw,
        stride=strides_hw,
        padding=(0,0) # Handled by F.fold structure based on kernel/stride
    )
    
    # Convert back to HWC: (B,C,H,W) -> (C,H,W) -> (H,W,C)
    output_canvas_hwc = output_canvas_bchw.squeeze(0).permute(1, 2, 0)

    # 5. Crop the output canvas
    pad_h_before = actual_paddings_list[0][0]
    pad_w_before = actual_paddings_list[1][0]
    
    final_output_hwc = output_canvas_hwc[
        pad_h_before : pad_h_before + InH,
        pad_w_before : pad_w_before + InW,
        :
    ]
    
    return final_output_hwc

# Helper for PyTorch activation
def _apply_activation_pytorch(
    values: torch.Tensor,
    func_callable: Optional[Callable[[torch.Tensor], torch.Tensor]]
) -> torch.Tensor:
    if func_callable is None:
        return values
    return func_callable(values)

def calculate_conv_unit_contribution_pytorch_cuda_batched(
    patch_tensor_batch: torch.Tensor, # Shape: (NumPatches, PatchD1, PatchD2, PatchD3)
    kernel_filters_tensor: torch.Tensor, # Shape: (PatchD1, PatchD2, PatchD3, C_out)
    kernel_bias_tensor: torch.Tensor, # Shape: (C_out,)
    output_channel_gain_tensor_batch: torch.Tensor, # Shape: (NumPatches, C_out)
    activation_config: Dict[str, Any],
    epsilon: float = 1e-9
) -> torch.Tensor:
    """
    Calculates a batch of contribution maps for input patches using PyTorch on CUDA.

    Args:
        patch_tensor_batch (torch.Tensor): Batch of input data patches.
            Shape: (NumPatches, PatchD1, PatchD2, PatchD3), on CUDA.
            PatchD1,D2,D3 usually correspond to KernelH, KernelW, InC.
        kernel_filters_tensor (torch.Tensor): Kernel filters, shared across batch.
            Shape: (PatchD1, PatchD2, PatchD3, C_out), on CUDA.
        kernel_bias_tensor (torch.Tensor): Bias for each output channel, shared.
            Shape: (C_out,), on CUDA.
        output_channel_gain_tensor_batch (torch.Tensor): Batch of gain factors.
            Shape: (NumPatches, C_out), on CUDA.
        activation_config (Dict[str, Any]): Configuration for activation.
        epsilon (float, optional): Small value for numerical stability.

    Returns:
        torch.Tensor: Batch of contribution maps.
            Shape: (NumPatches, PatchD1, PatchD2, PatchD3), on CUDA.
    """
    NumPatches = patch_tensor_batch.shape[0]
    C_out = kernel_filters_tensor.shape[-1]
    device = patch_tensor_batch.device
    dtype = patch_tensor_batch.dtype

    # Validate shapes and ensure all tensors are on the same device and dtype
    # Basic checks, more can be added if needed
    if kernel_filters_tensor.shape[:-1] != patch_tensor_batch.shape[1:]:
         raise ValueError("Spatial/InC dimensions of kernel_filters must match patch_tensor_batch.")
    if kernel_bias_tensor.shape[0] != C_out or output_channel_gain_tensor_batch.shape[1] != C_out:
        raise ValueError("C_out dimension mismatch for bias or output_channel_gain.")
    if output_channel_gain_tensor_batch.shape[0] != NumPatches:
        raise ValueError("NumPatches dimension mismatch for output_channel_gain_tensor_batch.")
    
    # Ensure other tensors are on the same device/dtype (or convert)
    kernel_filters_tensor = kernel_filters_tensor.to(device=device, dtype=dtype)
    kernel_bias_tensor = kernel_bias_tensor.to(device=device, dtype=dtype)
    output_channel_gain_tensor_batch = output_channel_gain_tensor_batch.to(device=device, dtype=dtype)


    # Bias processing (broadcasts to NumPatches later)
    # Shapes: (C_out,)
    bias_positive_part = torch.relu(kernel_bias_tensor)
    bias_negative_part = torch.relu(-kernel_bias_tensor)

    # Element-wise product (convolution-like)
    # patch_tensor_batch: (NP, D1, D2, D3) -> unsqueeze for C_out: (NP, D1, D2, D3, 1)
    # kernel_filters_tensor: (D1, D2, D3, C_out) -> unsqueeze for NP: (1, D1, D2, D3, C_out)
    # conv_out shape: (NP, D1, D2, D3, C_out)
    conv_out = kernel_filters_tensor.unsqueeze(0) * patch_tensor_batch.unsqueeze(-1)

    # Positive and negative contributions
    # Shapes: (NP, D1, D2, D3, C_out)
    positive_contributions = torch.relu(conv_out)
    negative_contributions_orig_sign = torch.min(conv_out, torch.tensor(0.0, device=device, dtype=dtype))

    # Sums per output channel, per patch in batch
    # Shapes: (NP, C_out)
    sum_positive = torch.sum(positive_contributions, dim=(1, 2, 3))
    sum_negative_abs = torch.sum(torch.abs(negative_contributions_orig_sign), dim=(1, 2, 3))
    sum_total_pre_bias = sum_positive - sum_negative_abs

    # Saturation flags
    # Shapes: (NP, C_out)
    positive_saturation_flags = sum_positive > epsilon
    negative_saturation_flags = sum_negative_abs > epsilon

    act_type = activation_config.get("type")
    act_range_config = activation_config.get("range", {})
    act_func_callable = activation_config.get("func") # PyTorch compatible callable
    lower_bound = act_range_config.get("l")
    upper_bound = act_range_config.get("u")

    if act_type == 'mono':
        if lower_bound is not None:
            positive_saturation_flags = positive_saturation_flags & (sum_total_pre_bias > lower_bound)
        if upper_bound is not None:
            negative_saturation_flags = negative_saturation_flags & (sum_total_pre_bias < upper_bound)
    elif act_type == 'non_mono':
        if act_func_callable is None:
            raise ValueError("'func' must be provided for 'non_mono' activation type.")
        
        # Shapes: (NP, C_out)
        activated_total_sum = _apply_activation_pytorch(sum_total_pre_bias, act_func_callable)
        # bias_positive_part/negative_part (C_out,) broadcast with sum_positive/negative_abs (NP, C_out)
        activated_positive_path = _apply_activation_pytorch(sum_positive + bias_positive_part.unsqueeze(0), act_func_callable)
        activated_negative_path = _apply_activation_pytorch(-1.0 * (sum_negative_abs + bias_negative_part.unsqueeze(0)), act_func_callable)

        if lower_bound is not None:
            positive_saturation_flags = positive_saturation_flags & (sum_total_pre_bias > lower_bound)
        if upper_bound is not None:
            negative_saturation_flags = negative_saturation_flags & (sum_total_pre_bias < upper_bound)
        
        negative_saturation_flags = negative_saturation_flags & (torch.abs(activated_total_sum - activated_positive_path) > 1e-5)
        positive_saturation_flags = positive_saturation_flags & (torch.abs(activated_total_sum - activated_negative_path) > 1e-5)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")

    # Aggregate weights
    # denominator shape: (NP, C_out)
    denominator = sum_positive + sum_negative_abs + \
                  bias_positive_part.unsqueeze(0) + bias_negative_part.unsqueeze(0)
    
    # p_agg_wt, n_agg_wt shapes: (NP, C_out)
    p_agg_wt = torch.zeros_like(denominator)
    n_agg_wt = torch.zeros_like(denominator)

    valid_denominator_mask = denominator > epsilon
    
    if torch.any(valid_denominator_mask):
        inv_denominator_safe = torch.zeros_like(denominator)
        inv_denominator_safe[valid_denominator_mask] = 1.0 / denominator[valid_denominator_mask]
        
        # output_channel_gain_tensor_batch already (NP, C_out)
        p_agg_wt = inv_denominator_safe * output_channel_gain_tensor_batch * positive_saturation_flags.to(dtype)
        n_agg_wt = inv_denominator_safe * output_channel_gain_tensor_batch * negative_saturation_flags.to(dtype)
        
    # Build contribution map per channel, per patch
    # contribution_map_per_channel shape: (NP, D1, D2, D3, C_out)
    contribution_map_per_channel = torch.zeros_like(conv_out) # Has shape (NP,D1,D2,D3,C_out)
    
    # p_agg_wt/n_agg_wt (NP, C_out) need to be unsqueezed for broadcasting: (NP, 1, 1, 1, C_out)
    contribution_map_per_channel += positive_contributions * p_agg_wt.view(NumPatches, 1, 1, 1, C_out)
    contribution_map_per_channel += torch.abs(negative_contributions_orig_sign) * n_agg_wt.view(NumPatches, 1, 1, 1, C_out)

    # Sum contributions over output channels (C_out dim) to get final map for each patch
    # final_contribution_map_batch shape: (NP, D1, D2, D3)
    final_contribution_map_batch = torch.sum(contribution_map_per_channel, dim=-1)
    
    return final_contribution_map_batch

def calculate_wt_conv_pytorch_cuda(
    input_tensor_ihwc: torch.Tensor,
    kernel_filters_tensor_khw_inc_outc: torch.Tensor,
    kernel_bias_tensor_outc: torch.Tensor,
    output_channel_gain_map_tensor_ohw_outc: torch.Tensor,
    strides_hw: Tuple[int, int],
    padding_mode_or_values: Union[str, Tuple[int, int], List[List[int]]],
    activation_config: Dict[str, Any],
    padding_constant_value: float = 0.0
) -> torch.Tensor:
    """
    Performs a custom weighted convolution contribution analysis using PyTorch on CUDA.

    Args:
        input_tensor_ihwc (torch.Tensor): Input tensor (InH, InW, InC) on CUDA.
        kernel_filters_tensor_khw_inc_outc (torch.Tensor): Kernel filters
            (KernelH, KernelW, InC, OutC) on CUDA.
        kernel_bias_tensor_outc (torch.Tensor): Bias (OutC,) on CUDA.
        output_channel_gain_map_tensor_ohw_outc (torch.Tensor): Gain map
            (OutH, OutW, OutC) on CUDA.
        strides_hw (Tuple[int, int]): Strides (StrideH, StrideW).
        padding_mode_or_values: Padding mode or values.
        activation_config (Dict[str, Any]): Activation configuration.
        padding_constant_value (float, optional): Value for padding. Defaults to 0.0.

    Returns:
        torch.Tensor: Resulting contribution map (InH, InW, InC) on CUDA.
    """
    device = input_tensor_ihwc.device
    dtype = input_tensor_ihwc.dtype

    # --- Input Validation (simplified) ---
    if input_tensor_ihwc.ndim != 3: raise ValueError("input_tensor_ihwc must be 3D")
    # ... ensure all tensors on same device, dtype, correct dimensions ...

    InH, InW, InC = input_tensor_ihwc.shape
    KernelH, KernelW, _, OutC = kernel_filters_tensor_khw_inc_outc.shape
    OutH, OutW, _ = output_channel_gain_map_tensor_ohw_outc.shape

    # 1. Pad the input (IHWC -> PaddedIHWC)
    input_padded_ihwc, actual_paddings_list = calculate_padding_pytorch_cuda(
        input_tensor=input_tensor_ihwc,
        kernel_shape=(KernelH, KernelW),
        strides=strides_hw,
        padding_mode=padding_mode_or_values,
        constant_value=padding_constant_value,
        spatial_dims_indices=(0, 1)
    )
    PaddedH, PaddedW, _ = input_padded_ihwc.shape

    # Convert input to BCHW for F.unfold: (H,W,C) -> (C,H,W) -> (1,C,H,W)
    input_padded_b_inc_h_w = input_padded_ihwc.permute(2, 0, 1).unsqueeze(0)

    # 2. Extract patches using F.unfold
    # Output: (B=1, InC * KernelH * KernelW, NumPatches_eff)
    patches_unfolded = F.unfold(
        input_padded_b_inc_h_w,
        kernel_size=(KernelH, KernelW),
        stride=strides_hw,
        padding=(0,0) # Padding already applied manually
    )
    _, _, NumPatches_eff = patches_unfolded.shape
    
    # Expected number of patches based on OutH, OutW from output_channel_gain_map
    if NumPatches_eff != OutH * OutW:
        raise ValueError(f"Mismatch in number of patches: F.unfold got {NumPatches_eff}, "
                         f"expected from output_channel_gain_map {OutH*OutW}. Check inputs/padding.")
    if NumPatches_eff == 0: # No patches could be extracted
        return torch.zeros_like(input_tensor_ihwc) # Return empty or zero contribution


    # Reshape patches_unfolded for the batched unit function:
    # (1, InC*KH*KW, NP) -> (1, InC, KH, KW, NP) -> (NP, KH, KW, InC)
    all_patches_np_khw_inc = patches_unfolded.view(1, InC, KernelH, KernelW, NumPatches_eff)
    all_patches_np_khw_inc = all_patches_np_khw_inc.permute(0, 4, 2, 3, 1).squeeze(0)

    # Reshape output_channel_gain_map for the batched unit function:
    # (OutH, OutW, OutC) -> (NumPatches_eff, OutC)
    output_channel_gain_batch_np_outc = output_channel_gain_map_tensor_ohw_outc.reshape(NumPatches_eff, OutC)

    # 3. Apply batched contribution calculation in chunks to manage memory
    # `all_patches_np_khw_inc` is (NP, KH, KW, InC)
    # `kernel_filters_tensor_khw_inc_outc` is (KH, KW, InC, OutC)
    # `kernel_bias_tensor_outc` is (OutC,)
    # `output_channel_gain_batch_np_outc` is (NP, OutC)
    # `updates_all_np_khw_inc` will be (NP, KH, KW, InC)

    # Determine a chunk size for processing patches
    # This size might need tuning. For VGG deep layers, NumPatches is small (e.g., 14*14=196).
    # For earlier layers, NumPatches is large (e.g., 224*224=50176).
    # The memory issue was with ~1.7GB for NP=196, InC=512, OutC=512. This suggests the unit function itself is heavy.
    # Let's try a small chunk size like 16 or 32 to be safe for now.
    # If NP=196, chunk_size=32, then 196/32 approx 7 chunks.
    # If conv_out was 1.7GB for NP=196, for NP=32 it would be 1.7GB * (32/196) = ~280MB, which might fit.
    chunk_size = 32 
    
    accumulated_updates_np_khw_inc = torch.zeros_like(all_patches_np_khw_inc)

    for i in range(0, NumPatches_eff, chunk_size):
        patch_chunk = all_patches_np_khw_inc[i:i+chunk_size]
        gain_chunk = output_channel_gain_batch_np_outc[i:i+chunk_size]
        
        # print(f"Processing chunk {i//chunk_size + 1}/{(NumPatches_eff + chunk_size - 1)//chunk_size}, patch_chunk shape: {patch_chunk.shape}")

        chunk_updates_np_khw_inc = calculate_conv_unit_contribution_pytorch_cuda_batched(
            patch_tensor_batch=patch_chunk,
        kernel_filters_tensor=kernel_filters_tensor_khw_inc_outc,
        kernel_bias_tensor=kernel_bias_tensor_outc,
            output_channel_gain_tensor_batch=gain_chunk,
        activation_config=activation_config
    )
        accumulated_updates_np_khw_inc[i:i+chunk_size] = chunk_updates_np_khw_inc
        del patch_chunk, gain_chunk, chunk_updates_np_khw_inc # Hint for GC
        if device.type == 'cuda':
            torch.cuda.empty_cache() # More aggressive cache clearing per chunk

    updates_all_np_khw_inc = accumulated_updates_np_khw_inc

    # 4. Accumulate updates using F.fold
    # Prepare updates_all for F.fold: (NP, KH, KW, InC) -> (1, InC * KH * KW, NP)
    updates_for_fold = updates_all_np_khw_inc.permute(3, 1, 2, 0).reshape(InC * KernelH * KernelW, NumPatches_eff)
    updates_for_fold = updates_for_fold.unsqueeze(0) # Add Batch dim: (1, InC*KH*KW, NP)

    output_canvas_b_inc_h_w = F.fold(
        updates_for_fold,
        output_size=(PaddedH, PaddedW), # Output size on padded dimensions
        kernel_size=(KernelH, KernelW),
        stride=strides_hw,
        padding=(0,0)
    )
    
    # Convert back to HWC: (1,InC,PaddedH,PaddedW) -> (InC,PaddedH,PaddedW) -> (PaddedH,PaddedW,InC)
    output_canvas_ihwc = output_canvas_b_inc_h_w.squeeze(0).permute(1, 2, 0)

    # 5. Crop the output canvas
    pad_h_before = actual_paddings_list[0][0]
    pad_w_before = actual_paddings_list[1][0]
    
    final_output_ihwc = output_canvas_ihwc[
        pad_h_before : pad_h_before + InH,
        pad_w_before : pad_w_before + InW,
        :
    ]
    
    return final_output_ihwc

def calculate_fc_input_relevance_pytorch_cuda(
    output_relevance: Any,      # R_j (NumOutputNeurons,) - Can be np.ndarray or torch.Tensor
    input_activations: Any,     # x_i (NumInputNeurons,) - Can be np.ndarray or torch.Tensor
    fc_weights: Any,            # W_ji (NumOutputNeurons, NumInputNeurons) - Can be np.ndarray or torch.Tensor
    fc_bias: Any,               # b_j (NumOutputNeurons,) - Can be np.ndarray or torch.Tensor
    activation_config: Dict[str, Any],
    epsilon: float = 1e-9
) -> torch.Tensor:
    """
    Calculates input relevance for an FC layer using LRP-like rules (PyTorch CUDA).
    Converts input NumPy arrays to PyTorch tensors and ensures device/dtype consistency.
    """
    # --- Convert NumPy inputs to PyTorch Tensors ---
    # Determine target device and dtype from output_relevance *after* ensuring it's a tensor.
    # If output_relevance itself is NumPy, we might need a default device/dtype or infer from others.
    # Let's assume output_relevance is the source of truth for device/dtype if it's a tensor.
    # If it's NumPy, we'll try to use a default device (e.g., 'cuda' if available, else 'cpu') and float32.

    potential_tensors = {
        "output_relevance": output_relevance,
        "input_activations": input_activations,
        "fc_weights": fc_weights,
        "fc_bias": fc_bias
    }
    
    processed_tensors = {}

    # Determine target device and dtype
    # Priority: 1. output_relevance if it's a Tensor. 2. Any other input if it's a Tensor. 3. Defaults.
    target_device = None
    target_dtype = None

    if isinstance(output_relevance, torch.Tensor):
        target_device = output_relevance.device
        target_dtype = output_relevance.dtype
    else: # Check other inputs or set defaults
        for val in [input_activations, fc_weights, fc_bias]: # Check in order
            if isinstance(val, torch.Tensor):
                target_device = val.device
                target_dtype = val.dtype
                break
        if target_device is None: # No input was a tensor, set defaults
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            target_dtype = torch.float32
            # print(f"Warning: No input tensors to infer device/dtype. Defaulting to {target_device}/{target_dtype}") # Optional warning


    for name, arr_or_tensor in potential_tensors.items():
        if isinstance(arr_or_tensor, np.ndarray):
            processed_tensors[name] = torch.from_numpy(arr_or_tensor).to(device=target_device, dtype=target_dtype)
        elif isinstance(arr_or_tensor, torch.Tensor):
            processed_tensors[name] = arr_or_tensor.to(device=target_device, dtype=target_dtype)
        else:
            raise TypeError(f"Argument '{name}' must be a NumPy array or PyTorch tensor, got {type(arr_or_tensor)}")

    # Assign back to original variable names for clarity
    output_relevance_t = processed_tensors["output_relevance"]
    input_activations_t = processed_tensors["input_activations"]
    fc_weights_t = processed_tensors["fc_weights"]
    fc_bias_t = processed_tensors["fc_bias"]
    
    # Squeeze batch dimension from output_relevance_t if it exists and is 1
    if output_relevance_t.ndim == 2 and output_relevance_t.shape[0] == 1:
        output_relevance_t = output_relevance_t.squeeze(0)
    # Squeeze batch dimension from input_activations_t if it exists and is 1 (less common for FC input but good for consistency)
    if input_activations_t.ndim == 2 and input_activations_t.shape[0] == 1:
        input_activations_t = input_activations_t.squeeze(0)
    
    # Now all inputs (output_relevance_t, etc.) are PyTorch tensors on the target_device and target_dtype.
    device = target_device # Use the determined target_device
    dtype = target_dtype  # Use the determined target_dtype

    # Validate shapes (after potential conversion, dtypes should match)
    num_output_neurons, num_input_neurons = fc_weights_t.shape
    
    expected_shapes = {
        "output_relevance": (num_output_neurons,),
        "input_activations": (num_input_neurons,),
        "fc_weights": (num_output_neurons, num_input_neurons),
        "fc_bias": (num_output_neurons,)
    }
    
    tensors_to_check_for_shape = {
        "output_relevance": output_relevance_t,
        "input_activations": input_activations_t,
        "fc_weights": fc_weights_t,
        "fc_bias": fc_bias_t
    }

    for t_name, t_val in tensors_to_check_for_shape.items():
        exp_shape = expected_shapes[t_name]
        if t_val.shape != exp_shape:
            raise ValueError(f"Shape mismatch for {t_name}: got {t_val.shape}, expected {exp_shape} (after conversion).")

    # --- Start of core logic (uses _t suffixed variables) ---
    individual_terms_pre_sum = fc_weights_t * input_activations_t.unsqueeze(0) 

    positive_terms_values = torch.relu(individual_terms_pre_sum) 
    negative_terms_values_orig_sign = torch.min(individual_terms_pre_sum, torch.tensor(0.0, device=device, dtype=dtype))

    sum_positive_terms = torch.sum(positive_terms_values, dim=1)
    sum_negative_terms_abs = torch.sum(torch.abs(negative_terms_values_orig_sign), dim=1)
    
    bias_positive = torch.relu(fc_bias_t)
    bias_negative_abs = torch.relu(-fc_bias_t)

    total_pre_activation = sum_positive_terms - sum_negative_terms_abs + fc_bias_t
    
    eff_sum_positive = sum_positive_terms.clone()
    eff_sum_negative_abs = sum_negative_terms_abs.clone()

    act_type = activation_config.get("type")
    act_range_config = activation_config.get("range", {})
    act_func_callable = activation_config.get("func") 
    lower_bound = act_range_config.get("l")
    upper_bound = act_range_config.get("u")

    if act_type == "mono":
        if lower_bound is not None:
            mono_l_mask = total_pre_activation < lower_bound
            eff_sum_positive[mono_l_mask] = 0.0
        if upper_bound is not None:
            mono_u_mask = total_pre_activation > upper_bound
            eff_sum_negative_abs[mono_u_mask] = 0.0
    elif act_type == "non_mono":
        if act_func_callable is None:
            raise ValueError("'func' must be provided for 'non_mono' activation type.")
        
        activated_total = _apply_activation_pytorch(total_pre_activation, act_func_callable)
        activated_positive_path = _apply_activation_pytorch(sum_positive_terms + bias_positive, act_func_callable)
        activated_negative_path = _apply_activation_pytorch(-1.0 * (sum_negative_terms_abs + bias_negative_abs), act_func_callable)

        if lower_bound is not None:
            nonmono_l_mask = total_pre_activation < lower_bound
            eff_sum_positive[nonmono_l_mask] = 0.0
        if upper_bound is not None:
            nonmono_u_mask = total_pre_activation > upper_bound
            eff_sum_negative_abs[nonmono_u_mask] = 0.0
        
        positive_explains_mask = torch.abs(activated_total - activated_positive_path) < epsilon
        eff_sum_negative_abs[positive_explains_mask] = 0.0
        
        negative_explains_mask = (torch.abs(activated_total - activated_negative_path) < epsilon) & (~positive_explains_mask)
        eff_sum_positive[negative_explains_mask] = 0.0
    else:
        raise ValueError(f"Unknown activation type: {act_type}")

    p_agg_wt = torch.zeros_like(sum_positive_terms)
    n_agg_wt = torch.zeros_like(sum_negative_terms_abs)
    
    denominator = eff_sum_positive + eff_sum_negative_abs + bias_positive + bias_negative_abs
    
    valid_p_mask_denom = (denominator > epsilon)
    valid_p_mask_sum = (eff_sum_positive > 0) 
    valid_p_mask = valid_p_mask_denom & valid_p_mask_sum

    if torch.any(valid_p_mask):
        pathway_sum_p_selected = eff_sum_positive[valid_p_mask] + bias_positive[valid_p_mask]
        valid_pathway_sum_p_mask_selected = pathway_sum_p_selected > epsilon
        
        prop_total_positive_pathway_selected = torch.zeros_like(pathway_sum_p_selected)
        prop_terms_in_pos_pathway_selected = torch.zeros_like(pathway_sum_p_selected)

        if torch.any(valid_pathway_sum_p_mask_selected):
            prop_total_positive_pathway_selected[valid_pathway_sum_p_mask_selected] = \
                pathway_sum_p_selected[valid_pathway_sum_p_mask_selected] / denominator[valid_p_mask][valid_pathway_sum_p_mask_selected]
            
            prop_terms_in_pos_pathway_selected[valid_pathway_sum_p_mask_selected] = \
                eff_sum_positive[valid_p_mask][valid_pathway_sum_p_mask_selected] / pathway_sum_p_selected[valid_pathway_sum_p_mask_selected]
        
        p_agg_wt[valid_p_mask] = prop_total_positive_pathway_selected * prop_terms_in_pos_pathway_selected

    valid_n_mask_denom = (denominator > epsilon)
    valid_n_mask_sum = (eff_sum_negative_abs > 0)
    valid_n_mask = valid_n_mask_denom & valid_n_mask_sum

    if torch.any(valid_n_mask):
        pathway_sum_n_selected = eff_sum_negative_abs[valid_n_mask] + bias_negative_abs[valid_n_mask]
        valid_pathway_sum_n_mask_selected = pathway_sum_n_selected > epsilon

        prop_total_negative_pathway_selected = torch.zeros_like(pathway_sum_n_selected)
        prop_terms_in_neg_pathway_selected = torch.zeros_like(pathway_sum_n_selected)
        
        if torch.any(valid_pathway_sum_n_mask_selected):
            prop_total_negative_pathway_selected[valid_pathway_sum_n_mask_selected] = \
                pathway_sum_n_selected[valid_pathway_sum_n_mask_selected] / denominator[valid_n_mask][valid_pathway_sum_n_mask_selected]

            prop_terms_in_neg_pathway_selected[valid_pathway_sum_n_mask_selected] = \
                eff_sum_negative_abs[valid_n_mask][valid_pathway_sum_n_mask_selected] / pathway_sum_n_selected[valid_pathway_sum_n_mask_selected]
                
        n_agg_wt[valid_n_mask] = prop_total_negative_pathway_selected * prop_terms_in_neg_pathway_selected

    relevance_contributions_ji = torch.zeros_like(individual_terms_pre_sum)

    prop_denom_p = sum_positive_terms.clone() 
    prop_denom_p[prop_denom_p < epsilon] = 1.0 
    
    prop_denom_n = sum_negative_terms_abs.clone() 
    prop_denom_n[prop_denom_n < epsilon] = 1.0

    # Unsqueeze terms that are (O,) to (O,1) for broadcasting
    output_relevance_b = output_relevance_t.unsqueeze(1)
    p_agg_wt_b = p_agg_wt.unsqueeze(1)
    n_agg_wt_b = n_agg_wt.unsqueeze(1)
    # prop_denom_p and prop_denom_n were already modified to avoid zeros by replacing small values with 1.0
    prop_denom_p_b = prop_denom_p.unsqueeze(1)
    prop_denom_n_b = prop_denom_n.unsqueeze(1)

    # Calculate for positive contributions
    # factor_pos_b has shape (O,1)
    factor_pos_b = (output_relevance_b * p_agg_wt_b) / prop_denom_p_b 
    # relevance_from_positive has shape (O,I)
    relevance_from_positive = positive_terms_values * factor_pos_b
    relevance_contributions_ji += relevance_from_positive
    del factor_pos_b, relevance_from_positive # Attempt to free memory for these intermediates
    
    # Calculate for negative contributions
    # factor_neg_b has shape (O,1)
    factor_neg_b = (output_relevance_b * n_agg_wt_b * -1.0) / prop_denom_n_b
    # relevance_from_negative has shape (O,I)
    relevance_from_negative = negative_terms_values_orig_sign * factor_neg_b
    relevance_contributions_ji += relevance_from_negative
    del factor_neg_b, relevance_from_negative # Attempt to free memory for these intermediates
    
    input_relevance_i = torch.sum(relevance_contributions_ji, dim=0)
    
    return input_relevance_i

def reshape_tensor_pytorch_cuda(
    tensor_to_reshape: torch.Tensor,
    target_shape_provider_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Reshapes an input PyTorch tensor to match the shape of a target shape provider tensor.

    This function assumes both tensors are or will be on a CUDA device if CUDA
    execution is intended for subsequent operations. The reshaping itself is
    device-agnostic but the context implies CUDA.

    The content of `target_shape_provider_tensor` is not used, only its `shape`.
    The total number of elements in `tensor_to_reshape` must be compatible
    with the total number of elements defined by `target_shape_provider_tensor.shape`.

    Args:
        tensor_to_reshape (torch.Tensor): The PyTorch tensor whose elements
            will be reshaped.
        target_shape_provider_tensor (torch.Tensor): A PyTorch tensor whose shape
            will be used as the target shape for reshaping `tensor_to_reshape`.

    Returns:
        torch.Tensor: A new tensor (often a view) with the same data as
                      `tensor_to_reshape` but with the shape of
                      `target_shape_provider_tensor`, on the same device
                      as `tensor_to_reshape`.

    Raises:
        TypeError: If inputs are not PyTorch tensors.
        RuntimeError: If reshaping is not possible due to incompatible sizes
                      (PyTorch typically raises RuntimeError for this).
    """
    if not isinstance(tensor_to_reshape, torch.Tensor):
        raise TypeError(f"Input 'tensor_to_reshape' must be a PyTorch tensor, got {type(tensor_to_reshape)}")
    if not isinstance(target_shape_provider_tensor, torch.Tensor):
        raise TypeError(f"Input 'target_shape_provider_tensor' must be a PyTorch tensor, got {type(target_shape_provider_tensor)}")

    target_shape: torch.Size = target_shape_provider_tensor.shape
    
    try:
        # tensor.reshape() or torch.reshape() can be used.
        # .reshape() is generally preferred.
        reshaped_tensor = tensor_to_reshape.reshape(target_shape)
    except RuntimeError as e:
        # PyTorch's reshape errors are usually quite descriptive.
        # We can add more context if desired.
        original_size = tensor_to_reshape.numel()
        target_numel = target_shape_provider_tensor.numel()
        raise RuntimeError(
            f"Cannot reshape tensor of size {original_size} (shape {tensor_to_reshape.shape}) "
            f"into shape {target_shape} (target size {target_numel}). {e}"
        ) from e
        
    return reshaped_tensor

# Example of original usage with renamed function:
# wts_tensor = torch.arange(12)
# inp_tensor_for_shape = torch.empty((3,4))
# reshaped_wts_tensor = reshape_tensor_pytorch_cuda(wts_tensor, inp_tensor_for_shape)
# print(reshaped_wts_tensor.shape) # Output: torch.Size([3, 4])

# Similar to NumPy, if the target shape is known as a tuple, passing it directly to
# tensor_to_reshape.reshape(shape_tuple) is more direct.
