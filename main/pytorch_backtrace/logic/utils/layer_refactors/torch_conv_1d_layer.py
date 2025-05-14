import torch
from typing import Dict, Union, Callable, Optional


def calculate_weighted_convolution_relevance_1d(
    patch: torch.Tensor,
    relevance_values: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    activation: Dict[str, Union[str, Dict[str, Optional[float]], Callable]]
) -> torch.Tensor:
    """
    Compute relevance for a single patch of the input tensor using vectorized operations
    with PyTorch CUDA acceleration.

    Parameters:
        patch (torch.Tensor): Patch of input corresponding to the receptive field of the kernel.
            Expected shape: [batch_size, input_features, *] (CUDA tensor)
        relevance_values (torch.Tensor): Relevance values from the next layer for this patch.
            Expected shape: [batch_size] or [*] (CUDA tensor)
        weights (torch.Tensor): Weights of the convolutional kernel.
            Expected shape: [output_features, input_features, kernel_size] (CUDA tensor)
        bias (torch.Tensor): Bias values for the convolution.
            Expected shape: [output_features] (CUDA tensor)
        activation (dict): Activation function details containing:
            - "type" (str): Type of activation ('mono' or 'non_mono').
            - "range" (dict): Dictionary with "l" (lower bound) and "u" (upper bound).
            - "func" (callable): Function to apply for activation.

    Returns:
        torch.Tensor: Weighted relevance matrix for the patch (CUDA tensor).
    """
    # Ensure inputs are on GPU
    patch = patch.cuda() if not patch.is_cuda else patch
    relevance_values = relevance_values.cuda() if not relevance_values.is_cuda else relevance_values
    weights = weights.cuda() if not weights.is_cuda else weights
    bias = bias.cuda() if not bias.is_cuda else bias
    
    # Compute convolution output using einsum for efficiency
    conv_out = torch.einsum("ijk,ij->ijk", weights, patch)
    
    # Separate positive and negative contributions using vectorized operations
    positive_mask = conv_out > 0
    positive_contributions = conv_out * positive_mask
    positive_sums = torch.einsum("ijk->k", positive_contributions)
    
    negative_mask = conv_out < 0
    negative_contributions = conv_out * negative_mask
    negative_sums = torch.einsum("ijk->k", negative_contributions) * -1.0
    
    total_sums = positive_sums + negative_sums
    
    # Handle positive and negative bias components
    bias_positive = bias * (bias > 0)
    bias_negative = bias * (bias < 0) * -1.0
    
    # Initialize saturation masks
    positive_saturate = positive_sums > 0
    negative_saturate = negative_sums > 0
    
    # Apply activation-specific logic
    if activation["type"] == 'mono':
        lower_bound = activation["range"]["l"]
        upper_bound = activation["range"]["u"]
        
        if lower_bound is not None:
            lower_bound = torch.tensor(lower_bound, device='cuda')
            positive_saturate = total_sums > lower_bound
            
        if upper_bound is not None:
            upper_bound = torch.tensor(upper_bound, device='cuda')
            negative_saturate = total_sums < upper_bound
            
    elif activation["type"] == 'non_mono':
        act_func = activation["func"]
        lower_bound = activation["range"]["l"]
        upper_bound = activation["range"]["u"]
        
        total_activated = act_func(total_sums)
        positive_activated = act_func(positive_sums + bias_positive)
        negative_activated = act_func(-1 * (negative_sums + bias_negative))
        
        if lower_bound is not None:
            lower_bound = torch.tensor(lower_bound, device='cuda')
            bound_mask = total_sums > lower_bound
            positive_saturate = positive_saturate & bound_mask
            
        if upper_bound is not None:
            upper_bound = torch.tensor(upper_bound, device='cuda')
            bound_mask = total_sums < upper_bound
            negative_saturate = negative_saturate & bound_mask
        
        diff_threshold = 1e-5
        positive_saturate = positive_saturate & (torch.abs(total_activated - negative_activated) > diff_threshold)
        negative_saturate = negative_saturate & (torch.abs(total_activated - positive_activated) > diff_threshold)
    
    # Calculate denominators safely
    denominators = positive_sums + negative_sums + bias_positive + bias_negative
    
    # Calculate aggregated weights
    positive_aggregated_weights = (1.0 / denominators) * relevance_values * positive_saturate
    negative_aggregated_weights = (1.0 / denominators) * relevance_values * negative_saturate
    
    # Apply weights to contributions and combine
    relevance_matrix = (positive_contributions * positive_aggregated_weights) + \
                       (negative_contributions * negative_aggregated_weights * -1.0)
    
    # Sum across the last dimension to get final relevance matrix
    return torch.sum(relevance_matrix, axis=-1)
