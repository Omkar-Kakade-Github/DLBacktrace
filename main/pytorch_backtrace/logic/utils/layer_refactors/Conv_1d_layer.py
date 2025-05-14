import numpy as np
from typing import Dict, Callable, Optional, Union


def calculate_weighted_convolution_relevance_1d(
   patch: np.ndarray,
   relevance_values: np.ndarray,
   weights: np.ndarray,
   bias: np.ndarray,
   activation: Dict[str, Union[str, Dict[str, Optional[float]], Callable]]
) -> np.ndarray:
   """
   Compute relevance for a single patch of the input tensor using vectorized operations.

   Parameters:
       patch (np.ndarray): Patch of input corresponding to the receptive field of the kernel.
       relevance_values (np.ndarray): Relevance values from the next layer for this patch.
       weights (np.ndarray): Weights of the convolutional kernel.
       bias (np.ndarray): Bias values for the convolution.
       activation (dict): Activation function details containing:
           - "type" (str): Type of activation ('mono' or 'non_mono').
           - "range" (dict): Dictionary with "l" (lower bound) and "u" (upper bound).
           - "func" (callable): Function to apply for activation.

   Returns:
       np.ndarray: Weighted relevance matrix for the patch.
   """
   # Compute convolution output using einsum for efficiency
   conv_out = np.einsum("ijk,ij->ijk", weights, patch)
   
   # Separate positive and negative contributions using vectorized operations
   positive_mask = conv_out > 0
   positive_contributions = conv_out * positive_mask
   positive_sums = np.einsum("ijk->k", positive_contributions)
   
   negative_mask = conv_out < 0
   negative_contributions = conv_out * negative_mask
   negative_sums = np.einsum("ijk->k", negative_contributions) * -1.0
   
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
           positive_saturate = total_sums > lower_bound
           
       if upper_bound is not None:
           negative_saturate = total_sums < upper_bound
           
   elif activation["type"] == 'non_mono':
       act_func = activation["func"]
       lower_bound = activation["range"]["l"]
       upper_bound = activation["range"]["u"]
       
       total_activated = act_func(total_sums)
       positive_activated = act_func(positive_sums + bias_positive)
       negative_activated = act_func(-1 * (negative_sums + bias_negative))
       
       if lower_bound is not None:
           bound_mask = total_sums > lower_bound
           positive_saturate = positive_saturate & bound_mask
           
       if upper_bound is not None:
           bound_mask = total_sums < upper_bound
           negative_saturate = negative_saturate & bound_mask
       
       diff_threshold = 1e-5
       positive_saturate = positive_saturate & (np.abs(total_activated - negative_activated) > diff_threshold)
       negative_saturate = negative_saturate & (np.abs(total_activated - positive_activated) > diff_threshold)
   
   # Calculate denominators safely
   denominators = positive_sums + negative_sums + bias_positive + bias_negative
   
   # Calculate aggregated weights
   positive_aggregated_weights = (1.0 / denominators) * relevance_values * positive_saturate
   negative_aggregated_weights = (1.0 / denominators) * relevance_values * negative_saturate
   
   # Apply weights to contributions and combine
   relevance_matrix = (positive_contributions * positive_aggregated_weights) + \
                      (negative_contributions * negative_aggregated_weights * -1.0)
   
   # Sum across the last dimension to get final relevance matrix
   return np.sum(relevance_matrix, axis=-1)
