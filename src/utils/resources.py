"""
    Utility functions for computing model resource consumption (number of parameters, FLOPs, latency)
    for a given input size and saving the results to a JSON file.
"""

from time import perf_counter_ns
import json
import os
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table


def compute_model_size(model):
    """
    Computes the total number of (trainable) parameters in the model.
    
    Parameters:
        :model: PyTorch model
    Returns:
        :num_params: total number of trainable parameters
        :num_trainable_params: total number of parameters that require gradients
    """
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params, num_trainable_params

@torch.no_grad()
def compute_latency_cuda(model, model_input, num_iterations):
    """
    Computes the latency of a forward pass through the model on CUDA.
    
    Parameters:
        :model: PyTorch model
        :model_input: dictionary containing the input tensors for the model
        :num_iterations: number of iterations to average latency over
    Returns:
        :latency: latency in milliseconds
    """
    
    # Warm-up runs
    for _ in range(20):
        _ = model(**model_input)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    
    for _ in range(num_iterations):
        _ = model(**model_input)
    
    end_event.record()

    torch.cuda.synchronize()  # Wait for all operations to finish
    
    latency = start_event.elapsed_time(end_event) / num_iterations  # Latency in milliseconds
    return latency

@torch.no_grad()
def compute_latency_mps(model, model_input, num_iterations):
    """
    Computes the latency of a forward pass through the model on MPS (Apple Silicon).
    
    Parameters:
        :model: PyTorch model
        :model_input: dictionary containing the input tensors for the model
        :num_iterations: number of iterations to average latency over
    Returns:
        :latency: latency in milliseconds
    """
    # Warm-up runs
    for _ in range(20):
        _ = model(**model_input)
    
    torch.mps.synchronize()  # Ensure all previous operations are complete before starting timing
    
    start_time = perf_counter_ns()
    for _ in range(num_iterations):
        _ = model(**model_input)
    
    torch.mps.synchronize()  # Ensure all operations are complete before stopping timing
    
    end_time = perf_counter_ns()
    
    latency = (end_time - start_time) / (1e6 * num_iterations)  # Latency in milliseconds
    return latency
    
@torch.no_grad()
def compute_latency_cpu(model, model_input, num_iterations):
    """
    Computes the latency of a forward pass through the model on CPU.
    
    Parameters:
        :model: PyTorch model
        :model_input: dictionary containing the input tensors for the model
        :num_iterations: number of iterations to average latency over
    Returns:
        :latency: latency in milliseconds
    """
    
    # Warm-up runs
    for _ in range(10):
        _ = model(**model_input)
    
    start_time = perf_counter_ns()
    
    for _ in range(num_iterations):
        _ = model(**model_input)
    
    end_time = perf_counter_ns()
    
    latency = (end_time - start_time) / (1e6 * num_iterations)  # Latency in milliseconds
    return latency

@torch.no_grad()
def compute_model_latency(model, model_input, device, num_runs):
    """
    Computes the average latency of a forward pass through the model.
    
    Parameters:
        :model: model to compute latency for
        :model_input: dictionary containing the input tensors for the model
        :num_runs: number of runs to average latency over
    Returns:
        :avg_latency: average latency in milliseconds
    """
    
    if device == "cuda":
        latency = compute_latency_cuda(model, model_input, num_runs)
    elif device == "mps":
        latency = compute_latency_mps(model, model_input, num_runs)
    else:
        latency = compute_latency_cpu(model, model_input, num_runs)
        
    return latency

@torch.no_grad()
def compute_flops(model, model_input, device):
    """
    Computes the FLOPs of the model for a given input size using fvcore.
    
    Parameters:
        :model: PyTorch model
        :model_input: dictionary containing the input tensors for the model
        :device: device to perform the computation on ("cpu", "cuda", "mps")
    Returns:
        :flops: total number of floating point operations for a forward pass
    """
    model = model.to(device)
    inputs = tuple(input_tensor.to(device) for input_tensor in model_input.values())
    flop_analyzer = FlopCountAnalysis(model, inputs=inputs)
    flops = flop_analyzer.total()

    return {
        "count": flops,
        "table": flop_count_table(flop_analyzer)
    }

@torch.no_grad()
def compute_resource_consumption(model, lidar, img_size, batch_size, device="mps", num_runs=100, save_to="./temp"):
    """
    Computes the number of parameters and FLOPs of the model for a given input size.
    
    Parameters:
        :model: the model to compute resource consumption for
        :lidar: whether the model uses LiDAR data
        :img_size: tuple representing the image size (W, H)
        :batch_size: size of the batch
        :device: device to perform the computation on ("cpu", "cuda", "mps")
        :num_runs: number of runs to average latency over
    Returns:
        :num_params: total number of trainable parameters in the model
        :flops: total number of floating point operations for a forward pass
    """
    model.eval()
    model = model.to(device)
    
    img_size = (img_size[1], img_size[0])  # Convert to (H, W) for input tensor shape
    input_img = torch.randn(batch_size, 3, *img_size).to(device)
    lidar_img = torch.randn(batch_size, 3, *img_size).to(device)
    lidar_mask = torch.ones(batch_size, 1, *img_size).to(device)
    
    if lidar:
        model_input = {"pixel_values": input_img, "lidar_values": lidar_img, "lidar_mask": lidar_mask}
    else:
        model_input = {"pixel_values": input_img}

    num_params, num_trainable_params = compute_model_size(model)
    latency = compute_model_latency(model, model_input, device, num_runs)
    flop_result = compute_flops(model, model_input, "cpu")
    
    result = {
        "num_params": num_params,
        "num_trainable_params": num_trainable_params,
        "latency": latency,
        "flops": flop_result
    }

    # Save the result to a JSON file
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        json_str = json.dumps(result, indent=4)
        with open(os.path.join(save_to, "resource_consumption.json"), "w") as f:
            f.write(json_str)

    return result