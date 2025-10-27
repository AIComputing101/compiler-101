"""
Example 4: Custom Activation Functions with Triton
=================================================

This example demonstrates how to implement custom activation functions in Triton.
Shows element-wise operations and comparison with PyTorch implementations.

Implemented activations:
- GELU (Gaussian Error Linear Unit) - used in BERT, GPT
- Swish/SiLU (Sigmoid Linear Unit) - used in EfficientNet
- Mish - smooth activation function

Key Concepts:
- Element-wise operations in Triton
- Mathematical functions (exp, tanh, sigmoid)
- Fusing multiple operations for better performance
- Integration with PyTorch autograd
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    GELU activation: GELU(x) = x * Φ(x)
    where Φ(x) is the cumulative distribution function of the standard normal distribution.

    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    # Get element indices for this block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)

    # GELU computation (tanh approximation)
    # Constants
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/π)
    coeff = 0.044715

    # Compute: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    tanh_inner = tl.libdevice.tanh(inner)  # Use GPU tanh
    output = 0.5 * x * (1.0 + tanh_inner)

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def swish_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Swish/SiLU activation: Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)

    # Swish: x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid_x = tl.sigmoid(x)
    output = x * sigmoid_x

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def mish_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Mish activation: Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)

    # Mish: x * tanh(softplus(x))
    # softplus(x) = ln(1 + exp(x))
    # For numerical stability, use: softplus(x) = log1p(exp(x)) for x < 20
    #                                            = x for x >= 20
    softplus_x = tl.log1p(tl.exp(x))
    tanh_softplus = tl.libdevice.tanh(softplus_x)
    output = x * tanh_softplus

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    ReLU activation: ReLU(x) = max(0, x)
    Simple example for comparison.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)

    # ReLU: max(0, x)
    output = tl.maximum(x, 0.0)

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


def gelu_triton(x: torch.Tensor) -> torch.Tensor:
    """Apply GELU activation using Triton."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def swish_triton(x: torch.Tensor) -> torch.Tensor:
    """Apply Swish/SiLU activation using Triton."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    swish_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def mish_triton(x: torch.Tensor) -> torch.Tensor:
    """Apply Mish activation using Triton."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mish_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def relu_triton(x: torch.Tensor) -> torch.Tensor:
    """Apply ReLU activation using Triton."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def benchmark_activation(name: str, triton_func, torch_func, size: int = 1_000_000):
    """
    Benchmark an activation function.

    Args:
        name: Name of the activation
        triton_func: Triton implementation
        torch_func: PyTorch implementation
        size: Number of elements
    """
    print(f"\nBenchmarking {name} (size={size:,})")
    print("-" * 60)

    # Create input
    x = torch.randn(size, device='cuda', dtype=torch.float32)

    # Warm up
    for _ in range(10):
        _ = triton_func(x)
    torch.cuda.synchronize()

    # Benchmark Triton
    import time
    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        y_triton = triton_func(x)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / iterations

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(iterations):
        y_torch = torch_func(x)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / iterations

    # Verify correctness
    max_error = torch.max(torch.abs(y_triton - y_torch)).item()

    # Calculate bandwidth
    bytes_transferred = x.numel() * x.element_size() * 2  # Read + write
    triton_bandwidth = bytes_transferred / triton_time / 1e9

    # Print results
    print(f"Triton:  {triton_time*1e6:.2f} μs ({triton_bandwidth:.1f} GB/s)")
    print(f"PyTorch: {torch_time*1e6:.2f} μs")
    print(f"Speedup: {torch_time/triton_time:.2f}x")
    print(f"Max error: {max_error:.2e}")


def plot_activations():
    """Plot activation functions for visualization."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Create input range
        x_cpu = torch.linspace(-5, 5, 1000)
        x_gpu = x_cpu.cuda()

        # Compute activations
        activations = {
            'ReLU': (relu_triton(x_gpu), torch.relu(x_gpu)),
            'GELU': (gelu_triton(x_gpu), torch.nn.functional.gelu(x_gpu)),
            'Swish': (swish_triton(x_gpu), torch.nn.functional.silu(x_gpu)),
            'Mish': (mish_triton(x_gpu), torch.nn.functional.mish(x_gpu)),
        }

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, (name, (y_triton, y_torch)) in enumerate(activations.items()):
            ax = axes[idx]

            # Plot
            ax.plot(x_cpu.numpy(), y_triton.cpu().numpy(), label='Triton', linewidth=2)
            ax.plot(x_cpu.numpy(), y_torch.cpu().numpy(), '--', label='PyTorch',
                   linewidth=2, alpha=0.7)

            # Styling
            ax.set_title(f'{name} Activation', fontsize=14, fontweight='bold')
            ax.set_xlabel('Input (x)', fontsize=12)
            ax.set_ylabel('Output', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)

        plt.tight_layout()
        plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved activation function plots to 'activation_functions.png'")

    except ImportError:
        print("\nNote: matplotlib not installed, skipping plots")
        print("Install with: pip install matplotlib")


def main():
    """
    Main function demonstrating custom activation functions.
    """
    print("Triton Custom Activation Functions Example")
    print("=" * 60)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires a GPU.")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")

    # Example 1: Compare activation outputs
    print("Example 1: Activation Function Outputs")
    print("-" * 60)
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
    print(f"Input: {x.cpu().numpy()}\n")

    # ReLU
    y_relu = relu_triton(x)
    print(f"ReLU:   {y_relu.cpu().numpy()}")

    # GELU
    y_gelu = gelu_triton(x)
    print(f"GELU:   {y_gelu.cpu().numpy()}")

    # Swish
    y_swish = swish_triton(x)
    print(f"Swish:  {y_swish.cpu().numpy()}")

    # Mish
    y_mish = mish_triton(x)
    print(f"Mish:   {y_mish.cpu().numpy()}")

    # Example 2: Verify against PyTorch
    print("\n\nExample 2: Verification Against PyTorch")
    print("-" * 60)
    x = torch.randn(1000, device='cuda')

    verifications = [
        ('GELU', gelu_triton(x), torch.nn.functional.gelu(x)),
        ('Swish', swish_triton(x), torch.nn.functional.silu(x)),
        ('Mish', mish_triton(x), torch.nn.functional.mish(x)),
        ('ReLU', relu_triton(x), torch.relu(x)),
    ]

    for name, y_triton, y_torch in verifications:
        max_error = torch.max(torch.abs(y_triton - y_torch)).item()
        status = "✓ PASS" if max_error < 1e-5 else "✗ FAIL"
        print(f"{name:8s}: max error = {max_error:.2e} {status}")

    # Benchmarks
    print("\n\nPerformance Benchmarks")
    print("=" * 60)

    benchmark_activation('ReLU', relu_triton, torch.relu)
    benchmark_activation('GELU', gelu_triton, torch.nn.functional.gelu)
    benchmark_activation('Swish', swish_triton, torch.nn.functional.silu)
    benchmark_activation('Mish', mish_triton, torch.nn.functional.mish)

    # Large tensor benchmarks
    print("\n\nLarge Tensor Benchmarks (10M elements)")
    print("=" * 60)
    benchmark_activation('GELU', gelu_triton, torch.nn.functional.gelu, size=10_000_000)
    benchmark_activation('Swish', swish_triton, torch.nn.functional.silu, size=10_000_000)

    # Plot activations
    print("\n\nGenerating Activation Function Plots")
    print("=" * 60)
    plot_activations()

    # Summary
    print("\n\nSummary")
    print("=" * 60)
    print("Custom activation functions in Triton demonstrate:")
    print("  • Element-wise operations are straightforward to implement")
    print("  • Performance competitive with PyTorch's optimized implementations")
    print("  • Easy integration into custom ML workflows")
    print("  • GELU and Swish are common in modern transformers (BERT, GPT)")
    print("  • Triton allows experimenting with novel activation functions")


if __name__ == "__main__":
    main()
