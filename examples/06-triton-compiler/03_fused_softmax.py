"""
Example 3: Fused Softmax with Triton
====================================

This example demonstrates kernel fusion in Triton with a softmax implementation.
It shows how Triton can fuse multiple operations into a single kernel for better performance.

Key Concepts:
- Kernel fusion: combining max, subtract, exp, and sum into one kernel
- Reduction operations with tl.max and tl.sum
- Online algorithms for numerical stability
- Row-wise processing for 2D tensors
- Performance benefits of fusion vs. separate kernels
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused softmax kernel that computes: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    This kernel performs all operations in a single pass:
    1. Find maximum value in the row (for numerical stability)
    2. Subtract max and compute exponentials
    3. Sum the exponentials
    4. Normalize by dividing by the sum

    Traditional approach would require 4 separate kernel launches:
    - Kernel 1: Find max
    - Kernel 2: Subtract max
    - Kernel 3: Exp and sum
    - Kernel 4: Divide by sum

    Triton's fusion eliminates intermediate global memory writes, significantly improving performance.
    """
    # Get the row index this program will process
    row_idx = tl.program_id(0)

    # Compute the starting pointer for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Create offsets for elements in this row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Mask for valid elements (handles cases where n_cols < BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load the row into SRAM (registers/shared memory)
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Step 1: Find the maximum value in the row (for numerical stability)
    # Using max normalization prevents overflow in exp()
    # softmax(x) = softmax(x - c) for any constant c
    row_max = tl.max(row, axis=0)

    # Step 2: Subtract max and compute exponentials
    # This is numerically stable: exp(x - max(x)) is always in (0, 1]
    row_minus_max = row - row_max
    numerator = tl.exp(row_minus_max)

    # Step 3: Compute the sum of exponentials (denominator)
    denominator = tl.sum(numerator, axis=0)

    # Step 4: Normalize to get softmax values
    softmax_output = numerator / denominator

    # Step 5: Write the result back to global memory
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax using Triton's fused kernel.

    Args:
        x: Input tensor of shape (..., n_cols)
           Will be flattened to (n_rows, n_cols) for processing

    Returns:
        Softmax output with same shape as input
    """
    # Save original shape for later reshaping
    original_shape = x.shape

    # Flatten to 2D: (n_rows, n_cols)
    # Each row will be processed independently
    x_2d = x.view(-1, x.shape[-1])

    n_rows, n_cols = x_2d.shape

    # Allocate output
    y = torch.empty_like(x_2d)

    # Choose block size (must be power of 2 >= n_cols)
    # Round up to nearest power of 2 for efficiency
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Limit block size to avoid excessive register usage
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)

    # Launch one program per row
    grid = (n_rows,)

    # Launch kernel
    softmax_kernel[grid](
        x_2d, y,
        x_2d.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape output to match input
    return y.view(original_shape)


@triton.jit
def softmax_kernel_online(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online softmax kernel for very large n_cols (> BLOCK_SIZE).

    This version processes the row in chunks, maintaining running max and sum.
    Uses the online algorithm to avoid storing the entire row in registers.

    Algorithm:
    - Process row in chunks of BLOCK_SIZE
    - Maintain running max (m) and running sum (l)
    - When max changes, rescale previous sum: l_new = l_old * exp(m_old - m_new)
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Initialize running values
    m = -float('inf')  # Running max
    l = 0.0  # Running sum of exp

    # First pass: compute max and sum
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row_block = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

        # Update running max
        block_max = tl.max(row_block, axis=0)
        m_new = tl.maximum(m, block_max)

        # Rescale previous sum when max changes
        # l = l * exp(m - m_new) + sum(exp(row_block - m_new))
        l = l * tl.exp(m - m_new)
        l += tl.sum(tl.exp(row_block - m_new), axis=0)

        m = m_new

    # Second pass: compute and write softmax
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row_block = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

        softmax_output = tl.exp(row_block - m) / l

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def benchmark_softmax(n_rows: int, n_cols: int):
    """
    Benchmark Triton fused softmax against PyTorch's implementation.

    Args:
        n_rows: Number of rows
        n_cols: Number of columns
    """
    print(f"\nBenchmarking Softmax ({n_rows} x {n_cols})")
    print("=" * 60)

    # Create random input on GPU
    x = torch.randn(n_rows, n_cols, device='cuda')

    # Warm up
    for _ in range(10):
        _ = softmax_triton(x)
    torch.cuda.synchronize()

    # Benchmark Triton
    import time
    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        y_triton = softmax_triton(x)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / iterations

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(iterations):
        y_torch = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / iterations

    # Verify correctness
    max_error = torch.max(torch.abs(y_triton - y_torch)).item()
    assert max_error < 1e-5, f"Results don't match! Max error: {max_error}"

    # Calculate bandwidth (bytes read + bytes written)
    # Read entire input, write entire output
    bytes_transferred = x.numel() * x.element_size() * 2
    triton_bandwidth = bytes_transferred / triton_time / 1e9
    torch_bandwidth = bytes_transferred / torch_time / 1e9

    # Print results
    print(f"Triton:  {triton_time*1e3:.2f} ms ({triton_bandwidth:.1f} GB/s)")
    print(f"PyTorch: {torch_time*1e3:.2f} ms ({torch_bandwidth:.1f} GB/s)")
    print(f"Speedup: {torch_time/triton_time:.2f}x")
    print(f"Max error: {max_error:.2e}")


def main():
    """
    Main function demonstrating Triton fused softmax.
    """
    print("Triton Fused Softmax Example")
    print("=" * 60)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires a GPU.")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")

    # Example 1: Simple softmax
    print("Example 1: Simple Softmax (1D)")
    print("-" * 60)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device='cuda')
    y = softmax_triton(x)
    print(f"Input:  {x.cpu().numpy()}")
    print(f"Output: {y.cpu().numpy()}")
    print(f"Sum:    {y.sum().item():.6f} (should be 1.0)")

    # Example 2: Batch softmax
    print("\n\nExample 2: Batch Softmax (2D)")
    print("-" * 60)
    x = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], device='cuda')
    y = softmax_triton(x)
    print(f"Input shape: {x.shape}")
    print(f"Input:\n{x.cpu().numpy()}")
    print(f"\nOutput:\n{y.cpu().numpy()}")
    print(f"Row sums: {y.sum(dim=-1).cpu().numpy()} (should all be 1.0)")

    # Example 3: Numerical stability test
    print("\n\nExample 3: Numerical Stability (Large Values)")
    print("-" * 60)
    x = torch.tensor([[1000.0, 2000.0, 3000.0]], device='cuda')
    y = softmax_triton(x)
    print(f"Input:  {x.cpu().numpy()}")
    print(f"Output: {y.cpu().numpy()}")
    print(f"Note: Triton's max normalization prevents overflow")

    # Example 4: Compare with PyTorch
    print("\n\nExample 4: Verification Against PyTorch")
    print("-" * 60)
    x = torch.randn(5, 10, device='cuda')
    y_triton = softmax_triton(x)
    y_torch = torch.softmax(x, dim=-1)
    max_error = torch.max(torch.abs(y_triton - y_torch)).item()
    print(f"Input shape: {x.shape}")
    print(f"Max error vs PyTorch: {max_error:.2e}")
    print(f"Verification: {'PASSED' if max_error < 1e-5 else 'FAILED'}")

    # Benchmarks
    print("\n\nPerformance Benchmarks")
    print("=" * 60)
    print("Note: Triton shines with kernel fusion, reducing memory bandwidth")

    # Small batch, small sequence
    benchmark_softmax(128, 512)

    # Medium batch, medium sequence (typical transformer layer)
    benchmark_softmax(256, 1024)

    # Large batch, large sequence
    benchmark_softmax(512, 2048)

    # Very wide (attention heads in transformers)
    benchmark_softmax(64, 4096)


if __name__ == "__main__":
    main()
