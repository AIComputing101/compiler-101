"""
Example 1: Vector Addition with Triton
======================================

This example demonstrates the basic Triton kernel for vector addition.
It shows how Triton simplifies GPU programming compared to CUDA/HIP.

Key Concepts:
- @triton.jit decorator for JIT compilation
- tl.program_id for thread block indexing
- tl.load/tl.store for memory operations
- Automatic vectorization and memory coalescing
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    a_ptr,  # Pointer to input tensor a
    b_ptr,  # Pointer to input tensor b
    c_ptr,  # Pointer to output tensor c
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block (compile-time constant)
):
    """
    Triton kernel for element-wise vector addition: c = a + b

    Each thread block processes BLOCK_SIZE elements.
    Triton automatically handles vectorization and memory coalescing.
    """
    # Get the current block's ID
    # This is similar to blockIdx.x in CUDA, but abstracted
    block_id = tl.program_id(axis=0)

    # Calculate the starting index for this block
    block_start = block_id * BLOCK_SIZE

    # Generate offsets for elements within this block
    # tl.arange creates [0, 1, 2, ..., BLOCK_SIZE-1]
    # Adding block_start gives us global indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to prevent out-of-bounds memory access
    # This handles cases where n_elements is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements

    # Load data from global memory
    # The mask ensures we don't read beyond the array bounds
    # Triton automatically coalesces these memory accesses for efficiency
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # Perform the computation
    # Triton automatically vectorizes this operation
    c = a + b

    # Store the result back to global memory
    tl.store(c_ptr + offsets, c, mask=mask)


def vector_add_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Host function to launch the Triton vector addition kernel.

    Args:
        a: First input tensor (must be on CUDA)
        b: Second input tensor (must be on CUDA)

    Returns:
        c: Output tensor containing a + b
    """
    # Ensure inputs are on the same device and have the same shape
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA device"
    assert a.shape == b.shape, "Input tensors must have the same shape"

    # Get total number of elements
    n_elements = a.numel()

    # Allocate output tensor
    c = torch.empty_like(a)

    # Choose block size (power of 2, typically 256-1024)
    # Triton will optimize for the specific GPU architecture
    BLOCK_SIZE = 1024

    # Calculate grid size (number of blocks needed)
    # We need enough blocks to cover all elements
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    # The grid parameter is a function that returns the grid dimensions
    # Triton will pass metadata (including BLOCK_SIZE) to this function
    vector_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)

    return c


def benchmark_vector_add(size: int = 1_000_000, dtype=torch.float32):
    """
    Benchmark Triton vector addition against PyTorch's native implementation.

    Args:
        size: Number of elements in vectors
        dtype: Data type for tensors
    """
    print(f"\nBenchmarking Vector Addition (size={size:,}, dtype={dtype})")
    print("=" * 60)

    # Create random input tensors on GPU
    a = torch.randn(size, device='cuda', dtype=dtype)
    b = torch.randn(size, device='cuda', dtype=dtype)

    # Warm up GPU
    for _ in range(10):
        _ = vector_add_triton(a, b)
    torch.cuda.synchronize()

    # Benchmark Triton implementation
    import time
    start = time.perf_counter()
    for _ in range(100):
        c_triton = vector_add_triton(a, b)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / 100

    # Benchmark PyTorch implementation
    start = time.perf_counter()
    for _ in range(100):
        c_torch = a + b
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / 100

    # Verify correctness
    assert torch.allclose(c_triton, c_torch, rtol=1e-5), "Results don't match!"

    # Print results
    print(f"Triton:  {triton_time*1e6:.2f} μs")
    print(f"PyTorch: {torch_time*1e6:.2f} μs")
    print(f"Speedup: {torch_time/triton_time:.2f}x")
    print(f"Bandwidth (Triton): {3*size*a.element_size()/triton_time/1e9:.2f} GB/s")


def main():
    """
    Main function demonstrating Triton vector addition.
    """
    print("Triton Vector Addition Example")
    print("=" * 60)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires a GPU.")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")

    # Example 1: Simple vector addition
    print("Example 1: Simple Vector Addition")
    print("-" * 60)
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    b = torch.tensor([5.0, 6.0, 7.0, 8.0], device='cuda')
    c = vector_add_triton(a, b)
    print(f"a = {a.cpu().numpy()}")
    print(f"b = {b.cpu().numpy()}")
    print(f"c = {c.cpu().numpy()}")

    # Example 2: Large vector addition
    print("\nExample 2: Large Vector Addition")
    print("-" * 60)
    size = 10_000_000
    a = torch.randn(size, device='cuda')
    b = torch.randn(size, device='cuda')
    c = vector_add_triton(a, b)
    print(f"Successfully added two vectors of size {size:,}")
    print(f"First 5 elements of result: {c[:5].cpu().numpy()}")

    # Example 3: Non-multiple of block size
    print("\nExample 3: Size Not Multiple of Block Size")
    print("-" * 60)
    size = 1337  # Not a multiple of 1024
    a = torch.randn(size, device='cuda')
    b = torch.randn(size, device='cuda')
    c = vector_add_triton(a, b)
    print(f"Successfully handled size {size} (not a multiple of BLOCK_SIZE=1024)")

    # Benchmark
    benchmark_vector_add(size=1_000_000, dtype=torch.float32)
    benchmark_vector_add(size=10_000_000, dtype=torch.float32)


if __name__ == "__main__":
    main()
