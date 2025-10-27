"""
Example 2: Matrix Multiplication with Triton
============================================

This example demonstrates matrix multiplication using Triton's tiling optimization.
It shows how Triton automatically uses shared memory for performance.

Key Concepts:
- 2D grid configuration with tl.program_id(0) and tl.program_id(1)
- Shared memory tiling for efficient matrix multiplication
- Block-level computation with tl.dot
- Memory coalescing and bank conflict avoidance
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,  # C is M x N, A is M x K, B is K x N
    # Strides for each dimension (for handling non-contiguous tensors)
    stride_am, stride_ak,  # A strides
    stride_bk, stride_bn,  # B strides
    stride_cm, stride_cn,  # C strides
    # Meta-parameters (compile-time constants)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  # For better L2 cache utilization
):
    """
    Triton kernel for matrix multiplication: C = A @ B

    This kernel uses tiling to optimize memory access:
    1. Load tiles of A and B into shared memory
    2. Compute partial products on tiles
    3. Accumulate results in registers
    4. Write final result to C

    The tiling strategy significantly improves performance by:
    - Reducing global memory accesses
    - Maximizing reuse of data in shared memory
    - Enabling better memory coalescing
    """
    # Program ID represents the position of the current block in the output matrix
    pid = tl.program_id(axis=0)

    # Swizzle the program ID for better L2 cache utilization
    # This groups blocks together that access nearby memory
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B
    # These represent the top-left corners of the tiles we'll load
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create pointers to A and B tiles
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator for the output tile
    # This will hold the partial sums as we process K dimension
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_SIZE_K
    # This is the core tiling loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next tiles of A and B into shared memory
        # Triton automatically manages shared memory allocation
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # Perform matrix multiplication on the tiles
        # tl.dot is optimized to use tensor cores on supported GPUs
        accumulator += tl.dot(a, b)

        # Advance pointers to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Convert accumulator to output dtype
    c = accumulator.to(tl.float32)

    # Write back the results to C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Host function to launch the Triton matrix multiplication kernel.

    Args:
        a: First input matrix of shape (M, K)
        b: Second input matrix of shape (K, N)

    Returns:
        c: Output matrix of shape (M, N) containing a @ b
    """
    # Check input constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA device"
    assert a.is_contiguous() and b.is_contiguous(), "Tensors must be contiguous"

    M, K = a.shape
    K, N = b.shape

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tile sizes (these can be tuned for different GPUs)
    # Larger tiles = more shared memory usage but better reuse
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8  # For L2 cache optimization

    # Create grid
    # We need enough blocks to cover the entire output matrix
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return c


def benchmark_matmul(M: int, N: int, K: int, dtype=torch.float32):
    """
    Benchmark Triton matrix multiplication against PyTorch's implementation.

    Args:
        M, N, K: Matrix dimensions (A: MxK, B: KxN, C: MxN)
        dtype: Data type for tensors
    """
    print(f"\nBenchmarking Matrix Multiplication ({M}x{K} @ {K}x{N}, dtype={dtype})")
    print("=" * 60)

    # Create random input matrices on GPU
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)

    # Warm up GPU
    for _ in range(10):
        _ = matmul_triton(a, b)
    torch.cuda.synchronize()

    # Benchmark Triton implementation
    import time
    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        c_triton = matmul_triton(a, b)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / iterations

    # Benchmark PyTorch implementation (uses cuBLAS)
    start = time.perf_counter()
    for _ in range(iterations):
        c_torch = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / iterations

    # Verify correctness
    assert torch.allclose(c_triton, c_torch, rtol=1e-2, atol=1e-2), "Results don't match!"

    # Calculate FLOPS (floating point operations per second)
    # Matrix multiplication performs 2*M*N*K operations (multiply and add)
    flops = 2.0 * M * N * K
    triton_tflops = flops / triton_time / 1e12
    torch_tflops = flops / torch_time / 1e12

    # Print results
    print(f"Triton:  {triton_time*1e3:.2f} ms ({triton_tflops:.2f} TFLOPS)")
    print(f"PyTorch: {torch_time*1e3:.2f} ms ({torch_tflops:.2f} TFLOPS)")
    print(f"Relative performance: {triton_tflops/torch_tflops*100:.1f}% of PyTorch")


def main():
    """
    Main function demonstrating Triton matrix multiplication.
    """
    print("Triton Matrix Multiplication Example")
    print("=" * 60)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires a GPU.")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")

    # Example 1: Small matrix multiplication
    print("Example 1: Small Matrix Multiplication")
    print("-" * 60)
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    c = matmul_triton(a, b)
    print(f"A ({a.shape}):")
    print(a.cpu().numpy())
    print(f"\nB ({b.shape}):")
    print(b.cpu().numpy())
    print(f"\nC = A @ B ({c.shape}):")
    print(c.cpu().numpy())

    # Verify against PyTorch
    c_torch = torch.matmul(a, b)
    print(f"\nVerification: max error = {torch.max(torch.abs(c - c_torch)).item():.2e}")

    # Example 2: Rectangular matrices
    print("\n\nExample 2: Rectangular Matrices")
    print("-" * 60)
    M, K, N = 512, 256, 1024
    a = torch.randn((M, K), device='cuda')
    b = torch.randn((K, N), device='cuda')
    c = matmul_triton(a, b)
    print(f"Successfully computed ({M}x{K}) @ ({K}x{N}) = ({M}x{N})")
    print(f"Sample output (top-left 3x3):")
    print(c[:3, :3].cpu().numpy())

    # Benchmarks
    print("\n\nPerformance Benchmarks")
    print("=" * 60)

    # Small matrix
    benchmark_matmul(512, 512, 512, dtype=torch.float32)

    # Medium matrix
    benchmark_matmul(1024, 1024, 1024, dtype=torch.float32)

    # Large matrix
    benchmark_matmul(2048, 2048, 2048, dtype=torch.float32)

    # Rectangular matrix (common in neural networks)
    benchmark_matmul(4096, 1024, 4096, dtype=torch.float32)


if __name__ == "__main__":
    main()
