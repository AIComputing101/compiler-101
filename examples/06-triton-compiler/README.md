# Module 6: Triton Compiler Examples

This directory contains practical examples demonstrating the Triton Compiler's capabilities for high-performance GPU programming with Python.

## Overview

Triton is a language and compiler for writing highly efficient custom Deep Learning primitives. The purpose of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but with comparable efficiency.

These examples progress from basic to advanced GPU programming concepts:

1. **Vector Addition** - Basic GPU kernel with automatic vectorization
2. **Matrix Multiplication** - Tiling and shared memory optimization
3. **Fused Softmax** - Kernel fusion for reduced memory bandwidth
4. **Custom Activations** - ML activation functions (GELU, Swish, Mish)
5. **AMD Optimization** - Advanced optimization with bank conflict avoidance

## Prerequisites

### System Requirements
- NVIDIA GPU with CUDA support (Compute Capability 7.0+) or AMD GPU with ROCm
- Linux or Windows with WSL2
- Python 3.8 or higher
- CUDA Toolkit 11.4+ (for NVIDIA) or ROCm 5.0+ (for AMD)

### Installation

1. **Install PyTorch with CUDA support:**
   ```bash
   # For NVIDIA GPUs (CUDA 11.8)
   pip install torch --index-url https://download.pytorch.org/whl/cu118

   # For AMD GPUs (ROCm 5.7)
   pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
   ```

2. **Install Triton:**
   ```bash
   pip install triton
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import triton; print(f'Triton version: {triton.__version__}')"
   ```

### GPU Check

Ensure your GPU is properly configured:

```bash
# For NVIDIA
nvidia-smi

# For AMD
rocm-smi
```

## Examples

### Example 1: Vector Addition

**File:** [`01_vector_addition.py`](01_vector_addition.py)

**Concepts:**
- `@triton.jit` decorator for JIT compilation
- `tl.program_id()` for block indexing
- `tl.load()` and `tl.store()` for memory operations
- Automatic masking for boundary conditions
- Automatic vectorization

**Run:**
```bash
python 01_vector_addition.py
```

**Expected Output:**
```
Triton Vector Addition Example
============================================================
Using GPU: NVIDIA GeForce RTX 3090

Example 1: Simple Vector Addition
------------------------------------------------------------
a = [1. 2. 3. 4.]
b = [5. 6. 7. 8.]
c = [ 6.  8. 10. 12.]

...

Benchmarking Vector Addition (size=1,000,000, dtype=torch.float32)
============================================================
Triton:  125.34 μs
PyTorch: 132.45 μs
Speedup: 1.06x
Bandwidth (Triton): 95.82 GB/s
```

**What You'll Learn:**
- How Triton simplifies GPU programming compared to CUDA
- Automatic memory coalescing and vectorization
- Handling non-aligned array sizes with masking

---

### Example 2: Matrix Multiplication

**File:** [`02_matrix_multiplication.py`](02_matrix_multiplication.py)

**Concepts:**
- 2D grid configuration with `tl.program_id(0)` and `tl.program_id(1)`
- Shared memory tiling for efficient data reuse
- `tl.dot()` for optimized matrix operations (uses tensor cores when available)
- L2 cache optimization with block swizzling
- Handling non-square and non-power-of-2 matrices

**Run:**
```bash
python 02_matrix_multiplication.py
```

**Expected Output:**
```
Triton Matrix Multiplication Example
============================================================
Using GPU: NVIDIA GeForce RTX 3090

Example 1: Small Matrix Multiplication
------------------------------------------------------------
A ((2, 2)):
[[1. 2.]
 [3. 4.]]

B ((2, 2)):
[[5. 6.]
 [7. 8.]]

C = A @ B ((2, 2)):
[[19. 22.]
 [43. 50.]]

...

Benchmarking Matrix Multiplication (1024x1024 @ 1024x1024, dtype=torch.float32)
============================================================
Triton:  1.23 ms (1.75 TFLOPS)
PyTorch: 0.95 ms (2.26 TFLOPS)
Relative performance: 77.4% of PyTorch
```

**What You'll Learn:**
- How tiling reduces global memory accesses
- Shared memory management (automatic by Triton)
- Performance comparison with highly-optimized cuBLAS
- Block-level parallelism for matrix operations

**Note:** PyTorch uses cuBLAS (NVIDIA's highly-optimized BLAS library), which is hard to beat. Triton typically achieves 70-90% of cuBLAS performance, which is excellent for a high-level approach.

---

### Example 3: Fused Softmax

**File:** [`03_fused_softmax.py`](03_fused_softmax.py)

**Concepts:**
- Kernel fusion: combining multiple operations into one kernel
- Reduction operations with `tl.max()` and `tl.sum()`
- Numerical stability with max normalization
- Online algorithms for processing large sequences
- Memory bandwidth optimization

**Run:**
```bash
python 03_fused_softmax.py
```

**Expected Output:**
```
Triton Fused Softmax Example
============================================================
Using GPU: NVIDIA GeForce RTX 3090

Example 1: Simple Softmax (1D)
------------------------------------------------------------
Input:  [[1. 2. 3. 4.]]
Output: [[0.0320586  0.08714432 0.23688282 0.64391426]]
Sum:    1.000000 (should be 1.0)

...

Benchmarking Softmax (256 x 1024)
============================================================
Triton:  0.15 ms (13.7 GB/s)
PyTorch: 0.18 ms (11.4 GB/s)
Speedup: 1.20x
Max error: 5.96e-08
```

**What You'll Learn:**
- Benefits of kernel fusion (eliminating intermediate memory writes)
- How to implement numerically stable algorithms
- Row-wise parallel processing
- Reduction operations in Triton
- Memory bandwidth as a performance bottleneck

**Why Fusion Matters:**
Traditional approach requires 4 kernel launches:
1. Find max → write to memory
2. Subtract max → write to memory
3. Exp and sum → write to memory
4. Divide → write to memory

Triton's fused kernel does all operations in one pass, reading input once and writing output once.

---

### Example 4: Custom Activation Functions

**File:** [`04_custom_activation.py`](04_custom_activation.py)

**Concepts:**
- Element-wise operations in Triton
- Mathematical functions (exp, tanh, sigmoid)
- Common ML activation functions (GELU, Swish, Mish)
- Integration with PyTorch models

**Run:**
```bash
python 04_custom_activation.py
```

**Expected Output:**
```
Triton Custom Activation Functions Example
============================================================
Using GPU: NVIDIA GeForce RTX 3090

Example 1: Activation Function Outputs
------------------------------------------------------------
Input: [-2. -1.  0.  1.  2.]

ReLU:   [0.    0.    0.    1.    2.   ]
GELU:   [-0.046  -0.159   0.     0.841  1.955]
Swish:  [-0.238  -0.269   0.     0.731  1.762]
Mish:   [-0.252  -0.303   0.     0.865  1.944]

Benchmarking GELU (size=1,000,000)
------------------------------------------------------------
Triton:  82.34 μs (97.2 GB/s)
PyTorch: 85.12 μs
Speedup: 1.03x
Max error: 3.24e-08
```

**What You'll Learn:**
- How to implement custom element-wise operations
- Using Triton's math library (`tl.libdevice`)
- Performance comparison with PyTorch's implementations
- Common activation functions in modern ML (used in BERT, GPT, EfficientNet)

---

### Example 5: AMD GPU Optimization (Advanced)

**File:** [`05_optimized_matmul_amd.py`](05_optimized_matmul_amd.py)

**Concepts:**
- Hardware-specific optimizations
- Shared memory bank conflict avoidance
- Conditional compilation for different GPU architectures
- Performance profiling and validation
- Following best practices from [OPTIMIZE.md](OPTIMIZE.md)

**Run:**
```bash
python 05_optimized_matmul_amd.py
```

**Expected Output (AMD GPU):**
```
Optimized Matrix Multiplication for AMD GPUs
======================================================================

Detected GPU: AMD Instinct MI250
Vendor: AMD

Benchmarking Matrix Multiplication (1024x1024 @ 1024x1024)
======================================================================

Performance Results:
----------------------------------------------------------------------
Implementation       Time (ms)    TFLOPS       vs Baseline
----------------------------------------------------------------------
Baseline (Triton)        8.523      0.25            1.00x
Optimized (Triton)       7.218      0.30            1.18x
PyTorch (rocBLAS)        5.124      0.42            1.66x

Optimization Impact:
----------------------------------------------------------------------
✓ Optimized version is 15.3% faster than baseline
  This improvement is due to reduced LDS bank conflicts
```

**What You'll Learn:**
- Understanding GPU memory architecture (AMD LDS, NVIDIA shared memory)
- How bank conflicts impact performance (up to 50% slowdown)
- Shared memory padding technique to avoid conflicts
- Hardware-aware kernel design
- Profiling with rocprof (AMD) or nsys (NVIDIA)

**Advanced Tools:**

1. **Profiling Script:**
   ```bash
   python profile_matmul.py --size 2048 --detailed
   ```

2. **Comprehensive Benchmark:**
   ```bash
   python bench_optimization.py --plot --output results.json
   ```

3. **Read the Guide:**
   See [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) for detailed explanation

**Key Optimization:**
```
Problem: AMD LDS bank conflicts reduce performance by 50%+
Solution: Add 1 column padding to shared memory tiles (16x17 instead of 16x16)
Result: 15-20% throughput improvement on AMD GPUs
Cost: 6.25% memory overhead, minimal impact on NVIDIA
```

---

## Performance Tips

### 1. Block Size Selection
- Choose block sizes as powers of 2 (256, 512, 1024)
- Larger blocks = more data reuse but higher register pressure
- Triton's `triton.cdiv()` helps calculate grid sizes

### 2. Memory Access Patterns
- Triton automatically coalesces memory accesses
- Keep data contiguous in memory when possible
- Use `tensor.contiguous()` before passing to kernels

### 3. Compile-Time Constants
- Use `tl.constexpr` for parameters known at compile time
- Allows Triton to generate optimized code for specific sizes

### 4. Debugging
- Start with small inputs to verify correctness
- Compare against PyTorch implementations
- Use `torch.allclose()` with appropriate tolerances (rtol=1e-5)

### 5. Benchmarking
- Warm up GPU before timing (10+ iterations)
- Use `torch.cuda.synchronize()` for accurate timing
- Run multiple iterations (100+) and take average

## Understanding Performance Results

### Vector Addition
- Memory-bound operation (limited by bandwidth, not compute)
- Triton comparable to PyTorch because both are bandwidth-limited
- Peak bandwidth depends on GPU model (RTX 3090 ≈ 936 GB/s)

### Matrix Multiplication
- Compute-bound operation (limited by FLOPS, not bandwidth)
- PyTorch uses cuBLAS, which is extremely optimized
- Triton achieving 70-90% of cuBLAS is excellent for a high-level approach
- Uses tensor cores on supported GPUs (Turing, Ampere, Ada)

### Fused Softmax
- Triton excels here due to kernel fusion
- Often 1.2-2x faster than PyTorch
- Demonstrates Triton's strength in custom operations

## Troubleshooting

### "CUDA out of memory"
- Reduce batch sizes or matrix dimensions
- Check GPU memory usage: `nvidia-smi` or `rocm-smi`
- Clear cache: `torch.cuda.empty_cache()`

### "No CUDA-capable device detected"
- Verify GPU with `nvidia-smi` or `rocm-smi`
- Check PyTorch CUDA installation: `torch.cuda.is_available()`
- Reinstall PyTorch with correct CUDA/ROCm version

### "Triton compilation failed"
- Update Triton: `pip install --upgrade triton`
- Check for unsupported GPU (need Compute Capability 7.0+)
- Verify CUDA/ROCm toolkit installation

### Performance lower than expected
- Ensure GPU is not thermal throttling: check temperatures
- Close other GPU-intensive applications
- Use larger problem sizes (small problems have overhead)
- Check power management settings (performance mode)

## Learning Path

1. **Start with Vector Addition**
   - Understand basic Triton syntax
   - Learn about program IDs and masking
   - Compare to simple CUDA kernels (Module 4)

2. **Progress to Matrix Multiplication**
   - Understand tiling and shared memory
   - Learn about 2D grids
   - Appreciate automatic optimizations

3. **Explore Fused Softmax**
   - Learn kernel fusion benefits
   - Understand reduction operations
   - See real-world ML applications

4. **Experiment Further**
   - Modify block sizes and observe performance changes
   - Try different tensor sizes
   - Implement custom operations (ReLU, layer norm, etc.)

## Additional Resources

### Official Documentation
- [Triton Documentation](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [PyTorch + Triton Tutorial](https://pytorch.org/tutorials/intermediate/triton_tutorial.html)

### Research Papers
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

### Related Modules
- **Module 4:** NVIDIA CUDA Compiler - Compare low-level CUDA to high-level Triton
- **Module 5:** AMD HIP Compiler - Understand cross-platform GPU programming
- **Module 1:** C Compiler Basics - Foundation for understanding compilation

### Community
- [Triton Discussions](https://github.com/openai/triton/discussions)
- [PyTorch Forums - GPU Section](https://discuss.pytorch.org/c/gpu/9)

## Next Steps

After completing these examples, you can:

1. **Implement Custom ML Operations**
   - Layer normalization
   - Fused attention mechanisms
   - Custom activation functions

2. **Optimize Real Workloads**
   - Integrate Triton kernels into PyTorch models
   - Profile and optimize bottlenecks
   - Compare against cuDNN/cuBLAS

3. **Explore Advanced Topics**
   - Multi-GPU programming
   - Mixed-precision training (FP16, BF16)
   - Sparse operations

4. **Study Production Use Cases**
   - FlashAttention (attention optimization)
   - PyTorch's `torch.compile()` (uses Triton internally)
   - xFormers library optimizations

## Contributing

Found issues or have improvements? Please contribute:
- Fix bugs in examples
- Add new examples demonstrating advanced concepts
- Improve documentation and comments
- Share performance results from different GPUs

## License

These examples are part of the Compiler-101 educational project.
See the main repository LICENSE file for details.
