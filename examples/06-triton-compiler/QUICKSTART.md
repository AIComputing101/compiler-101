# Quick Start Guide - Triton Compiler Examples

Get up and running with Triton GPU programming in 5 minutes!

## 1. Check Prerequisites

**Required:**
- NVIDIA GPU (Compute Capability 7.0+) or AMD GPU with ROCm
- Python 3.8 or higher
- CUDA Toolkit 11.4+ (for NVIDIA) or ROCm 5.0+ (for AMD)

**Verify GPU:**
```bash
# For NVIDIA
nvidia-smi

# For AMD
rocm-smi
```

## 2. Install Dependencies

**Step 1: Install PyTorch with CUDA/ROCm support**

Choose ONE based on your GPU:

```bash
# NVIDIA GPU (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# NVIDIA GPU (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# AMD GPU (ROCm 5.7)
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
```

**Step 2: Install Triton**

```bash
pip install triton
```

**Optional: Install from requirements.txt**

```bash
pip install -r requirements.txt
```

## 3. Verify Installation

Run the setup verification script:

```bash
python test_setup.py
```

**Expected output:**
```
✓ PyTorch is installed
✓ Triton is installed
✓ NumPy is installed
✓ CUDA is available
  GPU: NVIDIA GeForce RTX 3090
✓ Triton is installed
  Version: 2.1.0
✓ Simple kernel test PASSED
✓ All checks PASSED! You're ready to run the examples.
```

## 4. Run Your First Example

**Vector Addition (simplest):**
```bash
python 01_vector_addition.py
```

**Expected output:**
```
Triton Vector Addition Example
============================================================
Using GPU: NVIDIA GeForce RTX 3090

Example 1: Simple Vector Addition
------------------------------------------------------------
a = [1. 2. 3. 4.]
b = [5. 6. 7. 8.]
c = [ 6.  8. 10. 12.]

Benchmarking Vector Addition (size=1,000,000)
============================================================
Triton:  125.34 μs
PyTorch: 132.45 μs
Speedup: 1.06x
```

## 5. Explore More Examples

**Matrix Multiplication (demonstrates tiling):**
```bash
python 02_matrix_multiplication.py
```

**Fused Softmax (demonstrates kernel fusion):**
```bash
python 03_fused_softmax.py
```

**Custom Activations (ML focus):**
```bash
python 04_custom_activation.py
```

**Run all benchmarks:**
```bash
python benchmark_all.py
```

## 6. Understanding the Code

### Basic Triton Kernel Structure

```python
import triton
import triton.language as tl

@triton.jit  # Mark function for GPU compilation
def my_kernel(
    input_ptr,      # Pointer to input data
    output_ptr,     # Pointer to output data
    n_elements,     # Problem size
    BLOCK_SIZE: tl.constexpr,  # Compile-time constant
):
    # 1. Get block ID
    pid = tl.program_id(0)

    # 2. Calculate element indices for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 3. Create mask for boundary conditions
    mask = offsets < n_elements

    # 4. Load data
    x = tl.load(input_ptr + offsets, mask=mask)

    # 5. Compute
    y = x * 2  # Your operation here

    # 6. Store result
    tl.store(output_ptr + offsets, y, mask=mask)
```

### Launching a Kernel

```python
def my_function(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    output = torch.empty_like(x)

    # Define grid size (number of blocks)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    my_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)

    return output
```

## 7. Common Issues & Solutions

### "CUDA out of memory"
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Or reduce problem sizes in examples
```

### "Module not found: triton"
```bash
# Reinstall Triton
pip install --upgrade triton
```

### "CUDA not available"
```bash
# Check PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Performance lower than expected
- Ensure GPU is not thermal throttling (check with `nvidia-smi`)
- Close other GPU applications
- Use larger problem sizes (small problems have kernel launch overhead)
- Check GPU power settings (performance mode vs. power saving)

## 8. Next Steps

1. **Modify Examples**: Change block sizes, problem sizes, operations
2. **Implement New Kernels**: Try ReLU, layer normalization, etc.
3. **Profile Performance**: Use NVIDIA Nsight or AMD ROCProfiler
4. **Read Documentation**: [https://triton-lang.org/](https://triton-lang.org/)
5. **Explore Module Docs**: See [README.md](README.md) for detailed information

## 9. Learning Resources

**Official:**
- [Triton Documentation](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [PyTorch Triton Tutorial](https://pytorch.org/tutorials/intermediate/triton_tutorial.html)

**Related Course Modules:**
- Module 4: NVIDIA CUDA Compiler (compare to Triton)
- Module 5: AMD HIP Compiler (cross-platform GPU)
- Module 1: C Compiler Basics (compilation fundamentals)

## 10. Benchmarking Tips

**Good practice for benchmarking:**
```python
import torch
import time

# 1. Warm up GPU (10+ iterations)
for _ in range(10):
    result = my_kernel(input_data)
torch.cuda.synchronize()

# 2. Measure (100+ iterations)
start = time.perf_counter()
for _ in range(100):
    result = my_kernel(input_data)
torch.cuda.synchronize()  # IMPORTANT: wait for GPU
elapsed = (time.perf_counter() - start) / 100

# 3. Verify correctness
expected = reference_implementation(input_data)
assert torch.allclose(result, expected, rtol=1e-5)
```

## Getting Help

- Check [README.md](README.md) for detailed documentation
- Run `python test_setup.py` to diagnose installation issues
- See [TROUBLESHOOTING section in README.md](README.md#troubleshooting)
- Open an issue on GitHub if you find bugs

---

**Ready to go? Start with:**
```bash
python 01_vector_addition.py
```

Happy GPU programming! 🚀
