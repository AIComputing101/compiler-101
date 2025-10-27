# Module 6: Triton Compiler – Simplifying High-Performance GPU Programming with Python  

The Triton Compiler is a game-changing tool for GPU programming, designed to bridge the gap between **productivity** and **performance**. Unlike low-level GPU compilers (CUDA, HIP—Modules 4–5) that require deep hardware knowledge, Triton lets developers write high-performance GPU code in Python, abstracting away tedious details like thread block scheduling, memory coalescing, and vectorization. Created by researchers at OpenAI and now integrated into major ML frameworks (PyTorch, TensorFlow), Triton specializes in matrix operations and neural network workloads—critical for modern AI. This module explores how Triton reimagines GPU compiler design, leveraging Python’s readability and JIT (Just-In-Time) compilation to deliver performance on par with hand-tuned CUDA, while drastically reducing development time.  


## 6.1 Module Objective  
By the end of this module, you will understand how Triton:  
- Uses a Python frontend to enable intuitive GPU kernel writing, without sacrificing low-level control.  
- Translates Python code into a custom Intermediate Representation (Triton IR) optimized for parallel GPU workloads.  
- Applies automatic optimizations (memory coalescing, shared memory tiling, vectorization) to match or exceed hand-tuned GPU code.  
- JIT-compiles Triton IR to target GPU ISA (NVIDIA PTX, AMD GCN) at runtime, adapting to hardware specifics.  
- Integrates with ML frameworks to accelerate real-world workloads (e.g., transformers, attention mechanisms).  

The focus is on Triton’s core innovation: **making GPU programming accessible to non-experts** while retaining the performance needed for demanding applications like deep learning.  


## 6.2 Core Workflow: Python → Triton IR → JIT GPU ISA  
Triton’s pipeline is intentionally streamlined, avoiding the complex multi-stage workflows of traditional compilers (Modules 1–5) in favor of a developer-friendly, JIT-driven approach. The key stages are:  

1. **Python Frontend**: Developers write GPU kernels using Python syntax and Triton’s high-level API.  
2. **Triton IR Generation**: The compiler converts Python code into a hardware-agnostic IR designed for parallelism.  
3. **Optimization Passes**: Triton applies GPU-specific optimizations (memory coalescing, tiling) to the IR.  
4. **JIT Compilation**: The optimized IR is compiled to target GPU ISA (e.g., NVIDIA PTX) at runtime.  
5. **Kernel Execution**: The Triton runtime launches the compiled kernel on the GPU, managing memory and synchronization.  


### 6.2.1 Stage 1: Python Frontend – Writing Kernels with High-Level Abstractions  
Triton’s greatest strength is its Python frontend, which lets developers express GPU parallelism without manually managing threads, blocks, or memory. At the heart of this frontend is the `@triton.jit` decorator, which marks functions for JIT compilation to GPU, and the `triton.language` module (aliased `tl`), which provides GPU-specific primitives.  

#### Key Concepts in the Python Frontend  
- **Program IDs**: Triton uses `tl.program_id` to divide work across GPU thread blocks. For 2D workloads (e.g., matrix multiplication), you can use `tl.program_id(0)` (row) and `tl.program_id(1)` (column).  
- **Thread Indices**: `tl.arange` generates indices for threads within a block, replacing manual thread ID calculations (e.g., `threadIdx.x` in CUDA).  
- **Memory Access**: `tl.load` and `tl.store` handle GPU memory operations, with Triton automatically optimizing for global/shared memory.  
- **Vectorization**: Triton’s IR automatically vectorizes operations (e.g., adding two 16-element vectors at once), leveraging GPU SIMD (Single Instruction, Multiple Data) capabilities.  


#### Example 1: Simple Vector Addition in Triton  
Compare this Triton kernel to the CUDA/HIP equivalents (Modules 4–5)—notice how much boilerplate is eliminated:  

```python
import triton
import triton.language as tl
import torch  # For tensor handling (Triton integrates with PyTorch)

# Decorate with @triton.jit to mark for GPU compilation
@triton.jit
def vec_add_kernel(
    a_ptr: tl.pointer_type(tl.int32),  # Pointer to input tensor a
    b_ptr: tl.pointer_type(tl.int32),  # Pointer to input tensor b
    c_ptr: tl.pointer_type(tl.int32),  # Pointer to output tensor c
    n: tl.int32,                      # Length of vectors
    BLOCK_SIZE: tl.constexpr tl.int32  # Thread block size (compile-time constant)
):
    # 1. Get the thread block's ID (each block handles a chunk of the vector)
    block_id = tl.program_id(0)
    # 2. Calculate the start index of the chunk for this block
    block_start = block_id * BLOCK_SIZE
    # 3. Generate indices for threads within the block (0 to BLOCK_SIZE-1)
    thread_indices = block_start + tl.arange(0, BLOCK_SIZE)
    # 4. Mask out indices beyond the vector length (avoid out-of-bounds access)
    mask = thread_indices < n

    # 5. Load a[thread_indices] and b[thread_indices] from global memory
    a_vals = tl.load(a_ptr + thread_indices, mask=mask)
    b_vals = tl.load(b_ptr + thread_indices, mask=mask)

    # 6. Compute c = a + b (Triton auto-vectorizes this)
    c_vals = a_vals + b_vals

    # 7. Store results to c[thread_indices]
    tl.store(c_ptr + thread_indices, c_vals, mask=mask)

# Host function to launch the kernel
def triton_vec_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.numel()
    # Allocate output tensor on GPU
    c = torch.empty_like(a)
    # Define block size (1024 threads/block—Triton optimizes this for hardware)
    BLOCK_SIZE = 1024
    # Calculate number of blocks needed (ceil(n / BLOCK_SIZE))
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel: (num_blocks, 1, 1) = 1D grid of blocks
    vec_add_kernel[(num_blocks, 1, 1)](a, b, c, n, BLOCK_SIZE)
    return c

# Test the kernel
if __name__ == "__main__":
    a = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.int32)
    b = torch.tensor([5, 6, 7, 8], device="cuda", dtype=torch.int32)
    c = triton_vec_add(a, b)
    print("Result:", c)  # Output: tensor([ 6,  8, 10, 12], device='cuda:0', dtype=int32)
```  


#### Why This Is Revolutionary  
- **No Manual Thread Management**: Triton handles block/thread indices via `tl.program_id` and `tl.arange`—no need to calculate `threadIdx.x` or `blockIdx.x`.  
- **Auto-Masking**: The `mask` parameter prevents out-of-bounds memory access, eliminating manual `if (idx < n)` checks.  
- **Framework Integration**: Triton works seamlessly with PyTorch/TensorFlow tensors, avoiding tedious memory copying between host and device.  


### 6.2.2 Stage 2: Triton IR – A Parallelism-First Intermediate Representation  
Triton does not use LLVM IR (unlike HIP/Clang—Module 5) or PTX (unlike CUDA—Module 4). Instead, it uses a **custom IR** designed explicitly for GPU parallelism. This IR abstracts hardware details (e.g., register counts, memory banks) but retains enough low-level information to enable aggressive optimizations.  

#### Key Features of Triton IR  
- **Explicit Parallelism**: The IR models thread blocks, warps/wavefronts, and SIMD lanes as first-class constructs, making it easy to optimize for GPU parallelism.  
- **Memory Hierarchy Awareness**: The IR distinguishes between global, shared (LDS), and private memory, letting optimizations target specific memory spaces.  
- **Compile-Time Constants**: Parameters like `BLOCK_SIZE` are marked as compile-time constants (`tl.constexpr`), enabling Triton to generate hardware-specific code (e.g., 1024 threads/block for NVIDIA Ampere, 256 for older GPUs).  

#### Example Triton IR Snippet (Simplified for `vec_add_kernel`)  
The Python kernel above is converted to Triton IR that looks like this (pseudocode):  

```triton_ir
func @vec_add_kernel(
    %a_ptr: ptr<int32, global>,
    %b_ptr: ptr<int32, global>,
    %c_ptr: ptr<int32, global>,
    %n: i32,
    %BLOCK_SIZE: i32
) attributes {kernel} {
    // Get block ID (1D grid)
    %block_id = program_id 0 : i32
    // Calculate block start index
    %block_start = mul i32 %block_id, %BLOCK_SIZE : i32
    // Generate thread indices within block (0 to BLOCK_SIZE-1)
    %thread_indices = arange i32 0, %BLOCK_SIZE : i32
    // Add block start to get global indices
    %global_indices = add i32 %block_start, %thread_indices : i32
    // Create mask: global_indices < n
    %mask = cmp slt i32 %global_indices, %n : i1
    // Load a[global_indices] with mask
    %a_vals = load ptr<int32, global> %a_ptr[%global_indices], mask=%mask : i32
    %b_vals = load ptr<int32, global> %b_ptr[%global_indices], mask=%mask : i32
    // Compute sum
    %c_vals = add i32 %a_vals, %b_vals : i32
    // Store result
    store ptr<int32, global> %c_ptr[%global_indices], %c_vals, mask=%mask : i32
    return
}
```  

This IR is far more concise than LLVM IR or PTX, as it’s tailored to GPU parallelism rather than general-purpose computing.  


## 6.3 Stage 3: Automatic Optimizations – Triton’s "Secret Sauce"  
The biggest advantage of Triton over manual CUDA/HIP is its ability to automatically apply GPU-specific optimizations that would take hours to implement by hand. These optimizations are baked into the compiler’s pass pipeline and require no input from the developer.  


### 6.3.1 1. Global Memory Coalescing  
GPU global memory is fast only if threads in a warp/wavefront access **contiguous memory addresses** (Module 4–5). Triton’s optimizer ensures:  
- Thread indices are ordered to align with memory transaction sizes (32 bytes for NVIDIA, 64 bytes for AMD).  
- Strided accesses (e.g., `a[idx * 2]`) are reordered to form contiguous chunks where possible.  

In the `vec_add_kernel`, Triton automatically arranges `global_indices` to access `a` and `b` in contiguous blocks, maximizing bandwidth.  


### 6.3.2 2. Shared Memory Tiling for Large Workloads  
For large tensors (e.g., matrix multiplication), accessing global memory repeatedly is slow. Triton uses **shared memory tiling** (equivalent to NVIDIA’s shared memory or AMD’s LDS) to cache frequently used data:  
1. The compiler splits large tensors into small "tiles" (e.g., 16x16 elements).  
2. Each thread block loads a tile from global memory into shared memory.  
3. Computations are performed on the tile in shared memory (10–100x faster than global memory).  

#### Example: Matrix Multiplication with Tiling  
Triton’s built-in `tl.dot` function (for matrix multiplication) automatically uses tiling. Here’s a simplified version:  

```python
@triton.jit
def matmul_kernel(
    A_ptr: tl.pointer_type(tl.float32),
    B_ptr: tl.pointer_type(tl.float32),
    C_ptr: tl.pointer_type(tl.float32),
    M: tl.int32, N: tl.int32, K: tl.int32,  # A: MxK, B: KxN, C: MxN
    A_stride: tl.int32, B_stride: tl.int32, C_stride: tl.int32,
    BLOCK_SIZE_M: tl.constexpr tl.int32 = 16,
    BLOCK_SIZE_N: tl.constexpr tl.int32 = 16,
    BLOCK_SIZE_K: tl.constexpr tl.int32 = 16,
):
    # Program IDs for 2D grid (rows x columns)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load A tile into shared memory
    a_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_tile = tl.load(A_ptr + pid_m*BLOCK_SIZE_M*A_stride + k, mask=...)
        # Load B tile into shared memory
        b_tile = tl.load(B_ptr + k*B_stride + pid_n*BLOCK_SIZE_N, mask=...)
        # Compute dot product on tiles (Triton optimizes this)
        c_tile = tl.dot(a_tile, b_tile)
        # Store result to C
        tl.store(C_ptr + pid_m*BLOCK_SIZE_M*C_stride + pid_n*BLOCK_SIZE_N, c_tile, mask=...)
```  

Triton’s optimizer handles:  
- Tile size selection (16x16 is optimal for most GPUs).  
- Shared memory bank conflict avoidance (critical for AMD GPUs).  
- Loop unrolling to reduce overhead.  


### 6.3.3 3. Warp/Wavefront Optimization  
Triton optimizes for GPU execution units:  
- **Warp Specialization**: Threads in a warp (32 for NVIDIA, 64 for AMD) are grouped to minimize divergence (e.g., `if`/`else` branches).  
- **Vectorization**: Operations on small tensors (e.g., 16-element vectors) are mapped to GPU SIMD instructions (e.g., NVIDIA’s `wmma` for tensor cores, AMD’s `v_dot` for CDNA).  


### 6.3.4 4. Register Allocation  
Triton’s register allocator balances two goals:  
- Minimizing "spilling" (moving data from registers to slow local memory).  
- Maximizing occupancy (number of active warps per SM/CU).  

For the `vec_add_kernel`, Triton allocates just enough registers to store `a_vals`, `b_vals`, and `c_vals`, avoiding spills.  


## 6.4 Stage 4: JIT Compilation and Runtime Integration  
Triton uses JIT compilation to generate hardware-specific code at runtime. This is critical for ML workloads, where tensor shapes and hardware vary widely.  


### 6.4.1 How JIT Compilation Works  
1. When the `vec_add_kernel` is first called, Triton’s frontend parses the Python code and generates Triton IR.  
2. The optimizer applies passes (coalescing, tiling) to the IR.  
3. The code generator compiles the optimized IR to target ISA:  
   - **NVIDIA GPUs**: Generates PTX, which is further compiled to cubin by NVIDIA’s driver.  
   - **AMD GPUs**: Generates GCN/CDNA ISA via HIP (Module 5).  
4. The compiled kernel is cached in memory. Subsequent calls to `vec_add_kernel` reuse the cached code, avoiding recompilation.  


### 6.4.2 Triton Runtime  
The Triton runtime manages GPU resources and kernel launches:  
- **Grid/Block Configuration**: Automatically calculates the number of blocks (`num_blocks`) based on tensor size and `BLOCK_SIZE`.  
- **Memory Management**: Integrates with PyTorch/TensorFlow to use framework-allocated GPU memory, avoiding redundant copies.  
- **Synchronization**: Handles host-device synchronization (e.g., waiting for a kernel to finish before returning results).  


## 6.5 Real-World Impact: Triton in ML Frameworks  
Triton is not just a research project—it’s used in production to accelerate critical ML workloads:  
- **PyTorch FlashAttention**: A fast implementation of the attention mechanism (used in transformers) built with Triton. It outperforms hand-tuned CUDA by 2–4x.  
- **TensorFlow XLA**: Triton is integrated into XLA (Accelerated Linear Algebra) to optimize matrix operations.  
- **OpenAI GPT**: Triton is used to accelerate transformer layers in GPT-3 and GPT-4, reducing training time.  


## 6.6 Triton vs. Other GPU Compilers  
Triton fills a unique niche in the GPU compiler ecosystem, balancing accessibility and performance:  

| Feature                  | Triton Compiler                          | NVIDIA CUDA (Module 4)                     | AMD HIP (Module 5)                         |  
|--------------------------|------------------------------------------|---------------------------------------------|---------------------------------------------|  
| **Programming Language** | Python (high-level).                     | C/C++ (low-level).                          | C/C++ (low-level, portable).                |  
| **Optimization Effort**  | Automatic (no manual tuning).            | Manual (memory coalescing, tiling).         | Manual (similar to CUDA, but cross-vendor). |  
| **ML Integration**       | Native (PyTorch/TensorFlow).             | Requires bindings (e.g., PyTorch CUDA ops). | Requires HIP bindings.                      |  
| **Performance**          | On par with hand-tuned CUDA.             | Best (but requires expertise).              | Near CUDA (cross-vendor).                  |  
| **Learning Curve**       | Low (Python developers adapt quickly).   | High (requires GPU hardware knowledge).     | High (similar to CUDA).                     |  


## 6.7 Summary  
The Triton Compiler redefines GPU programming by putting productivity first, without compromising performance. Key takeaways:  
- **Python Frontend**: Makes GPU programming accessible to developers without low-level hardware expertise.  
- **Automatic Optimizations**: Coalescing, tiling, and warp optimization eliminate manual tuning.  
- **JIT Compilation**: Adapts to hardware and workloads at runtime, critical for ML.  
- **Framework Integration**: Seamless use with PyTorch/TensorFlow enables production deployment.  

For ML developers, Triton is a game-changer—it lets you focus on algorithm design, not GPU hardware details. For compiler designers, it demonstrates how domain-specific IRs (tailored to GPU parallelism) can outperform general-purpose IRs like LLVM for specialized workloads. As AI workloads grow more complex, Triton’s role in simplifying high-performance GPU programming will only become more critical.

Figure: Python Frontend + JIT + Automatic GPU Optimization
```
[Python Code (with @triton.jit)]  ← Input (Oval)
       ↓ "Code with `@triton.jit`, `tl.load`, `tl.dot` (matrix ops)"
┌─────────────────────────────┐
│ Frontend: Python Parser     │  ← Light Blue (Frontend)
│ (Parses Triton API:         │
│  • `tl.program_id` (grid/blocks) │
│  • `tl.load`/`tl.store` (memory) │
│  • `tl.dot` (matrix multiplication)) │
└────────────────┬────────────┘
                 ↓ "Triton-Python AST"
┌─────────────────────────────┐
│ IR Generation: Triton IR    │  ← Light Green (IR)
│ (Custom Parallel IR:        │
│  • Models thread blocks     │
│  • Explicit SIMD lanes)     │
└────────────────┬────────────┘
                 ↓ "Raw Triton IR"
┌─────────────────────────────┐
│ Auto-Optimization          │  ← Light Yellow (Optimization)
│ (No Manual Tuning!):        │
│  • Memory Coalescing (aligns global access) │
│  • Shared Memory Tiling (caches to LDS) │
│  • Vectorization (uses GPU SIMD) │
└────────────────┬────────────┘
                 ↓ "Optimized Triton IR"
┌─────────────────────────────┐
│ Backend: JIT Compiler      │  ← Light Orange (Backend)
│ (Runtime Compilation:       │
│  • NVIDIA → PTX → Cubin     │
│  • AMD → GCN ISA)           │
└────────────────┬────────────┘
                 ↓ "JIT-Generated GPU Code"
┌─────────────────────────────┐
│ Runtime: Triton Runtime     │  ← Light Purple (Runtime)
│ (Integrates with ML Frameworks: │
│  • PyTorch/TensorFlow tensor reuse │
│  • Kernel launch coordination) │
└────────────────┬────────────┘
                 ↓ "GPU-Accelerated ML"
[ML Workload (e.g., Transformer Inference)]  ← Output (Oval)
```


## 6.8 Contributing to the Triton Compiler (case study) 

This section walks through a practical, upstream-ready contribution to Triton: reducing shared-memory (LDS) bank conflicts for matrix multiplication (matmul) on AMD GPUs. It illustrates the full contributor workflow—profiling, designing an optimization, implementing it cleanly, validating it, and preparing a high‑quality PR.


### 6.8.1 Scenario: Improve matmul on AMD by reducing LDS bank conflicts
On AMD GPUs, shared memory (LDS) is divided into 32 banks. When many threads in a wavefront access addresses that map to the same bank, requests serialize and throughput drops. Triton’s tiled matmul (via `tl.dot`) relies on LDS; for some tile sizes (e.g., 16×16), access patterns can align poorly and trigger frequent bank conflicts.

Goal: adjust tiling to minimize conflicts on AMD—boosting throughput without hurting NVIDIA performance.


### 6.8.2 Step 1 — Identify the problem (profiling and benchmarking)
- Set up Triton in editable mode and prepare benchmarking tools.
- On AMD (ROCm), use `rocprof` to measure LDS bank conflict metrics while running matmul benchmarks.

Commands (example):

```bash
# Install Triton from source for dev
git clone https://github.com/openai/triton.git
cd triton
pip install -e ".[tests,benchmark]"

# Profile a matmul benchmark on AMD
rocprof --stats ./benchmarks/bench_matmul.py
```

What to look for: high "LDS Bank Conflict" counts with 16×16 tiles indicate the issue.


### 6.8.3 Step 2 — Design the optimization (padding to break alignment)
- AMD LDS has 32 banks, 4 bytes each → 32×4 = 128 bytes per bank cycle.
- If adjacent threads access addresses separated by multiples of 128 bytes, they collide on the same bank.
- Add a small padding column to shared tiles to shift addresses and de-align accesses.

Design choices:
- Keep logical tile size (e.g., 16×16) but allocate shared tiles as 16×17 (i.e., +1 column) on AMD.
- Gate the padding via a compile-time flag or target detection so NVIDIA remains unchanged.


### 6.8.4 Step 3 — Implement the optimization (kernel + IR hook)
In the Triton matmul kernel, conditionally allocate padded shared-memory tiles on AMD and adjust indices accordingly. Conceptually:

```python
import triton
import triton.language as tl

@triton.jit
def optimized_matmul(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    A_stride, B_stride, C_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    AMD_PAD: tl.constexpr = False  # Enable padding for AMD
):
    # Allocate shared tiles (add +1 col when AMD_PAD is enabled)
    if AMD_PAD:
        sA = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K + 1), dtype=tl.float32)
        sB = tl.zeros((BLOCK_SIZE_K + 1, BLOCK_SIZE_N), dtype=tl.float32)
    else:
        sA = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        sB = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

    # ... load tiles A/B into sA/sB, tl.sync(), tl.dot on the logical K span, store C ...
```

If you need IR-level control, add a small hook in the shared-memory allocator to conditionally pad 2D shapes when targeting AMD:

```python
# Pseudocode in Triton codegen
def allocate_shared(shape, dtype, amd_pad=False):
    if amd_pad and len(shape) == 2:
        shape = (shape[0], shape[1] + 1)
    return lower_to_backend_shared_alloc(shape, dtype)
```

Key requirements:
- Keep the logical math on K unchanged; only the physical LDS layout gets a padding column.
- Ensure index math skips the padding column during loads/compute/stores.
- Guard behind a flag or target check to avoid regressions on NVIDIA.


### 6.8.5 Step 4 — Validate correctness and measure performance
1) Unit tests: verify numerical equivalence vs. PyTorch.

```python
import torch
import triton

def test_optimized_matmul_amd():
    for M, N, K in [(16, 16, 16), (1024, 1024, 1024)]:
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)
        # Call your AMD-padded matmul kernel
        C_triton = /* launch optimized_matmul with AMD_PAD=True */
        C_torch = A @ B
        assert torch.allclose(C_triton, C_torch, atol=1e-3)
```

2) Benchmarks: compare baseline vs. padded tiles on AMD (e.g., MI250). Measure GFLOP/s across sizes like 1024, 2048, 4096.

Expected result: substantial drop in "LDS Bank Conflict" counters and ~15–20% throughput gains for large matrices on AMD when padding is enabled, with no regressions on NVIDIA (padding off).


### 6.8.6 Step 5 — Prepare a quality PR
- Open an issue summarizing the problem, design, and early data.
- Submit a PR with:
  - Kernel/IR changes guarded behind a clearly documented flag/target check.
  - Unit tests and a benchmark script.
  - Brief docs explaining padding and when/why it helps.
- Address feedback (e.g., tune tile sizes for other AMD cards, verify NVIDIA parity, consider auto-detection).


### 6.8.7 Try it in this repo
This course repo includes Triton examples in `examples/06-triton-compiler/`. To experiment locally:

```bash
# (Optional) use a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install Python deps for the examples
pip install -r examples/06-triton-compiler/requirements.txt

# Quick sanity checks and benchmarks (NVIDIA or AMD, where supported)
python examples/06-triton-compiler/01_vector_addition.py
python examples/06-triton-compiler/02_matrix_multiplication.py
python examples/06-triton-compiler/benchmark_all.py
```

If you’re developing the AMD padding idea, start from `02_matrix_multiplication.py`, prototype a padded tile, and compare timings. You can then port the idea to upstream Triton following the steps above.


### 6.8.8 Takeaways for impactful Triton contributions
- Focus on ML-critical ops (matmul, attention, softmax, layernorm).
- Use vendor profilers (`nsys`, `nvprof`, `rocprof`) to identify real bottlenecks.
- Design optimizations that are guarded or adaptive to avoid cross‑vendor regressions.
- Ship tests, benchmarks, and docs with your PR to ease review and ensure longevity.
