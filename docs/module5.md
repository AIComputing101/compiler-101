# Module 5: AMD ROCm/HIP Compiler – Cross-Vendor GPU Programming with Open Standards  

AMD’s ROCm (Radeon Open Compute Platform) and HIP (Heterogeneous-Compute Interface for Portability) form an open-source ecosystem designed to break vendor lock-in in GPU computing. Unlike NVIDIA’s CUDA (Module 4), which is proprietary to NVIDIA GPUs, HIP enables developers to write a single codebase that compiles to both AMD and NVIDIA GPUs. This module explores how the ROCm/HIP compiler toolkit extends C++ and CUDA-like syntax to support cross-vendor parallelism, leveraging Clang/LLVM for frontend processing and AMD-specific optimizations for backend code generation.  


## 5.1 Module Objective  
By the end of this module, you will understand how the AMD ROCm/HIP compiler:  
- Converts CUDA-style code to portable HIP code using automated tools, minimizing manual rewrites.  
- Uses Clang/LLVM as a frontend to parse HIP code, separating host and device logic while generating vendor-agnostic LLVM IR.  
- Optimizes LLVM IR for AMD GPU architectures (GCN/CDNA) via specialized passes, focusing on wavefront efficiency and memory hierarchy.  
- Generates AMD-specific assembly (GCN ISA) and integrates with the ROCm runtime for cross-vendor execution.  

The focus is on HIP’s core value: **portability without performance loss**—enabling the same code to run efficiently on AMD and NVIDIA GPUs through open standards and compiler innovation.  


## 5.2 Core Workflow: From HIP Code to Cross-Vendor Execution  
HIP’s workflow is designed to mirror CUDA’s (Module 4) for familiarity but adds layers for cross-vendor compatibility. The key stages are:  

1. **HIP Code Conversion**: Convert existing CUDA code to HIP (or write new HIP code directly).  
2. **Frontend Parsing**: Use Clang to parse HIP code, generate LLVM IR for both host and device.  
3. **Backend Optimization**: Optimize LLVM IR for target hardware (AMD GCN/CDNA or NVIDIA PTX).  
4. **Runtime Integration**: Link with ROCm (AMD) or CUDA (NVIDIA) runtimes for execution.  


### 5.2.1 Stage 1: HIP Code Conversion – From CUDA to Portable Code  
HIP is intentionally designed to resemble CUDA, making it easy to adapt existing CUDA code. For developers with CUDA codebases, AMD provides `hipify-clang`—a tool that automates conversion to HIP, replacing CUDA-specific APIs with HIP equivalents.  

#### Key HIP-CUDA Equivalents  
HIP retains CUDA’s kernel launch syntax and parallel primitives but replaces vendor-specific functions with portable alternatives:  

| CUDA Feature               | HIP Equivalent                          | Purpose                                  |  
|----------------------------|-----------------------------------------|------------------------------------------|  
| `cudaMalloc`               | `hipMalloc`                             | Allocate device memory.                  |  
| `cudaMemcpy`               | `hipMemcpy`                             | Copy data between host and device.       |  
| `__global__`               | `__global__` (unchanged)                | Mark kernel functions.                   |  
| `threadIdx`, `blockIdx`    | `threadIdx`, `blockIdx` (unchanged)     | Thread/block ID variables.               |  
| `cudaDeviceSynchronize`    | `hipDeviceSynchronize`                  | Synchronize host and device.             |  


#### Example: CUDA to HIP Conversion  
Consider a CUDA vector addition kernel. Using `hipify-clang`, the conversion is nearly automatic:  

**Original CUDA Code**:  
```cpp
#include <cuda_runtime.h>

__global__ void vec_add(const int* a, const int* b, int* c, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));
    vec_add<<<1, n>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    // ... cleanup ...
}
```  

**Converted HIP Code**:  
```cpp
#include <hip/hip_runtime.h>  // Replaces cuda_runtime.h

__global__ void vec_add(const int* a, const int* b, int* c, int n) {  // __global__ unchanged
    int idx = threadIdx.x;  // threadIdx unchanged
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int *d_a, *d_b, *d_c;
    hipMalloc(&d_a, n * sizeof(int));  // cudaMalloc → hipMalloc
    hipMalloc(&d_b, n * sizeof(int));
    hipMalloc(&d_c, n * sizeof(int));
    vec_add<<<1, n>>>(d_a, d_b, d_c, n);  // Launch syntax unchanged
    hipDeviceSynchronize();  // cudaDeviceSynchronize → hipDeviceSynchronize
    // ... cleanup ...
}
```  

This converted code compiles for both AMD and NVIDIA GPUs, eliminating the need for separate codebases.  


### 5.2.2 Stage 2: Frontend – Clang for HIP Parsing and LLVM IR Generation  
HIP leverages the Clang compiler (a C/C++ frontend for LLVM) to parse HIP code, extending Clang’s capabilities to handle GPU-specific constructs. This reuse of LLVM’s infrastructure (a core principle from *From Source Code to Machine Code*) ensures compatibility with modern C++ features and simplifies cross-vendor support.  


#### Parsing HIP-Specific Constructs  
Clang’s HIP frontend extends its C++ parser to recognize:  
- `__global__` and `__device__` keywords (same as CUDA) to identify device code.  
- Kernel launch syntax (`<<<...>>>`), which is desugared into HIP runtime calls (e.g., `hipLaunchKernelGGL`).  
- GPU-specific variables (`threadIdx`, `blockDim`, etc.), which map to hardware registers on AMD GPUs.  


#### Generating LLVM IR for Host and Device  
Like `nvcc`, the HIP frontend separates code into host and device streams:  
- **Host code**: Compiled to CPU-specific LLVM IR (e.g., x86-64) and later to CPU assembly.  
- **Device code**: Compiled to GPU-agnostic LLVM IR with AMD-specific intrinsics (e.g., for wavefront operations) or NVIDIA PTX-compatible IR.  

**Example LLVM IR for Device Code** (simplified for `vec_add` kernel):  
```llvm
; Device code LLVM IR for vec_add
define amdgpu_kernel void @_Z7vec_addPKiS0_Pii(i32* %a, i32* %b, i32* %c, i32 %n) {
entry:
  ; Get thread ID (threadIdx.x)
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  ; Check if thread ID < n
  %cmp = icmp slt i32 %tid, %n
  br i1 %cmp, label %loop, label %exit

loop:
  ; Calculate address: a + tid * 4 (sizeof(int))
  %a_addr = getelementptr inbounds i32, i32* %a, i32 %tid
  %b_addr = getelementptr inbounds i32, i32* %b, i32 %tid
  %c_addr = getelementptr inbounds i32, i32* %c, i32 %tid
  ; Load a[tid] and b[tid]
  %a_val = load i32, i32* %a_addr, align 4
  %b_val = load i32, i32* %b_addr, align 4
  ; Compute sum
  %sum = add i32 %a_val, %b_val
  ; Store result to c[tid]
  store i32 %sum, i32* %c_addr, align 4
  br label %exit

exit:
  ret void
}
```  

This IR includes AMD-specific intrinsics like `@llvm.amdgcn.workitem.id.x()` (to get thread ID) but remains structurally similar to CUDA-generated IR, enabling cross-vendor optimization.  


## 5.3 Stage 3: Backend – Optimizing for AMD GPU Architectures  
The HIP backend uses LLVM’s code generator to optimize device-specific LLVM IR for AMD’s GPU architectures, primarily **GCN (Graphics Core Next)** and its successor **CDNA (Compute DNA)**. These architectures are designed for high-throughput parallel computing, with features like wavefronts (AMD’s equivalent of NVIDIA’s warps) and a hierarchical memory system.  


### 5.3.1 AMD GPU Architecture Fundamentals  
To understand backend optimizations, it’s critical to grasp key AMD GPU concepts:  
- **Compute Units (CUs)**: AMD GPUs are composed of CUs—processing units analogous to NVIDIA’s SMs (Streaming Multiprocessors). Each CU contains:  
  - **SIMD Engines**: Execute instructions for wavefronts (groups of 64 threads, compared to NVIDIA’s 32-thread warps).  
  - **Local Data Share (LDS)**: Shared memory for threads within a CU (equivalent to NVIDIA’s shared memory).  
  - **Scalar Units**: Handle control flow and scalar operations.  
- **Memory Hierarchy**:  
  - **Global Memory**: Large (GBs) but high-latency, accessible to all CUs.  
  - **LDS (Local Data Share)**: Low-latency shared memory within a CU (KBs).  
  - **Private Memory**: Per-thread registers and local memory.  


### 5.3.2 Key Backend Optimizations for AMD GPUs  
The HIP backend (via LLVM) applies AMD-specific optimizations to maximize performance:  

#### 1. Wavefront Utilization  
AMD GPUs execute threads in wavefronts of 64. The backend optimizes control flow to minimize **wavefront divergence**—when threads in a wavefront follow different execution paths (e.g., `if`/`else`). Divergence forces the GPU to serialize execution, reducing throughput.  

- **Optimization**: Predicated execution (similar to PTX) is used to mask threads that would otherwise diverge, keeping the wavefront executing in lockstep.  


#### 2. Memory Coalescing for Global Memory  
Like NVIDIA GPUs, AMD GPUs require contiguous memory accesses from threads in a wavefront to maximize global memory bandwidth. The backend ensures:  
- Threads in a wavefront access consecutive 32-byte or 64-byte chunks (matching AMD’s memory transaction size).  
- Strided accesses (e.g., `a[idx * 2]`) are reordered where possible to align with memory transactions.  


#### 3. LDS (Shared Memory) Optimization  
The backend optimizes usage of LDS (AMD’s shared memory) to reduce global memory accesses:  
- **Bank Conflict Avoidance**: LDS is divided into memory banks (32 banks for GCN). The backend rearranges data layout to prevent multiple threads from accessing the same bank simultaneously, which causes delays.  
- **Tiling**: For algorithms like matrix multiplication, the backend promotes global memory accesses to LDS, reusing data across threads in a wavefront.  


#### 4. Register Allocation  
AMD GPUs have a fixed number of registers per CU (e.g., 65,536 32-bit registers for GCN). The backend optimizes register usage to:  
- Minimize "spilling" (moving excess data from registers to slower local memory).  
- Balance register usage across wavefronts to maximize CU occupancy (number of active wavefronts per CU).  


### 5.3.3 Generating GCN/CDNA ISA  
The final backend step converts optimized LLVM IR to **GCN (or CDNA) ISA**—AMD’s hardware-specific assembly. For the `vec_add` kernel, the GCN ISA (simplified) looks like:  

```asm
; GCN ISA for vec_add kernel (AMD MI250, CDNA architecture)
amdgcn_target cdna1
section .text

vec_add:
    ; Load thread ID (tid.x) into s0
    s_load_dword s0, s[0:1], 0x10  ; s0 = threadIdx.x

    ; Load n into s1
    s_load_dword s1, s[0:1], 0x14  ; s1 = n

    ; Compare tid.x < n (set s2 to 1 if true, 0 otherwise)
    s_cmp_lt_i32 s0, s1
    s_cselect_b32 s2, 1, 0

    ; If tid.x >= n, exit (s_endpgm)
    s_and_b32 s3, s2, 1
    s_cbranch_scc0 exit

    ; Calculate address offsets: tid.x * 4 (sizeof(int))
    v_mov_b32_e32 v0, s0
    v_lshlrev_b32_e32 v0, 2, v0  ; v0 = tid.x * 4

    ; Load a[tid.x] from global memory
    flat_load_dword v1, v[0:1]  ; v1 = a[tid.x]

    ; Load b[tid.x] from global memory
    flat_load_dword v2, v[2:3]  ; v2 = b[tid.x]

    ; Compute sum: a + b
    v_add_i32_e32 v3, v1, v2    ; v3 = v1 + v2

    ; Store result to c[tid.x]
    flat_store_dword v[4:5], v3 ; c[tid.x] = v3

exit:
    s_endpgm  ; Terminate kernel
```  

This assembly directly maps to CDNA hardware instructions, with explicit use of scalar registers (`s0-s3`) for control flow and vector registers (`v0-v5`) for data processing.  


## 5.4 Stage 4: ROCm Runtime – Cross-Vendor Execution  
The ROCm runtime is AMD’s equivalent to NVIDIA’s CUDA Runtime, providing APIs for device management, memory allocation, and kernel launches. It integrates with the compiled HIP code to enable execution on AMD GPUs, while maintaining compatibility with NVIDIA GPUs via CUDA Runtime.  


### 5.4.1 Key ROCm Runtime Components  
- **HIP Runtime**: Abstracts vendor-specific APIs, providing `hipMalloc`, `hipMemcpy`, and `hipLaunchKernelGGL` (for kernel launches).  
- **ROCt (ROCm Thunk)**: Low-level interface to GPU hardware, handling device initialization and command submission.  
- **HSA Runtime**: Manages GPU queues, memory pools, and signal synchronization (HSA = Heterogeneous System Architecture, an open standard for CPU-GPU interaction).  


### 5.4.2 Cross-Vendor Execution Flow  
For a HIP program targeting AMD:  
1. The host code calls `hipMalloc` to allocate global memory on the AMD GPU.  
2. `hipMemcpy` transfers data from host to GPU memory via HSA Runtime.  
3. `vec_add<<<1, n>>>` is translated to `hipLaunchKernelGGL`, which configures the grid/block dimensions and submits the kernel to the GPU via ROCt.  
4. The GPU executes the GCN/CDNA ISA code, with the runtime coordinating synchronization via `hipDeviceSynchronize`.  

For NVIDIA targets, the same HIP code uses the CUDA Runtime, replacing ROCt/HSA with NVIDIA’s driver interfaces.  


## 5.5 Key Differences: HIP vs. CUDA vs. Standard Compilers  
HIP’s design bridges the gap between vendor-specific and universal compilers:  

| Feature                  | AMD ROCm/HIP Compiler                   | NVIDIA nvcc (Module 4)                     | Standard C++ Compiler (Module 3)       |  
|--------------------------|------------------------------------------|---------------------------------------------|------------------------------------------|  
| **Vendor Lock-In**       | Open-source, cross-vendor (AMD/NVIDIA).  | Proprietary, NVIDIA-only.                   | Vendor-agnostic (CPU).                  |  
| **Frontend**             | Clang/LLVM (open-source).                | NVCC frontend + LLVM (hybrid).              | GCC/Clang/LLVM.                         |  
| **Device IR**            | LLVM IR with AMD/NVIDIA intrinsics.      | PTX (NVIDIA-specific).                      | LLVM IR/GIMPLE (CPU).                   |  
| **Target ISA**           | GCN/CDNA (AMD) or PTX (NVIDIA).          | Cubin (NVIDIA GPU-specific).                | x86-64/ARM (CPU).                       |  
| **Optimization Focus**   | Wavefront efficiency, LDS usage.         | Warp synchronization, shared memory.        | CPU cache, branch prediction.           |  


## 5.6 Summary  
The AMD ROCm/HIP compiler toolkit enables cross-vendor GPU programming by extending open-source compiler infrastructure (Clang/LLVM) and adopting CUDA-like syntax for familiarity. Key takeaways:  
- **Portability**: HIP code converts easily from CUDA and runs on both AMD and NVIDIA GPUs.  
- **Open Standards**: Leverages Clang/LLVM and HSA for frontend/backend processing, avoiding proprietary lock-in.  
- **AMD-Specific Optimizations**: Backend optimizations for GCN/CDNA architectures (wavefronts, LDS, memory coalescing) ensure performance parity with vendor-specific tools.  

By prioritizing open-source collaboration and cross-vendor compatibility, HIP and ROCm have become critical tools for heterogeneous computing, empowering developers to target the best GPU for their workload without rewriting code.

Figure: AMD ROCm/HIP Compiler Cross-Vendor Flow (AMD/NVIDIA) + Clang/LLVM
```
[CUDA/HIP Source Code]  ← Input (Oval)
       ↓ "Code with `hipMalloc`, `__global__ void vec_add()`"
┌─────────────────────────────┐
│ Optional: HIP Conversion    │  ← Dashed (Optional)
│ (`hipify-clang` tool:       │
│  • `cudaMalloc` → `hipMalloc` │
│  • No other code changes)   │
└────────────────┬────────────┘
                 ↓ "Pure HIP Code"
┌─────────────────────────────┐
│ Frontend: Clang for HIP     │  ← Light Blue (Frontend)
│ (Parses HIP; generates      │
│  LLVM IR with AMD/NVIDIA    │
│  intrinsics)                 │
└────────────────┬────────────┘
                 ↓ "HIP-Aware LLVM IR"
┌─────────────────────────────┐
│ Backend: LLVM Target Opt    │  ← Light Orange (Backend)
│ (Dual Paths for Cross-Vendor): │
│  • AMD Path: Optimize for GCN/CDNA (wavefronts, LDS) → GCN ISA │
│  • NVIDIA Path: Convert to PTX → Cubin │
└───────┬────────────┬────────┘
        ↓            ↓
        │ "AMD GCN ISA" │ "NVIDIA Cubin"
        ↓            ↓
┌──────────────┐  ┌──────────────┐
│ ROCm Runtime │  │ CUDA Runtime │  ← Light Purple (Runtime)
│ (AMD)        │  │ (NVIDIA)     │
└───────┬────────┘  └───────┬────────┘
        └────────────┬───────┘
                     ↓ "Unified HIP Executable"
[Cross-Vendor Executable (AMD/NVIDIA)]  ← Output (Oval)
```