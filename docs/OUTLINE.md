# Complete Tutorial: From Minimal C Compilers to Specialized Compilers (GPU & Quantum)
This tutorial integrates core principles from **Writing a C Compiler (2024.7)** (Nora Sandler) — which focuses on building C compilers from scratch — and **From Source Code to Machine Code (2023.5)** — which outlines universal compiler architecture (frontend → IR → backend) and target hardware adaptation. The content follows a progressive structure, starting with foundational C compilers and extending to specialized tools for C++, GPUs, and quantum-classical hybrid systems.


## 1. Minimal C Compiler – Build the Compiler Backbone
### Objective
Master the **four core compiler stages** (lexer → parser → IR → assembly) by building a minimal C compiler that handles simple integer expressions and return statements — the foundation of all compilers (per *Writing a C Compiler*, Chapter 1).

### Core Stages (From *Writing a C Compiler*, Chapter 1)
A minimal compiler transforms raw C source code into executable assembly by breaking the process into modular, reusable passes — a pattern reused in all advanced compilers (aligned with *From Source Code to Machine Code*’s "frontend-backend decoupling" principle).

#### 1.1 Stage 1: Lexer (Tokenization)
The lexer converts human-readable C code into a list of **tokens** (the "words" of the compiler). Tokens include keywords (`int`, `return`), identifiers (`main`), constants (`2`), and punctuation (`;`, `{`).  
- **Implementation**: Use regular expressions (per *Writing a C Compiler*, Table 1-1) to match token patterns. For example:
  - Identifiers: `[a-zA-Z_]\w*\b` (e.g., `main`).  
  - Constants: `[0-9]+\b` (e.g., `2`).  
  - Keywords: `int\b`, `return\b` (e.g., `return`).  
- **Example**: For `int main(void) { return 2; }`, the lexer outputs:  
  `[IntKeyword, Identifier("main"), LParen, VoidKeyword, RParen, LBrace, ReturnKeyword, Constant(2), Semicolon, RBrace]`.

#### 1.2 Stage 2: Parser (AST Construction)
The parser converts tokens into an **Abstract Syntax Tree (AST)** — a hierarchical representation of the program’s logic (per *Writing a C Compiler*, Chapter 1). The AST eliminates syntax details (e.g., parentheses) and retains only logical structure.  
- **Technique**: Use **recursive descent parsing** (per *Writing a C Compiler*, Chapter 1) — a handwritten approach where each non-terminal symbol (e.g., `<exp>`, `<statement>`) has a dedicated parsing function.  
- **Example**: The code `return 2 + 3;` becomes an AST:  
  `ReturnNode → BinaryNode(Add, ConstantNode(2), ConstantNode(3))`.  
- **Key Tool**: Formal grammar (EBNF) to define valid C constructs (per *Writing a C Compiler*, Listing 1-6). For minimal C:  
  `<statement> ::= "return" <exp> ";"`  
  `<exp> ::= <int> | <exp> "+" <exp>`

#### 1.3 Stage 3: Intermediate Representation (IR) Generation
The IR is a low-level, uniform format that bridges the AST (high-level logic) and assembly (hardware-specific code). For minimal C, we use **TACKY** (Three-Address Code for C-like languages) — a simple IR from *Writing a C Compiler*, Chapter 2.  
- **Why IR?**: Decouples the frontend (parsing) from the backend (assembly generation) — a universal compiler design choice (per *From Source Code to Machine Code*).  
- **Example**: The AST `BinaryNode(Add, 2, 3)` becomes TACKY:  
  `tmp0 = 2 + 3; return tmp0;`  
- **Rule**: TACKY avoids nested expressions; every instruction uses at most three operands (constants or temporary variables).

#### 1.4 Stage 4: Assembly Generation & Code Emission
The backend converts TACKY to **x64 assembly** (target ISA) and writes the final `.s` file. This stage follows the x64 System V ABI (per *Writing a C Compiler*, Chapter 9) for register usage and stack layout.  
- **Example**: TACKY `return 2;` becomes x64 assembly:  
  ```asm
  .globl main  ; Declare main as global (for linker)
  main:
    pushq %rbp  ; Function prologue: save base pointer
    movq %rsp, %rbp  ; Set up current stack frame
    movl $2, %eax  ; Store return value (2) in EAX (ABI requirement)
    popq %rbp  ; Function epilogue: restore base pointer
    ret  ; Return to caller (crt0 runtime)
  ```  
- **Key Detail**: On Linux, add `.section .note.GNU-stack,"",@progbits` to enable non-executable stack (per *Writing a C Compiler*, Chapter 1).

### Validation
Test the minimal compiler with `return 2;`:  
1. Lexer: Generates valid tokens.  
2. Parser: Builds a correct AST.  
3. IR: Emits `return Constant(2);`.  
4. Assembly: Runs via `gcc program.s -o program` and exits with code `2` (checked via `echo $?`).


## 2. Standard C Compiler – Add Full C Features
### Objective
Extend the minimal compiler to support real-world C features (variables, control flow, functions, file scope) by adding **semantic analysis** and optimizing backend passes — aligning with *Writing a C Compiler*’s Part I ("The Basics") and *From Source Code to Machine Code*’s "feature expansion" principles.

### Key Extensions (From *Writing a C Compiler*, Chapters 5–10)
#### 2.1 New Stage: Semantic Analysis
Semantic analysis validates the program’s logic (beyond syntax) and builds a **symbol table** to track identifiers (variables, functions). This stage is critical for standard C (per *Writing a C Compiler*, Chapter 5).  
- **Variable Resolution**: Ensure variables are declared before use and not duplicated in the same scope. For example:  
  ```c
  int main(void) {
    int a = 5;  // Valid: declared
    return a + b;  // Error: b is undeclared
  }
  ```  
- **Type Checking**: Validate that operations match types (e.g., no adding `int` to a pointer) and functions are called with the correct number of arguments (per *Writing a C Compiler*, Chapter 9). For example:  
  ```c
  int add(int a, int b) { return a + b; }
  int main(void) {
    return add(1);  // Error: add expects 2 arguments
  }
  ```

#### 2.2 Control Flow: if, Loops, and Break/Continue
Add support for control-flow statements using TACKY’s `Jump`/`JumpIfZero` instructions (per *Writing a C Compiler*, Chapters 6–8):  
- **if Statements**: Translate `if (a > 0) return 1; else return 0;` to TACKY:  
  ```tacky
  cmp a, 0 → tmp0;  // Compare a and 0
  JumpIfZero(tmp0, else_label);  // Jump if a ≤ 0
  return 1;
  Label(else_label);
  return 0;
  ```  
- **while Loops**: Use labels and jumps to repeat code. For `while (a > 0) a--;`:  
  ```tacky
  Label(loop_start);
  cmp a, 0 → tmp0;
  JumpIfZero(tmp0, loop_end);
  a = a - 1;
  Jump(loop_start);
  Label(loop_end);
  ```

#### 2.3 Functions: Calling Conventions
Implement function calls using the **x64 System V ABI** (per *Writing a C Compiler*, Chapter 9) — a set of rules for argument passing, return values, and stack management:  
- **Argument Passing**: First 6 integer arguments in registers (`EDI`, `ESI`, `EDX`, `ECX`, `R8D`, `R9D`); extra arguments pushed to the stack (reverse order).  
- **Return Values**: Store in `EAX` (32-bit int) or `RAX` (64-bit int).  
- **Example**: Calling `add(2, 3)`:  
  ```asm
  movl $2, %edi  ; 1st argument (2) in EDI
  movl $3, %esi  ; 2nd argument (3) in ESI
  call add  ; Invoke add
  ```

#### 2.4 File Scope & Storage-Class Specifiers
Handle `static` (internal linkage, persistent storage) and `extern` (external linkage, declared elsewhere) — per *Writing a C Compiler*, Chapter 10:  
- **static Variables**: Stored in the **data section** (not stack) and initialized once. For `static int a = 0;`:  
  ```asm
  .data  ; Switch to data section (persistent storage)
  .align 4  ; Align to 4 bytes (32-bit int)
  a: .long 0  ; Define a = 0
  ```  
- **extern Variables**: Declared but not defined (linker resolves from other files). For `extern int b;`, no assembly is emitted — only a symbol table entry.

### Standard Compiler Workflow
For `int add(int a, int b) { return a + b; } int main() { return add(2, 3); }`:  
1. Lexer → Tokens (e.g., `[IntKeyword, Identifier("add"), LParen, ...]`).  
2. Parser → AST (e.g., `FunctionNode("add") → BinaryNode(Add, VarNode("a"), VarNode("b"))`).  
3. Semantic Analysis → Validate `add`’s signature and `main`’s call to `add`.  
4. IR (TACKY) → Emit instructions for `add` and `main`.  
5. Assembly → Generate x64 code following System V ABI.  
6. Linking → `gcc program.s -o program` (links with crt0 runtime).


## 3. Modern C++ Compilers (GCC/Clang) – Extend C to C++
### Objective
Understand how modern C++ compilers (GCC 14+, Clang 18+) build on C compiler stages to support C++-specific features (OOP, templates, exceptions) — integrating *Writing a C Compiler*’s foundational passes with *From Source Code to Machine Code*’s "frontend extension" architecture.

### Core Extensions to C Compiler Stages
C++ compilers reuse C’s lexer, parser, IR, and assembly stages but extend each to handle C++’s richer syntax and semantics (per *Writing a C Compiler*’s Chapter 5–10 principles, expanded for C++).

#### 3.1 Lexer: C++-Specific Tokens
Extend the C lexer to recognize C++ keywords, operators, and literals (per *From Source Code to Machine Code*’s "token set expansion"):  
- **New Keywords**: `class`, `template`, `virtual`, `constexpr` (C++11+), `concepts` (C++20).  
- **New Operators**: `::` (scope resolution), `->*` (pointer-to-member), `sizeof...` (variadic template).  
- **New Literals**: Raw strings (`R"(multi-line)"`), user-defined literals (`123_km`), `nullptr` (C++11+).  
- **Implementation**: Update regex rules (per *Writing a C Compiler*, Table 1-1) to include C++ tokens. For example, `::` is matched by the regex `::\b`.

#### 3.2 Parser: C++-Aware ASTs
Build ASTs that represent OOP, templates, and nested scopes — extending C’s recursive descent parser (per *Writing a C Compiler*, Chapter 3):  
- **Class Nodes**: For `class Base { public: virtual int foo(); };`:  
  `ClassNode("Base") → Members[ VirtualFuncNode("foo", int) ] → AccessModifier(Public)`.  
- **Template Nodes**: For `template <typename T> T add(T a, T b) { return a + b; }`:  
  `TemplateNode(Param("T")) → FunctionNode("add", T, [VarNode("a", T), VarNode("b", T)])`.  
- **Lambda Nodes**: For `[=]() { return x + y; }`:  
  `LambdaNode(Capture(ByValue), Body[ BinaryNode(Add, VarNode("x"), VarNode("y")) ])`.  

#### 3.3 Semantic Analysis: C++ Logic Validation
Expand C’s semantic checks to handle C++’s unique rules (per *Writing a C Compiler*, Chapter 5’s symbol table logic):  
- **OOP Checks**: Ensure `virtual` functions override correctly (matching return types/parameters) and `private` members are not accessed externally.  
- **Template Checks**: Validate template instantiations (e.g., `add<int>(2, 3)` is valid; `add<int>("a", "b")` is not).  
- **Modern C++ Checks**: Enforce `constexpr` constraints (e.g., `constexpr int f() { return 2 + 3; }` is valid; `constexpr int f() { int x; return x; }` is not).

#### 3.4 IR: LLVM IR for C++
Modern C++ compilers use **LLVM IR** (instead of C’s TACKY) to handle complex features — aligning with *From Source Code to Machine Code*’s "universal IR" principle:  
- **Virtual Functions**: Translate `obj.foo()` (virtual) to an indirect call via a **vtable** (stored in the data section, per *Writing a C Compiler*, Chapter 10). Example LLVM IR:  
  ```llvm
  %vtable = load ptr, ptr %obj  ; Load vtable pointer from obj
  %foo_ptr = getelementptr inbounds ptr, ptr %vtable, i64 1  ; Get foo() from vtable
  call ptr %foo_ptr(%obj)  ; Indirect call
  ```  
- **Exceptions**: Use LLVM IR intrinsics (e.g., `llvm.eh.exception`) to handle try/catch blocks — extending C’s jump logic (per *Writing a C Compiler*, Chapter 4).

#### 3.5 Assembly: C++ ABI Compliance
Follow the **Itanium C++ ABI** (used by GCC/Clang) to ensure compatibility between compiled C++ code:  
- **Name Mangling**: Encode function names with type info to handle overloading (e.g., `void foo(int)` → `_Z3fooi`, `void foo(double)` → `_Z3food`).  
- **Vtable Layout**: Store vtables in the data section (per *Writing a C Compiler*, Chapter 10) as global arrays of function pointers.  
- **Example**: Assembly for `Base::foo()`:  
  ```asm
  .globl _Z4Base3fooEv  ; Mangled name for Base::foo()
  _Z4Base3fooEv:
    pushq %rbp
    movq %rsp, %rbp
    movl $42, %eax  ; Return 42
    popq %rbp
    ret
  ```

### Key Advantage: Backward Compatibility
GCC/Clang compile C code (via `-std=c99`/`-std=c17`) by disabling C++ extensions — reusing the C compiler’s core stages (per *Writing a C Compiler*’s Part I).


## 4. NVIDIA CUDA Compiler (nvcc) – Target GPU Hardware
### Objective
Learn how nvcc adapts C/C++ compiler stages to optimize for NVIDIA GPUs (CUDA Cores, SMs) — integrating *Writing a C Compiler*’s stage-based design with *From Source Code to Machine Code*’s "target hardware adaptation" principles.

### Core Workflow: Host + Device Code
CUDA compilers handle two code types:  
- **Host Code**: Runs on the CPU (compiled like standard C/C++).  
- **Device Code**: Runs on the GPU (compiled to GPU ISA via PTX IR).  

#### 4.1 Stage 1: Frontend – Separate Host/Device Code
The frontend parses CUDA code and identifies device code via keywords (`__global__` for kernels, `__device__` for device functions) — extending C++’s parser (per *Writing a C Compiler*, Chapter 3):  
- **Example**:  
  ```c
  __global__ void vec_add(int* a, int* b, int* c) {  // Device code (GPU)
    int idx = threadIdx.x;  // GPU thread ID
    c[idx] = a[idx] + b[idx];
  }
  int main() {  // Host code (CPU)
    // Allocate GPU memory + launch kernel
    vec_add<<<1, 1024>>>(a_dev, b_dev, c_dev);
  }
  ```

#### 4.2 Stage 2: IR – PTX (Parallel Thread Execution)
Translate device code to **PTX** — a portable IR for NVIDIA GPUs (analogous to C’s TACKY, per *Writing a C Compiler*, Chapter 2):  
- **PTX Abstraction**: Hides GPU hardware details (e.g., `add.s32` for 32-bit integer addition).  
- **Example PTX for `c[idx] = a[idx] + b[idx];`**:  
  ```ptx
  ld.global.s32 %r1, [%a+%idx*4];  // Load a[idx] from global memory
  ld.global.s32 %r2, [%b+%idx*4];  // Load b[idx] from global memory
  add.s32 %r3, %r1, %r2;  // Add
  st.global.s32 [%c+%idx*4], %r3;  // Store to c[idx]
  ```

#### 4.3 Stage 3: Backend – Optimize for GPU
Optimize PTX for GPU architecture (per *From Source Code to Machine Code*’s "hardware-specific optimization"):  
- **Memory Coalescing**: Reorder memory access to reduce global memory latency (critical for GPU performance).  
- **Thread Scheduling**: Map PTX threads to GPU hardware threads (warps) to maximize occupancy.  
- **Assembly Generation**: Translate optimized PTX to **cubin** (NVIDIA GPU ISA, e.g., Ampere, Hopper).

#### 4.4 Stage 4: Kernel Launch Setup
The host code generates a **kernel launch configuration** (e.g., `vec_add<<<1, 1024>>>`) — translated to low-level CUDA Runtime calls. The compiler ensures alignment with GPU thread hierarchy (grids → blocks → threads).

### Key Difference from C Compilers
- **Parallelism**: nvcc optimizes for thousands of concurrent threads (C compilers target single-threaded CPU execution).  
- **Memory Model**: Manages GPU memory spaces (global, shared, local) — extending C’s stack/data/BSS sections (per *Writing a C Compiler*, Chapter 10).


## 5. AMD ROCm/HIP Compiler – Cross-Vendor GPU Compilation
### Objective
Understand how AMD’s ROCm ecosystem and HIP (Heterogeneous-Compute Interface for Portability) enable cross-vendor GPU programming — building on *Writing a C Compiler*’s code conversion logic and *From Source Code to Machine Code*’s "portable IR" design.

### Core Workflow: HIP → LLVM IR → AMD ISA
HIP acts as a portable layer that translates CUDA-like code to AMD’s GPU ISA (GCN/CDNA) or NVIDIA’s ISA — reusing C++’s Clang/LLVM frontend (per *Writing a C Compiler*’s Chapter 3–5).

#### 5.1 Stage 1: HIP Code Conversion
Use `hipify-clang` to convert CUDA code to HIP code (no manual rewrite):  
- **Example**:  
  ```c
  // CUDA → HIP conversion
  cudaMalloc(&a_dev, size) → hipMalloc(&a_dev, size);
  __global__ void vec_add(...) → __global__ void vec_add(...);  // Unchanged
  ```

#### 5.2 Stage 2: Frontend – Clang for HIP
The Clang frontend parses HIP code, separates host/device code, and generates **LLVM IR** — extending C++’s frontend (per *Writing a C Compiler*, Chapter 3):  
- **Device Code Handling**: Identifies `__global__`/`__device__` functions and generates LLVM IR with GPU-specific intrinsics.  
- **Host Code Handling**: Compiles CPU code like standard C++ (follows x64 System V ABI).

#### 5.3 Stage 3: Backend – LLVM for AMD ISA
LLVM optimizes the IR (loop unrolling, memory coalescing) and compiles it to **AMD GCN/CDNA ISA** (per *From Source Code to Machine Code*’s "target ISA generation"):  
- **Optimizations**: Tailored for AMD GPU hardware (e.g., maximizing wavefront utilization).  
- **Example**: LLVM IR for `a[idx] + b[idx]` is translated to GCN ISA:  
  ```gcn
  s_load_dword s0, s[0:1], 0x0  ; Load a[idx]
  s_load_dword s1, s[0:1], 0x4  ; Load b[idx]
  v_add_i32 v0, s0, s1  ; Add
  s_store_dword s[0:1], 0x8, v0  ; Store result
  ```

#### 5.4 Stage 4: Runtime – ROCm Runtime
The ROCm runtime manages GPU memory, kernel launches, and host-device synchronization — analogous to CUDA Runtime but cross-vendor.

### Key Advantage: Portability
HIP code compiles to both AMD and NVIDIA GPUs, avoiding vendor lock-in — a major extension of C’s "single-target" compiler design (per *Writing a C Compiler*).


## 6. Triton Compiler – Simplify GPU Programming
### Objective
Learn how Triton (a Python-based compiler) abstracts low-level GPU details, letting developers write high-performance GPU code without hand-tuning — integrating *Writing a C Compiler*’s IR decoupling and *From Source Code to Machine Code*’s "JIT compilation" principles.

### Core Workflow: Python → Custom IR → JIT GPU ISA
Triton targets GPU matrix operations (ML workloads) and uses JIT compilation to generate optimized code on the fly.

#### 6.1 Stage 1: Frontend – Python API
Developers write GPU code in Python using Triton’s API (no raw PTX/ISA) — extending C’s text-based frontend (per *Writing a C Compiler*, Chapter 1):  
- **Example**: Matrix multiplication:  
  ```python
  import triton
  import triton.language as tl

  @triton.jit  # Mark for JIT compilation
  def matmul(a, b, c, M, N, K):
    i = tl.program_id(0)  # Kernel program ID (row)
    j = tl.program_id(1)  # Kernel program ID (column)
    a_block = tl.load(a[i*K + tl.arange(0, 32)])  # Load a block of a
    b_block = tl.load(b[j + tl.arange(0, 32)]*N)  # Load a block of b
    c[i*N + j] = tl.sum(a_block * b_block)  # Compute dot product
  ```

#### 6.2 Stage 2: IR – Triton Custom IR
Triton converts Python code to a **custom low-level IR** — analogous to C’s TACKY (per *Writing a C Compiler*, Chapter 2):  
- **IR Features**: Explicit thread block management, memory access primitives, and arithmetic operations.  
- **Optimizations**: Built-in optimizations (memory coalescing, thread scheduling) to maximize GPU performance.

#### 6.3 Stage 3: JIT Compilation
Triton JIT-compiles the IR to target GPU ISA (NVIDIA PTX/AMD GCN) at runtime — aligning with *From Source Code to Machine Code*’s "dynamic compilation" principles:  
- **Hardware Adaptation**: The JIT compiler detects the target GPU (e.g., NVIDIA A100, AMD MI250) and generates optimized ISA.  
- **Example**: For an NVIDIA GPU, the IR is compiled to PTX; for AMD, to GCN ISA.

### Key Advantage: Productivity
Triton eliminates manual PTX/ISA writing and thread block management — a major simplification of GPU compiler workflows (per *Writing a C Compiler*’s "simplicity-first" design for minimal C).


## 7. CUDA-Q Compiler – Quantum-Classical Hybrid Programming
### Objective
Explore CUDA-Q, NVIDIA’s compiler for quantum-classical hybrid programs, which run on quantum GPUs (NVIDIA Quantum Platform) — extending *Writing a C Compiler*’s multi-stage design and *From Source Code to Machine Code*’s "hybrid hardware" adaptation.

### Core Workflow: Quantum + Classical Code → Quantum IR → Hardware ISA
CUDA-Q integrates quantum code (for quantum processors) with classical GPU code (for pre/post-processing) — reusing CUDA’s host/device separation (per Chapter 4).

#### 7.1 Stage 1: Frontend – Quantum-Classical Code Parsing
The frontend parses code with both classical GPU kernels (CUDA) and quantum circuits (`__qpu__` functions) — extending C++’s parser (per *Writing a C Compiler*, Chapter 3):  
- **Example**:  
  ```c
  #include <cudaq.h>

  __qpu__ void entangle() {  // Quantum code (quantum processor)
    cudaq::qubit q0, q1;
    h(q0);        // Hadamard gate
    cx(q0, q1);   // CNOT gate (entangle q0 and q1)
    mz(q0, q1);   // Measure qubits
  }

  int main() {  // Classical code (CPU/GPU)
    entangle();  // Launch quantum circuit
    return 0;
  }
  ```

#### 7.2 Stage 2: IR – Quantum + Classical IR
- **Quantum IR**: Converts `__qpu__` functions to **OpenQASM 3.0** (a quantum IR) — analogous to C’s TACKY (per *Writing a C Compiler*, Chapter 2).  
- **Classical IR**: Compiles CUDA code to LLVM IR (per Chapter 3).

#### 7.3 Stage 3: Backend – Target-Specific Compilation
- **Quantum Code**: Translates OpenQASM 3.0 to quantum hardware ISA (e.g., NVIDIA Quantum Processor instructions).  
- **Classical Code**: Compiles LLVM IR to x64 (CPU) or NVIDIA PTX (GPU) — per *From Source Code to Machine Code*’s "multi-target" design.

#### 7.4 Stage 4: Runtime – Quantum-Classical Sync
The CUDA-Q runtime coordinates execution between classical GPUs and quantum processors (e.g., transfer quantum measurement results to the CPU/GPU).

### Key Extension from C Compilers
CUDA-Q adds **quantum IR** and **quantum hardware support** to the classical compiler pipeline — a paradigm shift from C’s "classical-only" focus (per *Writing a C Compiler*).


## Summary of Compiler Stages & Design Principles
| Compiler Type       | Core Stages (From C Foundation)      | Target Hardware       | Key Design Principle (From References)                          |
|---------------------|--------------------------------------|-----------------------|----------------------------------------------------------------|
| Minimal C           | Lexer → Parser → TACKY → x64 Assembly| CPU (x64)             | Stage-based decoupling (*Writing a C Compiler*, Chapter 1)      |
| Standard C          | + Semantic Analysis                  | CPU (x64)             | ABI compliance (*Writing a C Compiler*, Chapter 9)             |
| Modern C++ (GCC/Clang)| + C++ Tokens/AST/LLVM IR/Itanium ABI | CPU (x64/ARM)         | Frontend extension (*From Source Code to Machine Code*)         |
| NVIDIA nvcc         | + Host/Device Separation → PTX       | NVIDIA GPUs           | Target hardware adaptation (*From Source Code to Machine Code*) |
| AMD ROCm/HIP        | + HIP Conversion → LLVM → GCN ISA   | AMD/NVIDIA GPUs       | Cross-vendor portability (*Writing a C Compiler* code reuse)    |
| Triton              | Python → Custom IR → JIT GPU ISA    | GPUs (ML)             | JIT compilation (*From Source Code to Machine Code*)            |
| CUDA-Q              | + Quantum IR → Quantum ISA          | Quantum + Classical   | Hybrid hardware support (*From Source Code to Machine Code*)    |

All compilers build on the minimal C compiler’s "stage-based" foundation — proving the universality of the principles in *Writing a C Compiler* and *From Source Code to Machine Code*.