# Compiler-101 Examples

This directory contains practical, runnable examples for each module in the Compiler-101 course.

## Available Examples

### Module 1: Minimal C Compiler
**Directory:** [`01-minimal-compiler/`](01-minimal-compiler/)

Basic C compiler implementation demonstrating core compilation concepts.

---

### Module 6: Triton Compiler
**Directory:** [`06-triton-compiler/`](06-triton-compiler/)

High-performance GPU programming with Python using Triton compiler.

**Examples:**
- **01_vector_addition.py** - Basic GPU kernel with automatic vectorization
- **02_matrix_multiplication.py** - Tiling and shared memory optimization
- **03_fused_softmax.py** - Kernel fusion for reduced memory bandwidth
- **04_custom_activation.py** - Custom ML activation functions (GELU, Swish, Mish)

**Quick Start:**
```bash
cd 06-triton-compiler
pip install -r requirements.txt
python test_setup.py
python 01_vector_addition.py
```

See the [module README](06-triton-compiler/README.md) for detailed setup and usage instructions.

---

## Prerequisites

Each module may have different requirements. Check the individual module directories for:
- System requirements
- Installation instructions
- Dependencies
- Usage examples

## Learning Path

We recommend following the modules in order:
1. Start with Module 1 to understand basic compiler concepts
2. Progress through subsequent modules building on foundational knowledge
3. Module 6 demonstrates modern compiler techniques for GPU programming

## Contributing

Found issues or have improvements? Contributions are welcome:
- Fix bugs in examples
- Add new examples
- Improve documentation
- Share insights and optimizations
