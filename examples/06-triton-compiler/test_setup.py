"""
Installation Verification Script
================================

This script verifies that all dependencies for Module 6 examples are properly installed.

Usage:
    python test_setup.py
"""

import sys


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print(f"✗ CUDA is NOT available")
            print(f"  PyTorch version: {torch.__version__}")
            print(f"  CUDA compiled: {torch.version.cuda}")
            return False
    except ImportError:
        print(f"✗ PyTorch is not installed")
        return False


def check_triton():
    """Check Triton installation and version."""
    try:
        import triton
        print(f"✓ Triton is installed")
        print(f"  Version: {triton.__version__}")
        return True
    except ImportError:
        print(f"✗ Triton is NOT installed")
        return False


def test_simple_kernel():
    """Test a simple Triton kernel."""
    print("\nTesting simple Triton kernel...")
    try:
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

        # Test data
        size = 1024
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')
        output = torch.empty_like(x)

        # Launch kernel
        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)

        # Verify
        expected = x + y
        if torch.allclose(output, expected):
            print("✓ Simple kernel test PASSED")
            return True
        else:
            print("✗ Simple kernel test FAILED (incorrect results)")
            return False

    except Exception as e:
        print(f"✗ Simple kernel test FAILED: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("Triton Compiler Examples - Installation Verification")
    print("=" * 60)
    print()

    all_ok = True

    # Check core dependencies
    print("Checking core dependencies...")
    print("-" * 60)
    all_ok &= check_import('torch', 'PyTorch')
    all_ok &= check_import('triton', 'Triton')
    all_ok &= check_import('numpy', 'NumPy')
    print()

    # Check CUDA
    print("Checking CUDA availability...")
    print("-" * 60)
    cuda_ok = check_cuda()
    all_ok &= cuda_ok
    print()

    # Check Triton version
    print("Checking Triton installation...")
    print("-" * 60)
    triton_ok = check_triton()
    all_ok &= triton_ok
    print()

    # Test simple kernel if everything is available
    if cuda_ok and triton_ok:
        print("Running functional tests...")
        print("-" * 60)
        all_ok &= test_simple_kernel()
        print()

    # Summary
    print("=" * 60)
    if all_ok:
        print("✓ All checks PASSED! You're ready to run the examples.")
        print()
        print("Try running:")
        print("  python 01_vector_addition.py")
        print("  python 02_matrix_multiplication.py")
        print("  python 03_fused_softmax.py")
        print("  python benchmark_all.py")
    else:
        print("✗ Some checks FAILED. Please install missing dependencies.")
        print()
        print("Installation instructions:")
        print()
        print("1. Install PyTorch with CUDA support:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print()
        print("2. Install Triton:")
        print("   pip install triton")
        print()
        print("See README.md for detailed installation instructions.")

    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
