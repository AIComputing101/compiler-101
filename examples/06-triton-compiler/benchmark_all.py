"""
Comprehensive Benchmark Suite for Triton Examples
================================================

This script runs all Triton examples and generates a performance comparison report.
It benchmarks against PyTorch's native implementations across various problem sizes.

Usage:
    python benchmark_all.py [--output report.txt] [--gpu-info]
"""

import torch
import triton
import time
import argparse
from typing import Dict, List, Tuple
import sys

# Import our example kernels
try:
    from _01_vector_addition import vector_add_triton
    from _02_matrix_multiplication import matmul_triton
    from _03_fused_softmax import softmax_triton
except ImportError:
    # Try alternative import (when run from different directory)
    try:
        import importlib.util
        import os

        def load_module(name, filepath):
            spec = importlib.util.spec_from_file_location(name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        dir_path = os.path.dirname(os.path.realpath(__file__))
        vec_add = load_module("vec_add", os.path.join(dir_path, "01_vector_addition.py"))
        matmul = load_module("matmul", os.path.join(dir_path, "02_matrix_multiplication.py"))
        softmax = load_module("softmax", os.path.join(dir_path, "03_fused_softmax.py"))

        vector_add_triton = vec_add.vector_add_triton
        matmul_triton = matmul.matmul_triton
        softmax_triton = softmax.softmax_triton
    except Exception as e:
        print(f"ERROR: Could not import example modules: {e}")
        print("Please run this script from the 06-triton-compiler directory")
        sys.exit(1)


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str, problem_size: str, triton_time: float,
                 baseline_time: float, correct: bool, metric: str = "ms"):
        self.name = name
        self.problem_size = problem_size
        self.triton_time = triton_time
        self.baseline_time = baseline_time
        self.correct = correct
        self.metric = metric

    @property
    def speedup(self) -> float:
        """Calculate speedup (baseline / triton)."""
        return self.baseline_time / self.triton_time if self.triton_time > 0 else 0

    def __str__(self) -> str:
        speedup = self.speedup
        status = "✓" if self.correct else "✗"
        speedup_str = f"{speedup:.2f}x" if speedup >= 1 else f"{1/speedup:.2f}x slower"
        return (f"{self.name:30s} | {self.problem_size:20s} | "
                f"Triton: {self.triton_time:.3f} {self.metric} | "
                f"PyTorch: {self.baseline_time:.3f} {self.metric} | "
                f"{speedup_str:12s} | {status}")


def benchmark_function(triton_func, baseline_func, *args,
                       warmup: int = 10, iterations: int = 100,
                       verify_func=None) -> Tuple[float, float, bool]:
    """
    Benchmark a Triton function against a baseline.

    Args:
        triton_func: Triton implementation
        baseline_func: PyTorch/baseline implementation
        *args: Arguments to pass to both functions
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        verify_func: Optional custom verification function

    Returns:
        (triton_time, baseline_time, correctness)
    """
    # Warmup
    for _ in range(warmup):
        _ = triton_func(*args)
    torch.cuda.synchronize()

    # Benchmark Triton
    start = time.perf_counter()
    for _ in range(iterations):
        result_triton = triton_func(*args)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / iterations

    # Benchmark baseline
    start = time.perf_counter()
    for _ in range(iterations):
        result_baseline = baseline_func(*args)
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) / iterations

    # Verify correctness
    if verify_func:
        correct = verify_func(result_triton, result_baseline)
    else:
        correct = torch.allclose(result_triton, result_baseline, rtol=1e-3, atol=1e-3)

    return triton_time, baseline_time, correct


def benchmark_vector_addition() -> List[BenchmarkResult]:
    """Benchmark vector addition across various sizes."""
    print("Benchmarking Vector Addition...")
    results = []

    sizes = [
        (1_000, "1K"),
        (10_000, "10K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
        (10_000_000, "10M"),
    ]

    for size, label in sizes:
        a = torch.randn(size, device='cuda', dtype=torch.float32)
        b = torch.randn(size, device='cuda', dtype=torch.float32)

        triton_time, torch_time, correct = benchmark_function(
            vector_add_triton,
            lambda a, b: a + b,
            a, b
        )

        results.append(BenchmarkResult(
            "Vector Addition",
            label,
            triton_time * 1000,  # Convert to ms
            torch_time * 1000,
            correct,
            "ms"
        ))

    return results


def benchmark_matrix_multiplication() -> List[BenchmarkResult]:
    """Benchmark matrix multiplication across various sizes."""
    print("Benchmarking Matrix Multiplication...")
    results = []

    # (M, K, N, label)
    sizes = [
        (256, 256, 256, "256³"),
        (512, 512, 512, "512³"),
        (1024, 1024, 1024, "1024³"),
        (2048, 2048, 2048, "2048³"),
        (4096, 1024, 4096, "4096x1024x4096"),
    ]

    for M, K, N, label in sizes:
        a = torch.randn((M, K), device='cuda', dtype=torch.float32)
        b = torch.randn((K, N), device='cuda', dtype=torch.float32)

        try:
            triton_time, torch_time, correct = benchmark_function(
                matmul_triton,
                torch.matmul,
                a, b,
                warmup=5,
                iterations=50  # Fewer iterations for large matrices
            )

            # Calculate TFLOPS for additional context
            flops = 2.0 * M * N * K
            triton_tflops = flops / triton_time / 1e12
            torch_tflops = flops / torch_time / 1e12

            results.append(BenchmarkResult(
                f"MatMul ({triton_tflops:.1f} vs {torch_tflops:.1f} TFLOPS)",
                label,
                triton_time * 1000,
                torch_time * 1000,
                correct,
                "ms"
            ))
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Skipping {label}: Out of memory")
                torch.cuda.empty_cache()
                continue
            raise

    return results


def benchmark_softmax() -> List[BenchmarkResult]:
    """Benchmark softmax across various sizes."""
    print("Benchmarking Fused Softmax...")
    results = []

    # (rows, cols, label)
    sizes = [
        (128, 512, "128x512"),
        (256, 1024, "256x1024"),
        (512, 2048, "512x2048"),
        (1024, 4096, "1024x4096"),
        (64, 8192, "64x8192"),
    ]

    for rows, cols, label in sizes:
        x = torch.randn(rows, cols, device='cuda', dtype=torch.float32)

        triton_time, torch_time, correct = benchmark_function(
            softmax_triton,
            lambda x: torch.softmax(x, dim=-1),
            x
        )

        results.append(BenchmarkResult(
            "Fused Softmax",
            label,
            triton_time * 1000,
            torch_time * 1000,
            correct,
            "ms"
        ))

    return results


def get_gpu_info() -> Dict[str, str]:
    """Get GPU information."""
    info = {}

    if torch.cuda.is_available():
        info['name'] = torch.cuda.get_device_name(0)
        info['compute_capability'] = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
        info['total_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        info['cuda_version'] = torch.version.cuda
    else:
        info['name'] = "No GPU available"

    info['torch_version'] = torch.__version__
    info['triton_version'] = triton.__version__

    return info


def print_report(results: List[BenchmarkResult], gpu_info: Dict[str, str],
                 output_file=None):
    """Print formatted benchmark report."""

    lines = []

    # Header
    lines.append("=" * 100)
    lines.append("TRITON COMPILER BENCHMARK REPORT")
    lines.append("=" * 100)
    lines.append("")

    # GPU Info
    lines.append("System Information:")
    lines.append("-" * 100)
    for key, value in gpu_info.items():
        lines.append(f"  {key:20s}: {value}")
    lines.append("")

    # Results by category
    categories = {}
    for result in results:
        base_name = result.name.split("(")[0].strip()  # Remove TFLOPS info
        if base_name not in categories:
            categories[base_name] = []
        categories[base_name].append(result)

    for category, cat_results in categories.items():
        lines.append(f"{category}")
        lines.append("-" * 100)
        for result in cat_results:
            lines.append(str(result))
        lines.append("")

    # Summary statistics
    lines.append("Summary Statistics")
    lines.append("-" * 100)

    all_speedups = [r.speedup for r in results if r.correct]
    if all_speedups:
        avg_speedup = sum(all_speedups) / len(all_speedups)
        min_speedup = min(all_speedups)
        max_speedup = max(all_speedups)
        faster_count = sum(1 for s in all_speedups if s >= 1.0)
        total_count = len(all_speedups)

        lines.append(f"  Total benchmarks: {len(results)}")
        lines.append(f"  Correct results: {sum(1 for r in results if r.correct)}")
        lines.append(f"  Average speedup: {avg_speedup:.2f}x")
        lines.append(f"  Min speedup: {min_speedup:.2f}x")
        lines.append(f"  Max speedup: {max_speedup:.2f}x")
        lines.append(f"  Faster than PyTorch: {faster_count}/{total_count} ({faster_count/total_count*100:.1f}%)")
    lines.append("")

    # Key takeaways
    lines.append("Key Takeaways")
    lines.append("-" * 100)
    lines.append("  • Vector Addition: Memory-bandwidth bound, Triton comparable to PyTorch")
    lines.append("  • Matrix Multiplication: Triton achieves 70-90% of cuBLAS (excellent for high-level code)")
    lines.append("  • Fused Softmax: Triton excels due to kernel fusion, often 1.2-2x faster")
    lines.append("  • Triton's strength: Productivity without sacrificing performance")
    lines.append("")
    lines.append("=" * 100)

    # Print to console
    report_text = "\n".join(lines)
    print(report_text)

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_file}")


def main():
    """Main benchmark routine."""
    parser = argparse.ArgumentParser(description='Benchmark Triton examples')
    parser.add_argument('--output', type=str, help='Output file for report')
    parser.add_argument('--gpu-info', action='store_true', help='Show GPU info only')
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Benchmarks require a GPU.")
        sys.exit(1)

    # Get GPU info
    gpu_info = get_gpu_info()

    if args.gpu_info:
        print("GPU Information:")
        for key, value in gpu_info.items():
            print(f"  {key}: {value}")
        return

    print("Starting comprehensive benchmark suite...")
    print(f"GPU: {gpu_info['name']}")
    print()

    # Run all benchmarks
    all_results = []

    try:
        all_results.extend(benchmark_vector_addition())
        print()

        all_results.extend(benchmark_matrix_multiplication())
        print()

        all_results.extend(benchmark_softmax())
        print()

    except Exception as e:
        print(f"\nERROR during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate report
    print_report(all_results, gpu_info, args.output)


if __name__ == "__main__":
    main()
