"""
Comprehensive Benchmark for AMD Optimization
===========================================

This script provides detailed benchmarking of the AMD bank conflict optimization
across various matrix sizes, data types, and configurations.

Features:
- Multiple matrix sizes (square and rectangular)
- Different data types (FP32, FP16)
- Statistical analysis (mean, std, min, max)
- Comparison charts and reports
- Hardware detection and adaptive testing

Usage:
    # Basic benchmark
    python bench_optimization.py

    # Quick test (fewer iterations)
    python bench_optimization.py --quick

    # Save detailed report
    python bench_optimization.py --output benchmark_results.json

    # Generate plots
    python bench_optimization.py --plot
"""

import torch
import triton
import argparse
import json
import time
import sys
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

try:
    from _05_optimized_matmul_amd import matmul_triton, detect_gpu_vendor
except ImportError:
    print("ERROR: Could not import optimized matmul")
    print("Make sure 05_optimized_matmul_amd.py is in the same directory")
    sys.exit(1)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    M: int
    K: int
    N: int
    dtype: torch.dtype
    iterations: int = 100
    warmup: int = 10

    def __str__(self):
        dtype_str = str(self.dtype).split('.')[-1]
        return f"{self.M}x{self.K}x{self.N} ({dtype_str})"


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    baseline_mean: float
    baseline_std: float
    baseline_min: float
    baseline_max: float
    optimized_mean: float
    optimized_std: float
    optimized_min: float
    optimized_max: float
    torch_mean: float
    improvement_pct: float
    tflops_baseline: float
    tflops_optimized: float
    tflops_torch: float
    correctness_error: float

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['config'] = str(self.config)
        return result


class MatmulBenchmark:
    """Benchmark suite for matrix multiplication optimizations."""

    def __init__(self, use_cuda_events: bool = True):
        self.use_cuda_events = use_cuda_events
        self.vendor = detect_gpu_vendor()
        self.results: List[BenchmarkResult] = []

    def benchmark_single(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Benchmark a single configuration.

        Args:
            config: Benchmark configuration

        Returns:
            BenchmarkResult with timing and correctness data
        """
        print(f"  Benchmarking {config}...", end=' ', flush=True)

        # Create test data
        a = torch.randn((config.M, config.K), device='cuda', dtype=config.dtype)
        b = torch.randn((config.K, config.N), device='cuda', dtype=config.dtype)

        # Warm up
        for _ in range(config.warmup):
            _ = matmul_triton(a, b, use_amd_padding=False)
            _ = matmul_triton(a, b, use_amd_padding=True)
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Measure baseline
        baseline_times = self._measure_times(
            lambda: matmul_triton(a, b, use_amd_padding=False),
            config.iterations
        )

        # Measure optimized
        optimized_times = self._measure_times(
            lambda: matmul_triton(a, b, use_amd_padding=True),
            config.iterations
        )

        # Measure PyTorch
        torch_times = self._measure_times(
            lambda: torch.matmul(a, b),
            config.iterations
        )

        # Verify correctness
        c_baseline = matmul_triton(a, b, use_amd_padding=False)
        c_optimized = matmul_triton(a, b, use_amd_padding=True)
        c_torch = torch.matmul(a, b)

        error = torch.max(torch.abs(c_optimized - c_torch)).item()

        # Calculate TFLOPS
        flops = 2.0 * config.M * config.N * config.K
        baseline_mean = sum(baseline_times) / len(baseline_times)
        optimized_mean = sum(optimized_times) / len(optimized_times)
        torch_mean = sum(torch_times) / len(torch_times)

        tflops_baseline = flops / baseline_mean / 1e12
        tflops_optimized = flops / optimized_mean / 1e12
        tflops_torch = flops / torch_mean / 1e12

        improvement = (baseline_mean - optimized_mean) / baseline_mean * 100

        result = BenchmarkResult(
            config=config,
            baseline_mean=baseline_mean,
            baseline_std=self._std(baseline_times),
            baseline_min=min(baseline_times),
            baseline_max=max(baseline_times),
            optimized_mean=optimized_mean,
            optimized_std=self._std(optimized_times),
            optimized_min=min(optimized_times),
            optimized_max=max(optimized_times),
            torch_mean=torch_mean,
            improvement_pct=improvement,
            tflops_baseline=tflops_baseline,
            tflops_optimized=tflops_optimized,
            tflops_torch=tflops_torch,
            correctness_error=error,
        )

        print(f"{improvement:+.1f}% ({tflops_optimized:.1f} TFLOPS)")

        return result

    def _measure_times(self, func, iterations: int) -> List[float]:
        """Measure execution times for multiple iterations."""
        times = []

        if self.use_cuda_events:
            # Use CUDA events for precise GPU timing
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                func()
                end.record()

                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) / 1000)  # Convert to seconds

        else:
            # Use CPU timing (less accurate)
            for _ in range(iterations):
                start = time.perf_counter()
                func()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        return times

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def run_suite(self, configs: List[BenchmarkConfig]):
        """Run benchmark suite for all configurations."""
        print(f"\nRunning Benchmark Suite")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Vendor: {self.vendor}")
        print("=" * 70)

        for config in configs:
            try:
                result = self.benchmark_single(config)
                self.results.append(result)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ✗ OOM - skipping")
                    torch.cuda.empty_cache()
                else:
                    raise

    def print_summary(self):
        """Print benchmark summary."""
        if not self.results:
            print("No results to summarize")
            return

        print("\n\nBenchmark Summary")
        print("=" * 90)
        print(f"{'Configuration':<25} {'Baseline':<12} {'Optimized':<12} {'Improvement':<12} {'Error':<10}")
        print("-" * 90)

        total_improvement = 0
        positive_improvements = 0

        for result in self.results:
            print(f"{str(result.config):<25} "
                  f"{result.baseline_mean*1e3:>10.2f}ms "
                  f"{result.optimized_mean*1e3:>10.2f}ms "
                  f"{result.improvement_pct:>+10.1f}% "
                  f"{result.correctness_error:>8.2e}")

            total_improvement += result.improvement_pct
            if result.improvement_pct > 0:
                positive_improvements += 1

        print("-" * 90)
        avg_improvement = total_improvement / len(self.results)
        print(f"Average improvement: {avg_improvement:+.1f}%")
        print(f"Configurations with speedup: {positive_improvements}/{len(self.results)}")

        # Analysis
        print("\n\nAnalysis")
        print("=" * 90)

        if self.vendor == 'AMD':
            if avg_improvement > 10:
                print("✓ Excellent results! AMD optimization is effective.")
                print("  LDS bank conflicts are significantly reduced.")
            elif avg_improvement > 0:
                print("✓ Positive results. AMD optimization provides benefit.")
                print("  Some configurations may need tuning for optimal performance.")
            else:
                print("⚠ Optimization not showing expected gains.")
                print("  Possible reasons:")
                print("  - GPU model may handle bank conflicts differently")
                print("  - Matrix sizes may be too small to show benefit")
                print("  - Padding overhead may outweigh conflict reduction")
        else:
            print(f"ℹ Running on {self.vendor} GPU")
            print("  This optimization primarily targets AMD architecture.")
            print("  Results on NVIDIA GPUs may show minimal impact or slight overhead.")

        # Find best and worst cases
        best = max(self.results, key=lambda r: r.improvement_pct)
        worst = min(self.results, key=lambda r: r.improvement_pct)

        print(f"\nBest case:  {best.config} ({best.improvement_pct:+.1f}%)")
        print(f"Worst case: {worst.config} ({worst.improvement_pct:+.1f}%)")

    def save_results(self, filename: str):
        """Save results to JSON file."""
        data = {
            'gpu': torch.cuda.get_device_name(0),
            'vendor': self.vendor,
            'results': [r.to_dict() for r in self.results],
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Results saved to {filename}")

    def plot_results(self):
        """Plot benchmark results (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            if not self.results:
                print("No results to plot")
                return

            # Extract data
            configs = [str(r.config) for r in self.results]
            baseline = [r.tflops_baseline for r in self.results]
            optimized = [r.tflops_optimized for r in self.results]
            pytorch = [r.tflops_torch for r in self.results]

            # Create bar chart
            x = np.arange(len(configs))
            width = 0.25

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Plot 1: TFLOPS comparison
            ax1.bar(x - width, baseline, width, label='Baseline', alpha=0.8)
            ax1.bar(x, optimized, width, label='Optimized', alpha=0.8)
            ax1.bar(x + width, pytorch, width, label='PyTorch', alpha=0.8)

            ax1.set_ylabel('TFLOPS', fontsize=12)
            ax1.set_title(f'Matrix Multiplication Performance ({self.vendor} GPU)',
                         fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(configs, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Improvement percentage
            improvements = [r.improvement_pct for r in self.results]
            colors = ['green' if i > 0 else 'red' for i in improvements]

            ax2.bar(x, improvements, color=colors, alpha=0.6)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax2.set_ylabel('Improvement (%)', fontsize=12)
            ax2.set_title('Optimization Impact', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(configs, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('optimization_benchmark.png', dpi=150, bbox_inches='tight')
            print("\n✓ Plot saved to 'optimization_benchmark.png'")

        except ImportError:
            print("\n⚠ matplotlib not installed. Install with: pip install matplotlib")


def create_benchmark_configs(quick: bool = False) -> List[BenchmarkConfig]:
    """Create list of benchmark configurations."""
    if quick:
        # Quick test with fewer sizes and iterations
        sizes = [(512, 512, 512), (1024, 1024, 1024)]
        dtypes = [torch.float32]
        iterations = 20
    else:
        # Comprehensive test
        sizes = [
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 1024, 4096),  # Rectangular (common in ML)
            (1024, 4096, 1024),  # Rectangular (common in ML)
        ]
        dtypes = [torch.float32]
        iterations = 100

    configs = []
    for M, K, N in sizes:
        for dtype in dtypes:
            configs.append(BenchmarkConfig(M, K, N, dtype, iterations=iterations))

    return configs


def main():
    """Main benchmark routine."""
    parser = argparse.ArgumentParser(
        description='Benchmark AMD GPU optimization for matrix multiplication'
    )
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test (fewer iterations)')
    parser.add_argument('--output', type=str,
                       help='Save results to JSON file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate performance plots')
    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)

    print("AMD Optimization Benchmark Suite")
    print("=" * 70)

    # Create benchmark suite
    benchmark = MatmulBenchmark(use_cuda_events=True)

    # Create configurations
    configs = create_benchmark_configs(quick=args.quick)
    print(f"Testing {len(configs)} configurations")

    # Run benchmarks
    benchmark.run_suite(configs)

    # Print summary
    benchmark.print_summary()

    # Save results
    if args.output:
        benchmark.save_results(args.output)

    # Plot results
    if args.plot:
        benchmark.plot_results()

    print("\n" + "=" * 70)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
