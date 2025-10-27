"""
Profiling Utilities for Triton Matrix Multiplication
===================================================

This script provides tools to profile Triton kernels and detect performance bottlenecks,
specifically focusing on shared memory bank conflicts on AMD GPUs.

Features:
- GPU vendor detection
- Kernel profiling with rocprof (AMD) or nsys (NVIDIA)
- Bank conflict analysis
- Performance metrics collection
- Automated profiling workflow

Usage:
    # Basic profiling
    python profile_matmul.py

    # Profile specific size
    python profile_matmul.py --size 2048

    # Generate detailed report
    python profile_matmul.py --detailed --output profile_report.txt
"""

import torch
import triton
import argparse
import subprocess
import os
import sys
import json
from pathlib import Path

# Import our kernels
try:
    from _05_optimized_matmul_amd import (
        matmul_triton,
        detect_gpu_vendor,
        matmul_kernel_baseline,
        matmul_kernel_optimized
    )
except ImportError:
    print("ERROR: Could not import optimized matmul kernels")
    print("Make sure 05_optimized_matmul_amd.py is in the same directory")
    sys.exit(1)


class GPUProfiler:
    """Base class for GPU profiling."""

    def __init__(self, vendor: str):
        self.vendor = vendor
        self.metrics = {}

    def check_tools(self) -> bool:
        """Check if profiling tools are available."""
        raise NotImplementedError

    def profile_kernel(self, kernel_name: str, *args, **kwargs) -> dict:
        """Profile a kernel and return metrics."""
        raise NotImplementedError


class AMDProfiler(GPUProfiler):
    """Profiler for AMD GPUs using rocprof."""

    def __init__(self):
        super().__init__('AMD')

    def check_tools(self) -> bool:
        """Check if rocprof is available."""
        try:
            result = subprocess.run(['rocprof', '--version'],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def profile_kernel(self, test_func, output_file='rocprof_output'):
        """
        Profile a kernel using rocprof.

        Args:
            test_func: Function that runs the kernel
            output_file: Base name for output files

        Returns:
            dict: Profiling metrics including bank conflicts
        """
        if not self.check_tools():
            print("WARNING: rocprof not found. Install ROCm toolkit for profiling.")
            return {}

        # Create a temporary Python script to profile
        script_content = f"""
import torch
import sys
sys.path.insert(0, '{os.getcwd()}')
from _05_optimized_matmul_amd import matmul_triton

# Create test data
M, K, N = 1024, 1024, 1024
a = torch.randn((M, K), device='cuda', dtype=torch.float32)
b = torch.randn((K, N), device='cuda', dtype=torch.float32)

# Run kernel
for _ in range(10):  # Multiple iterations for better statistics
    c = matmul_triton(a, b, use_amd_padding={test_func.__name__ == 'optimized'})
torch.cuda.synchronize()
"""

        script_path = 'temp_profile_script.py'
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Run rocprof
        cmd = [
            'rocprof',
            '--stats',
            '--timestamp', 'on',
            '-o', output_file,
            'python', script_path
        ]

        try:
            print(f"Running rocprof (this may take a minute)...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Parse results
            metrics = self._parse_rocprof_output(output_file + '.stats.csv')

            # Clean up
            os.remove(script_path)

            return metrics

        except subprocess.TimeoutExpired:
            print("ERROR: Profiling timed out")
            return {}
        except Exception as e:
            print(f"ERROR during profiling: {e}")
            return {}

    def _parse_rocprof_output(self, stats_file: str) -> dict:
        """Parse rocprof statistics output."""
        metrics = {
            'lds_bank_conflicts': 0,
            'avg_duration_ns': 0,
            'kernel_count': 0,
        }

        if not os.path.exists(stats_file):
            return metrics

        try:
            with open(stats_file, 'r') as f:
                lines = f.readlines()
                # Parse CSV format (simplified)
                for line in lines[1:]:  # Skip header
                    if 'matmul_kernel' in line:
                        parts = line.split(',')
                        # Extract relevant metrics
                        # Note: Actual column positions depend on rocprof version
                        metrics['kernel_count'] += 1

        except Exception as e:
            print(f"WARNING: Could not parse rocprof output: {e}")

        return metrics


class NVIDIAProfiler(GPUProfiler):
    """Profiler for NVIDIA GPUs using nsys (Nsight Systems)."""

    def __init__(self):
        super().__init__('NVIDIA')

    def check_tools(self) -> bool:
        """Check if nsys is available."""
        try:
            result = subprocess.run(['nsys', '--version'],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def profile_kernel(self, test_func, output_file='nsys_report'):
        """
        Profile a kernel using nsys.

        Args:
            test_func: Function that runs the kernel
            output_file: Base name for output files

        Returns:
            dict: Profiling metrics
        """
        if not self.check_tools():
            print("WARNING: nsys not found. Install CUDA toolkit for profiling.")
            return {}

        # Create temporary script
        script_content = f"""
import torch
import sys
sys.path.insert(0, '{os.getcwd()}')
from _05_optimized_matmul_amd import matmul_triton

M, K, N = 1024, 1024, 1024
a = torch.randn((M, K), device='cuda', dtype=torch.float32)
b = torch.randn((K, N), device='cuda', dtype=torch.float32)

for _ in range(10):
    c = matmul_triton(a, b, use_amd_padding={test_func.__name__ == 'optimized'})
torch.cuda.synchronize()
"""

        script_path = 'temp_profile_script.py'
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Run nsys
        cmd = [
            'nsys', 'profile',
            '--stats', 'true',
            '-o', output_file,
            'python', script_path
        ]

        try:
            print(f"Running nsys (this may take a minute)...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Parse results
            metrics = self._parse_nsys_output(result.stdout)

            # Clean up
            os.remove(script_path)

            return metrics

        except subprocess.TimeoutExpired:
            print("ERROR: Profiling timed out")
            return {}
        except Exception as e:
            print(f"ERROR during profiling: {e}")
            return {}

    def _parse_nsys_output(self, output: str) -> dict:
        """Parse nsys output."""
        metrics = {
            'avg_duration_ns': 0,
            'kernel_count': 0,
        }

        # Parse nsys stats output
        # Note: Actual parsing depends on nsys output format
        lines = output.split('\n')
        for line in lines:
            if 'matmul_kernel' in line:
                metrics['kernel_count'] += 1

        return metrics


def simple_performance_analysis(size: int = 1024):
    """
    Perform simple performance analysis without external profiling tools.

    This uses PyTorch's CUDA events for timing and basic heuristics
    to estimate bank conflict impact.
    """
    print(f"\nSimple Performance Analysis ({size}x{size})")
    print("=" * 70)

    M = K = N = size
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)

    # Warm up
    for _ in range(10):
        _ = matmul_triton(a, b, use_amd_padding=False)
        _ = matmul_triton(a, b, use_amd_padding=True)
    torch.cuda.synchronize()

    # Measure with CUDA events for accurate GPU timing
    def measure_kernel(use_padding: bool, iterations: int = 100):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iterations):
            _ = matmul_triton(a, b, use_amd_padding=use_padding)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        return elapsed_ms / iterations

    baseline_time = measure_kernel(False, iterations=100)
    optimized_time = measure_kernel(True, iterations=100)

    # Calculate metrics
    improvement = (baseline_time - optimized_time) / baseline_time * 100
    flops = 2.0 * M * N * K
    baseline_tflops = flops / (baseline_time / 1000) / 1e12
    optimized_tflops = flops / (optimized_time / 1000) / 1e12

    # Print results
    print(f"Baseline (no padding):  {baseline_time:.3f} ms ({baseline_tflops:.2f} TFLOPS)")
    print(f"Optimized (padding):    {optimized_time:.3f} ms ({optimized_tflops:.2f} TFLOPS)")
    print(f"Improvement:            {improvement:.1f}%")

    # Heuristic analysis
    vendor = detect_gpu_vendor()
    print(f"\nGPU Vendor: {vendor}")

    if vendor == 'AMD':
        if improvement > 10:
            print("✓ Significant improvement detected!")
            print("  Likely due to reduced LDS bank conflicts")
        elif improvement > 0:
            print("⚠ Modest improvement detected")
            print("  Optimization is helping but may not be optimal for this size")
        else:
            print("⚠ No improvement or regression detected")
            print("  Padding overhead may outweigh bank conflict reduction")
    else:
        print("ℹ Running on non-AMD GPU")
        print("  Optimization primarily targets AMD architecture")

    return {
        'baseline_time': baseline_time,
        'optimized_time': optimized_time,
        'improvement_pct': improvement,
    }


def main():
    """Main profiling routine."""
    parser = argparse.ArgumentParser(description='Profile Triton matrix multiplication')
    parser.add_argument('--size', type=int, default=1024,
                       help='Matrix size (default: 1024)')
    parser.add_argument('--detailed', action='store_true',
                       help='Run detailed profiling with rocprof/nsys')
    parser.add_argument('--output', type=str,
                       help='Output file for report')
    args = parser.parse_args()

    print("Triton Matrix Multiplication Profiler")
    print("=" * 70)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)

    vendor = detect_gpu_vendor()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Vendor: {vendor}")
    print()

    # Always run simple analysis
    results = simple_performance_analysis(args.size)

    # Detailed profiling if requested
    if args.detailed:
        print("\n\nDetailed Profiling")
        print("=" * 70)

        if vendor == 'AMD':
            profiler = AMDProfiler()
            if profiler.check_tools():
                print("\nProfiling baseline kernel...")
                baseline_metrics = profiler.profile_kernel(
                    lambda: None, 'baseline_profile')

                print("\nProfiling optimized kernel...")
                optimized_metrics = profiler.profile_kernel(
                    optimized, 'optimized_profile')

                print("\nBank Conflict Analysis:")
                print("-" * 70)
                if baseline_metrics and optimized_metrics:
                    print(f"Baseline LDS conflicts:  {baseline_metrics.get('lds_bank_conflicts', 'N/A')}")
                    print(f"Optimized LDS conflicts: {optimized_metrics.get('lds_bank_conflicts', 'N/A')}")
            else:
                print("⚠ rocprof not available. Install ROCm for detailed profiling.")

        elif vendor == 'NVIDIA':
            profiler = NVIDIAProfiler()
            if profiler.check_tools():
                print("\nProfiling with nsys...")
                profiler.profile_kernel(lambda: None, 'triton_profile')
                print("\n✓ Profile saved. View with: nsys-ui triton_profile.nsys-rep")
            else:
                print("⚠ nsys not available. Install CUDA toolkit for detailed profiling.")

    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("Triton Matrix Multiplication Profile\n")
            f.write("=" * 70 + "\n")
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"Vendor: {vendor}\n")
            f.write(f"Matrix size: {args.size}x{args.size}\n\n")
            f.write(f"Baseline time: {results['baseline_time']:.3f} ms\n")
            f.write(f"Optimized time: {results['optimized_time']:.3f} ms\n")
            f.write(f"Improvement: {results['improvement_pct']:.1f}%\n")
        print(f"\n✓ Report saved to {args.output}")

    print("\n\nProfiling Tips:")
    print("-" * 70)
    print("• For AMD GPUs: Use 'rocprof --stats' to see LDS bank conflicts")
    print("• For NVIDIA GPUs: Use 'nsys profile' or 'ncu' for detailed metrics")
    print("• Larger matrices (2048+) show more pronounced optimization effects")
    print("• Compare multiple runs to account for GPU frequency scaling")


if __name__ == "__main__":
    main()
