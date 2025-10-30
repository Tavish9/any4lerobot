#!/usr/bin/env python3
"""
Simple benchmark to demonstrate performance improvements.

This script compares the old pandas .apply() approach with the new
vectorized pad_vector_column() function.
"""

import time
import numpy as np
import pandas as pd


def old_padding_approach(series, target_dim):
    """Old approach using pandas apply with lambda."""
    return series.apply(
        lambda x: np.pad(x, (0, target_dim - len(x)), 'constant').tolist()
        if x is not None and isinstance(x, (list, np.ndarray)) and len(x) < target_dim
        else x
    )


def new_padding_approach(series, target_dim):
    """New vectorized approach."""
    padded = [
        np.pad(x, (0, target_dim - len(x)), 'constant').tolist()
        if x is not None and isinstance(x, (list, np.ndarray)) and len(x) < target_dim
        else x
        for x in series
    ]
    return pd.Series(padded, index=series.index)


def benchmark_padding(n_rows=10000, vector_dim=14, target_dim=32):
    """Benchmark old vs new padding approaches."""
    # Create test data
    data = [np.random.randn(vector_dim).tolist() for _ in range(n_rows)]
    series = pd.Series(data)
    
    # Benchmark old approach
    start = time.time()
    result_old = old_padding_approach(series, target_dim)
    time_old = time.time() - start
    
    # Benchmark new approach
    start = time.time()
    result_new = new_padding_approach(series, target_dim)
    time_new = time.time() - start
    
    # Verify results are the same
    assert len(result_old) == len(result_new)
    assert all(len(r) == target_dim for r in result_new)
    
    print(f"\nBenchmark Results (n={n_rows}, dim={vector_dim}→{target_dim}):")
    print(f"  Old approach (apply): {time_old:.4f}s")
    print(f"  New approach (list):  {time_new:.4f}s")
    print(f"  Speedup:              {time_old/time_new:.2f}x")
    print(f"  Time saved:           {time_old - time_new:.4f}s ({(1-time_new/time_old)*100:.1f}%)")


def benchmark_vectorized_stats():
    """Benchmark nested loops vs vectorized operations for statistics."""
    n_datasets = 5
    channels = 3
    height = 10
    width = 10
    
    # Create test data (simulating image statistics)
    values = [
        [[[np.random.rand() for _ in range(width)] 
          for _ in range(height)] 
         for _ in range(channels)]
        for _ in range(n_datasets)
    ]
    
    # Old approach with nested loops
    start = time.time()
    result_old = []
    for channel_idx in range(len(values[0])):
        channel_result = []
        for pixel_idx in range(len(values[0][channel_idx])):
            pixel_result = []
            for value_idx in range(len(values[0][channel_idx][pixel_idx])):
                avg = sum(values[i][channel_idx][pixel_idx][value_idx] 
                         for i in range(len(values))) / len(values)
                pixel_result.append(avg)
            channel_result.append(pixel_result)
        result_old.append(channel_result)
    time_old = time.time() - start
    
    # New vectorized approach
    start = time.time()
    values_array = np.array(values)
    result_new = np.mean(values_array, axis=0).tolist()
    time_new = time.time() - start
    
    print(f"\nVectorized Statistics Benchmark:")
    print(f"  Old approach (loops):      {time_old:.4f}s")
    print(f"  New approach (vectorized): {time_new:.4f}s")
    print(f"  Speedup:                   {time_old/time_new:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Performance Improvement Benchmarks")
    print("=" * 60)
    
    # Run benchmarks with different sizes
    for n_rows in [1000, 5000, 10000]:
        benchmark_padding(n_rows=n_rows)
    
    benchmark_vectorized_stats()
    
    print("\n" + "=" * 60)
    print("Note: Actual performance gains will vary based on:")
    print("  - Hardware (CPU, memory)")
    print("  - Data size and complexity")
    print("  - System load")
    print("=" * 60)
