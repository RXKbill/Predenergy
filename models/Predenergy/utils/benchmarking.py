"""
Performance Benchmarking and Model Comparison Utilities
This module provides comprehensive benchmarking tools for Predenergy models.
"""

import time
import psutil
import paddle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import gc
from contextlib import contextmanager
import threading


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    dataset_name: str
    inference_time: float
    memory_usage: float
    gpu_memory_usage: float
    throughput: float
    batch_size: int
    sequence_length: int
    prediction_length: int
    metrics: Dict[str, float]
    system_info: Dict[str, Any]
    timestamp: str


class SystemMonitor:
    """Monitor system resources during benchmarking."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        
    def start_monitoring(self):
        """Start system monitoring in a separate thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring and return statistics."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        
        return {
            'cpu_usage_avg': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'cpu_usage_max': np.max(self.cpu_usage) if self.cpu_usage else 0,
            'memory_usage_avg': np.mean(self.memory_usage) if self.memory_usage else 0,
            'memory_usage_max': np.max(self.memory_usage) if self.memory_usage else 0,
            'gpu_usage_avg': np.mean(self.gpu_usage) if self.gpu_usage else 0,
            'gpu_usage_max': np.max(self.gpu_usage) if self.gpu_usage else 0,
            'gpu_memory_avg': np.mean(self.gpu_memory) if self.gpu_memory else 0,
            'gpu_memory_max': np.max(self.gpu_memory) if self.gpu_memory else 0,
        }
    
    def _monitor_loop(self):
        """Monitor system resources continuously."""
        while self.monitoring:
            try:
                # CPU and Memory
                self.cpu_usage.append(psutil.cpu_percent())
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.percent)
                
                # GPU monitoring (if available)
                if paddle.device.cuda.device_count() > 0:
                    try:
                        # Note: This is a simplified GPU monitoring
                        # In practice, you might want to use nvidia-ml-py
                        gpu_memory = paddle.device.cuda.memory_usage() / paddle.device.cuda.max_memory_allocated()
                        self.gpu_memory.append(gpu_memory * 100)
                        self.gpu_usage.append(50.0)  # Placeholder
                    except:
                        pass
                
                time.sleep(self.interval)
            except:
                break


@contextmanager
def memory_profiler():
    """Context manager for memory profiling."""
    gc.collect()
    
    if paddle.device.cuda.device_count() > 0:
        paddle.device.cuda.empty_cache()
        initial_gpu_memory = paddle.device.cuda.memory_usage()
    else:
        initial_gpu_memory = 0
    
    initial_cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    gc.collect()
    
    final_cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    if paddle.device.cuda.device_count() > 0:
        paddle.device.cuda.empty_cache()
        final_gpu_memory = paddle.device.cuda.memory_usage()
    else:
        final_gpu_memory = 0
    
    return {
        'cpu_memory_delta': final_cpu_memory - initial_cpu_memory,
        'gpu_memory_delta': final_gpu_memory - initial_gpu_memory,
        'peak_cpu_memory': final_cpu_memory,
        'peak_gpu_memory': final_gpu_memory
    }


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for Predenergy models."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def benchmark_model(
        self,
        model: Any,
        test_data: np.ndarray,
        model_name: str = "Predenergy",
        dataset_name: str = "test_dataset",
        num_runs: int = 5,
        warmup_runs: int = 2
    ) -> BenchmarkResult:
        """
        Benchmark a model's performance comprehensively.
        
        Args:
            model: The model to benchmark
            test_data: Test data for inference
            model_name: Name of the model
            dataset_name: Name of the dataset
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
        
        Returns:
            BenchmarkResult object
        """
        print(f"Benchmarking {model_name} on {dataset_name}...")
        
        # System info
        system_info = self._get_system_info()
        
        # Data info
        batch_size, sequence_length = test_data.shape[:2]
        prediction_length = getattr(model.config, 'horizon', 24)
        
        # Convert to tensor
        if not isinstance(test_data, paddle.Tensor):
            test_data = paddle.to_tensor(test_data, dtype='float32')
        
        # Warmup runs
        print(f"Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            with paddle.no_grad():
                _ = model(test_data)
        
        # Actual benchmark runs
        print(f"Running {num_runs} benchmark iterations...")
        
        inference_times = []
        memory_usage_stats = []
        
        monitor = SystemMonitor()
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")
            
            # Start monitoring
            monitor.start_monitoring()
            
            # Memory profiling
            with memory_profiler() as memory_stats:
                start_time = time.perf_counter()
                
                with paddle.no_grad():
                    predictions = model(test_data)
                
                # Ensure computation is complete
                if paddle.device.cuda.device_count() > 0:
                    paddle.device.cuda.synchronize()
                
                end_time = time.perf_counter()
            
            # Stop monitoring
            system_stats = monitor.stop_monitoring()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            memory_usage_stats.append({
                **memory_stats,
                **system_stats
            })
            
            # Clean up
            del predictions
            gc.collect()
            if paddle.device.cuda.device_count() > 0:
                paddle.device.cuda.empty_cache()
        
        # Calculate statistics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        throughput = batch_size / avg_inference_time  # samples per second
        
        avg_memory_usage = np.mean([stats['peak_cpu_memory'] for stats in memory_usage_stats])
        avg_gpu_memory_usage = np.mean([stats['peak_gpu_memory'] for stats in memory_usage_stats])
        
        # Basic performance metrics
        metrics = {
            'avg_inference_time': avg_inference_time,
            'std_inference_time': std_inference_time,
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'throughput': throughput,
            'latency_per_sample': avg_inference_time / batch_size,
        }
        
        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            dataset_name=dataset_name,
            inference_time=avg_inference_time,
            memory_usage=avg_memory_usage,
            gpu_memory_usage=avg_gpu_memory_usage,
            throughput=throughput,
            batch_size=batch_size,
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            metrics=metrics,
            system_info=system_info,
            timestamp=pd.Timestamp.now().isoformat()
        )
        
        self.results.append(result)
        print(f"✓ Benchmark completed. Throughput: {throughput:.2f} samples/sec")
        
        return result
    
    def benchmark_batch_sizes(
        self,
        model: Any,
        test_data: np.ndarray,
        batch_sizes: List[int],
        model_name: str = "Predenergy"
    ) -> Dict[int, BenchmarkResult]:
        """Benchmark model with different batch sizes."""
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nBenchmarking batch size: {batch_size}")
            
            # Create batch
            if len(test_data) >= batch_size:
                batch_data = test_data[:batch_size]
            else:
                # Repeat data to reach batch size
                repeats = (batch_size + len(test_data) - 1) // len(test_data)
                repeated_data = np.tile(test_data, (repeats, 1, 1))
                batch_data = repeated_data[:batch_size]
            
            try:
                result = self.benchmark_model(
                    model, batch_data, 
                    model_name=f"{model_name}_bs{batch_size}",
                    dataset_name=f"batch_size_{batch_size}"
                )
                results[batch_size] = result
                
            except Exception as e:
                print(f"  ⚠️ Failed to benchmark batch size {batch_size}: {e}")
                continue
        
        return results
    
    def benchmark_sequence_lengths(
        self,
        model: Any,
        test_data: np.ndarray,
        sequence_lengths: List[int],
        model_name: str = "Predenergy"
    ) -> Dict[int, BenchmarkResult]:
        """Benchmark model with different sequence lengths."""
        
        results = {}
        
        for seq_len in sequence_lengths:
            print(f"\nBenchmarking sequence length: {seq_len}")
            
            if test_data.shape[1] >= seq_len:
                seq_data = test_data[:, :seq_len]
            else:
                print(f"  ⚠️ Skipping sequence length {seq_len} (exceeds available data)")
                continue
            
            try:
                result = self.benchmark_model(
                    model, seq_data,
                    model_name=f"{model_name}_seq{seq_len}",
                    dataset_name=f"sequence_length_{seq_len}"
                )
                results[seq_len] = result
                
            except Exception as e:
                print(f"  ⚠️ Failed to benchmark sequence length {seq_len}: {e}")
                continue
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, Any],
        test_data: np.ndarray,
        dataset_name: str = "comparison_dataset"
    ) -> pd.DataFrame:
        """Compare multiple models on the same dataset."""
        
        comparison_results = []
        
        for model_name, model in models.items():
            print(f"\nBenchmarking {model_name}...")
            
            try:
                result = self.benchmark_model(model, test_data, model_name, dataset_name)
                
                comparison_results.append({
                    'model_name': model_name,
                    'inference_time': result.inference_time,
                    'memory_usage': result.memory_usage,
                    'gpu_memory_usage': result.gpu_memory_usage,
                    'throughput': result.throughput,
                    'latency_per_sample': result.metrics['latency_per_sample'],
                })
                
            except Exception as e:
                print(f"  ⚠️ Failed to benchmark {model_name}: {e}")
                continue
        
        df = pd.DataFrame(comparison_results)
        
        if not df.empty:
            # Add relative performance metrics
            fastest_time = df['inference_time'].min()
            highest_throughput = df['throughput'].max()
            lowest_memory = df['memory_usage'].min()
            
            df['speed_relative'] = fastest_time / df['inference_time']
            df['throughput_relative'] = df['throughput'] / highest_throughput
            df['memory_efficiency'] = lowest_memory / df['memory_usage']
            
            # Overall score (simple weighted average)
            df['overall_score'] = (
                0.4 * df['speed_relative'] +
                0.3 * df['throughput_relative'] +
                0.3 * df['memory_efficiency']
            )
            
            # Sort by overall score
            df = df.sort_values('overall_score', ascending=False).reset_index(drop=True)
        
        return df
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        
        system_info = {
            'platform': psutil.sys.platform,
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'python_version': psutil.sys.version,
            'paddle_version': paddle.__version__,
        }
        
        # GPU information
        if paddle.device.cuda.device_count() > 0:
            system_info.update({
                'gpu_count': paddle.device.cuda.device_count(),
                'gpu_memory_total': paddle.device.cuda.get_device_properties().total_memory / (1024**3),  # GB
                'cuda_version': paddle.version.cuda() if hasattr(paddle.version, 'cuda') else 'Unknown',
            })
        else:
            system_info.update({
                'gpu_count': 0,
                'gpu_memory_total': 0,
                'cuda_version': 'N/A',
            })
        
        return system_info
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save benchmark results to file."""
        
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = {
                'model_name': result.model_name,
                'dataset_name': result.dataset_name,
                'inference_time': result.inference_time,
                'memory_usage': result.memory_usage,
                'gpu_memory_usage': result.gpu_memory_usage,
                'throughput': result.throughput,
                'batch_size': result.batch_size,
                'sequence_length': result.sequence_length,
                'prediction_length': result.prediction_length,
                'metrics': result.metrics,
                'system_info': result.system_info,
                'timestamp': result.timestamp,
            }
            serializable_results.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📁 Results saved to: {filepath}")
        return str(filepath)
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        
        if not self.results:
            return "No benchmark results available."
        
        report = []
        report.append("# Predenergy Performance Benchmark Report")
        report.append(f"Generated on: {pd.Timestamp.now()}")
        report.append("")
        
        # Summary statistics
        report.append("## Summary")
        report.append(f"Total benchmarks run: {len(self.results)}")
        
        throughputs = [r.throughput for r in self.results]
        inference_times = [r.inference_time for r in self.results]
        
        report.append(f"Average throughput: {np.mean(throughputs):.2f} samples/sec")
        report.append(f"Best throughput: {np.max(throughputs):.2f} samples/sec")
        report.append(f"Average inference time: {np.mean(inference_times):.4f} seconds")
        report.append(f"Best inference time: {np.min(inference_times):.4f} seconds")
        report.append("")
        
        # Individual results
        report.append("## Individual Results")
        for result in self.results:
            report.append(f"### {result.model_name} - {result.dataset_name}")
            report.append(f"- Inference time: {result.inference_time:.4f} seconds")
            report.append(f"- Throughput: {result.throughput:.2f} samples/sec")
            report.append(f"- Memory usage: {result.memory_usage:.2f} MB")
            report.append(f"- GPU memory usage: {result.gpu_memory_usage:.2f} MB")
            report.append(f"- Batch size: {result.batch_size}")
            report.append(f"- Sequence length: {result.sequence_length}")
            report.append("")
        
        # System information (from first result)
        if self.results:
            system_info = self.results[0].system_info
            report.append("## System Information")
            for key, value in system_info.items():
                report.append(f"- {key}: {value}")
            report.append("")
        
        return "\n".join(report)


def quick_benchmark(model: Any, test_data: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """Quick benchmark function for simple performance testing."""
    
    print(f"Quick benchmark for {model_name}...")
    
    # Convert to tensor if needed
    if not isinstance(test_data, paddle.Tensor):
        test_data = paddle.to_tensor(test_data, dtype='float32')
    
    # Warmup
    with paddle.no_grad():
        _ = model(test_data)
    
    # Benchmark
    times = []
    for _ in range(5):
        start_time = time.perf_counter()
        
        with paddle.no_grad():
            _ = model(test_data)
        
        if paddle.device.cuda.device_count() > 0:
            paddle.device.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    throughput = test_data.shape[0] / avg_time
    
    results = {
        'avg_inference_time': avg_time,
        'throughput': throughput,
        'latency_per_sample': avg_time / test_data.shape[0],
    }
    
    print(f"✓ Throughput: {throughput:.2f} samples/sec")
    print(f"✓ Latency: {results['latency_per_sample']*1000:.2f} ms/sample")
    
    return results