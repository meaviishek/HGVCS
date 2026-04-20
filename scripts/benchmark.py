#!/usr/bin/env python3
"""
HGVCS Performance Benchmark Script

This script benchmarks the performance of various HGVCS components.

Usage:
    python scripts/benchmark.py --all
    python scripts/benchmark.py --gesture
    python scripts/benchmark.py --voice
    python scripts/benchmark.py --network
"""

import os
import sys
import time
import argparse
import statistics
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

def print_result(name, value, unit="", threshold=None):
    """Print a benchmark result with color coding."""
    if threshold:
        if value <= threshold:
            color = Colors.GREEN
        else:
            color = Colors.FAIL
    else:
        color = Colors.CYAN
    
    print(f"  {name:.<40} {color}{value:>10.2f} {unit}{Colors.ENDC}")

def benchmark_gesture_recognition(num_iterations=100):
    """Benchmark gesture recognition performance."""
    print_header("Gesture Recognition Benchmark")
    
    try:
        import cv2
        import numpy as np
        import mediapipe as mp
        
        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            rgb = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
            hands.process(rgb)
        
        # Benchmark detection
        detection_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            rgb = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
            hands.process(rgb)
            end = time.perf_counter()
            detection_times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = statistics.mean(detection_times)
        min_time = min(detection_times)
        max_time = max(detection_times)
        std_dev = statistics.stdev(detection_times) if len(detection_times) > 1 else 0
        
        print_result("Average Detection Time", avg_time, "ms", 50)
        print_result("Minimum Detection Time", min_time, "ms")
        print_result("Maximum Detection Time", max_time, "ms")
        print_result("Standard Deviation", std_dev, "ms")
        print_result("FPS", 1000 / avg_time, "fps")
        
        hands.close()
        
        return {
            'avg_latency': avg_time,
            'min_latency': min_time,
            'max_latency': max_time,
            'std_dev': std_dev,
            'fps': 1000 / avg_time
        }
        
    except Exception as e:
        print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
        return None

def benchmark_voice_recognition(num_iterations=10):
    """Benchmark voice recognition performance."""
    print_header("Voice Recognition Benchmark")
    
    try:
        import whisper
        import numpy as np
        
        # Load model
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        
        # Create dummy audio (3 seconds at 16kHz)
        dummy_audio = np.random.randn(16000 * 3).astype(np.float32)
        
        # Warmup
        print("Warming up...")
        for _ in range(3):
            model.transcribe(dummy_audio, language='en')
        
        # Benchmark
        transcribe_times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            result = model.transcribe(dummy_audio, language='en')
            end = time.perf_counter()
            transcribe_times.append((end - start) * 1000)
            print(f"  Iteration {i+1}/{num_iterations}", end='\r')
        
        print()
        
        avg_time = statistics.mean(transcribe_times)
        min_time = min(transcribe_times)
        max_time = max(transcribe_times)
        std_dev = statistics.stdev(transcribe_times) if len(transcribe_times) > 1 else 0
        
        print_result("Average Transcription Time", avg_time, "ms", 500)
        print_result("Minimum Transcription Time", min_time, "ms")
        print_result("Maximum Transcription Time", max_time, "ms")
        print_result("Standard Deviation", std_dev, "ms")
        print_result("Real-time Factor", avg_time / 3000, "x")
        
        return {
            'avg_latency': avg_time,
            'min_latency': min_time,
            'max_latency': max_time,
            'std_dev': std_dev,
            'rtf': avg_time / 3000
        }
        
    except Exception as e:
        print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
        return None

def benchmark_model_inference(num_iterations=100):
    """Benchmark gesture model inference."""
    print_header("Model Inference Benchmark")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        # Create a simple model for benchmarking
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(30, 67)),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(25, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Create dummy input
        dummy_input = np.random.randn(1, 30, 67).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            model.predict(dummy_input, verbose=0)
        
        # Benchmark
        inference_times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            model.predict(dummy_input, verbose=0)
            end = time.perf_counter()
            inference_times.append((end - start) * 1000)
            print(f"  Iteration {i+1}/{num_iterations}", end='\r')
        
        print()
        
        avg_time = statistics.mean(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        std_dev = statistics.stdev(inference_times) if len(inference_times) > 1 else 0
        
        print_result("Average Inference Time", avg_time, "ms", 50)
        print_result("Minimum Inference Time", min_time, "ms")
        print_result("Maximum Inference Time", max_time, "ms")
        print_result("Standard Deviation", std_dev, "ms")
        print_result("Inferences/Second", 1000 / avg_time, "inf/s")
        
        return {
            'avg_latency': avg_time,
            'min_latency': min_time,
            'max_latency': max_time,
            'std_dev': std_dev,
            'ips': 1000 / avg_time
        }
        
    except Exception as e:
        print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
        return None

def benchmark_file_transfer(file_size_mb=100):
    """Benchmark file transfer speed."""
    print_header("File Transfer Benchmark")
    
    try:
        import tempfile
        import hashlib
        
        # Create temporary file
        file_size = file_size_mb * 1024 * 1024
        chunk_size = 65536  # 64KB
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
            # Write random data
            bytes_written = 0
            while bytes_written < file_size:
                chunk = os.urandom(min(chunk_size, file_size - bytes_written))
                f.write(chunk)
                bytes_written += len(chunk)
            
            temp_path = f.name
        
        # Benchmark read speed
        start = time.perf_counter()
        with open(temp_path, 'rb') as f:
            while f.read(chunk_size):
                pass
        read_time = time.perf_counter() - start
        
        # Benchmark checksum calculation
        start = time.perf_counter()
        sha256 = hashlib.sha256()
        with open(temp_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        checksum_time = time.perf_counter() - start
        
        # Cleanup
        os.unlink(temp_path)
        
        read_speed = file_size_mb / read_time
        
        print_result("File Read Speed", read_speed, "MB/s")
        print_result("Read Time", read_time * 1000, "ms")
        print_result("Checksum Time", checksum_time * 1000, "ms")
        
        return {
            'read_speed_mbps': read_speed,
            'read_time_ms': read_time * 1000,
            'checksum_time_ms': checksum_time * 1000
        }
        
    except Exception as e:
        print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
        return None

def benchmark_memory_usage():
    """Benchmark memory usage."""
    print_header("Memory Usage Benchmark")
    
    try:
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Baseline
        gc.collect()
        baseline = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load gesture components
        import cv2
        import mediapipe as mp
        import numpy as np
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        
        gesture_memory = process.memory_info().rss / 1024 / 1024
        
        # Load voice components
        import whisper
        model = whisper.load_model("base")
        
        voice_memory = process.memory_info().rss / 1024 / 1024
        
        # Cleanup
        hands.close()
        del model
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        print_result("Baseline Memory", baseline, "MB")
        print_result("After Gesture Load", gesture_memory, "MB")
        print_result("After Voice Load", voice_memory, "MB")
        print_result("Gesture Overhead", gesture_memory - baseline, "MB")
        print_result("Voice Overhead", voice_memory - gesture_memory, "MB")
        print_result("Total Peak", voice_memory, "MB", 500)
        print_result("After Cleanup", final_memory, "MB")
        
        return {
            'baseline_mb': baseline,
            'gesture_mb': gesture_memory - baseline,
            'voice_mb': voice_memory - gesture_memory,
            'peak_mb': voice_memory,
            'final_mb': final_memory
        }
        
    except Exception as e:
        print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
        return None

def generate_report(results):
    """Generate benchmark report."""
    print_header("Benchmark Summary")
    
    report = []
    report.append("HGVCS Performance Benchmark Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("")
    
    for test_name, result in results.items():
        if result:
            report.append(f"\n{test_name}:")
            for key, value in result.items():
                report.append(f"  {key}: {value:.2f}")
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save to file
    report_path = f"logs/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("logs", exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n{Colors.GREEN}Report saved to: {report_path}{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description='HGVCS Performance Benchmark')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--gesture', action='store_true', help='Benchmark gesture recognition')
    parser.add_argument('--voice', action='store_true', help='Benchmark voice recognition')
    parser.add_argument('--model', action='store_true', help='Benchmark model inference')
    parser.add_argument('--network', action='store_true', help='Benchmark file transfer')
    parser.add_argument('--memory', action='store_true', help='Benchmark memory usage')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    
    args = parser.parse_args()
    
    # If no specific benchmark selected, run all
    if not any([args.gesture, args.voice, args.model, args.network, args.memory]):
        args.all = True
    
    results = {}
    
    print_header("HGVCS Performance Benchmark")
    print(f"Iterations: {args.iterations}")
    print(f"Platform: {os.name}")
    print(f"Python: {sys.version}")
    
    if args.all or args.gesture:
        results['Gesture Recognition'] = benchmark_gesture_recognition(args.iterations)
    
    if args.all or args.model:
        results['Model Inference'] = benchmark_model_inference(args.iterations)
    
    if args.all or args.voice:
        results['Voice Recognition'] = benchmark_voice_recognition(min(args.iterations // 10, 10))
    
    if args.all or args.network:
        results['File Transfer'] = benchmark_file_transfer()
    
    if args.all or args.memory:
        results['Memory Usage'] = benchmark_memory_usage()
    
    generate_report(results)

if __name__ == "__main__":
    main()
