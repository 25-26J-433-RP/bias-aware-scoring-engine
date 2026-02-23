
import requests
import time
import statistics
import json

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8001/score-sinhala-ml" 

def generate_sentence(words_count):
    # Basic Sinhala words to simulate length
    words = ["මම", "පාසල්", "යනවා", "අම්මා", "බත්", "උයනවා", "ලස්සන", "මල්", "පන්සල", "අහස"]
    sentence = " ".join([words[i % len(words)] for i in range(words_count)])
    return sentence

def benchmark_latency():
    # Test lengths: 100, 500, 1000 words
    test_lengths = [100, 500, 1000]
    runs_per_length = 6 # 1 warmup + 5 measured
    
    print("="*50)
    print("PERFORMANCE BENCHMARK: WARM LATENCY VS ESSAY LENGTH")
    print("="*50)
    
    final_results = {}

    for length in test_lengths:
        print(f"\nTesting {length} words...")
        essay_text = generate_sentence(length)
        latencies = []
        
        for i in range(runs_per_length):
            payload = {
                "text": essay_text, "grade": 7, "dyslexic_flag": False
            }
            try:
                start_time = time.time()
                headers = {"X-API-KEY": "akura-research-secret-2026"}
                response = requests.post(API_URL, json=payload, headers=headers)
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    if i > 0: # Skip first run (Warmup/Cold start)
                        latencies.append(elapsed)
                        print(f"  Run {i}: {elapsed:.3f}s")
                    else:
                        print(f"  Warmup: {elapsed:.3f}s")
                else:
                    print(f"  Run {i+1}: Failed ({response.status_code})")
            except Exception as e:
                print(f"  Run {i+1}: Error {str(e)}")
        
        if latencies:
            final_results[length] = {
                "avg": statistics.mean(latencies),
                "p95": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 0 else 0,
                "max": max(latencies)
            }

    print("\n" + "="*50)
    print("FINAL PERFORMANCE SUMMARY (WARM LATENCY)")
    print("="*50)
    print(f"{'Words':<10} | {'Avg (s)':<10} | {'P95 (s)':<10} | {'Status'}")
    print("-" * 50)
    
    for length, metrics in final_results.items():
        # Warm latency target is < 2.0s
        status = "PASS" if metrics['avg'] < 2.0 else "FAIL"
        print(f"{length:<10} | {metrics['avg']:<10.3f} | {metrics['p95']:<10.3f} | {status}")
    print("="*50)

if __name__ == "__main__":
    benchmark_latency()
