
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
    runs_per_length = 5
    
    print("="*50)
    print("PERFORMANCE BENCHMARK: LATENCY VS ESSAY LENGTH")
    print("="*50)
    
    final_results = {}

    for length in test_lengths:
        print(f"\nTesting {length} words...")
        essay_text = generate_sentence(length)
        latencies = []
        
        for i in range(runs_per_length):
            payload = {
                "text": essay_text,
                "grade": 7,
                "dyslexic_flag": False
            }
            
            try:
                start_time = time.time()
                response = requests.post(API_URL, json=payload)
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    latencies.append(elapsed)
                    print(f"  Run {i+1}: {elapsed:.3f}s")
                else:
                    print(f"  Run {i+1}: Failed (500 Error)")
            except Exception as e:
                print(f"  Run {i+1}: Error {str(e)}")
        
        if latencies:
            final_results[length] = {
                "avg": statistics.mean(latencies),
                "p95": sorted(latencies)[int(len(latencies) * 0.95)],
                "max": max(latencies)
            }

    print("\n" + "="*50)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"{'Words':<10} | {'Avg (s)':<10} | {'P95 (s)':<10} | {'Status'}")
    print("-" * 50)
    
    for length, metrics in final_results.items():
        status = "PASS" if metrics['p95'] < 2.0 else "FAIL"
        print(f"{length:<10} | {metrics['avg']:<10.3f} | {metrics['p95']:<10.3f} | {status}")
    print("="*50)

if __name__ == "__main__":
    benchmark_latency()
