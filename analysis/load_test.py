import requests
import time
import concurrent.futures
import statistics

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8001/score-sinhala-ml"
API_KEY = "akura-research-secret-2026"
HEADERS = {"X-API-KEY": API_KEY}
CONCURRENT_USERS = 20  # Simulate 20 users hitting the API at once
TOTAL_REQUESTS = 100    # Total requests for the test

ESSAY_SAMPLE = "මම පාසල් යනවා. මගේ පාසල ලස්සනයි. ගුරුවරුන් අපිට හොඳින් උගන්වනවා. අපි පාසල් යන්නේ සතුටින්."

def send_request(request_id):
    payload = {
        "text": ESSAY_SAMPLE,
        "grade": 6,
        "dyslexic_flag": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        elapsed = time.time() - start_time
        
        return {
            "id": request_id,
            "status": response.status_code,
            "elapsed": elapsed
        }
    except Exception as e:
        return {
            "id": request_id,
            "status": "ERROR",
            "error": str(e)
        }

def run_load_test():
    print(f"Starting Load Test...")
    print(f"Concurrent Users: {CONCURRENT_USERS}")
    print(f"Total Requests: {TOTAL_REQUESTS}")
    print("-" * 30)

    results = []
    start_test = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
        future_to_req = {executor.submit(send_request, i): i for i in range(TOTAL_REQUESTS)}
        for future in concurrent.futures.as_completed(future_to_req):
            res = future.result()
            results.append(res)
            if len(results) % 10 == 0:
                print(f"Processed {len(results)}/{TOTAL_REQUESTS} requests...")

    total_time = time.time() - start_test
    
    # Process Results
    successes = [r for r in results if r['status'] == 200]
    failures = [r for r in results if r['status'] != 200]
    latencies = [r['elapsed'] for r in successes]

    print("\n" + "="*50)
    print("LOAD TEST SUMMARY")
    print("="*50)
    print(f"Total Requests: {TOTAL_REQUESTS}")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")
    print(f"Total Test Time: {total_time:.2f}s")
    print(f"Throughput: {len(successes)/total_time:.2f} req/s")
    
    if latencies:
        print(f"Average Latency: {statistics.mean(latencies):.3f}s")
        print(f"P95 Latency: {sorted(latencies)[int(len(latencies) * 0.95)]:.3f}s")
        print(f"Max Latency: {max(latencies):.3f}s")
    
    # Save Report
    with open("analysis/LOAD_TEST_REPORT.md", "w") as f:
        f.write("# Load Test Report\n\n")
        f.write(f"- **Concurrent Users:** {CONCURRENT_USERS}\n")
        f.write(f"- **Total Requests:** {TOTAL_REQUESTS}\n")
        f.write(f"- **Success Rate:** {(len(successes)/TOTAL_REQUESTS)*100:.1f}%\n")
        f.write(f"- **Average Latency:** {statistics.mean(latencies):.3f}s\n")
        f.write(f"- **P95 Latency:** {sorted(latencies)[int(len(latencies) * 0.95)]:.3f}s\n")
        f.write(f"- **Throughput:** {len(successes)/total_time:.2f} req/s\n")

    print("\nReport generated: analysis/LOAD_TEST_REPORT.md")

if __name__ == "__main__":
    run_load_test()
