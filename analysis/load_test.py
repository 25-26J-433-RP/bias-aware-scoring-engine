import argparse
import concurrent.futures
import statistics
import threading
import time
from collections import Counter

import requests

# --- DEFAULT CONFIGURATION ---
API_URL = "http://127.0.0.1:8001/score-sinhala-ml"
API_KEY = "akura-research-secret-2026"
HEADERS = {"X-API-KEY": API_KEY}

DEFAULT_CONCURRENT_USERS = 20
DEFAULT_TOTAL_REQUESTS = 100
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 0
DEFAULT_BACKOFF_SECONDS = 0.5

# Sinhala text with explicit Unicode escapes to avoid console/file encoding issues.
ESSAY_SAMPLE = (
    "\u0db8\u0dd9\u0dba \u0db4\u0dbb\u0dd3\u0d9a\u0dca\u0dc2\u0dab "
    "\u0dbb\u0da0\u0db1\u0dba\u0d9a\u0dd2. "
    "\u0db8\u0dd9\u0dba \u0dad\u0dc0\u0dad\u0dca "
    "\u0dc0\u0dcf\u0d9a\u0dca\u0dba\u0dba\u0d9a\u0dd2."
)


class RatePacer:
    def __init__(self, target_rps: float) -> None:
        self.interval = 1.0 / target_rps if target_rps > 0 else 0.0
        self.next_slot = time.perf_counter()
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.perf_counter()
            if now < self.next_slot:
                time.sleep(self.next_slot - now)
            self.next_slot = max(self.next_slot, now) + self.interval


SESSION = requests.Session()


def send_request(
    request_id: int,
    timeout: float,
    max_retries: int,
    backoff_seconds: float,
) -> dict:
    payload = {
        "text": ESSAY_SAMPLE,
        "grade": 6,
        "dyslexic_flag": False,
    }

    attempts = 0
    started = time.perf_counter()

    while True:
        attempts += 1
        try:
            response = SESSION.post(API_URL, json=payload, headers=HEADERS, timeout=timeout)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            return {
                "id": request_id,
                "status": "ERROR",
                "elapsed": elapsed,
                "attempts": attempts,
                "error": str(exc),
            }

        if response.status_code != 429 or attempts > (max_retries + 1):
            elapsed = time.perf_counter() - started
            return {
                "id": request_id,
                "status": response.status_code,
                "elapsed": elapsed,
                "attempts": attempts,
                "body": response.text[:200],
            }

        retry_after = response.headers.get("Retry-After")
        wait_seconds = backoff_seconds * (2 ** (attempts - 1))
        if retry_after:
            try:
                wait_seconds = max(wait_seconds, float(retry_after))
            except ValueError:
                pass
        time.sleep(wait_seconds)


def run_load_test(
    concurrent_users: int,
    total_requests: int,
    target_rps: float,
    timeout: float,
    max_retries: int,
    backoff_seconds: float,
) -> None:
    print("Starting Load Test...")
    print(f"Concurrent Users: {concurrent_users}")
    print(f"Total Requests: {total_requests}")
    if target_rps > 0:
        print(f"Target Rate: {target_rps:.2f} req/s ({target_rps * 60:.1f}/min)")
    else:
        print("Target Rate: burst (unpaced)")
    print("-" * 30)

    results = []
    start_test = time.perf_counter()
    pacer = RatePacer(target_rps)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = []
        for request_id in range(total_requests):
            pacer.wait()
            futures.append(
                executor.submit(
                    send_request,
                    request_id,
                    timeout,
                    max_retries,
                    backoff_seconds,
                )
            )

        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if idx % 10 == 0 or idx == total_requests:
                print(f"Processed {idx}/{total_requests} requests...")

    total_time = time.perf_counter() - start_test

    successes = [r for r in results if r["status"] == 200]
    failures = [r for r in results if r["status"] != 200]
    latencies = [r["elapsed"] for r in successes]
    attempts_total = sum(r.get("attempts", 1) for r in results)
    status_counts = Counter(r["status"] for r in results)

    print("\n" + "=" * 50)
    print("LOAD TEST SUMMARY")
    print("=" * 50)
    print(f"Total Requests: {total_requests}")
    print(f"HTTP Attempts: {attempts_total}")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")
    print(f"Total Test Time: {total_time:.2f}s")
    print(f"Throughput: {len(successes) / total_time:.2f} req/s")
    print(f"Status Breakdown: {dict(status_counts)}")

    if latencies:
        latencies_sorted = sorted(latencies)
        p95_index = min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.95))
        print(f"Average Latency: {statistics.mean(latencies):.3f}s")
        print(f"P95 Latency: {latencies_sorted[p95_index]:.3f}s")
        print(f"Max Latency: {max(latencies):.3f}s")
    else:
        print("Average Latency: N/A (no successful requests)")
        print("P95 Latency: N/A (no successful requests)")
        print("Max Latency: N/A (no successful requests)")

    if failures:
        first_failure = failures[0]
        if first_failure.get("status") == "ERROR":
            print(f"First error: {first_failure.get('error')}")
        else:
            print(
                f"First non-200 response: {first_failure.get('status')} "
                f"- {first_failure.get('body', '')}"
            )

    report_path = "analysis/LOAD_TEST_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Load Test Report\n\n")
        handle.write(f"- **Concurrent Users:** {concurrent_users}\n")
        handle.write(f"- **Total Requests:** {total_requests}\n")
        handle.write(f"- **Target Rate (req/s):** {target_rps if target_rps > 0 else 'burst'}\n")
        handle.write(f"- **HTTP Attempts:** {attempts_total}\n")
        handle.write(f"- **Success Rate:** {(len(successes) / total_requests) * 100:.1f}%\n")
        if latencies:
            latencies_sorted = sorted(latencies)
            p95_index = min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.95))
            handle.write(f"- **Average Latency:** {statistics.mean(latencies):.3f}s\n")
            handle.write(f"- **P95 Latency:** {latencies_sorted[p95_index]:.3f}s\n")
        else:
            handle.write("- **Average Latency:** N/A (no successful requests)\n")
            handle.write("- **P95 Latency:** N/A (no successful requests)\n")
        handle.write(f"- **Throughput:** {len(successes) / total_time:.2f} req/s\n")
        handle.write(f"- **Status Breakdown:** {dict(status_counts)}\n")

    print(f"\nReport generated: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rate-aware API load test for /score-sinhala-ml")
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT_USERS, help="Max worker threads")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL_REQUESTS, help="Total requests")
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="Target requests/second. Use 0 for unpaced burst mode.",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-request timeout (seconds)")
    parser.add_argument("--retries", type=int, default=DEFAULT_MAX_RETRIES, help="Retries on HTTP 429")
    parser.add_argument(
        "--backoff",
        type=float,
        default=DEFAULT_BACKOFF_SECONDS,
        help="Initial retry backoff in seconds (exponential)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_load_test(
        concurrent_users=max(1, args.concurrent),
        total_requests=max(1, args.total),
        target_rps=max(0.0, args.rate),
        timeout=max(1.0, args.timeout),
        max_retries=max(0, args.retries),
        backoff_seconds=max(0.0, args.backoff),
    )
