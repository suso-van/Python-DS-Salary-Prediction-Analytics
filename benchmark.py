# benchmark.py
import time
import requests
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8000/predict"
TOTAL_REQUESTS = 200
CONCURRENT_WORKERS = 10
HEADERS = {"X-API-KEY": "ghost_architect_secure_token_2026"} # Added Secure Tokens

PAYLOAD = {
    "work_year": 2026,
    "experience_level": "SE",
    "employment_type": "FT",
    "job_title": "Machine Learning Engineer",
    "employee_residence": "US",
    "remote_ratio": 100,
    "company_location": "US",
    "company_size": "M"
}

def send_prediction_request(request_id):
    """Fires a post request containing valid security clearance headers."""
    start_time = time.perf_counter()
    try:
        # Injected headers mapping parameter
        response = requests.post(URL, json=PAYLOAD, headers=HEADERS, timeout=5)
        latency = time.perf_counter() - start_time
        return response.status_code, latency
    except requests.exceptions.RequestException:
        return 500, time.perf_counter() - start_time

def run_stress_test():
    print(f"[Benchmarking] Target Boundary: {URL}")
    print(f"[Benchmarking] Injected Volume: {TOTAL_REQUESTS} total requests across {CONCURRENT_WORKERS} concurrent threads...")
    print("---------------------------------------------------------------------------")

    try:
        requests.post(URL, json=PAYLOAD, headers=HEADERS, timeout=5)
    except Exception:
        print("[Error] Server offline! Make sure orchestration bounds are aligned.")
        return

    start_bench = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        results = list(executor.map(send_prediction_request, range(TOTAL_REQUESTS)))
        
    total_time = time.perf_counter() - start_bench

    status_codes = [r[0] for r in results]
    latencies = [r[1] for r in results]
    
    success_count = status_codes.count(200)
    failed_count = len(status_codes) - success_count
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    throughput = TOTAL_REQUESTS / total_time

    print("\n======================= STRESS TEST METRICS REPORT =======================")
    print(f" Performance Throughput : {throughput:.2f} requests / second")
    print(f" Average Latency per Pass: {avg_latency_ms:.2f} ms")
    print(f" Total Benchmark Time    : {total_time:.3f} seconds")
    print("---------------------------------------------------------------------------")
    print(f" Status Summary          : Success (200 OK): {success_count} | Failed: {failed_count}")
    print("===========================================================================\n")

if __name__ == "__main__":
    run_stress_test()