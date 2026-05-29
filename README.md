# Python-DS-Salary-Prediction-Analytics
# ⚡ PROD-MLOPS // Automated Inference & Governance Engine

An enterprise-grade, high-concurrency MLOps microservice architecture built to serve, audit, and monitor Data Science salary evaluation models. Engineered with a focus on bridging low-level systems reliability with high-level statistical observability, this system achieves a throughput of **1,700+ requests/sec** with an average latency of **~5ms** via non-blocking asynchronous event loops.

---

## 🏗️ System Architecture & Lifecycle

The platform is decoupled into three primary engineering operational layers:
1. **The Brain (ML Pipeline):** Features structural target-leakage protection, logarithmic transformations for highly skewed numerical targets, categorical frequency vector mapping, and serialized ensemble optimization.
2. **The Gateway (Web Infrastructure):** An asynchronous FastAPI microservice locked down behind a token-authenticated firewall (`X-API-KEY`) featuring volatile RAM hot-reloading and non-blocking background I/O logging.
3. **The Radar (Governance Matrix):** Continuous multi-threaded performance stress testing paired with an automated distribution drift engine analyzing real-time traffic divergence from training baselines.

---

## 🛠️ Repository Topology

```text
salary_predict_engine/
│
├── data/                          # Local Storage Boundary (Raw Datasets)
├── models/                        # Artifact Storage (Serialized Joblib Pipelines)
│
├── src/
│   ├── pipeline.py                # Data Engineering (Leakage prevention & Log Transforms)
│   ├── train.py                   # Core ML (Ensemble Model Optimization Loop)
│   └── evaluate.py                # Math Layer (Validation & Array Evaluation)
│
├── tests/
│   ├── test_app.py                # Gateway Quality Guard (HTTP Route Assertions)
│   └── test_pipeline.py           # Pipeline Quality Guard (Mock I/O tests via pytest)
│
├── app.py                         # Web Gateway (FastAPI Framework + Async SQLite Engine)
├── benchmark.py                   # Performance Testing (Multi-threaded Stress Generator)
├── monitor.py                     # Governance Layer (Statistical Distribution Drift Analyzer)
├── run_system.sh                  # Shell Orchestration Harness (Master Automation script)
│
├── .env                           # Security Config (Encrypted Token Vault)
├── requirements.txt               # Dependency Lockfile (Deterministic Package Bounds)
└── Dockerfile                     # Systems Layer (Slim Linux Container Schema)
```

🚀 System Requirements & Quickstart
1. Environment Provisioning
Clone the repository and spin up a virtual environment via Conda or Virtualenv:

Bash
git clone [https://github.com/suso-van/Python-DS-Salary-Prediction-Analytics.git](https://github.com/suso-van/Python-DS-Salary-Prediction-Analytics.git)
cd Python-DS-Salary-Prediction-Analytics
conda activate base
pip install -r requirements.txt
2. Vault Token Configuration
Create a .env file in the root directory to store your secure access tokens:

```Code snippet
API_ACCESS_TOKEN=ghost_architect_secure_token_2026
```
3. Execution via the Orchestration Harness
The entire lifecycle is fully automated. You can test code safety, train the active model, mount the API gateway, flood it with stress traffic, and measure data drift distributions with a single command:

```Bash
chmod +x run_system.sh
./run_system.sh
```
🔒 API Specifications & Gateway Hardening
Every operational route (except the public root health check) is actively protected by a custom header-token injection requirement. Unauthorized hits are rejected immediately with an HTTP 403 Forbidden status code.

📥 1. Compute Inference
Endpoint: POST /predict

Headers: X-API-KEY: <your_secret_token>

Payload Structure:
```
JSON
{
  "work_year": 2026,
  "experience_level": "SE",
  "employment_type": "FT",
  "job_title": "Machine Learning Engineer",
  "employee_residence": "US",
  "remote_ratio": 100,
  "company_location": "US",
  "company_size": "M"
}
```
System Action: Evaluates input through the in-RAM ensemble matrix, returns the calculated valuation instantly, and hands off the relational write to a BackgroundTasks worker thread pool to eliminate disk I/O network blocks.

🔄 2. Zero-Downtime Hot-Reload
Endpoint: POST /model/reload

Headers: X-API-KEY: <your_secret_token>

System Action: Flushes out-of-date runtime model arrays from volatile memory and safely swaps in a newly trained model pipeline artifact straight from the disk layer without taking down the web server or interrupting live incoming connections.

📊 Governance, Observability & Data Drift
To mitigate systemic model decay under shifting production market conditions, the monitoring utility (monitor.py) tracks live user trends against the training dataset baseline.

If incoming requests deviate past a strict 15% mathematical distribution shift threshold, an automatic warning matrix is triggered to prompt a pipeline retraining sequence:
```
Plaintext
==========================================================================
                MLOPS AUTOMATED ECOSYSTEM HEALTH MONITOR                  
==========================================================================
Checking system stability (Max Drift Tolerance: 15.0%)...

!!!! [ DRIFT DETECTED ] !!!! experience_level     -> Max Shift: 53.87%
 [ OK ]  employment_type      -> Max Shift: 3.13%
!!!! [ DRIFT DETECTED ] !!!! company_size         -> Max Shift: 46.29%
!!!! [ DRIFT DETECTED ] !!!! remote_ratio         -> Max Shift: 37.23%
--------------------------------------------------------------------------
 WARNING: Substantial data drift detected in production input streams.
 Action Required: Retrain the predictive pipeline with recent log data.
```
🧪 Structural Test Coverage
Quality assurance and structural integrity guards are fully implemented through pytest. The test suite covers:

Pipeline Isolation: Validates target transformations, inverse calculations, and target leakage prevention constraints.

HTTP Assertions: Tests authorization boundaries, input format fallback mechanisms (422 Unprocessable Entity), and route status code responses.

Run testing blocks manually using:
```
Bash
pytest tests/ -v
```
Ecosystem Focus: High-Concurrency MLOps / Systems-Level AI Deployment
