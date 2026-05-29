#!/bin/bash

# Force script to exit immediately if any individual step crashes
set -e

echo "=========================================================================="
echo "       STARTING SYSTEM-WIDE MLOPS MICROSERVICE COMPLETION RUN             "
echo "=========================================================================="

# Step 1: Execute the Test Suite
echo -e "\n[Step 1/5] Running pytest suite to verify structural integrity..."
pytest tests/

# Step 2: Trigger Training Pipeline
echo -e "\n[Step 2/5] Ingesting telemetry data and training ensemble model..."
if [ -f main.py ]; then
    python main.py
else
    python src/train.py
fi

# Step 3: Launch the Authenticated FastAPI Server in the background
echo -e "\n[Step 3/5] Spinning up FastAPI Web Gateway asynchronously..."
uvicorn app:app --port 8000 --reload > server_log.txt 2>&1 &
SERVER_PID=$!

# Give the server 3 seconds to fully initialize and bind to port 8000
sleep 3

# Ensure the background server process gets terminated cleanly even if this script fails midway
trap "echo 'Shutting down server (PID: $SERVER_PID)...'; kill $SERVER_PID" EXIT

# Step 4: Inject Concurrent Live Production Traffic via the Benchmark Tool
echo -e "\n[Step 4/5] Flooding inference gateway with multi-threaded benchmark traffic..."
python benchmark.py

# Step 5: Execute Data Drift Governance Analysis
echo -e "\n[Step 5/5] Analyzing production data drift matrices against baseline distributions..."
python monitor.py

echo "=========================================================================="
echo "      SUCCESS: END-TO-END SYSTEM COMPLETION CONCLUDED TRANSITIONALLY     "
echo "=========================================================================="