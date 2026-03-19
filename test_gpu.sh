#!/bin/bash
set -e

# Create a test parquet with more examples in tmp/ to see progress
echo "Creating test data in tmp/..."
mkdir -p tmp
uv run python3 <<EOF
import pandas as pd
data = []
for i in range(20):
    data.append({
        "transcript": f"This is video transcript number {i}. It discusses technology and AI in various contexts."
    })
df = pd.DataFrame(data)
df.to_parquet("tmp/test_input.parquet")
EOF

# Create a test config with debug_sleep to force visible progress
TIMESTAMP=$(date +%s)
OUTPUT_FILE="tmp/test_output_${TIMESTAMP}.parquet"
CONFIG_FILE="tmp/config_test.yaml"

echo "Creating test config: ${CONFIG_FILE}..."
cat > ${CONFIG_FILE} <<EOF
input_file: "tmp/test_input.parquet"
output_file: "${OUTPUT_FILE}"
prompt_file: "prompt.txt"
model_id: "meta-llama/Llama-3.1-8B-Instruct"
max_new_tokens: 128
batch_size: 8
device: "auto"
gpu_memory_utilization: 0.8 # Llama 8B is larger, let's be safe with memory during test
EOF

# Run the classification for 20 examples
echo "Running Llama classification for 20 examples..."
uv run python3 src/main.py ${CONFIG_FILE}

echo "Test complete. Results in ${OUTPUT_FILE}"
