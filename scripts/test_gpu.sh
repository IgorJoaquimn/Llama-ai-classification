#!/bin/bash
set -e

# Create a test directory if it doesn't exist
mkdir -p tmp

echo "Step 1: Creating test data in tmp/..."
uv run python3 <<EOF
import pandas as pd
data = []
for i in range(5):
    data.append({
        "video_id": f"test_{i}",
        "title": f"Artificial Intelligence in Healthcare Part {i}",
        "transcript": f"This is a test transcript about how AI is revolutionizing medical diagnostics and patient care in hospital number {i}."
    })
df = pd.DataFrame(data)
df.to_parquet("tmp/test_input.parquet")
EOF

# Create a test config
TIMESTAMP=$(date +%s)
OUTPUT_FILE="tmp/test_output_${TIMESTAMP}.parquet"
CONFIG_FILE="tmp/config_test.yaml"

echo "Step 2: Creating test config: ${CONFIG_FILE}..."
cat > ${CONFIG_FILE} <<EOF
input_file: "tmp/test_input.parquet"
output_file: "${OUTPUT_FILE}"
prompt_file: "prompts/individual.txt"
model_id: "Qwen/Qwen2.5-0.5B-Instruct"
max_new_tokens: 256
gpu_memory_utilization: 0.5
row_limit: 0
chunk_size: 5
EOF

# Run the classification for test examples
echo "Step 3: Running Llama classification (vLLM) for 5 test examples..."
uv run python src/main.py ${CONFIG_FILE}

echo ""
echo "Step 4: Verifying results..."
uv run python3 <<EOF
import pandas as pd
df = pd.read_parquet("${OUTPUT_FILE}")
print("\n--- Test Results Summary ---")
print(df[['title', 'is_ai_related', 'conf_classification', 'topics']].to_string())
EOF

echo ""
echo "✅ Test complete. Detailed results in ${OUTPUT_FILE}"
