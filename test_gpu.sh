#!/bin/bash
set -e

# Create a small test parquet with 5 examples in tmp/
echo "Creating test data in tmp/..."
mkdir -p tmp
uv run python3 <<EOF
import pandas as pd
df = pd.DataFrame({
    "transcript": [
        "This is a video about AI and robots in the future.",
        "A tutorial on how to use Python for data science.",
        "Breaking news: New Llama 3 model released today!",
        "How to bake a chocolate cake in 10 minutes.",
        "The impact of artificial intelligence on the job market."
    ]
})
df.to_parquet("tmp/test_input.parquet")
EOF

# Create a test config with timestamped output in tmp/
TIMESTAMP=$(date +%s)
OUTPUT_FILE="tmp/test_output_${TIMESTAMP}.parquet"
CONFIG_FILE="tmp/config_test.yaml"

echo "Creating test config: ${CONFIG_FILE}..."
cat > ${CONFIG_FILE} <<EOF
input_file: "tmp/test_input.parquet"
output_file: "${OUTPUT_FILE}"
prompt_file: "prompt.txt"
model_id: "Qwen/Qwen2.5-0.5B-Instruct"
max_new_tokens: 256
batch_size: 8
device: "auto"
EOF

# Run the classification for 5 examples
echo "Running Llama classification for 5 examples..."
uv run python3 src/main.py ${CONFIG_FILE}

# Display the results
echo "Displaying results from ${OUTPUT_FILE}..."
uv run python3 <<EOF
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()
df = pd.read_parquet("${OUTPUT_FILE}")

table = Table(title="Classification Results Sample", show_header=True, header_style="bold magenta")
table.add_column("Index", style="dim", width=6)
table.add_column("Summary", width=40)
table.add_column("Topics", width=20)
table.add_column("AI?", justify="center")

for i, row in df.iterrows():
    table.add_row(
        str(i),
        str(row['summary'])[:150] + "...",
        ", ".join(row['topics']) if isinstance(row['topics'], list) else str(row['topics']),
        "[green]Yes[/green]" if row['is_ai_related'] else "[red]No[/red]"
    )

console.print(table)
EOF

echo "Test complete. Results in ${OUTPUT_FILE}"
