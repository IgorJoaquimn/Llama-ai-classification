# Llama AI Classification

This repository uses local Llama models via `transformers` to classify text transcripts stored in Parquet files. It is managed using `uv`.

## File Structure

```text
.
├── src/
│   ├── main.py            # Core classification logic
│   └── download_model.py  # Script to pre-download model to cache
├── data/                  # Input and output Parquet files
├── tmp/                   # Temporary test and runtime files
├── config.yaml            # Main project configuration
├── prompt.txt             # Classification system prompt
├── test_gpu.sh            # End-to-end test script with 5 samples
├── pyproject.toml         # uv project configuration
└── README.md              # Documentation
```

## Setup

The easiest way to get started is by using the automated setup script:

```bash
./setup.sh
```

This script will:
1.  **Check for uv**: If not installed, it will guide you through the process.
2.  **Sync Dependencies**: Install Python and all required packages.
3.  **Prepare Workspace**: Create necessary `data/` and `tmp/` folders.
4.  **Hugging Face Login**: Prompt for login if you need gated models.
5.  **Download Model**: Optionally pre-cache the model to avoid slow runs.

Alternatively, you can follow these manual steps:

## Usage

1.  **Prepare Input**:
    Place your input data in a file named `input.parquet` (or configure via `config.yaml`). It should have a column named `transcript`.

2.  **Run Classification**:
    ```bash
    uv run python src/main.py
    ```

## Test

To verify your environment (especially GPU/CUDA), run the provided test script:
```bash
./test_gpu.sh
```
This will create a temporary dataset, run 5 examples, and display a summary table of the results.
