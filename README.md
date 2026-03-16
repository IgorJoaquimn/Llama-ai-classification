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

1.  **Install uv**:
    If you haven't already, install [uv](https://github.com/astral-sh/uv).

2.  **Sync Dependencies**:
    ```bash
    uv sync
    ```

3.  **Hugging Face Login**:
    Required to access Llama models or gated repositories.
    ```bash
    uv run hf auth login
    ```

4.  **Download Model (Optional but recommended)**:
    Pre-cache the model to avoid slow downloads during execution.
    ```bash
    uv run python src/download_model.py
    ```

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
