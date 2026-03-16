# Llama AI Classification

This repository uses local Llama models via `transformers` to classify text transcripts stored in Parquet files. It is managed using `uv`.

## Setup

1.  **Install uv**:
    If you haven't already, install [uv](https://github.com/astral-sh/uv).

2.  **Sync Dependencies**:
    ```bash
    uv sync
    ```

3.  **Hugging Face Login** (Optional, but recommended for model access):
    ```bash
    uv run huggingface-cli login
    ```

## Usage

1.  **Prepare Input**:
    Place your input data in a file named `input.parquet`. It should have a column named `transcript`.

2.  **Run Classification**:
    ```bash
    uv run main.py
    ```

The script will:
- Load the specified Llama model (default: `meta-llama/Llama-3.2-1B-Instruct`).
- Read the prompt from `prompt.txt`.
- Process each row in `input.parquet`.
- Save results to `output.parquet`.
- **Resume Capability**: If the script is stopped, it will resume from the last processed row by checking `output.parquet`.

## Configuration

You can adjust the following in `main.py`:
- `INPUT_FILE`: Name of the input Parquet file.
- `OUTPUT_FILE`: Name of the output Parquet file.
- `MODEL_ID`: The Hugging Face model ID to use.
- `PROMPT_FILE`: The file containing the system prompt.
