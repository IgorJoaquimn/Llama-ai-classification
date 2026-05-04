# Llama AI Classification Pipeline

This project is a high-performance tool designed to classify YouTube video comments and transcripts using Artificial Intelligence. It leverages a local Llama 3.1 8B model and the **vLLM** engine to process thousands of rows efficiently.

## 🚀 Getting Started

We use **uv**, a modern and extremely fast Python package manager.

### 1. Install `uv`
If you don't have it yet, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup the Project
The easiest way to set up everything (Python, libraries, and folders) is to run the setup script:
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

---

## 🛠️ Verifying your GPU

Before running the full dataset, verify your environment with the test script:

```bash
chmod +x scripts/test_gpu.sh
./scripts/test_gpu.sh
```
This script runs a small classification test. If you see a results table, you're ready!

---

## 📂 Project Structure

- **`src/`**: Core Python scripts.
  - `main.py`: Main classification pipeline.
  - `generate_report.py`: Generates HTML reports from results.
  - `preprocess_comments_v2.py`: Advanced comment cleaning and filtering.
- **`prompts/`**: AI instruction templates.
  - `individual.txt`: Prompt for classifying individual comments.
- **`notebooks/`**: Data analysis and exploratory notebooks.
- **`scripts/`**: Utility scripts for setup and testing.
- **`reports/`**: Generated HTML classification reports.
- **`data/`**: Project data (Parquet/CSV files).
- **`config.yaml`**: Main configuration file for paths and model settings.
- **`docs/`**: Reference papers and documentation.

---

## ⚙️ Configuration (`config.yaml`)

Edit `config.yaml` to customize the pipeline:

- **`input_file`**: Path to your input Parquet file.
- **`output_file`**: Path where results will be saved.
- **`model_id`**: Hugging Face model ID (e.g., `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`).
- **`gpu_memory_utilization`**: Fraction of VRAM to reserve (default `0.9`).
- **`row_limit`**: Limit processing to N rows (set `0` for all).
- **`chunk_size`**: Save progress every N prompts.

---

## 🏃 Running the Pipeline

### 1. Run Classification
```bash
uv run python src/main.py
```

### 2. Generate Report
```bash
uv run python src/generate_report.py
```
The report will be saved in `reports/report_individual.html`.

---

## 📊 Monitoring

- **GPU**: `watch -n 1 nvidia-smi`
- **System**: `btop`

---

## 🧠 Results Summary

The output file adds classification metadata:
- **`is_ai_related`**: Boolean flag.
- **`topics`**: Identified topics (e.g., "AI ethics", "creativity").
- **`keywords`**: Extracted keywords from the comment.
- **`is_ai_generated_content`**: Whether the comment itself was likely AI-generated.
