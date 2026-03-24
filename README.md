# Llama AI Classification Pipeline

This project is a high-performance tool designed to classify YouTube video titles and transcripts using Artificial Intelligence. It uses a local Llama 3.1 8B model and the **vLLM** engine to process thousands of rows quickly and accurately.

## 🚀 Getting Started

We use **uv**, a modern and extremely fast Python package manager. It handles everything: installing Python, managing libraries, and running the project in a "virtual environment" so it doesn't mess up your computer's system settings.

### 1. Install `uv`
If you don't have it yet, run this command in your terminal:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*After installing, close and reopen your terminal.*

### 2. Setup the Project
The easiest way to set up everything (Python, libraries, and folders) is to run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```
This script will:
- Install project dependencies using `uv sync`.
- Create the `data/`, `logs/`, and `tmp/` folders.
- Optionally log you into Hugging Face to access Llama models.
- Optionally pre-download the model to save time later.

---

## 🛠️ Verifying your GPU

Before running the full dataset, you should verify that your graphics card (GPU) is working correctly with the AI model. We provide a test script for this:

```bash
chmod +x test_gpu.sh
./test_gpu.sh
```
This script creates a small fake dataset, runs 5 classifications, and prints a summary table. If you see the table with results, your environment is perfect!

---

## 📂 Project Organization

- **`data/input/`**: Place your `.parquet` or `.csv` files here.
- **`data/output/`**: This is where the results (`output_classification.parquet`) will be saved.
- **`src/main.py`**: The "brain" of the project that runs the classification.
- **`config.yaml`**: The "control panel" where you change settings.
- **`prompt.txt`**: The "instructions" we give to the AI model.
- **`logs/`**: Detailed records of every run for troubleshooting.
- **`eda_classification.ipynb`**: A Jupyter Notebook to visualize and analyze your results.

---

## ⚙️ How to Use the Configuration (`config.yaml`)

You don't need to touch the code to change how the pipeline works. Just open `config.yaml` in any text editor:

- **`input_file`**: The path to your data (e.g., `"data/input/my_videos.parquet"`).
- **`output_file`**: Where you want the results saved.
- **`model_id`**: Which AI model to use (default is Llama 3.1 8B).
- **`gpu_memory_utilization`**: Set this to `0.9` (90%) to give the AI plenty of "thinking space" on your GPU.
- **`row_limit`**: Set to `0` to process the entire file, or a small number (like `10`) for a quick test.
- **`chunk_size`**: How many rows to process before saving (default is `100`).

---

## 🏃 Running the Pipeline

To start the classification, use:
```bash
uv run python src/main.py
```

### Running in the Background
If you have a large dataset (thousands of rows) and want to let it run while you do other things (or even close your terminal), use:
```bash
nohup uv run python src/main.py > logs/classification.log 2>&1 &
```
You can check the progress anytime by looking at the log:
```bash
tail -f logs/classification.log
```

---

## 📊 Monitoring the Process

Since AI classification is "heavy" on your computer's hardware, it is important to monitor your resources:

### 1. Monitoring the GPU (`nvidia-smi`)
This tool shows you how much Video RAM (VRAM) the model is using and how hard the graphics card is working.
```bash
watch -n 1 nvidia-smi
```
*Look for "Memory-Usage" and "Volatile Gpu-Util".*

### 2. Monitoring the System (`btop`)
This is a beautiful, interactive dashboard for your CPU, RAM, and disk usage.
```bash
btop
```
*If you don't have it, you can install it on Linux with `sudo apt install btop`.*

---

## 🧠 Understanding the Results

The output file will contain your original data plus several new columns created by the AI:

- **`summary`**: A concise overview of the video content.
- **`is_ai_related`**: A simple `True` or `False`.
- **`topics`**: Categories assigned to the video (e.g., "Business", "Hardware").
- **`conf_classification`**: **Model Confidence**. A score from 0 to 1. 
    - `0.95+`: The model is very certain.
    - `0.70-0.90`: Generally reliable.
    - `Below 0.60`: You should probably double-check this one manually!
- **`conf_rationale`**: How "sure" the model felt while writing its reasoning (Chain of Thought).

---

## 📈 Analyzing the Data
Open the `eda_classification.ipynb` file in VS Code or any Jupyter environment to see charts, distributions, and a deep-dive analysis of your classified dataset.
