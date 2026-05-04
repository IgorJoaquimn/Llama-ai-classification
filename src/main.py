import os
import json
import yaml
import sys
import re
import pandas as pd
import torch
import warnings
from typing import List, Dict, Any, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, logging as hf_logging
from rich.console import Console
from rich.rule import Rule

# Configuration
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()

console = Console(force_terminal=True)

class LlamaClassifier:
    """Handles LLM-based classification using vLLM."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_id = config.get("model_id", "meta-llama/Llama-3.1-8B-Instruct")
        self.max_tokens = config.get("max_new_tokens", 512)
        self.gpu_utilization = config.get("gpu_memory_utilization", 0.9)
        self.max_model_len = config.get("max_model_len", 8192)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.llm = self._init_llm()
        
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.max_tokens,
        )

    def _init_llm(self) -> LLM:
        """Initializes the vLLM engine."""
        console.print(f"[bold blue]Step 3:[/bold blue] Initializing vLLM model [cyan]{self.model_id}[/cyan]...")
        quantization = self.config.get("quantization", "none")
        vllm_quant = quantization if quantization in ["awq", "gptq", "fp8", "marlin", "compressed-tensors"] else None
        
        return LLM(
            model=self.model_id,
            trust_remote_code=True,
            quantization=vllm_quant,
            gpu_memory_utilization=self.gpu_utilization,
            max_model_len=self.max_model_len,
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
        )

    def _parse_json(self, text: str) -> Any:
        """Extracts and parses JSON from model output."""
        try:
            if "```json" in text:
                content = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                content = text.split("```")[1].split("```")[0]
            else:
                start = text.find('[') if '[' in text else text.find('{')
                end = text.rfind(']') + 1 if ']' in text else text.rfind('}') + 1
                content = text[start:end] if start != -1 else text
                
            return json.loads(content.strip())
        except Exception:
            return []

def main():
    console.print(Rule(title="[bold magenta]AI Comment Classification Pipeline[/bold magenta]"))
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    input_path = "data/comments_clean.parquet"
    output_path = config.get("output_file", "data/output/comment_classification.parquet")
    prompt_file = "prompt_individual.txt"
    
    if not os.path.exists(input_path):
        console.print(f"[red]Error: {input_path} not found. Run preprocessing first.[/red]")
        return
        
    with open(prompt_file, "r") as f:
        prompt_template = f.read()

    # 1. Load and Filter Data
    console.print(f"[blue]Loading {input_path}...[/blue]")
    df = pd.read_parquet(input_path)
    
    # Filter top 10 comments per video by like_count
    console.print("[blue]Filtering top 10 liked comments per video...[/blue]")
    df = df.sort_values(['video_id', 'like_count'], ascending=[True, False])
    df_top = df.groupby('video_id').head(10).copy()
    
    # 2. Check for resume
    if os.path.exists(output_path):
        df_done = pd.read_parquet(output_path)
        done_ids = df_done['comment_id'].unique()
        df_todo = df_top[~df_top['comment_id'].isin(done_ids)].copy()
        df_output = df_done
    else:
        df_todo = df_top
        df_output = pd.DataFrame()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if len(df_todo) == 0:
        console.print("[green]All comments already processed![/green]")
        return

    # Apply row limit for smoke tests
    row_limit = config.get("row_limit", 0)
    if row_limit > 0:
        df_todo = df_todo.head(row_limit)
        console.print(f"[yellow]Limiting to {row_limit} comments for this run.[/yellow]")

    # 3. Initialize Classifier
    classifier = LlamaClassifier(config)
    
    # 4. Processing Loop
    # We group comments into batches of N to match the prompt_individual.txt format
    BATCH_LLM_SIZE = 5 
    chunk_size = config.get("chunk_size", 100) # LLM calls before saving
    
    all_rows = df_todo.to_dict('records')
    
    for i in range(0, len(all_rows), BATCH_LLM_SIZE * chunk_size):
        current_slice = all_rows[i : i + BATCH_LLM_SIZE * chunk_size]
        
        batch_prompts = []
        batch_comment_groups = []
        
        for j in range(0, len(current_slice), BATCH_LLM_SIZE):
            group = current_slice[j : j + BATCH_LLM_SIZE]
            batch_comment_groups.append(group)
            
            comments_text = "\n".join([f"{k+1}. \"{c['text']}\"" for k, c in enumerate(group)])
            
            messages = [
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": f"Comments:\n{comments_text}"}
            ]
            prompt = classifier.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            batch_prompts.append(prompt)
            
        # LLM Generation
        outputs = classifier.llm.generate(batch_prompts, classifier.sampling_params, use_tqdm=True)
        
        new_data = []
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            group_results = classifier._parse_json(generated_text)
            
            # Ensure group_results is a list
            if not isinstance(group_results, list):
                group_results = []
                
            original_group = batch_comment_groups[j]
            for k, row in enumerate(original_group):
                res = group_results[k] if k < len(group_results) else {}
                row.update(res)
                new_data.append(row)
        
        df_output = pd.concat([df_output, pd.DataFrame(new_data)], ignore_index=True)
        
        # Incremental save
        temp_path = f"{output_path}.tmp"
        df_output.to_parquet(temp_path, index=False)
        os.replace(temp_path, output_path)

    console.print(Rule(title="[bold green]Pipeline Complete[/bold green]"))

if __name__ == "__main__":
    main()
