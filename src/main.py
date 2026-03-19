import os
import json
import yaml
import sys
import re
import numpy as np
import pandas as pd
import torch
import warnings
from typing import List, Dict, Any, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, logging as hf_logging
from rich.console import Console
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

# Configuration
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()

console = Console(force_terminal=True)

class LlamaClassifier:
    """Handles LLM-based classification using vLLM with granular confidence metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_id = config.get("model_id", "meta-llama/Llama-3.1-8B-Instruct")
        self.max_tokens = config.get("max_new_tokens", 512)
        self.gpu_utilization = config.get("gpu_memory_utilization", 0.9)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.llm = self._init_llm()
        
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.max_tokens,
            logprobs=1 
        )

    def _init_llm(self) -> LLM:
        """Initializes the vLLM engine."""
        console.print(f"[bold blue]Step 3:[/bold blue] Initializing vLLM model [cyan]{self.model_id}[/cyan]...")
        quantization = self.config.get("quantization", "none")
        vllm_quant = quantization if quantization in ["awq", "gptq", "fp8", "marlin"] else None
        
        return LLM(
            model=self.model_id,
            trust_remote_code=True,
            quantization=vllm_quant,
            gpu_memory_utilization=self.gpu_utilization,
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
        )

    def format_prompts(self, df: pd.DataFrame, system_prompt: str) -> List[str]:
        """Converts dataframe rows into chat-templated prompts."""
        prompts = []
        for _, row in df.iterrows():
            transcript = str(row.get('transcript', ''))[:4000]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript:\n{transcript}"}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompts.append(prompt)
        return prompts

    def _calculate_detailed_confidences(self, text: str, generated_output) -> Dict[str, float]:
        """Calculates segmented confidence scores using token logprobs and offsets."""
        token_ids = generated_output.token_ids
        logprobs_list = generated_output.logprobs
        
        # 1. Map all logprobs to a flat list
        all_lps = []
        for lp, tid in zip(logprobs_list, token_ids):
            if lp and tid in lp:
                all_lps.append(lp[tid].logprob)
            else:
                # Fallback if vLLM output structure varies slightly or token missing
                all_lps.append(0.0)

        if not all_lps:
            return {"conf_total": 0.0, "conf_rationale": 0.0, "conf_classification": 0.0}

        # 2. Token Offset Mapping
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding['offset_mapping']
        
        # 3. Identify Segment Boundaries
        json_start = text.find("```json")
        if json_start == -1:
            json_start = text.find("{")
        if json_start == -1:
            json_start = len(text)

        # Search for classification value: "is_ai_related": true/false
        match = re.search(r'"is_ai_related":\s*(true|false)', text)
        class_start, class_end = (-1, -1)
        if match:
            class_start, class_end = match.start(1), match.end(1)

        # 4. Filter logprobs by segment
        rationale_lps = []
        classification_lps = []
        
        # Match tokens to text segments
        for i, (start, end) in enumerate(offsets):
            if i >= len(all_lps): break
            lp = all_lps[i]
            
            # Rationale (everything before JSON)
            if end <= json_start:
                rationale_lps.append(lp)
            
            # Classification Token (the specific true/false value)
            if class_start != -1 and start >= class_start and end <= class_end:
                classification_lps.append(lp)

        # 5. Convert Logprobs to Probabilities (Exp Mean)
        def exp_mean(lps):
            return float(np.exp(np.mean(lps))) if lps else 0.0

        return {
            "conf_total": exp_mean(all_lps),
            "conf_rationale": exp_mean(rationale_lps),
            "conf_classification": exp_mean(classification_lps)
        }

    def process_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Generates and parses responses for a batch of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        results = []
        
        for output in outputs:
            generated_output = output.outputs[0]
            text = generated_output.text.strip()
            
            # Calculate granular confidences
            conf_metrics = self._calculate_detailed_confidences(text, generated_output)
            
            # Parse JSON and merge metrics
            parsed = self._parse_json(text)
            parsed.update(conf_metrics)
            results.append(parsed)
            
        return results

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extracts and parses JSON from model output."""
        try:
            if "```json" in text:
                content = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                content = text.split("```")[1].split("```")[0]
            else:
                start = text.find('{')
                end = text.rfind('}') + 1
                content = text[start:end] if start != -1 else text
                
            return json.loads(content.strip())
        except Exception:
            return {
                "summary": "Error parsing JSON", 
                "keywords": [], 
                "is_ai_related": None, 
                "topics": [],
                "raw_output": text[:1000]
            }

def load_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return f.read()

def main():
    console.print(Rule(title="[bold magenta]AI Classification Pipeline[/bold magenta]"))
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    prompt_template = load_file(config.get("prompt_file", "prompt.txt"))
    input_path = config.get("input_file", "data/input.parquet")
    output_path = config.get("output_file", "data/output.parquet")
    
    classifier = LlamaClassifier(config)
    
    df_input = pd.read_parquet(input_path)
    if os.path.exists(output_path):
        df_output = pd.read_parquet(output_path)
        processed_ids = df_output.index.tolist()
    else:
        df_output = pd.DataFrame()
        processed_ids = []

    df_todo = df_input[~df_input.index.isin(processed_ids)]
    
    limit = config.get("row_limit", 1000)
    if limit and not df_todo.empty:
        df_todo = df_todo.head(limit)
        console.print(f"[yellow]Processing next {len(df_todo)} rows...[/yellow]")

    if df_todo.empty:
        console.print("[bold green]All transcripts processed![/bold green]")
        return

    all_prompts = classifier.format_prompts(df_todo, prompt_template)
    chunk_size = config.get("chunk_size", 100)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[yellow]Classifying...", total=len(all_prompts))
        
        for i in range(0, len(all_prompts), chunk_size):
            end = i + chunk_size
            batch_prompts = all_prompts[i:end]
            batch_indices = df_todo.index[i:end]
            batch_rows = df_todo.iloc[i:end]
            
            results = classifier.process_batch(batch_prompts)
            
            new_data = []
            for j, res in enumerate(results):
                row_dict = batch_rows.iloc[j].to_dict()
                row_dict.update(res)
                new_data.append(pd.DataFrame([row_dict], index=[batch_indices[j]]))
            
            df_output = pd.concat([df_output] + new_data)
            
            temp_path = f"{output_path}.tmp"
            df_output.to_parquet(temp_path)
            os.replace(temp_path, output_path)
            
            progress.update(task, advance=len(batch_prompts))

    console.print(Rule(title="[bold green]Pipeline Complete[/bold green]"))

if __name__ == "__main__":
    main()
