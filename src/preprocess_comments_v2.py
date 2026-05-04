import os
import pandas as pd
import torch
import yaml
import re
import json
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from rich.console import Console
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

console = Console()

class CommentSpamFilter:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_id = model_id
        self.gpu_utilization = 0.8 # Higher for production
        self.max_model_len = 2048
        self.llm = self._init_llm()
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=80,
        )

    def _init_llm(self) -> LLM:
        return LLM(
            model=self.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_utilization,
            max_model_len=self.max_model_len,
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
            disable_log_stats=True
        )

    def classify_spam(self, comments: List[str]) -> List[bool]:
        """Classifies comments as spam (True) or not spam (False)."""
        if not comments: return []
        
        prompts = []
        for text in comments:
            clean_text = str(text).replace("\n", " ")[:400]
            prompt = f"<|im_start|>system\nYou are a strict YouTube moderator. Your job is to identify SCAMS and ADS.\nVALID: Opinions, questions, real discussion.\nSPAM: Links to scams, crypto ads, all-caps urgent 'FREE MONEY' offers.\n<|im_start|>user\nAnalyze this comment for spam:\nComment: \"{clean_text}\"\n<|im_start|>assistant\nAnalysis:\n- Indicators (links, scams, ads):"
            prompts.append(prompt)

        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip().lower()
            is_spam = "final judgment: spam" in generated_text or \
                      ("is spam" in generated_text and "is not spam" not in generated_text) or \
                      ("contains a scam" in generated_text) or \
                      ("is an advertisement" in generated_text)
            
            if "yes" in generated_text and "spam" in generated_text and "not spam" not in generated_text:
                is_spam = True

            results.append(is_spam)
        return results

def process_all_comments(configs: List[Dict[str, str]], output_parquet: str, chunk_size: int = 5000):
    console.print(Rule(title="[bold green]Starting Full Comment Preprocessing[/bold green]"))
    
    filter_engine = CommentSpamFilter()
    all_dfs = []

    for config in configs:
        path = config['path']
        lang = config['lang']
        
        if not os.path.exists(path):
            console.print(f"[red]Skipping {path} (not found)[/red]")
            continue
            
        console.print(f"[blue]Processing {lang.upper()} comments from {path}...[/blue]")
        
        # Read in chunks to save memory
        reader = pd.read_csv(path, low_memory=False, chunksize=chunk_size)
        
        # We'll use a local progress bar for each file
        for chunk in reader:
            # 1. Length Filter
            chunk['word_count'] = chunk['text'].fillna('').str.split().str.len()
            df_valid_len = chunk[chunk['word_count'] > 3].copy()
            
            if df_valid_len.empty:
                continue
                
            # 2. LLM Spam Filter
            texts = df_valid_len['text'].tolist()
            is_spam_list = filter_engine.classify_spam(texts)
            df_valid_len['is_spam'] = is_spam_list
            
            # Keep only non-spam
            df_clean = df_valid_len[~df_valid_len['is_spam']].copy()
            df_clean['lang'] = lang
            
            # Select relevant columns to save space
            cols_to_keep = ['video_id', 'comment_id', 'text', 'like_count', 'published_at', 'lang']
            all_dfs.append(df_clean[cols_to_keep])
            
            # Incremental save or just keep in list if memory allows
            # For 3M comments, we might want to save to temporary parquets
            
    if all_dfs:
        df_final = pd.concat(all_dfs, ignore_index=True)
        df_final.to_parquet(output_parquet)
        console.print(f"[bold green]Successfully saved {len(df_final)} clean comments to {output_parquet}[/bold green]")

if __name__ == "__main__":
    CONFIGS = [
        {'path': 'data/comments/comments_pt.csv', 'lang': 'pt'},
        {'path': 'data/comments/comments_en.csv', 'lang': 'en'}
    ]
    OUTPUT_PARQUET = 'data/comments_clean.parquet'
    
    # NOTE: 3 million rows with LLM classification will take hours.
    # We might want to run this in the background.
    process_all_comments(CONFIGS, OUTPUT_PARQUET)
