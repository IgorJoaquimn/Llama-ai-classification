import os
import pandas as pd
import torch
import re
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from rich.console import Console
from rich.rule import Rule

console = Console()

# Fixed System Prompt for Prefix Caching
SYSTEM_PROMPT = """You are a social media analyst. Analyze the comment for SPAM and LANGUAGE.
SPAM: Yes if ad/scam/bot, No if real discussion.
LANGUAGE: Primary language (e.g., English, Portuguese, Spanish)."""

class CommentProcessor:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_id = model_id
        self.gpu_utilization = 0.90  # Maximize VRAM usage
        self.max_model_len = 1024     # Comments are short, save KV cache
        self.llm = self._init_llm()
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=64, # Sufficient for "Reasoning + Language + Judgment"
        )

    def _init_llm(self) -> LLM:
        return LLM(
            model=self.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_utilization,
            max_model_len=self.max_model_len,
            enable_prefix_caching=True, # OPTIMIZATION: Cache the system prompt
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
            disable_log_stats=True
        )

    def process_batch(self, comments: List[str]) -> List[Dict[str, Any]]:
        """Detects spam and language using batched inference with prefix caching."""
        if not comments: return []
        
        prompts = []
        for text in comments:
            clean_text = str(text).replace("\n", " ")[:400]
            # Use ChatML format which Qwen expects for best prefix caching performance
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nAnalyze: \"{clean_text}\"<|im_end|>\n<|im_start|>assistant\nAnalysis:\n- Indicators:"
            prompts.append(prompt)

        # use_tqdm=True restores the vLLM progress bar for this batch
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=True)
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip().lower()
            
            # 1. Improved Spam Detection Logic (looking for judgment keywords)
            is_spam = "final judgment: spam" in generated_text or \
                      ("is spam" in generated_text and "is not spam" not in generated_text) or \
                      ("contains a scam" in generated_text) or \
                      ("is an advertisement" in generated_text) or \
                      ("spam: yes" in generated_text)
            
            if not is_spam and (" yes" in generated_text or "yes," in generated_text) and "spam" in generated_text and "not spam" not in generated_text:
                is_spam = True

            # 2. Optimized Language Extraction
            lang = "unknown"
            # Fast regex for common languages
            if "english" in generated_text: lang = "en"
            elif "portuguese" in generated_text or "português" in generated_text: lang = "pt"
            elif "spanish" in generated_text or "español" in generated_text: lang = "es"
            else:
                lang_match = re.search(r"language:\s*(\w+)", generated_text)
                if lang_match: lang = lang_match.group(1)[:2] # Keep it short (en, pt, es)

            results.append({
                "is_spam": is_spam,
                "detected_lang": lang
            })
        return results

def run_production_preprocessing(configs: List[Dict[str, str]], output_parquet: str, chunk_size: int = 200000):
    console.print(Rule(title="[bold green]High-Performance Comment Preprocessing[/bold green]"))
    
    processor = CommentProcessor()
    
    for config in configs:
        path = config['path']
        if not os.path.exists(path): continue
            
        console.print(f"[blue]Processing {path} in large chunks...[/blue]")
        
        # Large chunks because 64GB RAM can handle it
        reader = pd.read_csv(path, low_memory=False, chunksize=chunk_size)
        
        for i, chunk in enumerate(reader):
            # 1. Fast Pre-filtering (CPU/Pandas)
            chunk['word_count'] = chunk['text'].fillna('').str.split().str.len()
            df_batch = chunk[chunk['word_count'] > 3].copy()
            
            if df_batch.empty: continue
            
            # 2. LLM Processing (GPU/vLLM)
            texts = df_batch['text'].tolist()
            llm_results = processor.process_batch(texts)
            
            df_batch['is_spam'] = [r['is_spam'] for r in llm_results]
            df_batch['detected_lang'] = [r['detected_lang'] for r in llm_results]
            
            # 3. Filtering and Saving
            df_clean = df_batch[~df_batch['is_spam']].copy()
            cols = ['video_id', 'comment_id', 'text', 'like_count', 'published_at', 'detected_lang']
            
            # Incremental save to Parquet
            # We use a simple strategy: if it's the first chunk of the first file, write new. 
            # Otherwise, we'll write separate parquets to a folder for best speed, then combine.
            output_dir = "data/processed_chunks"
            os.makedirs(output_dir, exist_ok=True)
            chunk_file = f"{output_dir}/chunk_{config['lang']}_{i}.parquet"
            df_clean[cols].to_parquet(chunk_file, index=False)
            
            console.print(f"[dim]Finished chunk {i} ({len(df_clean)} rows saved to {chunk_file})[/dim]")

    console.print(f"[bold green]All chunks processed! You can now combine files in {output_dir}[/bold green]")

if __name__ == "__main__":
    CONFIGS = [
        {'path': 'data/comments/comments_pt.csv', 'lang': 'pt'},
        {'path': 'data/comments/comments_en.csv', 'lang': 'en'}
    ]
    OUTPUT_PARQUET = 'data/comments_clean.parquet'
    
    # Run the full production pipeline
    run_production_preprocessing(CONFIGS, OUTPUT_PARQUET)
