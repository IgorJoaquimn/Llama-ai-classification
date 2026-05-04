import pandas as pd
import torch
import os
from transformers import pipeline
from torch.utils.data import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

console = Console()

# Dataset class for more efficient pipeline processing
class CommentDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        return self.texts[i]

def preprocess_files(file_configs, output_path, batch_size=256, min_comments_per_video=5):
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        console.print("[bold green]GPU detected! Using CUDA with Dataset iterator.[/bold green]")
    else:
        console.print("[bold yellow]No GPU detected. Using CPU.[/bold yellow]")

    # Initialize the language detection pipeline
    lang_pipe = pipeline(
        "text-classification",
        model="papluca/xlm-roberta-base-language-detection",
        device=device,
        batch_size=batch_size
    )

    all_dfs = []

    for config in file_configs:
        input_path = config['path']
        target_lang = config['lang']
        
        if not os.path.exists(input_path):
            console.print(f"[bold red]File not found: {input_path}. Skipping.[/bold red]")
            continue

        console.print(f"\n[bold blue]Processing {input_path} (Target: {target_lang})...[/bold blue]")
        
        # Load data
        df = pd.read_csv(input_path, low_memory=False)
        initial_count = len(df)
        
        # 1. Filter by length (minimum 10 chars)
        df = df[df['text'].fillna('').str.strip().str.len() >= 10].copy()
        after_len_count = len(df)
        console.print(f"Removed {initial_count - after_len_count} short comments.")

        # 2. Language Detection using Dataset Iterator
        texts = df['text'].astype(str).tolist()
        dataset = CommentDataset(texts)
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Detecting language for {target_lang}...", total=len(texts))
            
            # Pass the dataset directly to the pipeline
            # We iterate over the results to update the progress bar
            for i, out in enumerate(lang_pipe(dataset, truncation=True, max_length=128)):
                results.append(out['label'])
                if i % 100 == 0 or i == len(texts) - 1:
                    progress.update(task, completed=i+1)

        df['detected_lang'] = results
        
        # Filter for the correct language
        df_filtered = df[df['detected_lang'] == target_lang].copy()
        console.print(f"Kept {len(df_filtered)} validated '{target_lang}' comments.")
        all_dfs.append(df_filtered)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # 3. Filter videos with at least N valid comments
        console.print(f"\n[bold blue]Filtering videos with at least {min_comments_per_video} comments...[/bold blue]")
        counts = final_df['video_id'].value_counts()
        valid_video_ids = counts[counts >= min_comments_per_video].index
        
        final_df = final_df[final_df['video_id'].isin(valid_video_ids)].copy()
        
        console.print(f"Videos remaining: {len(valid_video_ids)}")
        console.print(f"Total comments remaining: {len(final_df)}")

        # 4. Save to Parquet
        console.print(f"\n[bold blue]Saving cleaned data to {output_path}...[/bold blue]")
        final_df.to_parquet(output_path, engine='fastparquet', index=False)
        console.print(f"[bold green]Success![/bold green]")
    else:
        console.print("[bold red]No data processed.[/bold red]")

if __name__ == "__main__":
    CONFIGS = [
        {'path': 'data/comments/comments_pt.csv', 'lang': 'pt'},
        {'path': 'data/comments/comments_en.csv', 'lang': 'en'}
    ]
    OUTPUT_PARQUET = 'data/comments_clean.parquet'
    
    preprocess_files(CONFIGS, OUTPUT_PARQUET)
