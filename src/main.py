import os
import json
import yaml
import sys
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

def load_config(config_path="config.yaml"):
    """Loads project configuration from a YAML file."""
    console.print(f"[bold blue]Step 1:[/bold blue] Loading configuration from [cyan]{config_path}[/cyan]...")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    console.print("[green]✓[/green] Configuration loaded successfully.")
    return config

def load_prompt(prompt_file):
    """Loads the system prompt from a text file."""
    console.print(f"[bold blue]Step 2:[/bold blue] Loading system prompt from [cyan]{prompt_file}[/cyan]...")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_file, "r") as f:
        content = f.read()
    console.print("[green]✓[/green] Prompt loaded successfully.")
    return content

def main():
    console.print(Rule(title="[bold magenta]Llama AI Classification Pipeline[/bold magenta]"))
    
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    try:
        config = load_config(config_path)
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")
        return

    input_file = config.get("input_file", "input.parquet")
    output_file = config.get("output_file", "output.parquet")
    model_id = config.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")
    prompt_file = config.get("prompt_file", "prompt.txt")
    max_new_tokens = config.get("max_new_tokens", 512)
    batch_size = config.get("batch_size", 1)
    quantization = config.get("quantization", "none")
    use_flash_attention = config.get("use_flash_attention", False)
    device_setting = config.get("device", "auto")

    # Determine device
    if device_setting == "auto":
        device_map = "auto" if torch.cuda.is_available() else None
    else:
        device_map = device_setting

    # Load prompt
    try:
        prompt_template = load_prompt(prompt_file)
    except Exception as e:
        console.print(f"[bold red]Error loading prompt:[/bold red] {e}")
        return

    # Load model and tokenizer
    console.print(f"[bold blue]Step 3:[/bold blue] Initializing model [cyan]{model_id}[/cyan]...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Fix padding for batching
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Using model_kwargs for dtype and quantization
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        if quantization == "4bit":
            model_kwargs["load_in_4bit"] = True
        elif quantization == "8bit":
            model_kwargs["load_in_8bit"] = True
        
        pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs=model_kwargs,
            device_map=device_map,
        )
        console.print("[green]✓[/green] Model and pipeline initialized.")
    except Exception as e:
        console.print(f"[bold red]Error initializing model:[/bold red] {e}")
        return

    # Load data
    console.print(f"[bold blue]Step 4:[/bold blue] Loading input data from [cyan]{input_file}[/cyan]...")
    if not os.path.exists(input_file):
        console.print("[yellow]![/yellow] Input file not found. Creating a dummy one for demonstration.")
        df_dummy = pd.DataFrame({"transcript": ["This is a video about AI and robots in the future.", "A tutorial on how to use Python for data science."] * 5})
        df_dummy.to_parquet(input_file)

    df_input = pd.read_parquet(input_file)
    console.print(f"[green]✓[/green] Loaded {len(df_input)} rows.")
    
    # Resume capability
    console.print(f"[bold blue]Step 5:[/bold blue] Checking for existing results in [cyan]{output_file}[/cyan]...")
    if os.path.exists(output_file):
        df_output = pd.read_parquet(output_file)
        processed_indices = df_output.index.tolist()
        console.print(f"[yellow]![/yellow] Resuming from {len(processed_indices)} already processed rows.")
    else:
        df_output = pd.DataFrame()
        processed_indices = []
        console.print("[green]✓[/green] Starting fresh processing.")

    # Filter only new data to process
    df_to_process = df_input[~df_input.index.isin(processed_indices)]
    
    if df_to_process.empty:
        console.print("[green]All rows already processed![/green]")
    else:
        # Processing Loop
        console.print(Rule(title=f"[bold yellow]Processing Transcripts (Batch Size: {batch_size})[/bold yellow]"))
        
        def data_generator():
            for _, row in df_to_process.iterrows():
                messages = [
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": f"Transcript:\n{row['transcript']}"}
                ]
                yield tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        results_iterator = pipe(
            data_generator(),
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            max_length=None,
            return_full_text=False
        )

        for (index, row), result in zip(df_to_process.iterrows(), tqdm(results_iterator, total=len(df_to_process), desc="Classifying")):
            response_text = result[0]["generated_text"].strip()

            try:
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                parsed_result = json.loads(response_text)
            except Exception as e:
                console.print(f"[yellow]![/yellow] Error parsing JSON for row {index}: {e}")
                parsed_result = {
                    "summary": "Error parsing",
                    "keywords": [],
                    "is_ai_related": None,
                    "is_ai_generated_content": None,
                    "topics": []
                }

            row_result = row.to_dict()
            row_result.update(parsed_result)
            
            df_new_row = pd.DataFrame([row_result], index=[index])
            df_output = pd.concat([df_output, df_new_row])
            df_output.to_parquet(output_file)

    console.print(Rule(title="[bold green]Processing Complete[/bold green]"))
    console.print(Panel(f"Results saved to [bold cyan]{output_file}[/bold cyan]", title="Success", border_style="green"))

if __name__ == "__main__":
    main()
