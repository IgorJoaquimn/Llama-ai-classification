import os
import yaml
from huggingface_hub import snapshot_download
from rich.console import Console

console = Console()

def main():
    if not os.path.exists("config.yaml"):
        console.print("[red]Error: config.yaml not found.[/red]")
        return

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_id = config.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")
    
    console.print(f"[bold blue]Downloading model:[/bold blue] [cyan]{model_id}[/cyan]")
    console.print("[yellow]This might take a while depending on your internet speed...[/yellow]")
    
    try:
        # This downloads the model to the local cache (~/.cache/huggingface/hub)
        snapshot_download(repo_id=model_id, repo_type="model")
        console.print(f"\n[bold green]✓ Model {model_id} is now fully cached![/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]Download failed:[/bold red] {e}")

if __name__ == "__main__":
    main()
