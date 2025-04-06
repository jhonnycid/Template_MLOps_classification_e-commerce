# dashboard.py
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from time import sleep

console = Console()

steps = [
    ("🔧 Build & Train Model", "make train"),
    ("📦 Predict on Test Set", "make predict"),
    ("📊 Generate Monitoring Report", "make monitor"),
    ("🚀 Launch API FastAPI", "make api"),
]

def run_step(label, command):
    console.rule(f"[bold cyan]{label}")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task(f"[white]Running: {command}", start=False)
        progress.start_task(task)
        try:
            result = subprocess.run(command, shell=True, check=True)
            progress.update(task, description=f"[green]✓ {label}")
        except subprocess.CalledProcessError as e:
            progress.update(task, description=f"[red]✗ {label} Failed")
            console.print(Panel(f"[red]Error while executing: {command}\n\n{e}", title="Execution Error"))
            raise SystemExit(1)
        sleep(0.5)

def main():
    console.print(Panel("[bold magenta]Rakuten MLOps Pipeline[/bold magenta]", subtitle="Interactive Dashboard", expand=False))
    for label, command in steps:
        run_step(label, command)
    console.print(Panel("[bold green]✅ All steps completed successfully!", expand=False))

if __name__ == "__main__":
    main()
