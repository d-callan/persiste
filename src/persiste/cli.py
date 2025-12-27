"""Command-line interface."""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="PERSISTE: Constraint detection framework")
console = Console()


@app.command()
def version():
    """Show PERSISTE version."""
    from persiste import __version__
    console.print(f"PERSISTE version {__version__}")


@app.command()
def plugins():
    """List available plugins."""
    from persiste.plugins import plugins as plugin_registry
    
    plugin_list = plugin_registry.list()
    
    if not plugin_list:
        console.print("[yellow]No plugins installed[/yellow]")
        console.print("\nInstall plugins with:")
        console.print("  pip install persiste-phylo")
        console.print("  pip install persiste-assembly")
        return
    
    table = Table(title="Available Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    
    for name in plugin_list:
        plugin = plugin_registry.load(name)
        table.add_row(name, plugin.version)
    
    console.print(table)


@app.command()
def info(plugin_name: str):
    """Show information about a plugin."""
    from persiste.plugins import plugins as plugin_registry
    
    try:
        plugin = plugin_registry.load(plugin_name)
    except KeyError as e:
        console.print(f"[red]Error: {e}[/red]")
        return
    
    console.print(f"\n[bold]{plugin.name}[/bold] v{plugin.version}")
    console.print(f"\n{plugin.__doc__ or 'No description available'}")
    
    if plugin.state_spaces:
        console.print(f"\n[cyan]State Spaces:[/cyan] {', '.join(plugin.state_spaces.keys())}")
    if plugin.baselines:
        console.print(f"[cyan]Baselines:[/cyan] {', '.join(plugin.baselines.keys())}")
    if plugin.analyses:
        console.print(f"[cyan]Analyses:[/cyan] {', '.join(plugin.analyses.keys())}")


if __name__ == "__main__":
    app()
