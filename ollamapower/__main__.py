"""Command-line entry point.

Phase 0 only ships the package skeleton; ``hub``, ``agent``, ``init``,
``doctor`` and the management commands arrive with the phases that build
them (see the spec, Section 14.3).
"""

from __future__ import annotations

import typer

from . import __version__

cli = typer.Typer(add_completion=False, no_args_is_help=True, help="OllamaPower - cluster-boosted Ollama")


@cli.callback()
def _root() -> None:
    """OllamaPower - cluster-boosted Ollama."""


@cli.command()
def version() -> None:
    """Show the OllamaPower version."""
    typer.echo(f"ollamapower {__version__}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
