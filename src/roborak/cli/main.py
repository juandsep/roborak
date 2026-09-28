"""roborak's command-line entry point."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from roborak import __version__
from roborak.cli.commands import ask as ask_cmd
from roborak.cli.commands import describe as describe_cmd
from roborak.cli.commands import fix as fix_cmd
from roborak.cli.commands import improve as improve_cmd
from roborak.cli.commands import review as review_cmd
from roborak.cli.commands import setup_cmd
from roborak.cli.commands.config_cmd import config_app
from roborak.cli.commands.rules import rules_app
from roborak.cli.shared import CliContext, set_cli_context

app = typer.Typer(
    name="roborak",
    help="AI code review from the terminal - local diffs, GitLab MRs, and GitHub PRs.",
    no_args_is_help=False,
    add_completion=False,
    rich_markup_mode="rich",
)

app.command("setup")(setup_cmd.setup)
app.command("review")(review_cmd.review)
app.command("describe")(describe_cmd.describe)
app.command("improve")(improve_cmd.improve)
app.command("fix")(fix_cmd.fix)
app.command("ask")(ask_cmd.ask)
app.add_typer(rules_app, name="rules")
app.add_typer(config_app, name="config")

console = Console(stderr=True)


def _version(value: bool) -> None:
    """Print the version and exit before the bare invocation falls through to ``review``."""
    if value:
        typer.echo(f"roborak {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="-v shows INFO logs; -vv adds DEBUG.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Errors only."),
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version,
        is_eager=True,
    ),
) -> None:
    """Run ``review`` when no subcommand is given, matching ``cr``'s bare invocation."""
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    elif quiet:
        level = logging.ERROR
    else:
        level = logging.WARNING
    # One handler on the same stderr console the spinner uses, so a log line prints
    # cleanly above a live stage rather than landing in the middle of it. `force`
    # replaces any handler a re-entrant invocation (or a test) left behind.
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_time=False, show_path=False, markup=False)],
        force=True,
    )

    set_cli_context(CliContext(console=console, quiet=quiet, verbose=verbose))

    if ctx.invoked_subcommand is None:
        ctx.invoke(review_cmd.review, repo=Path.cwd())


if __name__ == "__main__":
    app()
