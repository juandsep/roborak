"""Pieces every command needs.

Loading configuration, working out where the change comes from, and emitting the
result all behave identically across ``review``, ``describe``, ``improve`` and
``ask``, so they live here rather than being reimplemented four times.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console

from roborak.core.config import Config, ForgeConfig, ReviewProfile, load_config
from roborak.core.models import ChangeSet, Issue, ReviewResult, ReviewStatus
from roborak.core.severity import Severity
from roborak.core.verdict import blocking_findings
from roborak.llm.client import LLMClient, missing_credentials
from roborak.render import json_out, markdown, prompt_only, rich_report, terminal
from roborak.sources.base import SourceError
from roborak.sources.forge import (
    Provider,
    Target,
    detect_provider,
    get_token,
    parse_target,
    provider_from_url,
    resolve_host,
)
from roborak.sources.github import GitHubSource
from roborak.sources.gitlab import GitLabSource
from roborak.sources.issue import load_issue, resolve_linked_change
from roborak.sources.local_git import LocalGitSource, Scope, is_git_repo
from roborak.sources.paths import PathsSource
from roborak.supply.analyzer import is_supply_asset

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2

TOKEN_HELP = {
    "gitlab": "GITLAB_TOKEN (or ROBORAK_GITLAB_TOKEN), or put it in the config as "
    "forge.tokens.gitlab",
    "github": "GITHUB_TOKEN (or ROBORAK_GITHUB_TOKEN), put it in the config as "
    "forge.tokens.github, or sign in with `gh auth login`",
}


@dataclass
class CliContext:
    """What the top-level callback resolved, shared with every subcommand.

    The one stderr console lives here so logging and the spinner write to the same
    stream in coordination rather than racing each other, and the verbosity the
    callback parsed rides along so a command can honour ``--quiet`` for its own
    chrome."""

    console: Console
    quiet: bool = False
    verbose: int = 0


_active_context: CliContext | None = None


def set_cli_context(context: CliContext) -> None:
    """Record what the top-level callback resolved, for the command it dispatches to.

    Held here rather than threaded through a ``typer.Context`` parameter because the
    bare invocation (no subcommand) reaches ``review`` through ``ctx.invoke`` on the
    raw function, which does not inject a context param -- one process runs one
    command, so a module-level handoff is both simpler and reliable."""
    global _active_context
    _active_context = context


def cli_context() -> CliContext:
    """The context the callback built, or a plain stderr default if it never ran.

    The fallback keeps a command usable when it is driven directly, e.g. a test that
    calls the function without going through the app callback."""
    return _active_context or CliContext(console=Console(stderr=True))


_STAGE_MARK = {"ok": ("✓", "green"), "skip": ("⚠", "yellow"), "fail": ("✗", "red")}
_STAGE_LABEL_WIDTH = 18


@dataclass
class Stage:
    """One step of a run, with the outcome it leaves behind on stderr.

    A stage starts assuming it will succeed; the body overrides that with
    :meth:`skip` or :meth:`fail` and fills in :attr:`detail`. :meth:`progress`
    reaches back into the live spinner so a long stage can show where it is."""

    label: str
    detail: str = ""
    status: str = "ok"
    elapsed: float = 0.0
    _update: Callable[[str], None] | None = field(default=None, repr=False)

    def skip(self, reason: str) -> None:
        self.status = "skip"
        self.detail = reason

    def fail(self, detail: str) -> None:
        self.status = "fail"
        self.detail = detail

    def progress(self, text: str) -> None:
        if self._update is not None:
            self._update(text)


@dataclass
class StageLog:
    """The run's stage lines: a spinner while each runs, a record once it is done.

    On a terminal a stage animates a spinner and then leaves one permanent line;
    off one (a pipe, CI) it leaves the same line with no spinner or control codes;
    under ``--quiet`` or a machine-readable mode it leaves nothing. Everything here
    is stderr, so stdout stays the product."""

    console: Console
    quiet: bool = False
    stages: list[Stage] = field(default_factory=list)

    @contextmanager
    def stage(self, label: str, *, spinner_text: str | None = None) -> Iterator[Stage]:
        st = Stage(label=label)
        self.stages.append(st)
        start = time.monotonic()
        text = spinner_text or f"{label}…"
        use_spinner = not self.quiet and self.console.is_terminal
        spinner = (
            self.console.status(f"[dim]{text}[/]", spinner="dots") if use_spinner else nullcontext()
        )
        completed = False
        try:
            with spinner as handle:
                if handle is not None:
                    st._update = lambda t: handle.update(f"[dim]{t}[/]")
                yield st
                completed = True
        finally:
            st._update = None
            st.elapsed = time.monotonic() - start
            if not completed and st.status == "ok":
                st.status = "fail"
            if not self.quiet:
                self._print(st)

    def _print(self, st: Stage) -> None:
        mark, colour = _STAGE_MARK[st.status]
        label = st.label.ljust(_STAGE_LABEL_WIDTH)
        line = f"[{colour}]{mark}[/] {label}"
        if st.detail:
            line += f" [dim]{st.detail}[/]"
        # A skip carries no duration: the example in the issue leaves that column
        # empty, and the time a stage took to decide not to run tells no one anything.
        if st.status != "skip":
            line += f"  [dim]{st.elapsed:.1f}s[/]"
        self.console.print(line, highlight=False)

    def summary(self, result: ReviewResult | None = None) -> None:
        """A closing line so a non-fatal skip or failure is not lost to scrollback."""
        if self.quiet or not self.stages:
            return
        total = sum(st.elapsed for st in self.stages)
        parts = [f"{len(self.stages)} stage(s) in {total:.1f}s"]
        if result is not None and result.usage:
            parts.append(f"{len(result.usage)} model call(s), {result.tokens_used} tokens")
        skipped = [st.label for st in self.stages if st.status == "skip"]
        failed = [st.label for st in self.stages if st.status == "fail"]
        self.console.print()
        self.console.print(f"[dim]{'  ·  '.join(parts)}[/]", highlight=False)
        if skipped:
            self.console.print(f"[dim]skipped: {', '.join(skipped)}[/]", highlight=False)
        if failed:
            self.console.print(f"[yellow]failed: {', '.join(failed)}[/]", highlight=False)


def stdout_is_a_terminal() -> bool:
    """Whether the report is being read or captured.

    The two want opposite things -- a person wants it rendered, a pipe wants the
    markdown -- and this is the only honest way to tell them apart.
    """
    return sys.stdout.isatty()


def is_interactive() -> bool:
    """Whether there is a human here to answer a question.

    Both halves matter: a redirected stdout means the output is being captured
    rather than read, and a non-tty stdin means nobody could answer anyway.
    Typer's ``CliRunner`` and every CI runner fail the second test, which is what
    keeps them from hanging on a prompt. Asked of the streams directly rather
    than of a console, because a console may be stderr, and stderr staying a
    terminal says nothing about whether stdout was piped into a file.
    """
    return sys.stdout.isatty() and sys.stdin.isatty()


def fail(console: Console, message: str) -> NoReturn:
    console.print(f"[bold red]error[/] {message}")
    raise typer.Exit(EXIT_ERROR)


@dataclass
class Session:
    """Everything resolved from the flags, before any review work happens."""

    console: Console
    repo: Path
    config: Config
    changeset: ChangeSet
    llm: LLMClient | None
    target: Target | None
    token: str | None
    issue: Issue | None = None
    """What the change is supposed to solve, when ``--issue`` supplied one."""


def start(
    console: Console,
    *,
    repo: Path | None,
    mr: str | None,
    pr: str | None,
    issue: str | None = None,
    base: str | None = None,
    committed: bool = False,
    uncommitted: bool = False,
    include_untracked: bool = False,
    no_discussions: bool = False,
    config_path: Path | None = None,
    profile: ReviewProfile | None = None,
    model: str | None = None,
    no_llm: bool = False,
    quiet_status: bool = False,
    stages: StageLog | None = None,
) -> Session:
    """Resolve config, credentials and the changeset, failing fast and clearly."""
    repo = (repo or Path.cwd()).resolve()
    stages = stages or StageLog(console, quiet=quiet_status)

    if mr and pr:
        fail(console, "--mr and --pr are mutually exclusive.")
    if committed and uncommitted:
        fail(console, "--committed and --uncommitted are mutually exclusive.")

    try:
        config = load_config(repo, config_path, profile=profile)
    except (OSError, ValueError) as exc:
        fail(console, f"config error: {exc}")

    if model:
        config.llm.model = model

    if not no_llm and (missing := missing_credentials(config.model, config.llm)):
        fail(
            console,
            f"{config.model} needs [bold]{missing}[/] to be set.\n"
            "[dim]Set it, pick another model with --model, or run --no-llm.[/]",
        )

    provider: Provider | None = "gitlab" if mr else "github" if pr else None
    target: Target | None = None
    token: str | None = None

    loaded_issue: Issue | None = None
    if issue:
        stated_target = _has_target(
            mr=mr,
            pr=pr,
            base=base,
            committed=committed,
            uncommitted=uncommitted,
            include_untracked=include_untracked,
        )
        loaded_issue, linked = _load_issue(
            console,
            issue,
            repo=repo,
            forge=config.forge,
            mr=mr,
            pr=pr,
            stages=stages,
            resolve_link=not stated_target,
        )
        if linked is not None:
            provider, target = linked.provider, linked
            label = "merge request" if provider == "gitlab" else "pull request"
            console.print(f"[dim]issue #{loaded_issue.number} → {label} #{target.number}[/]")

    if provider is not None:
        token = get_token(provider, config.forge)
        if token is None:
            fail(console, f"No {provider} token found. Set {TOKEN_HELP[provider]}.")
        if target is None:
            try:
                target = parse_target(
                    (mr or pr or "").strip(),
                    provider,
                    host=resolve_host(provider, config.forge, repo=repo),
                    repo=repo,
                )
            except SourceError as exc:
                fail(console, str(exc))

    try:
        changeset = _load_changeset(
            console,
            repo,
            provider,
            target,
            token,
            max_recovered_file_bytes=config.forge.max_recovered_file_bytes,
            ignore_paths=config.ignore_paths,
            base=base,
            committed=committed,
            uncommitted=uncommitted,
            include_untracked=include_untracked,
            include_discussions=config.review.include_discussions and not no_discussions,
            stages=stages,
        )
    except SourceError as exc:
        fail(console, str(exc))

    return Session(
        console=console,
        repo=repo,
        config=config,
        changeset=changeset,
        llm=None if no_llm else LLMClient(config.llm),
        target=target,
        token=token,
        issue=loaded_issue,
    )


def _has_target(
    *,
    mr: str | None,
    pr: str | None,
    base: str | None,
    committed: bool,
    uncommitted: bool,
    include_untracked: bool,
) -> bool:
    """Whether the user already said what to review.

    ``--issue`` only picks the target when nothing else did; asking for an issue
    *and* a base ref means the issue is context for the diff you named.
    """
    return bool(mr or pr or base or committed or uncommitted or include_untracked)


def _issue_provider(
    console: Console,
    reference: str,
    *,
    mr: str | None,
    pr: str | None,
    repo: Path,
    forge: ForgeConfig,
) -> Provider:
    """Work out which forge holds the issue, or fail saying how to be explicit."""
    provider = (
        ("gitlab" if mr else "github" if pr else None)
        or provider_from_url(reference)
        or detect_provider(repo=repo, forge=forge)
    )
    if provider is None:
        fail(
            console,
            "Could not tell which forge issue "
            f"[bold]{reference}[/] lives on.\n"
            "[dim]Pass the full issue URL, add --mr/--pr to say which forge, or name "
            "the host in the config as forge.hosts.gitlab / forge.hosts.github.[/]",
        )
    return provider


def _load_issue(
    console: Console,
    reference: str,
    *,
    repo: Path,
    forge: ForgeConfig,
    mr: str | None,
    pr: str | None,
    stages: StageLog,
    resolve_link: bool,
) -> tuple[Issue, Target | None]:
    """Fetch the issue, and the change implementing it when we are choosing one."""
    reference = reference.strip()
    provider = _issue_provider(console, reference, mr=mr, pr=pr, repo=repo, forge=forge)

    try:
        issue_target = parse_target(
            reference,
            provider,
            host=resolve_host(provider, forge, repo=repo),
            kind="issue",
            repo=repo,
        )
    except SourceError as exc:
        fail(console, str(exc))

    token = get_token(provider, forge)
    if token is None:
        fail(console, f"--issue needs a {provider} token. Set {TOKEN_HELP[provider]}.")

    try:
        with stages.stage("issue") as st:
            issue = load_issue(issue_target, token)
            st.detail = f"#{issue.number}"

        if not resolve_link:
            return issue, None

        with stages.stage("linked change") as st:
            linked = resolve_linked_change(issue_target, token)
            st.detail = f"→ #{linked.number}" if linked is not None else "none found"
    except SourceError as exc:
        fail(console, str(exc))

    if linked is None:
        log.debug("issue #%d has no linked change; reviewing the local diff", issue.number)
        return issue, None

    return issue, Target(provider, issue_target.host, issue_target.project, linked.number)


def _load_changeset(
    console: Console,
    repo: Path,
    provider: Provider | None,
    target: Target | None,
    token: str | None,
    *,
    max_recovered_file_bytes: int,
    ignore_paths: list[str],
    base: str | None,
    committed: bool,
    uncommitted: bool,
    include_untracked: bool,
    include_discussions: bool,
    stages: StageLog,
) -> ChangeSet:
    if provider is not None:
        assert target is not None and token is not None
        label = "merge request" if provider == "gitlab" else "pull request"
        source = GitLabSource if provider == "gitlab" else GitHubSource
        instance = source(target=target, token=token)
        instance.max_recovered_file_bytes = max_recovered_file_bytes
        instance.include_discussions = include_discussions
        with stages.stage("fetch change", spinner_text=f"fetching {label}…") as st:
            changeset = instance.load()
            st.detail = f"{label} #{target.number}, {len(changeset.files)} file(s)"
            return changeset

    if not is_git_repo(repo):
        return _load_paths(
            console,
            repo,
            ignore_paths=ignore_paths,
            base=base,
            committed=committed,
            uncommitted=uncommitted,
            include_untracked=include_untracked,
            stages=stages,
        )

    scope = Scope.COMMITTED if committed else Scope.UNCOMMITTED if uncommitted else Scope.ALL
    return LocalGitSource(
        repo=repo, scope=scope, base=base, include_untracked=include_untracked
    ).load()


GIT_ONLY_FLAGS = ("--base", "--committed", "--uncommitted", "--include-untracked")


def _load_paths(
    console: Console,
    repo: Path,
    *,
    ignore_paths: list[str],
    base: str | None,
    committed: bool,
    uncommitted: bool,
    include_untracked: bool,
    stages: StageLog,
) -> ChangeSet:
    """Review a plain directory, whole file by file.

    Reached only when there is no repository to diff. The flags that name a diff
    have no meaning here, so they are refused rather than quietly reinterpreted:
    a user who asked for their uncommitted work should not be handed every file
    in the tree instead.
    """
    stated = [
        flag
        for flag, given in zip(
            GIT_ONLY_FLAGS,
            (base is not None, committed, uncommitted, include_untracked),
            strict=True,
        )
        if given
    ]
    if stated:
        raise SourceError(
            f"{repo} is not a git repository, so {', '.join(stated)} has nothing to compare. "
            "Drop it to review every file in the directory instead."
        )

    # `ignore_paths` is applied during the walk so that ignored noise cannot spend
    # the `max_files` budget an actual source file needed. Dependency assets are
    # exempt: the supply-chain stage still reads them, and Reviewer drops them
    # again before any of it reaches the model prompt.
    source = PathsSource(root=repo, ignore_paths=list(ignore_paths), keep=is_supply_asset)
    with stages.stage("read files", spinner_text=f"reading every file under {repo}…") as st:
        changeset = source.load()
        st.detail = f"no git repository; {len(changeset.files)} file(s)"
        return changeset


def emit(
    session: Session,
    result: ReviewResult,
    *,
    as_json: bool = False,
    agent: bool = False,
    prompt_only_mode: bool = False,
    markdown_path: Path | None = None,
    panels: bool = False,
    full: bool = False,
) -> None:
    """Write the result to whichever surfaces were asked for.

    stdout carries the report and nothing else. Everything roborak says *about*
    the run -- spinners, errors, the closing question -- goes to stderr, so
    ``roborak review > review.md`` produces the report rather than the report with
    a spinner smeared through it. The machine-readable modes rely on the same
    property, which they always did.
    """
    console = session.console

    if markdown_path is not None:
        try:
            markdown_path.write_text(markdown.render(result), encoding="utf-8")
        except OSError as exc:
            console.print(f"[bold red]error[/] could not write {markdown_path}: {exc}")

    if agent or as_json:
        print(json_out.render(result, agent=agent))
        return
    if prompt_only_mode:
        print(prompt_only.render(result))
        return

    if panels:
        terminal.render(result, console, session.repo)
    elif stdout_is_a_terminal():
        rich_report.print_report(result, session.repo, full=full)
    else:
        print(markdown.render(result))

    if markdown_path is not None:
        console.print(f"[dim]report written to {markdown_path}[/]")


def finish(result: ReviewResult, fail_on: Severity | None) -> None:
    """Translate the result into an exit code.

    Shares ``blocking_findings`` with the rendered pre-merge block and the forge
    status, so the three can never read the same findings differently. It stays
    keyed on ``fail_on`` rather than ``result.block_on``: the configured default
    exists so a report always has a verdict to state, and letting it move the
    exit code would start failing every CI job that runs roborak without a gate.
    """
    if result.errors or result.status is not ReviewStatus.COMPLETE:
        raise typer.Exit(EXIT_ERROR)
    if fail_on is not None and blocking_findings(result, fail_on):
        raise typer.Exit(EXIT_FINDINGS)
    raise typer.Exit(EXIT_OK)
