#!/usr/bin/env python3
"""Synchronize GitHub workflow changes across local and GitHub repositories.

The first implemented patch is ``nuget-login``. It updates workflow files that call
``xoofx/.github/.github/actions/dotnet-releaser-action`` so they use NuGet Trusted
Publishing via ``NuGet/login@v1`` and pass the resulting API key to the action.
It deliberately edits caller workflows (for example ``ci.yml``), not the shared
composite ``action.yml``.

The script is intentionally conservative:

* dry-run is the default; pass ``--apply`` to write local files;
* local discovery only includes git repositories with a GitHub remote;
* remote GitHub repositories discovered with ``gh`` are inspected in dry-run mode
  unless a local clone is available;
* workflow edits are targeted text patches, so comments and unrelated formatting are
  preserved as much as possible.
"""

from __future__ import annotations

import argparse
import base64
import codecs
import dataclasses
import difflib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

try:  # Rich is optional; the script remains usable without extra dependencies.
    from rich.console import Console
    from rich.syntax import Syntax

    HAVE_RICH = True
except ImportError:  # pragma: no cover - exercised when rich is not installed.
    Console = None  # type: ignore[assignment]
    Syntax = None  # type: ignore[assignment]
    HAVE_RICH = False


DEFAULT_WORKFLOWS = (".github/workflows/ci.yml", ".github/workflows/ci.yaml")
DEFAULT_ACTION_USES = "xoofx/.github/.github/actions/dotnet-releaser-action"
DEFAULT_REUSABLE_WORKFLOWS = (
    "xoofx/.github/.github/workflows/dotnet.yml",
    "xoofx/.github/.github/workflows/dotnet-multi.yml",
)
DEFAULT_REQUIRED_PERMISSIONS = {
    "id-token": "write",
    # dotnet-releaser commonly creates releases/tags and the reusable workflows in
    # this repository already request these scopes. Keeping them explicit avoids
    # accidentally reducing GITHUB_TOKEN capabilities when a permissions block is
    # introduced for Trusted Publishing.
    "actions": "write",
    "contents": "write",
}


@dataclasses.dataclass(frozen=True)
class RepoIdentity:
    """Best-effort identity for a repository target."""

    label: str
    owner: str | None = None
    name: str | None = None

    @property
    def full_name(self) -> str | None:
        if self.owner and self.name:
            return f"{self.owner}/{self.name}"
        return None


@dataclasses.dataclass(frozen=True)
class RepoTarget:
    """A local or remote repository to inspect."""

    identity: RepoIdentity
    path: Path | None = None
    remote_only: bool = False

    @property
    def label(self) -> str:
        return self.identity.full_name or self.identity.label


@dataclasses.dataclass
class WorkflowChange:
    repo: str
    workflow: str
    status: str
    message: str
    nuget_user: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    repo_path: str = ""

    @property
    def changed(self) -> bool:
        return self.old_text is not None and self.new_text is not None and self.old_text != self.new_text


@dataclasses.dataclass(frozen=True)
class JobBlock:
    name: str
    start: int
    end: int
    indent: int


@dataclasses.dataclass(frozen=True)
class StepsBlock:
    start: int
    end: int
    step_indent: int


@dataclasses.dataclass(frozen=True)
class StepBlock:
    start: int
    end: int
    step_indent: int


@dataclasses.dataclass(frozen=True)
class NugetLoginOptions:
    action_uses: str
    reusable_workflow_uses: tuple[str, ...]
    login_step_id: str
    login_step_name: str
    login_action: str
    required_permissions: dict[str, str]


class Printer:
    def __init__(self) -> None:
        self.console = Console() if HAVE_RICH else None

    def print(self, message: str = "") -> None:
        if self.console:
            self.console.print(message)
        else:
            print(strip_rich_markup(message))

    def print_table(self, rows: Sequence[WorkflowChange]) -> None:
        headers = ("Repo", "Path", "Workflow", "User", "Status", "Message")
        print(markdown_row(headers))
        print(markdown_row("---" for _ in headers))
        for row in rows:
            print(
                markdown_row(
                    (
                        row.repo,
                        row.repo_path or "—",
                        row.workflow,
                        row.nuget_user or "",
                        row.status,
                        row.message,
                    )
                )
            )

    def print_diff(self, title: str, diff_text: str) -> None:
        if not diff_text:
            return
        if self.console and Syntax:
            self.console.print(f"\n[bold]{title}[/bold]")
            self.console.print(Syntax(diff_text, "diff", word_wrap=False))
        else:
            print(f"\n{title}")
            print(diff_text)


def strip_rich_markup(value: str) -> str:
    return re.sub(r"\[/?[a-zA-Z0-9_ #=;:-]+\]", "", value)


def markdown_cell(value: object) -> str:
    text = str(value)
    return text.replace("|", r"\|").replace("\r\n", "<br>").replace("\n", "<br>")


def markdown_row(values: Iterable[object]) -> str:
    return "| " + " | ".join(markdown_cell(value) for value in values) + " |"


def run_command(args: Sequence[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(args),
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        joined = " ".join(args)
        raise RuntimeError(f"{joined} failed with exit code {result.returncode}: {result.stderr.strip()}")
    return result


def count_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def parse_key_line(line: str, indent: int | None = None) -> tuple[str, str] | None:
    if indent is not None and count_indent(line) != indent:
        return None
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return None
    match = re.match(r"([A-Za-z0-9_.-]+)\s*:\s*(.*)$", stripped)
    if not match:
        return None
    return match.group(1), match.group(2)


def split_text(text: str) -> tuple[list[str], str, bool]:
    newline = "\r\n" if "\r\n" in text else "\n"
    had_trailing_newline = text.endswith("\n")
    return text.splitlines(), newline, had_trailing_newline


def join_text(lines: Sequence[str], newline: str, had_trailing_newline: bool) -> str:
    result = newline.join(lines)
    if had_trailing_newline:
        result += newline
    return result


def next_block_end(lines: Sequence[str], start: int, parent_indent: int, limit: int | None = None) -> int:
    limit = len(lines) if limit is None else limit
    i = start + 1
    while i < limit:
        line = lines[i]
        if line.strip() and count_indent(line) <= parent_indent:
            break
        i += 1
    return i


def find_root_key(lines: Sequence[str], key: str) -> int | None:
    for index, line in enumerate(lines):
        parsed = parse_key_line(line, 0)
        if parsed and parsed[0] == key:
            return index
    return None


def find_jobs(lines: Sequence[str]) -> list[JobBlock]:
    jobs_index = find_root_key(lines, "jobs")
    if jobs_index is None:
        return []

    jobs_end = next_block_end(lines, jobs_index, 0)
    jobs: list[JobBlock] = []
    i = jobs_index + 1
    while i < jobs_end:
        parsed = parse_key_line(lines[i], 2)
        if parsed:
            name, value = parsed
            # A job key has a mapping value. Skip malformed or inline scalar nodes.
            if not value or value.startswith("#"):
                start = i
                end = jobs_end
                j = i + 1
                while j < jobs_end:
                    if parse_key_line(lines[j], 2):
                        end = j
                        break
                    j += 1
                jobs.append(JobBlock(name=name, start=start, end=end, indent=2))
                i = end
                continue
        i += 1
    return jobs


def find_job_by_name(lines: Sequence[str], name: str) -> JobBlock | None:
    return next((job for job in find_jobs(lines) if job.name == name), None)


def find_steps_block(lines: Sequence[str], job: JobBlock) -> StepsBlock | None:
    child_indent = job.indent + 2
    for i in range(job.start + 1, job.end):
        parsed = parse_key_line(lines[i], child_indent)
        if parsed and parsed[0] == "steps":
            return StepsBlock(start=i, end=next_block_end(lines, i, child_indent, job.end), step_indent=child_indent + 2)
    return None


def iter_step_blocks(lines: Sequence[str], steps: StepsBlock) -> Iterable[StepBlock]:
    i = steps.start + 1
    while i < steps.end:
        if count_indent(lines[i]) == steps.step_indent and lines[i].lstrip().startswith("- "):
            start = i
            end = steps.end
            j = i + 1
            while j < steps.end:
                if count_indent(lines[j]) == steps.step_indent and lines[j].lstrip().startswith("- "):
                    end = j
                    break
                j += 1
            yield StepBlock(start=start, end=end, step_indent=steps.step_indent)
            i = end
        else:
            i += 1


def normalize_uses(value: str) -> str:
    value = value.strip().strip("'\"")
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    return value


def uses_line_value(line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("- "):
        stripped = stripped[2:].strip()
    match = re.match(r"uses\s*:\s*(.+?)\s*$", stripped)
    if not match:
        return None
    return normalize_uses(match.group(1))


def step_uses(lines: Sequence[str], step: StepBlock, prefix: str) -> bool:
    for i in range(step.start, step.end):
        if count_indent(lines[i]) not in {step.step_indent, step.step_indent + 2}:
            continue
        value = uses_line_value(lines[i])
        if value and value.lower().startswith(prefix.lower()):
            return True
    return False


def workflow_uses_prefix(text: str, prefix: str) -> bool:
    lowered = prefix.lower()
    for line in text.splitlines():
        value = uses_line_value(line)
        if value and value.lower().startswith(lowered):
            return True
    return False


def find_action_jobs(lines: Sequence[str], action_uses: str) -> list[str]:
    names: list[str] = []
    for job in find_jobs(lines):
        steps = find_steps_block(lines, job)
        if not steps:
            continue
        if any(step_uses(lines, step, action_uses) for step in iter_step_blocks(lines, steps)):
            names.append(job.name)
    return names


def collect_step_ids(lines: Sequence[str], steps: StepsBlock) -> set[str]:
    ids: set[str] = set()
    property_indent = steps.step_indent + 2
    for step in iter_step_blocks(lines, steps):
        for i in range(step.start, step.end):
            parsed = parse_key_line(lines[i], property_indent)
            if parsed and parsed[0] == "id":
                value = parsed[1].split("#", 1)[0].strip().strip("'\"")
                if value:
                    ids.add(value)
    return ids


def unique_step_id(desired: str, existing_ids: set[str]) -> str:
    if desired not in existing_ids:
        return desired
    counter = 2
    while f"{desired}-{counter}" in existing_ids:
        counter += 1
    return f"{desired}-{counter}"


def find_step_property(lines: Sequence[str], step: StepBlock, key: str) -> int | None:
    property_indent = step.step_indent + 2
    for i in range(step.start, step.end):
        parsed = parse_key_line(lines[i], property_indent)
        if parsed and parsed[0] == key:
            return i
    return None


def find_step_uses_line(lines: Sequence[str], step: StepBlock) -> int | None:
    for i in range(step.start, step.end):
        if count_indent(lines[i]) not in {step.step_indent, step.step_indent + 2}:
            continue
        if uses_line_value(lines[i]) is not None:
            return i
    return None


def get_step_id(lines: Sequence[str], step: StepBlock) -> str | None:
    id_line = find_step_property(lines, step, "id")
    if id_line is None:
        return None
    parsed = parse_key_line(lines[id_line], step.step_indent + 2)
    if not parsed:
        return None
    return parsed[1].split("#", 1)[0].strip().strip("'\"") or None


def ensure_step_id(lines: list[str], step: StepBlock, step_id: str) -> bool:
    current_id_line = find_step_property(lines, step, "id")
    property_indent = step.step_indent + 2
    if current_id_line is not None:
        desired_line = " " * property_indent + f"id: {step_id}"
        if lines[current_id_line] != desired_line:
            lines[current_id_line] = desired_line
            return True
        return False

    uses_index = find_step_uses_line(lines, step)
    insert_at = (uses_index + 1) if uses_index is not None else (step.start + 1)
    lines.insert(insert_at, " " * property_indent + f"id: {step_id}")
    return True


def find_with_block(lines: Sequence[str], step: StepBlock) -> tuple[int, int, int] | None:
    property_indent = step.step_indent + 2
    for i in range(step.start, step.end):
        parsed = parse_key_line(lines[i], property_indent)
        if parsed and parsed[0] == "with":
            return i, next_block_end(lines, i, property_indent, step.end), property_indent
    return None


def set_step_with_value(lines: list[str], step: StepBlock, key: str, value: str) -> bool:
    property_indent = step.step_indent + 2
    child_indent = property_indent + 2
    with_block = find_with_block(lines, step)
    if with_block is None:
        lines.insert(step.end, " " * property_indent + "with:")
        lines.insert(step.end + 1, " " * child_indent + f"{key}: {value}")
        return True

    with_index, with_end, _ = with_block
    key_line_index: int | None = None
    for i in range(with_index + 1, with_end):
        parsed = parse_key_line(lines[i], child_indent)
        if parsed and parsed[0] == key:
            key_line_index = i
            break

    desired_line = " " * child_indent + f"{key}: {value}"
    if key_line_index is not None:
        if lines[key_line_index] == desired_line:
            return False
        lines[key_line_index] = desired_line
        return True

    lines.insert(with_end, desired_line)
    return True


def find_nuget_login_steps(lines: Sequence[str], job: JobBlock, login_action: str) -> list[StepBlock]:
    steps = find_steps_block(lines, job)
    if not steps:
        return []
    return [step for step in iter_step_blocks(lines, steps) if step_uses(lines, step, login_action)]


def find_action_steps(lines: Sequence[str], job: JobBlock, action_uses: str) -> list[StepBlock]:
    steps = find_steps_block(lines, job)
    if not steps:
        return []
    return [step for step in iter_step_blocks(lines, steps) if step_uses(lines, step, action_uses)]


def ensure_permissions_mapping(
    lines: list[str],
    permissions_index: int,
    indent: int,
    required_permissions: dict[str, str],
) -> tuple[bool, str | None]:
    parsed = parse_key_line(lines[permissions_index], indent)
    if not parsed or parsed[0] != "permissions":
        return False, "internal error: permissions block not found"

    inline_value = parsed[1].split("#", 1)[0].strip()
    if inline_value:
        return False, f"permissions is scalar ({inline_value!r}); not rewriting it automatically"

    changed = False
    child_indent = indent + 2
    block_end = next_block_end(lines, permissions_index, indent)
    existing: dict[str, int] = {}
    for i in range(permissions_index + 1, block_end):
        parsed_child = parse_key_line(lines[i], child_indent)
        if parsed_child:
            existing[parsed_child[0]] = i

    for key, value in required_permissions.items():
        desired_line = " " * child_indent + f"{key}: {value}"
        existing_index = existing.get(key)
        if existing_index is None:
            lines.insert(block_end, desired_line)
            block_end += 1
            changed = True
            continue

        parsed_child = parse_key_line(lines[existing_index], child_indent)
        current_value = parsed_child[1].split("#", 1)[0].strip() if parsed_child else ""
        if current_value != value:
            # Trusted publishing requires id-token: write. The defaults for this
            # patch also intentionally keep actions/contents at write.
            lines[existing_index] = desired_line
            changed = True

    return changed, None


def ensure_job_permissions(lines: list[str], job_name: str, required_permissions: dict[str, str]) -> tuple[bool, str | None]:
    job = find_job_by_name(lines, job_name)
    if job is None:
        return False, f"job {job_name!r} disappeared while patching"

    child_indent = job.indent + 2
    for i in range(job.start + 1, job.end):
        parsed = parse_key_line(lines[i], child_indent)
        if parsed and parsed[0] == "permissions":
            return ensure_permissions_mapping(lines, i, child_indent, required_permissions)

    insert_at = job.start + 1
    lines.insert(insert_at, " " * child_indent + "permissions:")
    for offset, (key, value) in enumerate(required_permissions.items(), start=1):
        lines.insert(insert_at + offset, " " * (child_indent + 2) + f"{key}: {value}")
    return True, None


def job_has_permissions(lines: Sequence[str], job_name: str) -> bool:
    job = find_job_by_name(lines, job_name)
    if job is None:
        return False
    child_indent = job.indent + 2
    for i in range(job.start + 1, job.end):
        parsed = parse_key_line(lines[i], child_indent)
        if parsed and parsed[0] == "permissions":
            return True
    return False


def ensure_workflow_permissions(lines: list[str], required_permissions: dict[str, str]) -> tuple[bool, str | None, bool]:
    """Update top-level permissions if present.

    Returns (changed, message, handled). handled is False when there is no top-level
    permissions block and callers should add permissions at job level instead.
    """

    permissions_index = find_root_key(lines, "permissions")
    if permissions_index is None:
        return False, None, False
    changed, message = ensure_permissions_mapping(lines, permissions_index, 0, required_permissions)
    if message:
        # A scalar top-level permissions value such as read-all or write-all is
        # valid YAML/GitHub Actions syntax. Leave it untouched and let callers add
        # a job-level mapping instead of broadening or narrowing it implicitly.
        return False, None, False
    return changed, message, True


def ensure_login_step(
    lines: list[str],
    job_name: str,
    nuget_user: str,
    options: NugetLoginOptions,
) -> tuple[str | None, bool, str | None]:
    job = find_job_by_name(lines, job_name)
    if job is None:
        return None, False, f"job {job_name!r} disappeared while inserting NuGet login"

    steps = find_steps_block(lines, job)
    if not steps:
        return None, False, f"job {job_name!r} has no steps block"

    action_steps = find_action_steps(lines, job, options.action_uses)
    if not action_steps:
        return None, False, f"job {job_name!r} no longer contains the dotnet-releaser action"

    first_action_start = min(step.start for step in action_steps)
    login_steps = find_nuget_login_steps(lines, job, options.login_action)
    existing_before_action = [step for step in login_steps if step.start < first_action_start]
    changed = False

    if existing_before_action:
        login_step = existing_before_action[0]
        step_id = get_step_id(lines, login_step)
        if step_id is None:
            step_ids = collect_step_ids(lines, steps)
            step_id = unique_step_id(options.login_step_id, step_ids)
            changed |= ensure_step_id(lines, login_step, step_id)
            # Recompute after insertion.
            job = find_job_by_name(lines, job_name)
            assert job is not None
            login_step = find_nuget_login_steps(lines, job, options.login_action)[0]
        changed |= set_step_with_value(lines, login_step, "user", nuget_user)
        return step_id, changed, None

    if login_steps:
        return (
            None,
            False,
            "NuGet/login step exists after the dotnet-releaser action; refusing to reorder steps automatically",
        )

    step_ids = collect_step_ids(lines, steps)
    step_id = unique_step_id(options.login_step_id, step_ids)
    indent = " " * steps.step_indent
    login_lines = [
        f"{indent}- name: {options.login_step_name}",
        f"{indent}  uses: {options.login_action}",
        f"{indent}  id: {step_id}",
        f"{indent}  with:",
        f"{indent}    user: {nuget_user}",
    ]
    lines[first_action_start:first_action_start] = login_lines
    return step_id, True, None


def update_action_tokens(lines: list[str], job_name: str, step_id: str, options: NugetLoginOptions) -> tuple[bool, str | None]:
    job = find_job_by_name(lines, job_name)
    if job is None:
        return False, f"job {job_name!r} disappeared while updating NUGET_TOKEN"

    action_steps = find_action_steps(lines, job, options.action_uses)
    if not action_steps:
        return False, f"job {job_name!r} no longer contains the dotnet-releaser action"

    changed = False
    token_expression = "${{ steps." + step_id + ".outputs.NUGET_API_KEY }}"
    for step in sorted(action_steps, key=lambda s: s.start, reverse=True):
        changed |= set_step_with_value(lines, step, "NUGET_TOKEN", token_expression)
    return changed, None


def apply_nuget_login_patch(text: str, nuget_user: str, options: NugetLoginOptions) -> tuple[str, str, list[str]]:
    """Patch one workflow file.

    Returns ``(new_text, status, messages)``. status is one of ``changed``,
    ``unchanged``, or ``skipped``.
    """

    lines, newline, had_trailing_newline = split_text(text)
    action_job_names = find_action_jobs(lines, options.action_uses)
    if not action_job_names:
        for reusable in options.reusable_workflow_uses:
            if workflow_uses_prefix(text, reusable):
                return (
                    text,
                    "skipped",
                    [
                        "uses a reusable xoofx/.github workflow rather than the composite action; "
                        "this patch only rewrites direct action steps"
                    ],
                )
        return text, "skipped", ["dotnet-releaser composite action pattern not found"]

    messages: list[str] = []
    changed = False

    permissions_changed, permission_message, top_level_permissions_handled = ensure_workflow_permissions(
        lines, options.required_permissions
    )
    changed |= permissions_changed
    if permission_message:
        return text, "skipped", [permission_message]

    # Process jobs from bottom to top. Insertions within a lower job then do not
    # invalidate the start indexes of jobs above it; each helper also recomputes by
    # job name before editing.
    for job_name in reversed(action_job_names):
        if not top_level_permissions_handled or job_has_permissions(lines, job_name):
            job_permissions_changed, job_permission_message = ensure_job_permissions(
                lines, job_name, options.required_permissions
            )
            changed |= job_permissions_changed
            if job_permission_message:
                return text, "skipped", [job_permission_message]

        step_id, login_changed, login_message = ensure_login_step(lines, job_name, nuget_user, options)
        changed |= login_changed
        if login_message:
            return text, "skipped", [login_message]
        if step_id is None:
            return text, "skipped", [f"could not determine NuGet login step id for job {job_name!r}"]

        token_changed, token_message = update_action_tokens(lines, job_name, step_id, options)
        changed |= token_changed
        if token_message:
            return text, "skipped", [token_message]

    if changed:
        messages.append(
            "added/updated NuGet Trusted Publishing login and NUGET_TOKEN wiring "
            f"for job(s): {', '.join(action_job_names)}"
        )
    else:
        messages.append("already configured for NuGet Trusted Publishing")

    return join_text(lines, newline, had_trailing_newline), "changed" if changed else "unchanged", messages


def parse_github_remote_url(url: str) -> tuple[str, str] | None:
    url = url.strip()
    patterns = [
        r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$",
        r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$",
        r"ssh://git@github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group("owner"), match.group("repo")
    return None


def local_repo_identity(path: Path) -> RepoIdentity:
    urls: list[str] = []
    try:
        result = run_command(["git", "remote", "get-url", "origin"], cwd=path, check=False)
        if result.returncode == 0:
            urls.append(result.stdout.strip())

        result = run_command(["git", "remote", "-v"], cwd=path, check=False)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    urls.append(parts[1])
    except OSError:
        pass

    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        parsed = parse_github_remote_url(url)
        if parsed:
            owner, name = parsed
            return RepoIdentity(label=str(path), owner=owner, name=name)

    # A repository without a GitHub remote is intentionally left without a
    # full_name so discovery can ignore it. This avoids surfacing unrelated local
    # git repositories in the synchronization report.
    return RepoIdentity(label=str(path))


def discover_git_repos(root: Path, max_depth: int) -> list[Path]:
    root = root.resolve()
    repos: list[Path] = []
    if not root.exists():
        return repos

    for current, dirs, _files in os.walk(root):
        current_path = Path(current)
        try:
            relative = current_path.relative_to(root)
        except ValueError:
            continue
        depth = 0 if str(relative) == "." else len(relative.parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        if ".git" in dirs:
            repos.append(current_path)
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in {".git", "bin", "obj", "node_modules", ".vs", ".idea"}]
    return sorted(repos, key=lambda p: str(p).lower())


def is_owner_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value.strip()))


def load_repo_list(path: Path) -> list[str]:
    result: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        result.append(line)
    return result


def list_github_owner_repos(owner: str, limit: int, include_archived: bool) -> list[str]:
    args = [
        "gh",
        "repo",
        "list",
        owner,
        "--limit",
        str(limit),
        "--json",
        "nameWithOwner,isArchived",
    ]
    result = run_command(args)
    data = json.loads(result.stdout or "[]")
    repos: list[str] = []
    for item in data:
        if item.get("isArchived") and not include_archived:
            continue
        name_with_owner = item.get("nameWithOwner")
        if name_with_owner:
            repos.append(name_with_owner)
    return repos


def clone_repo(full_name: str, clone_root: Path) -> Path:
    owner, name = full_name.split("/", 1)
    destination = clone_root / owner / name
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command(["gh", "repo", "clone", full_name, str(destination)])
    return destination


def resolve_targets(args: argparse.Namespace) -> list[RepoTarget]:
    explicit_values: list[str] = []
    explicit_values.extend(args.repo or [])
    for list_file in args.repo_list or []:
        explicit_values.extend(load_repo_list(Path(list_file)))

    discover_roots = [Path(p) for p in (args.discover_root or [])]
    if (not explicit_values or args.github_owner) and not args.no_default_discovery:
        # This repository lives at C:\code\.github in the author's setup; the
        # parent directory is a useful, safe dry-run default for local clones. Keep
        # discovering it when --github-owner is used so existing clones are reused
        # instead of treated as remote-only/missing.
        discover_roots.append(Path(__file__).resolve().parent.parent)

    local_paths: dict[Path, RepoTarget] = {}
    remote_names: set[str] = set()

    for value in explicit_values:
        if is_owner_name(value):
            remote_names.add(value)
            continue
        path = Path(value).expanduser().resolve()
        identity = local_repo_identity(path)
        if identity.full_name:
            local_paths[path] = RepoTarget(identity=identity, path=path)

    for root in discover_roots:
        for repo_path in discover_git_repos(root.expanduser(), args.max_depth):
            identity = local_repo_identity(repo_path)
            if identity.full_name:
                local_paths.setdefault(repo_path.resolve(), RepoTarget(identity=identity, path=repo_path.resolve()))

    for owner in args.github_owner or []:
        for full_name in list_github_owner_repos(owner, args.github_limit, args.include_archived):
            remote_names.add(full_name)

    local_by_full_name: dict[str, RepoTarget] = {}
    for target in local_paths.values():
        if target.identity.full_name:
            local_by_full_name[target.identity.full_name] = target

    targets = list(local_paths.values())
    existing_paths = {target.path for target in targets if target.path is not None}
    for full_name in sorted(remote_names):
        if full_name in local_by_full_name:
            continue
        if args.clone_missing:
            clone_root = Path(args.clone_root).expanduser().resolve()
            cloned_path = clone_repo(full_name, clone_root)
            if cloned_path.resolve() not in existing_paths:
                identity = local_repo_identity(cloned_path)
                if identity.full_name:
                    targets.append(RepoTarget(identity=identity, path=cloned_path.resolve()))
                    existing_paths.add(cloned_path.resolve())
            continue
        owner, name = full_name.split("/", 1)
        targets.append(RepoTarget(identity=RepoIdentity(label=full_name, owner=owner, name=name), remote_only=True))

    return sorted(targets, key=lambda target: target.label.lower())


def parse_user_maps(values: Sequence[str] | None) -> dict[str, str]:
    mapping = {"XenoAtom": "XenoAtom"}
    for raw_value in values or []:
        if "=" not in raw_value:
            raise ValueError(f"--user-map expects OWNER=USER, got {raw_value!r}")
        owner, user = raw_value.split("=", 1)
        owner = owner.strip()
        user = user.strip()
        if not owner or not user:
            raise ValueError(f"--user-map expects OWNER=USER, got {raw_value!r}")
        mapping[owner] = user
    return mapping


def infer_nuget_user(target: RepoTarget, user_map: dict[str, str], default_user: str) -> str:
    if target.identity.owner and target.identity.owner in user_map:
        return user_map[target.identity.owner]
    if target.path:
        parts = {part.lower(): part for part in target.path.parts}
        for owner, user in user_map.items():
            if owner.lower() in parts:
                return user
    return default_user


def parse_required_permissions(values: Sequence[str] | None) -> dict[str, str]:
    permissions = dict(DEFAULT_REQUIRED_PERMISSIONS)
    for raw_value in values or []:
        if "=" not in raw_value:
            raise ValueError(f"--permission expects KEY=VALUE, got {raw_value!r}")
        key, value = raw_value.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"--permission expects KEY=VALUE, got {raw_value!r}")
        permissions[key] = value
    if permissions.get("id-token") != "write":
        raise ValueError("nuget-login patch requires permission id-token=write")
    return permissions


def read_remote_workflow(full_name: str, workflow_path: str, ref: str) -> str | None:
    api_path = f"repos/{full_name}/contents/{workflow_path}"
    args = ["gh", "api", "--method", "GET", api_path]
    if ref:
        args.extend(["-f", f"ref={ref}"])
    result = run_command(args, check=False)
    if result.returncode != 0:
        return None
    data = json.loads(result.stdout)
    if isinstance(data, list) or data.get("type") != "file" or "content" not in data:
        return None
    encoded = data["content"]
    if data.get("encoding") != "base64":
        return None
    return base64.b64decode(encoded).decode("utf-8")


def unified_diff(old_text: str, new_text: str, from_file: str, to_file: str) -> str:
    # Normalize display-only diffs to LF so CRLF workflow files do not render with
    # extra blank lines in terminals. The actual patcher preserves the original
    # newline style when writing files.
    old_lines = [line + "\n" for line in old_text.splitlines()]
    new_lines = [line + "\n" for line in new_text.splitlines()]
    return "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=from_file,
            tofile=to_file,
        )
    )


def read_utf8_preserve_bom(path: Path) -> tuple[str, bool]:
    raw = path.read_bytes()
    has_bom = raw.startswith(codecs.BOM_UTF8)
    if has_bom:
        raw = raw[len(codecs.BOM_UTF8) :]
    return raw.decode("utf-8"), has_bom


def write_utf8_preserve_bom(path: Path, text: str, has_bom: bool) -> None:
    raw = text.encode("utf-8")
    if has_bom:
        raw = codecs.BOM_UTF8 + raw
    path.write_bytes(raw)


def repo_path_label(target: RepoTarget) -> str:
    return str(target.path) if target.path else "—"


def process_local_workflow(
    target: RepoTarget,
    workflow_path: str,
    nuget_user: str,
    options: NugetLoginOptions,
    apply: bool,
) -> WorkflowChange:
    assert target.path is not None
    repo_path = repo_path_label(target)
    absolute_workflow = target.path / workflow_path
    if not absolute_workflow.exists():
        return WorkflowChange(
            target.label,
            workflow_path,
            "missing",
            "workflow file not found",
            nuget_user,
            repo_path=repo_path,
        )

    old_text, has_bom = read_utf8_preserve_bom(absolute_workflow)
    new_text, patch_status, messages = apply_nuget_login_patch(old_text, nuget_user, options)
    message = "; ".join(messages)
    if patch_status == "changed":
        if apply:
            write_utf8_preserve_bom(absolute_workflow, new_text, has_bom)
            message = "wrote workflow update"
        else:
            message = "would update workflow"
        return WorkflowChange(
            target.label,
            workflow_path,
            "changed",
            message,
            nuget_user,
            old_text,
            new_text,
            repo_path=repo_path,
        )
    return WorkflowChange(target.label, workflow_path, patch_status, message, nuget_user, repo_path=repo_path)


def process_remote_workflow(
    target: RepoTarget,
    workflow_path: str,
    nuget_user: str,
    options: NugetLoginOptions,
    ref: str,
) -> WorkflowChange:
    full_name = target.identity.full_name
    repo_path = repo_path_label(target)
    if full_name is None:
        return WorkflowChange(
            target.label,
            workflow_path,
            "error",
            "remote target has no owner/name",
            nuget_user,
            repo_path=repo_path,
        )
    old_text = read_remote_workflow(full_name, workflow_path, ref)
    if old_text is None:
        return WorkflowChange(
            target.label,
            workflow_path,
            "missing",
            "workflow file not found via gh",
            nuget_user,
            repo_path=repo_path,
        )
    new_text, patch_status, messages = apply_nuget_login_patch(old_text, nuget_user, options)
    if patch_status == "changed":
        return WorkflowChange(
            target.label,
            workflow_path,
            "changed",
            "would update remote workflow; clone locally or use --clone-missing with --apply to write",
            nuget_user,
            old_text,
            new_text,
            repo_path=repo_path,
        )
    return WorkflowChange(target.label, workflow_path, patch_status, "; ".join(messages), nuget_user, repo_path=repo_path)


def process_targets(
    targets: Sequence[RepoTarget],
    workflows: Sequence[str],
    user_map: dict[str, str],
    default_nuget_user: str,
    options: NugetLoginOptions,
    apply: bool,
    remote_ref: str,
) -> list[WorkflowChange]:
    rows: list[WorkflowChange] = []
    for target in targets:
        nuget_user = infer_nuget_user(target, user_map, default_nuget_user)
        for workflow_path in workflows:
            try:
                if target.remote_only:
                    rows.append(process_remote_workflow(target, workflow_path, nuget_user, options, remote_ref))
                else:
                    rows.append(process_local_workflow(target, workflow_path, nuget_user, options, apply))
            except Exception as ex:  # Keep processing other repositories.
                rows.append(
                    WorkflowChange(
                        target.label,
                        workflow_path,
                        "error",
                        str(ex),
                        nuget_user,
                        repo_path=repo_path_label(target),
                    )
                )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize GitHub workflow patches across repositories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        default=argparse.SUPPRESS,
        help="preview changes without writing files (the default)",
    )
    mode.add_argument("--apply", dest="apply", action="store_true", help="write changes to local workflow files")
    parser.set_defaults(apply=False)

    parser.add_argument("--patch", choices=["nuget-login"], default="nuget-login", help="patch to apply")
    parser.add_argument("--repo", action="append", help="local repository path or GitHub owner/name; can be repeated")
    parser.add_argument("--repo-list", action="append", help="file containing repository paths or owner/name values")
    parser.add_argument("--discover-root", action="append", help="discover local git repositories under this directory")
    parser.add_argument("--no-default-discovery", action="store_true", help="do not auto-discover repositories under the script parent directory")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="maximum depth for local repository discovery (2 covers ROOT\\repo and ROOT\\group\\repo)",
    )
    parser.add_argument(
        "--workflow",
        action="append",
        help="workflow path inside each repository (defaults to ci.yml and ci.yaml)",
    )

    parser.add_argument("--github-owner", action="append", help="discover repositories with gh repo list OWNER")
    parser.add_argument("--github-limit", type=int, default=1000, help="maximum repositories per GitHub owner")
    parser.add_argument("--include-archived", action="store_true", help="include archived repositories from gh discovery")
    parser.add_argument("--remote-ref", default="", help="ref used when reading remote workflows through gh api")
    parser.add_argument("--clone-missing", action="store_true", help="clone owner/name targets that are not already local")
    parser.add_argument("--clone-root", default=str(Path(__file__).resolve().parent.parent), help="root used with --clone-missing")

    parser.add_argument("--default-nuget-user", default="xoofx", help="NuGet username for repositories not matched by --user-map")
    parser.add_argument("--user-map", action="append", help="map GitHub owner or path segment to NuGet user, e.g. XenoAtom=XenoAtom")
    parser.add_argument("--login-step-id", default="nuget-login", help="step id for inserted NuGet/login step")
    parser.add_argument("--login-step-name", default="NuGet Login", help="name for inserted NuGet/login step")
    parser.add_argument("--login-action", default="NuGet/login@v1", help="NuGet login action reference")
    parser.add_argument("--action-uses", default=DEFAULT_ACTION_USES, help="composite action uses prefix to patch")
    parser.add_argument(
        "--reusable-workflow-uses",
        action="append",
        help="reusable workflow uses prefix reported as unsupported by this patch (defaults to xoofx dotnet workflows)",
    )
    parser.add_argument(
        "--permission",
        action="append",
        help="permission KEY=VALUE required for patched jobs; defaults include id-token/actions/contents write",
    )
    parser.add_argument("--show-diff", action="store_true", help="print unified diffs for changed workflows")
    parser.add_argument("--only-changed", action="store_true", help="only show changed/error rows in the markdown summary table")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    printer = Printer()

    try:
        user_map = parse_user_maps(args.user_map)
        required_permissions = parse_required_permissions(args.permission)
        targets = resolve_targets(args)
    except Exception as ex:
        printer.print(f"[red]error:[/red] {ex}")
        return 2

    if not targets:
        printer.print("[yellow]No repository targets found.[/yellow]")
        return 1

    apply = bool(args.apply)
    workflows = tuple(args.workflow or DEFAULT_WORKFLOWS)
    reusable_workflow_uses = tuple(args.reusable_workflow_uses or DEFAULT_REUSABLE_WORKFLOWS)
    options = NugetLoginOptions(
        action_uses=args.action_uses,
        reusable_workflow_uses=reusable_workflow_uses,
        login_step_id=args.login_step_id,
        login_step_name=args.login_step_name,
        login_action=args.login_action,
        required_permissions=required_permissions,
    )

    rows = process_targets(
        targets=targets,
        workflows=workflows,
        user_map=user_map,
        default_nuget_user=args.default_nuget_user,
        options=options,
        apply=apply,
        remote_ref=args.remote_ref,
    )

    visible_rows = rows
    if args.only_changed:
        visible_rows = [row for row in rows if row.status in {"changed", "error"}]
    printer.print_table(visible_rows)

    if args.show_diff:
        for row in rows:
            if row.changed:
                diff_text = unified_diff(
                    row.old_text or "",
                    row.new_text or "",
                    f"{row.repo}/{row.workflow}",
                    f"{row.repo}/{row.workflow}",
                )
                printer.print_diff(f"Diff: {row.repo}/{row.workflow}", diff_text)

    changed_count = sum(1 for row in rows if row.status == "changed")
    error_count = sum(1 for row in rows if row.status == "error")
    if apply:
        printer.print(f"\nApplied changes to {changed_count} workflow(s); {error_count} error(s).")
    else:
        printer.print(f"\nDry run: {changed_count} workflow(s) would change; {error_count} error(s).")
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
