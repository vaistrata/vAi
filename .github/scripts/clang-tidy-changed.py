"""clang-tidy findings on the lines a branch changed.

    python .github/scripts/clang-tidy-changed.py --build build --base origin/master

Runs clang-tidy (.clang-tidy at the root) over every source of the compile
database, then keeps the findings whose line the branch added or changed since
its merge base with --base. Most of vAi is headers, and a header is only read
through a source that includes it: so every source runs, whatever changed.

--all lists every finding of the tree instead and fails on none of them: what
the tree still has, and whether clang-tidy reads every source.

Exit codes: 0 no finding on a changed line, 1 findings, 2 clang-tidy could not
read a source.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys

EXTENSIONS = ("*.h", "*.hpp", "*.cpp")
# As .clang-format-ignore: not ours to rename.
NOT_OURS = (
    ":!external",
    ":!vai/shaders",
    ":!vai/utils/stb_image.h",
    ":!vai/utils/stb_image_write.h",
    ":!vai/utils/tiny-json.h",
)
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
FINDING = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+): (?P<level>warning|error): (?P<text>.*)$"
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def changed_lines(base: str) -> dict[str, set[int]]:
    """Repo-relative path -> the line numbers the branch added or changed."""
    merge_base = git("merge-base", base, "HEAD").strip()
    diff = git("diff", "-U0", "--diff-filter=ACMR", merge_base, "--", *EXTENSIONS, *NOT_OURS)
    lines: dict[str, set[int]] = {}
    path = None
    for row in diff.splitlines():
        if row.startswith("+++ "):
            path = row[6:] if row.startswith("+++ b/") else None
        elif path and (hunk := HUNK.match(row)):
            start, count = int(hunk[1]), int(hunk[2] or 1)
            lines.setdefault(path, set()).update(range(start, start + count))
    return lines


def sources(build: str, root: str) -> list[str]:
    with open(os.path.join(build, "compile_commands.json"), encoding="utf-8") as f:
        files = {os.path.normpath(entry["file"]) for entry in json.load(f)}
    external = os.path.join(root, "external") + os.sep
    return sorted(f for f in files if f.startswith(root + os.sep) and not f.startswith(external))


def run_tidy(clang_tidy: str, build: str, source: str) -> tuple[str, int, str]:
    done = subprocess.run(
        [clang_tidy, "-p", build, "--quiet", source],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return source, done.returncode, done.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", required=True, help="directory of compile_commands.json")
    parser.add_argument("--base", help="the ref the branch is compared with")
    parser.add_argument("--all", action="store_true", help="list every finding, fail on none")
    parser.add_argument("--clang-tidy", default="clang-tidy")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    if not args.all and not args.base:
        parser.error("--base or --all")

    # A code excerpt may hold what a console's own encoding cannot print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = os.path.normpath(git("rev-parse", "--show-toplevel").strip())
    changed = {} if args.all else changed_lines(args.base)
    if not args.all and not changed:
        print("clang-tidy: no C++ lines changed")
        return 0

    files = sources(args.build, root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        runs = list(pool.map(lambda source: run_tidy(args.clang_tidy, args.build, source), files))

    # A header's finding comes back once per source that includes it: keep one.
    findings: dict[tuple[str, int, int, str], list[str]] = {}
    unread = []
    for source, _, output in runs:
        block = None
        for row in output.splitlines():
            found = FINDING.match(row)
            if not found:
                if block is not None:  # the code excerpt and notes under a finding
                    block.append(row)
                continue
            path = os.path.normpath(os.path.join(root, found["path"]))
            inside = path.startswith(root + os.sep)
            relative = os.path.relpath(path, root).replace(os.sep, "/") if inside else path
            line = int(found["line"])
            block = None
            if "[clang-diagnostic-error]" in found["text"]:
                unread.append(f"{relative}:{line}: {found['text']} (reading {source})")
            elif args.all or line in changed.get(relative, ()):
                key = (relative, line, int(found["col"]), found["text"])
                block = [] if key in findings else findings.setdefault(key, [])

    if unread:
        print("clang-tidy could not read these: its findings would be incomplete.")
        print("\n".join(sorted(set(unread))))
        return 2

    for (relative, line, col, text), excerpt in sorted(findings.items()):
        print(f"{relative}:{line}:{col}: {text}")
        print("\n".join(excerpt))
        if os.environ.get("GITHUB_ACTIONS") and not args.all:
            print(f"::error file={relative},line={line},col={col}::{text}")

    where = "in the tree" if args.all else "on changed lines"
    print(f"clang-tidy: {len(findings)} finding(s) {where}, {len(files)} sources read")
    return 1 if findings and not args.all else 0


if __name__ == "__main__":
    sys.exit(main())
