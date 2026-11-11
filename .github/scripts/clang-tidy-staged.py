"""The pre-commit hook: clang-tidy on the staged C++ lines.

The check is LLVM's clang-tidy-diff, as in CI. This adds what a clone needs
around it:

- the compile database. clang-tidy reads a file as the compiler does, and a
  Visual Studio build has no compile_commands.json. The first run makes one in
  build-tidy/ (a configure, not a build; VAI_TIDY_BUILD names another place).
- on Windows, the diff's paths with `\\`: clang-tidy-diff matches git's `/`
  paths against clang-tidy's `\\` ones and so would pass every change.

Exit codes: 0 clean, or no compile database could be made (the reason is
printed; CI still checks the PR). 1 a finding on a staged line.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

DIRECTORIES = ("vai", "models", "examples")  # what the configure below covers
NOT_OURS = (
    ":!vai/shaders",
    ":!vai/utils/stb_image.h",
    ":!vai/utils/stb_image_write.h",
    ":!vai/utils/tiny-json.h",
)
CONFIGURE = (
    "-G",
    "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-DVAI_BUILD_EXAMPLES=ON",
)
VSWHERE = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def configure(root: str, build: str) -> str | None:
    """Make the compile database. Returns why it could not, or None."""
    command = ["cmake", "-S", root, "-B", build, *CONFIGURE]
    if os.name != "nt":
        if not (shutil.which("cmake") and shutil.which("ninja")):
            return "cmake and ninja are needed on PATH"
        done = subprocess.run(command, capture_output=True, text=True)
    else:
        # Ninja needs the compiler's environment, which only a developer prompt
        # has: borrow it from the newest Visual Studio with the C++ tools.
        if not os.path.exists(VSWHERE):
            return "Visual Studio was not found (no vswhere.exe)"
        found = subprocess.run(
            [VSWHERE, "-latest", "-products", "*", "-property", "installationPath"]
            + ["-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        vcvars = os.path.join(found, "VC", "Auxiliary", "Build", "vcvars64.bat")
        if not found or not os.path.exists(vcvars):
            return "no Visual Studio with the C++ tools was found"
        command += ["-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl"]
        line = f'call "{vcvars}" >nul && {subprocess.list2cmdline(command)}'
        done = subprocess.run(line, shell=True, capture_output=True, text=True)
    if done.returncode != 0 or not os.path.exists(os.path.join(build, "compile_commands.json")):
        return "cmake could not configure:\n" + (done.stdout + done.stderr)[-1500:]
    return None


def main() -> int:
    import clang_tidy  # the wheel the hook installs; it carries clang-tidy-diff.py

    # pre-commit reads the hook's output as UTF-8, whatever the console's own is.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tools = os.path.join(os.path.dirname(clang_tidy.__file__), "data", "bin")
    root = os.path.normpath(git("rev-parse", "--show-toplevel").strip())
    os.chdir(root)

    diff = git("diff", "-U0", "--staged", "--diff-filter=ACMR", "--", *DIRECTORIES, *NOT_OURS)
    if "+++ b/" not in diff:
        return 0

    build = os.environ.get("VAI_TIDY_BUILD") or os.path.join(root, "build-tidy")
    if not os.path.exists(os.path.join(build, "compile_commands.json")):
        print(f"clang-tidy: making the compile database in {build} (once) ...", flush=True)
        if why := configure(root, build):
            print(f"clang-tidy: SKIPPED, {why}")
            print("CI checks the PR all the same. To have it here, configure once:")
            print(f"  cmake -S . -B build-tidy {' '.join(CONFIGURE)}")
            return 0

    if os.name == "nt":
        rows = diff.splitlines(keepends=True)
        diff = "".join(
            "+++ b/" + row[6:].replace("/", "\\") if row.startswith("+++ b/") else row
            for row in rows
        )

    binary = os.path.join(tools, "clang-tidy.exe" if os.name == "nt" else "clang-tidy")
    command = [sys.executable, os.path.join(tools, "clang-tidy-diff.py"), "-p1", "-path", build]
    command += ["-clang-tidy-binary", binary, "-j", str(os.cpu_count() or 1)]
    command += ["-quiet", "-hide-progress", "-warnings-as-errors=*"]
    return subprocess.run(command, input=diff, text=True, encoding="utf-8").returncode


if __name__ == "__main__":
    sys.exit(main())
