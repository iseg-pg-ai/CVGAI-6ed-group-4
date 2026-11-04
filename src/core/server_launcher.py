"""Cross-platform launcher for llama-server.

Uniform Python entry point that resolves configuration from Settings, environment variables,
and CLI flags.

Ref: Docs/implementation_plan.md, section 1.1 and WP-08.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from src.core.settings import Settings, get_settings


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the llama-server launcher.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="poe serve",
        description="Launch local llama-server for receipt extraction and verifiable Q&A.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Path to the GGUF model file (overrides MODEL / LLAMA_MODEL_PATH in .env).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number for llama-server (default: from .env or 8080).",
    )
    parser.add_argument(
        "-ngl",
        "--ngl",
        type=int,
        default=None,
        dest="ngl",
        help="Number of GPU layers to offload (default: from .env or 0).",
    )
    parser.add_argument(
        "-c",
        "--ctx",
        type=int,
        default=None,
        dest="ctx",
        help="Context size in tokens (default: from .env or 8192).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without starting the server.",
    )
    return parser


def find_llama_server_binary() -> str | None:
    """Find the llama-server executable in PATH.

    Returns:
        Resolved executable path as string, or None if not found.
    """
    return shutil.which("llama-server") or shutil.which("llama-server.exe")


def main(
    argv: Sequence[str] | None = None,
    settings_override: Settings | None = None,
) -> int:
    """Launch llama-server using resolved settings and CLI arguments.

    Args:
        argv: Command line arguments. None uses sys.argv[1:].
        settings_override: Optional explicit Settings instance for testing.

    Returns:
        Process exit code (0 for success, non-zero for error).
    """
    settings = settings_override or get_settings()
    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. Resolve model path
    model_str = args.model or os.getenv("MODEL") or os.getenv("LLAMA_MODEL_PATH")
    if model_str:
        candidate_path = Path(model_str)
        model_path = candidate_path if candidate_path.is_absolute() else (settings.repo_root / candidate_path).resolve()
    else:
        model_path = settings.llama_model_path or (
            settings.repo_root / "models" / "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"
        )

    # 2. Resolve port, ngl, ctx
    port = args.port if args.port is not None else settings.llama_port
    ngl = args.ngl if args.ngl is not None else settings.llama_ngl
    ctx = args.ctx if args.ctx is not None else settings.llama_ctx

    # 3. Locate binary
    binary = find_llama_server_binary()
    if not binary and not args.dry_run:
        print("Error: 'llama-server' executable not found on system PATH.", file=sys.stderr)
        print("Download llama.cpp binaries and ensure 'llama-server' is on your PATH.", file=sys.stderr)
        return 1

    # 4. Check model file existence
    if not model_path.is_file() and not args.dry_run:
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        print(
            "Please configure LLAMA_MODEL_PATH in .env, place your .gguf model in models/,\n"
            "or specify path: poe serve <path/to/model.gguf>",
            file=sys.stderr,
        )
        return 1

    cmd = [
        binary or "llama-server",
        "-m",
        str(model_path),
        "--port",
        str(port),
        "-ngl",
        str(ngl),
        "-c",
        str(ctx),
    ]

    print(f"Starting llama-server with model: {model_path} on port: {port} (NGL: {ngl}, CTX: {ctx})")

    if args.dry_run:
        print(f"Dry run command: {' '.join(cmd)}")
        return 0

    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\nllama-server stopped by user.")
        return 0
    except OSError as err:
        print(f"Error launching llama-server: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
