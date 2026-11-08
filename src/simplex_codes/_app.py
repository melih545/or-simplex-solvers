"""Command-line launcher for the local simplex workbench."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local simplex workbench.")
    parser.add_argument("--port", type=int, default=8765, help="local HTTP port")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65_535:
        parser.error("--port must be between 0 and 65535.")
    try:
        import uvicorn

        from ._api import app
    except ImportError:
        parser.exit(1, 'Install the workbench dependencies with pip install "simplex-codes[api,viz]".\n')
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
