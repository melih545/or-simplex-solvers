"""Command-line launcher for the local simplex workbench."""

from __future__ import annotations

import argparse
from copy import deepcopy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local simplex workbench.")
    parser.add_argument("--port", type=int, default=8765, help="local HTTP port")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65_535:
        parser.error("--port must be between 0 and 65535.")
    try:
        import uvicorn

        from ._api import app
    except ImportError:
        parser.exit(1, 'Install the workbench dependencies with pip install "simplex-codes[api,viz]".\n')
    log_config = deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["formatters"]["solve_json"] = {"format": "%(message)s"}
    log_config["handlers"]["solve_json"] = {
        "class": "logging.StreamHandler", "formatter": "solve_json", "stream": "ext://sys.stdout",
    }
    log_config["loggers"]["simplex.solve"] = {
        "handlers": ["solve_json"], "level": "INFO", "propagate": False,
    }
    uvicorn.run(app, host=args.host, port=args.port, log_config=log_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
