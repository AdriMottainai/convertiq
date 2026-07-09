"""Command-line entry point for ytnotes.

    ytnotes run                         # poll all channels, write new notes
    ytnotes run --dry-run               # print notes instead of writing/state
    ytnotes run --force                 # ignore state (reprocess)
    ytnotes run --video <url> --dry-run # process one video (local verification)
    ytnotes run --video <url> --force-whisper   # exercise the whisper path
"""

from __future__ import annotations

import argparse
import sys

from .config import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_STATE_PATH,
    DEFAULT_VAULT_DIR,
    load_settings,
)
from .pipeline import run, run_single_video


def _load(args) -> object:
    return load_settings(
        args.config_dir,
        state_path=args.state,
        vault_dir=args.vault,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ytnotes", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="poll channels and generate notes")
    run_p.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    run_p.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    run_p.add_argument("--vault", default=str(DEFAULT_VAULT_DIR))
    run_p.add_argument("--dry-run", action="store_true", help="print notes; don't write files or state")
    run_p.add_argument("--force", action="store_true", help="ignore processed-state and reprocess")
    run_p.add_argument("--video", help="process a single video URL/id instead of polling")
    run_p.add_argument("--force-whisper", action="store_true", help="with --video: force the whisper path")

    args = parser.parse_args(argv)

    if args.command == "run":
        settings = _load(args)
        try:
            if args.video:
                run_single_video(settings, args.video, dry_run=args.dry_run, force_whisper=args.force_whisper)
            else:
                run(settings, dry_run=args.dry_run, force=args.force)
        except KeyboardInterrupt:
            print("interrupted", file=sys.stderr)
            return 130
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
