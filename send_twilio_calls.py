#!/usr/bin/env python3
"""
Trigger Twilio voice reminders for ClickUp tasks.

This CLI reuses the Telegram reminder service to fetch pending tasks and then
initiates grouped voice calls using the configured Twilio account.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from telegram_reminder_service import ConfigurationError, TelegramReminderService


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call pending ClickUp reminders via Twilio.")
    parser.add_argument(
        "--assignee",
        action="append",
        help="Limit calls to specific assignees. Can be provided multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum amount of tasks to fetch from ClickUp (default: no limit).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned calls without contacting Twilio.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        service = TelegramReminderService.from_environment()
        deliveries = service.send_voice_reminders(
            assignees=args.assignee,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    except ConfigurationError as exc:
        logging.error("Configuration error: %s", exc)
        return 2
    except Exception as exc:  # pragma: no cover - Twilio/network guard
        logging.exception("Failed to dispatch Twilio calls: %s", exc)
        return 1

    if not deliveries:
        logging.info("No Twilio calls queued.")
        return 0

    for entry in deliveries:
        logging.info(
            "Call %s → %s tasks (%s)",
            entry["phone"],
            len(entry["task_ids"]),
            ", ".join(entry["assignees"]),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
