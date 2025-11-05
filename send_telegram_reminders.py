#!/usr/bin/env python3
"""
One-off helper that sends ClickUp reminders to Telegram.

Designed for automation environments (e.g. GitHub Actions) where we need to
push the current list of pending reminders before executing the rest of the
workflow.
"""

from __future__ import annotations

import argparse
import logging
import sys

from telegram_reminder_service import ConfigurationError, TelegramReminderService


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send current ClickUp reminders to Telegram.")
    parser.add_argument(
        "--chat-id",
        help="Override target chat id. Falls back to TELEGRAM_CHAT_ID or config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit amount of tasks to send (default: all pending).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        service = TelegramReminderService.from_environment()
        tasks = service.send_reminders(
            chat_id=args.chat_id,
            limit=args.limit,
            broadcast_all=bool(args.chat_id),
        )
    except ConfigurationError as exc:
        logging.error("Configuration error: %s", exc)
        return 2
    except Exception as exc:  # pragma: no cover - defensive automation guard
        logging.exception("Unexpected failure while sending reminders: %s", exc)
        return 1

    logging.info("Telegram reminders dispatched: %s", len(tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
