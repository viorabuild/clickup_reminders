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
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help=(
            "Poll Telegram for callbacks before and after sending reminders during this duration (seconds). "
            "Pass 0 to skip polling."
        ),
    )
    parser.add_argument(
        "--final-poll-seconds",
        type=float,
        default=30.0,
        help=(
            "Extra polling window (seconds) after the main run to wait for user actions "
            "before exiting. Pass 0 to disable the final wait."
        ),
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
        service.ensure_callback_comments()
        processed_total = 0
        if args.poll_seconds > 0:
            logging.info(
                "Polling Telegram for callbacks during %.1f seconds before sending reminders...",
                args.poll_seconds,
            )
            processed_before = service.poll_updates_for(duration=args.poll_seconds)
            logging.info("Processed updates before sending: %s", processed_before)
            processed_total += processed_before

        tasks = service.send_reminders(
            chat_id=args.chat_id,
            limit=args.limit,
            broadcast_all=False,
        )
        if args.poll_seconds > 0:
            logging.info(
                "Polling Telegram for callbacks during %.1f seconds after sending reminders...",
                args.poll_seconds,
            )
            processed_after = service.poll_updates_for(duration=args.poll_seconds)
            logging.info("Processed updates after sending: %s", processed_after)
            processed_total += processed_after

        if args.final_poll_seconds > 0:
            logging.info(
                "Final polling window for %.1f seconds to capture late user actions...",
                args.final_poll_seconds,
            )
            processed_final = service.poll_updates_for(duration=args.final_poll_seconds)
            logging.info("Processed updates during final wait: %s", processed_final)
            processed_total += processed_final

        if processed_total == 0:
            logging.info("No Telegram actions detected; triggering Twilio voice fallback for Alex.")
            try:
                deliveries = service.send_voice_reminders(assignees=["Alex"], limit=args.limit)
            except ConfigurationError as exc:
                logging.warning("Twilio fallback skipped due to configuration issue: %s", exc)
            except Exception as exc:  # pragma: no cover - defensive automation guard
                logging.exception("Twilio fallback failed: %s", exc)
            else:
                if deliveries:
                    logging.info("Twilio fallback completed: %s call(s) initiated.", len(deliveries))
                    summary_lines = [
                        "📞 Alex не ответил(а) в Telegram — запущен звонок:",
                    ]
                    for delivery in deliveries:
                        assignees = ", ".join(delivery.get("assignees") or ["—"])
                        call_result = delivery.get("call_result")
                        if call_result is None:
                            status_text = "без данных"
                        elif getattr(call_result, "success", False):
                            status_text = "успех"
                        else:
                            error = getattr(call_result, "error", None) or getattr(call_result, "status", "ошибка")
                            status_text = f"ошибка ({error})"
                        summary_lines.append(f"• {assignees} — {status_text}")

                    target_chat = args.chat_id or getattr(service, "default_chat_id", None)
                    if not target_chat:
                        try:
                            target_chat = service._resolve_target_chat()  # pylint: disable=protected-access
                        except Exception:  # pragma: no cover - best effort
                            target_chat = None

                    if target_chat:
                        try:
                            service.send_plain_message(target_chat, "\n".join(summary_lines))
                        except Exception as exc:  # pragma: no cover - best effort
                            logging.warning("Failed to send fallback summary to Telegram: %s", exc)
                else:
                    logging.info("Twilio fallback completed: нет задач для звонков.")
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
