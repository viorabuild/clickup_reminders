#!/usr/bin/env python3
"""Run the Telegram reminders workflow locally, mirroring the GitHub Actions job."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _expand(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    return candidate.resolve()


def _pick_existing(candidates: Iterable[str | Path]) -> Path | None:
    for candidate in candidates:
        path = _expand(candidate)
        if path.exists():
            return path
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
        Helper that prepares CONFIG_PATH/SECRETS_PATH and runs send_telegram_reminders.py,
        reproducing the GitHub Actions workflow locally.
        """.strip()
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Путь до конфигурации. По умолчанию используется config.json в корне проекта.",
    )
    parser.add_argument(
        "--example-config",
        default="config.example.json",
        help="Fallback конфигурация, если основной файл не найден.",
    )
    parser.add_argument(
        "--secrets",
        help="Явный путь до secrets.json. Если не указан, используются переменные окружения или стандартные пути.",
    )
    parser.add_argument(
        "--base-dir",
        default=str(ROOT_DIR),
        help="BASE_DIR, который будет проброшен в окружение (по умолчанию корень репозитория).",
    )
    parser.add_argument(
        "--chat-id",
        help=(
            "Проброс параметра --chat-id в send_telegram_reminders.py. "
            "Если не указан, используются значения из окружения или кеша."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Проброс параметра --limit в send_telegram_reminders.py.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        help="Проброс параметра --poll-seconds в send_telegram_reminders.py (по умолчанию как в оригинальном скрипте).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Включить подробный лог как в send_telegram_reminders.py.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Только проверить наличие конфигурации и секретов без отправки напоминаний.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Пропустить проверку секретов (если уверены в окружении).",
    )
    return parser.parse_args(argv)


def prepare_environment(args: argparse.Namespace) -> None:
    config_path = _pick_existing((args.config,))
    if not config_path:
        config_path = _pick_existing((args.example_config,))
    if not config_path:
        raise FileNotFoundError(
            "Не удалось найти config.json или config.example.json. Укажите путь через --config или --example-config."
        )

    os.environ["CONFIG_PATH"] = str(config_path)
    os.environ["BASE_DIR"] = str(_expand(args.base_dir))
    if args.chat_id:
        os.environ["TELEGRAM_CHAT_ID"] = str(args.chat_id)

    if args.secrets:
        secrets_path = _expand(args.secrets)
        if not secrets_path.exists():
            raise FileNotFoundError(f"Секреты по пути {secrets_path} не найдены")
        os.environ["SECRETS_PATH"] = str(secrets_path)


def verify_secrets() -> None:
    from telegram_reminder_service import (  # local import to ensure CONFIG_PATH is already установлен
        ConfigurationError,
        load_raw_config,
        load_runtime_credentials,
    )

    config = load_raw_config()
    credentials = load_runtime_credentials(config)

    def _status(value: bool, success: str, failure: str) -> str:
        return f"✅ {success}" if value else f"❌ {failure}"

    print("Проверка окружения:")
    print(_status(bool(credentials.get("clickup_api_key")), "CLICKUP_API_KEY задан", "CLICKUP_API_KEY отсутствует"))
    has_team = bool(credentials.get("clickup_team_id")) or bool(credentials.get("clickup_team_ids"))
    print(_status(has_team, "ClickUp team/workspace указан", "CLICKUP_TEAM_ID(S) отсутствует"))
    print(_status(bool(credentials.get("telegram_bot_token")), "TELEGRAM_BOT_TOKEN задан", "TELEGRAM_BOT_TOKEN отсутствует"))
    chat_id = credentials.get("telegram_chat_id")
    if chat_id:
        print(f"ℹ️  TELEGRAM_CHAT_ID: {chat_id}")
    else:
        print("ℹ️  TELEGRAM_CHAT_ID не указан — будет использован кешированный chat_id или значения из config.json")

    if not credentials.get("clickup_api_key"):
        raise ConfigurationError("CLICKUP_API_KEY не найден")
    if not has_team:
        raise ConfigurationError("CLICKUP_TEAM_ID(S) не найдены")
    if not credentials.get("telegram_bot_token"):
        raise ConfigurationError("TELEGRAM_BOT_TOKEN не найден")


def run_send_reminders(args: argparse.Namespace) -> int:
    from send_telegram_reminders import main as send_main

    cli_args: list[str] = []
    if args.chat_id:
        cli_args.extend(["--chat-id", args.chat_id])
    if args.limit is not None:
        cli_args.extend(["--limit", str(args.limit)])
    if args.poll_seconds is not None:
        cli_args.extend(["--poll-seconds", str(args.poll_seconds)])
    if args.verbose:
        cli_args.append("--verbose")
    return send_main(cli_args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        prepare_environment(args)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1

    if not args.skip_check:
        try:
            verify_secrets()
        except Exception as exc:  # noqa: BLE001 - выводим пользователю оригинальную ошибку
            print(f"❌ Ошибка проверки: {exc}")
            return 2

    if args.check_only:
        print("✅ Проверка завершена, запуск пропущен (--check-only)")
        return 0

    return run_send_reminders(args)


if __name__ == "__main__":
    sys.exit(main())
