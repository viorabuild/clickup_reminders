#!/usr/bin/env python3
"""
High level Telegram reminder helpers.

This module centralises the logic that fetches ClickUp tasks, pushes reminder
messages to Telegram (with inline actions), and processes callback events coming
from the Telegram Bot API.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pytz
import requests

from clickup import ClickUpClient

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
CHAT_ID_CACHE_PATH = PROJECT_ROOT / "var" / "telegram_chat_id.txt"
DEFAULT_SECRETS_CANDIDATES: Tuple[Path, ...] = (
    PROJECT_ROOT / ".venv" / "bin" / "secrets.json",
    PROJECT_ROOT.parent / ".venv" / "bin" / "secrets.json",
)


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass
class ReminderTask:
    """Normalized structure representing a ClickUp reminder task."""

    task_id: str
    name: str
    status: str
    due_human: str
    assignee: str
    url: str


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_config_path() -> Path:
    env_override = os.getenv("CONFIG_PATH")
    if env_override:
        candidate = Path(env_override).expanduser()
        if candidate.exists():
            return candidate
        raise ConfigurationError(f"CONFIG_PATH points to missing file: {candidate}")
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    raise ConfigurationError("config.json not found. Set CONFIG_PATH or create config.json.")


def load_raw_config() -> Dict[str, Any]:
    """Load project configuration as a raw dict."""
    return _load_json(_resolve_config_path())


def _iter_secret_candidates() -> Iterable[Path]:
    env_secret = os.getenv("SECRETS_PATH")
    if env_secret:
        yield Path(env_secret).expanduser()
    for candidate in DEFAULT_SECRETS_CANDIDATES:
        yield candidate.expanduser()


def _extract_nested(payload: Dict[str, Any], paths: Sequence[Sequence[str]]) -> Optional[str]:
    """Return the first non-empty string located at any of the provided paths."""
    for path in paths:
        node: Any = payload
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                node = None
                break
        if node is None:
            continue
        if isinstance(node, dict) and "value" in node:
            node = node["value"]
        if isinstance(node, str) and node:
            return node
    return None


def load_runtime_credentials(config: Dict[str, Any]) -> Dict[str, str]:
    """Load ClickUp and Telegram credentials from env/secrets."""
    env = {
        "clickup_api_key": os.getenv("CLICKUP_API_KEY"),
        "clickup_team_id": os.getenv("CLICKUP_TEAM_ID") or config.get("clickup_workspace_id"),
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID") or (config.get("telegram", {}) or {}).get("chat_id"),
    }

    missing = {key for key, value in env.items() if key != "telegram_chat_id" and not value}
    secrets_payload: Optional[Dict[str, Any]] = None
    if missing or (not env.get("telegram_bot_token")) or (not env.get("telegram_chat_id")):
        for candidate in _iter_secret_candidates():
            if candidate.exists():
                try:
                    secrets_payload = _load_json(candidate)
                    break
                except Exception as exc:  # pragma: no cover - defensive guard
                    LOGGER.warning("Failed to read secrets file %s: %s", candidate, exc)
                    continue

    if secrets_payload:
        secret_mappings = {
            "clickup_api_key": (("clickup", "api_key"), ("telegram", "secrets", "clickup_api_key")),
            "clickup_team_id": (("clickup", "team_id"), ("telegram", "secrets", "clickup_team_id")),
            "telegram_bot_token": (
                ("telegram", "bot_token"),
                ("telegram", "secrets", "bot_token"),
            ),
            "telegram_chat_id": (
                ("telegram", "chat_id"),
                ("telegram", "secrets", "chat_id"),
            ),
        }
        for key, paths in secret_mappings.items():
            if not env.get(key):
                env_value = _extract_nested(secrets_payload, paths)
                if env_value:
                    env[key] = env_value

    if not env.get("clickup_api_key"):
        raise ConfigurationError("CLICKUP_API_KEY not provided via env or secrets.")
    if not env.get("clickup_team_id"):
        raise ConfigurationError("CLICKUP_TEAM_ID not provided via env or secrets/config.")
    if not env.get("telegram_bot_token"):
        raise ConfigurationError("TELEGRAM_BOT_TOKEN not provided via env or secrets.")

    return env  # contains telegram_chat_id which may still be None


def _format_due(due_raw: Any, timezone_name: str) -> str:
    if not due_raw:
        return "Не указан"
    try:
        timestamp = int(due_raw) / 1000
        tz = pytz.timezone(timezone_name)
        dt = datetime.fromtimestamp(timestamp, tz)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(due_raw)


def _primary_assignee(task: Dict[str, Any]) -> str:
    assignees = task.get("assignees") or []
    if assignees and isinstance(assignees, list):
        first = assignees[0]
        name = first.get("username") or first.get("email") or first.get("name")
        if name:
            return str(name)
    custom_fields = task.get("custom_fields") or []
    for field in custom_fields:
        if isinstance(field, dict) and field.get("name", "").lower() == "assignee":
            value = field.get("value")
            if isinstance(value, str) and value:
                return value
    return "—"


class TelegramReminderService:
    """Business logic that orchestrates ClickUp and Telegram interactions."""

    ACTION_CODES = {"d": "ВЫПОЛНЕНО", "n": "НЕ_ВЫПОЛНЕНО", "p": "В_РАБОТЕ"}

    def __init__(
        self,
        config: Dict[str, Any],
        credentials: Dict[str, str],
        session: Optional[requests.Session] = None,
    ):
        self.config = config
        self.credentials = credentials
        self.session = session or requests.Session()
        self.bot_token = credentials["telegram_bot_token"]
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.default_chat_id = credentials.get("telegram_chat_id") or self._load_cached_chat_id()

        self.clickup_client = ClickUpClient(
            api_key=credentials["clickup_api_key"],
            team_id=str(credentials["clickup_team_id"]),
        )

        self.status_mapping = self._build_status_mapping()
        self.completed_statuses = self._build_completed_statuses()
        tz_name = config.get("working_hours", {}).get("timezone") or "UTC"
        self.timezone_name = tz_name

    @classmethod
    def from_environment(cls) -> "TelegramReminderService":
        config = load_raw_config()
        credentials = load_runtime_credentials(config)
        return cls(config=config, credentials=credentials)

    def _build_status_mapping(self) -> Dict[str, str]:
        clickup_section = self.config.get("clickup", {})
        mapping = {key.upper(): value for key, value in clickup_section.get("status_mapping", {}).items()}
        mapping.setdefault("ВЫПОЛНЕНО", clickup_section.get("completed_status", "complete"))
        mapping.setdefault("НЕ_ВЫПОЛНЕНО", clickup_section.get("pending_status", "to do"))
        mapping.setdefault("В_РАБОТЕ", clickup_section.get("in_progress_status", "in progress"))
        return mapping

    def _build_completed_statuses(self) -> set[str]:
        statuses = {self.status_mapping.get("ВЫПОЛНЕНО", "complete").lower()}
        clickup_section = self.config.get("clickup", {})
        completed = clickup_section.get("completed_status")
        if completed:
            statuses.add(str(completed).lower())
        statuses.add("complete")
        statuses.add("done")
        return statuses

    def _load_cached_chat_id(self) -> Optional[str]:
        try:
            if CHAT_ID_CACHE_PATH.exists():
                value = CHAT_ID_CACHE_PATH.read_text(encoding="utf-8").strip()
                return value or None
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.debug("Failed to read cached chat id: %s", exc)
        return None

    def _persist_chat_id(self, chat_id: str) -> None:
        try:
            CHAT_ID_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CHAT_ID_CACHE_PATH.write_text(str(chat_id), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning("Failed to persist chat id %s: %s", chat_id, exc)

    def _resolve_target_chat(self, override: Optional[str] = None) -> Optional[str]:
        if override:
            return str(override)
        if self.default_chat_id:
            return str(self.default_chat_id)
        cached = self._load_cached_chat_id()
        if cached:
            self.default_chat_id = cached
            return cached
        return None

    # --------------------------------------------------------------------- #
    # ClickUp helpers
    # --------------------------------------------------------------------- #
    def fetch_pending_tasks(self, limit: Optional[int] = None) -> List[ReminderTask]:
        """Return tasks from ClickUp that match the reminder filters and are not completed."""
        clickup_cfg = self.config.get("clickup", {})

        tags_cfg = (
            clickup_cfg.get("reminder_tags")
            or clickup_cfg.get("reminder_tag")
            or self.config.get("reminder_tags")
            or self.config.get("reminder_tag")
        )
        reminder_tags: List[str] = []
        if isinstance(tags_cfg, str):
            reminder_tags = [tags_cfg]
        elif isinstance(tags_cfg, (list, tuple)):
            reminder_tags = [str(tag) for tag in tags_cfg if str(tag).strip()]

        tasks_raw: List[Dict[str, Any]] = []
        try:
            if reminder_tags:
                seen_ids: set[str] = set()
                for tag in reminder_tags:
                    for task in self.clickup_client.fetch_tasks_by_tag(tag):
                        task_id = str(task.get("id") or "").strip()
                        if not task_id or task_id in seen_ids:
                            continue
                        tasks_raw.append(task)
                        seen_ids.add(task_id)
            else:
                list_id = clickup_cfg.get("list_id") or self.config.get("clickup_list_id")
                list_name = (
                    clickup_cfg.get("reminders_list_name")
                    or self.config.get("reminder_list_name")
                    or "Напоминания"
                )
                tasks_raw = self.clickup_client.fetch_tasks(
                    list_name=list_name if not list_id else None,
                    list_id=list_id,
                )
        except Exception as exc:
            LOGGER.error("Failed to fetch tasks from ClickUp: %s", exc)
            raise

        pending: List[ReminderTask] = []
        for task in tasks_raw:
            status_obj = task.get("status") or {}
            status_value = str(status_obj.get("status") or status_obj.get("name") or "").lower()
            if status_value in self.completed_statuses or status_obj.get("type") == "done":
                continue

            pending.append(
                ReminderTask(
                    task_id=str(task["id"]),
                    name=str(task.get("name", "Без названия")),
                    status=status_obj.get("status") or status_obj.get("name") or "—",
                    due_human=_format_due(task.get("due_date"), self.timezone_name),
                    assignee=_primary_assignee(task),
                    url=f"https://app.clickup.com/t/{task['id']}",
                )
            )

        pending.sort(key=lambda t: (t.due_human, t.name))

        if limit:
            pending = pending[:limit]
        return pending

    def update_clickup_status(self, task_id: str, status_key: str) -> None:
        """Update ClickUp task status using the configured mapping."""
        status_key = status_key.upper()
        target = self.status_mapping.get(status_key)
        if not target:
            raise ValueError(f"Unsupported status key: {status_key}")
        self.clickup_client.update_status(task_id, target)

        comment = f"Статус обновлен через Telegram-бота: {status_key}"
        try:
            self.clickup_client.add_comment(task_id, comment)
        except Exception as exc:
            LOGGER.warning("Failed to add ClickUp comment for %s: %s", task_id, exc)

    def fetch_task_details(self, task_id: str) -> Dict[str, Any]:
        try:
            return self.clickup_client.fetch_task(task_id)
        except Exception as exc:
            LOGGER.warning("Failed to fetch task %s details: %s", task_id, exc)
            return {"id": task_id, "name": f"Задача {task_id}"}

    # --------------------------------------------------------------------- #
    # Telegram helpers
    # --------------------------------------------------------------------- #
    def _telegram_post(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{method}"
        response = self.session.post(url, json=payload, timeout=15)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network guard
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = response.text
            LOGGER.error("Telegram API error (%s): %s | payload=%s", method, exc, error_payload)
            raise
        data = response.json()
        if not data.get("ok"):
            LOGGER.error("Telegram API call %s failed: %s", method, data)
        return data

    def send_plain_message(self, chat_id: str, text: str) -> Dict[str, Any]:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        return self._telegram_post("sendMessage", payload)

    def send_task_message(self, chat_id: str, task: ReminderTask, ordinal: int) -> Dict[str, Any]:
        text = (
            f"🔔 <b>Напоминание #{ordinal}</b>\n\n"
            f"📋 <b>Задача:</b> {task.name}\n"
            f"👤 <b>Исполнитель:</b> {task.assignee}\n"
            f"📊 <b>Статус:</b> {task.status}\n"
            f"⏰ <b>Срок:</b> {task.due_human}\n"
            f"🔗 <a href=\"{task.url}\">Открыть задачу</a>"
        )
        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": "✅ Выполнено", "callback_data": f"s:{task.task_id}:d"},
                    {"text": "❌ Не выполнено", "callback_data": f"s:{task.task_id}:n"},
                ],
                [
                    {"text": "🔄 В работе", "callback_data": f"s:{task.task_id}:p"},
                ],
            ]
        }
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "reply_markup": reply_markup,
            "disable_web_page_preview": True,
        }
        return self._telegram_post("sendMessage", payload)

    def remove_inline_keyboard(self, chat_id: str, message_id: int) -> None:
        self._telegram_post(
            "editMessageReplyMarkup",
            {"chat_id": chat_id, "message_id": message_id, "reply_markup": {"inline_keyboard": []}},
        )

    def answer_callback(self, callback_id: str, text: str, show_alert: bool = False) -> None:
        payload = {"callback_query_id": callback_id}
        if text:
            payload["text"] = text
        if show_alert:
            payload["show_alert"] = True
        self._telegram_post("answerCallbackQuery", payload)

    def send_reminders(self, chat_id: Optional[str] = None, limit: Optional[int] = None) -> List[ReminderTask]:
        target_chat = self._resolve_target_chat(chat_id)
        if not target_chat:
            raise ConfigurationError(
                "Chat id not supplied. Отправьте /start боту или передайте --chat-id для send_telegram_reminders.py."
            )

        if not self.default_chat_id:
            self.default_chat_id = target_chat
            self._persist_chat_id(target_chat)

        tasks = self.fetch_pending_tasks(limit=limit)
        if not tasks:
            self.send_plain_message(target_chat, "✅ На данный момент нет задач, требующих внимания.")
            return []

        preface = (
            f"📌 Найдено задач: {len(tasks)}. "
            "Отметьте статус прямо в боте — выбор обновит задачу в ClickUp."
        )
        self.send_plain_message(target_chat, preface)

        for idx, task in enumerate(tasks, start=1):
            try:
                self.send_task_message(target_chat, task, idx)
            except Exception as exc:  # pragma: no cover - network guard
                LOGGER.error("Failed to send task %s to Telegram: %s", task.task_id, exc)
                continue

        return tasks

    # ------------------------------------------------------------------ #
    # Update handlers
    # ------------------------------------------------------------------ #
    def handle_message(self, message: Dict[str, Any]) -> None:
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id"))
        text = (message.get("text") or "").strip().lower()

        if chat_id:
            self.default_chat_id = chat_id
            self._persist_chat_id(chat_id)

        if text in ("/start", "start", "/remind", "напомни", "напоминания"):
            self.send_plain_message(
                chat_id,
                "👋 Привет! Я бот напоминаний. Вот список актуальных задач:",
            )
            self.send_reminders(chat_id=chat_id)
        elif text in ("/help", "help", "помощь"):
            self.send_plain_message(
                chat_id,
                "ℹ️ Используйте /start, чтобы получить текущие напоминания и изменить статус задач кнопками.",
            )
        else:
            self.send_plain_message(
                chat_id,
                "⚠️ Неизвестная команда. Отправьте /start, чтобы получить список напоминаний.",
            )

    def handle_callback(self, callback: Dict[str, Any]) -> None:
        data = callback.get("data") or ""
        callback_id = callback.get("id")
        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        raw_chat_id = chat.get("id")
        chat_id = str(raw_chat_id) if raw_chat_id is not None else ""
        message_id = message.get("message_id")

        if not data.startswith("s:") or data.count(":") != 2:
            if callback_id:
                self.answer_callback(callback_id, "Неизвестное действие", show_alert=True)
            return

        _, task_id, action_code = data.split(":")
        status_key = self.ACTION_CODES.get(action_code)
        if not status_key:
            if callback_id:
                self.answer_callback(callback_id, "Действие не поддерживается", show_alert=True)
            return

        if chat_id:
            self.default_chat_id = chat_id
            self._persist_chat_id(chat_id)

        try:
            self.update_clickup_status(task_id, status_key)
        except Exception as exc:
            LOGGER.error("Failed to update task %s status: %s", task_id, exc)
            if callback_id:
                try:
                    self.answer_callback(
                        callback_id,
                        "❌ Не удалось обновить задачу в ClickUp",
                        show_alert=True,
                    )
                except Exception:  # pragma: no cover - best effort
                    pass
            if chat_id:
                self.send_plain_message(
                    chat_id,
                    f"⚠️ Не удалось обновить задачу <b>{task_id}</b>: {exc}",
                )
            return

        if not chat_id:
            chat_id = self._resolve_target_chat()

        if raw_chat_id is not None and chat_id and message_id:
            try:
                self.remove_inline_keyboard(chat_id, message_id)
            except Exception as exc:  # pragma: no cover - best effort
                LOGGER.debug("Failed to clear inline keyboard for message %s: %s", message_id, exc)

        if not chat_id:
            LOGGER.warning("No chat id available to notify about task %s update", task_id)
            return

        if callback_id:
            try:
                self.answer_callback(callback_id, "Готово!")
            except Exception as exc:  # pragma: no cover - network guard
                LOGGER.debug("Failed to send callback ack for task %s: %s", task_id, exc)
        task_payload = self.fetch_task_details(task_id)
        task_name = task_payload.get("name", f"Задача {task_id}")
        self.send_plain_message(
            chat_id,
            f"✅ Задача <b>{task_name}</b> отмечена как: <b>{status_key}</b>",
        )

    def get_updates(self, offset: Optional[int] = None, timeout: int = 30) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            payload["offset"] = offset
        response = self.session.post(f"{self.base_url}/getUpdates", json=payload, timeout=timeout + 5)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            LOGGER.error("getUpdates failed: %s", data)
            return []
        return data.get("result", [])
