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
import time
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
CALLBACK_LOG_PATH = PROJECT_ROOT / "var" / "telegram_callback_log.jsonl"
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


def _extract_nested(payload: Dict[str, Any], paths: Sequence[Sequence[str]]) -> Optional[Any]:
    """Return the first non-empty value located at any of the provided paths."""
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
        if isinstance(node, (list, tuple)):
            collected = []
            for item in node:
                if isinstance(item, str) and item.strip():
                    collected.append(item.strip())
            if collected:
                return collected
    return None


def _normalise_ids(values: Iterable[Any]) -> List[str]:
    cleaned: List[str] = []
    for candidate in values:
        if candidate is None:
            continue
        normalized = str(candidate).strip()
        if not normalized:
            continue
        if "<" in normalized or ">" in normalized:
            continue
        if normalized.lower().startswith("optional"):
            continue
        if not normalized.replace("-", "").isdigit():
            continue
        if normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def load_runtime_credentials(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load ClickUp and Telegram credentials from env/secrets."""
    env: Dict[str, Any] = {
        "clickup_api_key": os.getenv("CLICKUP_API_KEY"),
        "clickup_team_id": os.getenv("CLICKUP_TEAM_ID") or config.get("clickup_workspace_id"),
        "clickup_space_ids": os.getenv("CLICKUP_SPACE_IDS"),
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID") or (config.get("telegram", {}) or {}).get("chat_id"),
    }

    team_ids: List[str] = []
    env_team_ids = os.getenv("CLICKUP_TEAM_IDS")
    if env_team_ids:
        for candidate in env_team_ids.split(","):
            normalized = candidate.strip()
            if normalized:
                team_ids.append(normalized)

    clickup_cfg = config.get("clickup", {}) or {}
    for source in (
        clickup_cfg.get("team_ids"),
        clickup_cfg.get("workspace_ids"),
        config.get("clickup_team_ids"),
        config.get("clickup_workspace_ids"),
    ):
        if not source:
            continue
        if isinstance(source, (list, tuple, set)):
            iterable = source
        else:
            iterable = [source]
        for candidate in iterable:
            normalized = str(candidate).strip()
            if normalized and normalized not in team_ids:
                team_ids.append(normalized)

    env["clickup_team_ids"] = _normalise_ids(team_ids)

    space_candidates: List[Any] = []
    if env.get("clickup_space_ids"):
        space_candidates.extend(str(env["clickup_space_ids"]).split(","))
    for source in (
        clickup_cfg.get("space_ids"),
        config.get("clickup_space_ids"),
    ):
        if not source:
            continue
        if isinstance(source, (list, tuple, set)):
            values = source
        else:
            values = [source]
        space_candidates.extend(values)
    env["clickup_space_ids"] = _normalise_ids(space_candidates)

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
            "clickup_team_ids": (
                ("clickup", "team_ids"),
                ("clickup", "workspace_ids"),
                ("telegram", "secrets", "clickup_team_ids"),
            ),
            "clickup_space_ids": (
                ("clickup", "space_ids"),
                ("clickup", "workspace_ids"),
                ("telegram", "secrets", "clickup_space_ids"),
            ),
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
    if not env.get("clickup_team_id") and not env.get("clickup_team_ids"):
        raise ConfigurationError("CLICKUP_TEAM_ID(S) not provided via env or secrets/config.")
    if not env.get("telegram_bot_token"):
        raise ConfigurationError("TELEGRAM_BOT_TOKEN not provided via env or secrets.")

    team_ids_from_env = env.get("clickup_team_ids")
    normalized_team_ids: List[str] = []
    if isinstance(team_ids_from_env, (list, tuple, set)):
        normalized_team_ids = _normalise_ids(team_ids_from_env)
    elif isinstance(team_ids_from_env, str):
        normalized_team_ids = _normalise_ids([team_ids_from_env])

    if not normalized_team_ids:
        fallback_team_id = env.get("clickup_team_id")
        if fallback_team_id:
            normalized_team_ids = [str(fallback_team_id).strip()]
    else:
        fallback_team_id = env.get("clickup_team_id")
        fallback_candidates: List[str] = []
        if fallback_team_id:
            fallback_candidates = _normalise_ids([fallback_team_id])
        for candidate in fallback_candidates:
            if candidate not in normalized_team_ids:
                normalized_team_ids.insert(0, candidate)
        if normalized_team_ids:
            env["clickup_team_id"] = normalized_team_ids[0]

    env["clickup_team_ids"] = normalized_team_ids

    space_ids = env.get("clickup_space_ids")
    if isinstance(space_ids, str):
        env["clickup_space_ids"] = _normalise_ids(space_ids.split(","))

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

        self.team_ids = self._resolve_team_ids()
        if not self.team_ids:
            raise ConfigurationError("Не указаны идентификаторы ClickUp workspace/team.")

        self.clickup_clients = [
            ClickUpClient(
                api_key=credentials["clickup_api_key"],
                team_id=team_id,
            )
            for team_id in self.team_ids
        ]
        self.clickup_client = self.clickup_clients[0]

        self.clickup_config = self.config.get("clickup", {}) or {}
        self.status_mapping = self._build_status_mapping()
        self.completed_statuses = self._build_completed_statuses()
        self.reminder_tags = self._resolve_reminder_tags()
        self.reminders_list_id = self.clickup_config.get("list_id") or self.config.get("clickup_list_id")
        self.reminders_list_name = (
            self.clickup_config.get("reminders_list_name")
            or self.config.get("reminder_list_name")
            or "Напоминания"
        )
        self.space_ids = self._resolve_space_ids()
        self.assignee_chat_map = self._build_assignee_chat_map()
        tz_name = config.get("working_hours", {}).get("timezone") or "UTC"
        self.timezone_name = tz_name
        self.callback_log_path = self._resolve_callback_log_path()

    @classmethod
    def from_environment(cls) -> "TelegramReminderService":
        config = load_raw_config()
        credentials = load_runtime_credentials(config)
        return cls(config=config, credentials=credentials)

    def _resolve_team_ids(self) -> List[str]:
        team_ids_raw = self.credentials.get("clickup_team_ids")
        team_ids: List[str] = []

        if isinstance(team_ids_raw, (list, tuple, set)):
            candidates = team_ids_raw
        elif isinstance(team_ids_raw, str):
            candidates = [team_ids_raw]
        else:
            candidates = []

        for candidate in candidates:
            normalized = str(candidate).strip()
            if normalized and normalized not in team_ids:
                team_ids.append(normalized)

        fallback = self.credentials.get("clickup_team_id")
        if fallback:
            fallback_str = str(fallback).strip()
            if fallback_str and fallback_str not in team_ids:
                team_ids.insert(0, fallback_str)
        return team_ids

    def _resolve_space_ids(self) -> List[str]:
        candidates: List[Any] = []
        for source in (
            self.clickup_config.get("space_ids"),
            self.config.get("clickup_space_ids"),
            self.credentials.get("clickup_space_ids"),
        ):
            if not source:
                continue
            if isinstance(source, (list, tuple, set)):
                values = source
            else:
                values = [source]
            candidates.extend(values)
        return _normalise_ids(candidates)

    def _build_status_mapping(self) -> Dict[str, str]:
        clickup_section = self.clickup_config
        mapping = {key.upper(): value for key, value in clickup_section.get("status_mapping", {}).items()}
        mapping.setdefault("ВЫПОЛНЕНО", clickup_section.get("completed_status", "complete"))
        mapping.setdefault("НЕ_ВЫПОЛНЕНО", clickup_section.get("pending_status", "to do"))
        mapping.setdefault("В_РАБОТЕ", clickup_section.get("in_progress_status", "in progress"))
        return mapping

    def _build_completed_statuses(self) -> set[str]:
        statuses = {self.status_mapping.get("ВЫПОЛНЕНО", "complete").lower()}
        completed = self.clickup_config.get("completed_status")
        if completed:
            statuses.add(str(completed).lower())
        statuses.add("complete")
        statuses.add("done")
        return statuses

    def _resolve_reminder_tags(self) -> List[str]:
        tags_cfg = (
            self.clickup_config.get("reminder_tags")
            or self.clickup_config.get("reminder_tag")
            or self.config.get("reminder_tags")
            or self.config.get("reminder_tag")
        )

        if isinstance(tags_cfg, str):
            raw_tags: Iterable[str] = (tags_cfg,)
        elif isinstance(tags_cfg, (list, tuple, set)):
            raw_tags = (str(tag) for tag in tags_cfg)
        else:
            raw_tags = ()

        tags: List[str] = []
        seen: set[str] = set()
        for candidate in raw_tags:
            normalized = str(candidate).strip()
            if not normalized or normalized in seen:
                continue
            tags.append(normalized)
            seen.add(normalized)
        return tags

    def _build_assignee_chat_map(self) -> Dict[str, Tuple[str, ...]]:
        telegram_cfg = self.config.get("telegram") or {}
        mapping_cfg = telegram_cfg.get("assignee_chat_map") or telegram_cfg.get("assignee_chats") or {}
        if not isinstance(mapping_cfg, dict):
            return {}

        result: Dict[str, Tuple[str, ...]] = {}
        for raw_name, raw_chat_ids in mapping_cfg.items():
            if raw_name is None:
                continue

            if isinstance(raw_name, str):
                name_candidates = [part.strip() for part in raw_name.split("|") if part.strip()]
            else:
                name_candidates = [str(raw_name).strip()]

            if not name_candidates:
                continue

            if isinstance(raw_chat_ids, (list, tuple, set)):
                chat_iterable = raw_chat_ids
            else:
                chat_iterable = (raw_chat_ids,)

            chats: List[str] = []
            seen_chat: set[str] = set()
            for chat in chat_iterable:
                chat_str = str(chat).strip()
                if not chat_str or chat_str in seen_chat:
                    continue
                chats.append(chat_str)
                seen_chat.add(chat_str)

            if not chats:
                continue

            chat_tuple = tuple(chats)
            for name in name_candidates:
                normalized = self._normalize_assignee_name(name)
                if normalized:
                    result[normalized] = chat_tuple

        return result

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

    def _ensure_default_chat(self, chat_id: str) -> None:
        if not self.default_chat_id:
            self.default_chat_id = chat_id
            self._persist_chat_id(chat_id)

    @staticmethod
    def _normalize_assignee_name(name: str) -> str:
        normalized = str(name or "").strip().lower()
        if not normalized:
            return ""
        return " ".join(normalized.split())

    def _chat_targets_for_task(self, task: ReminderTask) -> Tuple[str, ...]:
        assignee = self._normalize_assignee_name(task.assignee)
        if not assignee:
            return ()

        direct_mapping = self.assignee_chat_map.get(assignee)
        if direct_mapping:
            return direct_mapping

        if "(" in assignee:
            trimmed = assignee.split("(", 1)[0].strip()
            if trimmed and trimmed in self.assignee_chat_map:
                return self.assignee_chat_map[trimmed]

        return ()

    def _resolve_callback_log_path(self) -> Optional[Path]:
        telegram_cfg = self.config.get("telegram") or {}
        raw_path = telegram_cfg.get("callback_log_path")
        if raw_path is None:
            return CALLBACK_LOG_PATH
        raw_str = str(raw_path).strip()
        if not raw_str:
            return None
        try:
            return Path(raw_str).expanduser()
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Failed to resolve callback log path %s: %s", raw_path, exc)
            return CALLBACK_LOG_PATH

    def _append_callback_log(self, entry: Dict[str, Any]) -> None:
        if not self.callback_log_path:
            return
        try:
            self.callback_log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(entry, ensure_ascii=True)
            with self.callback_log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.write("\n")
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.debug("Failed to append callback log: %s", exc)

    # --------------------------------------------------------------------- #
    # ClickUp helpers
    # --------------------------------------------------------------------- #
    def fetch_pending_tasks(self, limit: Optional[int] = None) -> List[ReminderTask]:
        """Return tasks from ClickUp that match the reminder filters and are not completed."""
        tasks_raw: List[Dict[str, Any]] = []
        try:
            if self.reminder_tags:
                seen_ids: set[str] = set()
                for client in self.clickup_clients:
                    for tag in self.reminder_tags:
                        for task in client.fetch_tasks_by_tag(tag, space_ids=self.space_ids or None):
                            task_id = str(task.get("id") or "").strip()
                            if not task_id or task_id in seen_ids:
                                continue
                            tasks_raw.append(task)
                            seen_ids.add(task_id)
            else:
                seen_ids = set()
                for client in self.clickup_clients:
                    batch = client.fetch_tasks(
                        list_name=self.reminders_list_name if not self.reminders_list_id else None,
                        list_id=self.reminders_list_id,
                    )
                    for task in batch:
                        task_id = str(task.get("id") or "").strip()
                        if not task_id or task_id in seen_ids:
                            continue
                        tasks_raw.append(task)
                        seen_ids.add(task_id)
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

    def _group_tasks_by_chat(
        self,
        tasks: Sequence[ReminderTask],
        fallback_chat: Optional[str],
    ) -> Dict[str, List[ReminderTask]]:
        deliveries: Dict[str, List[ReminderTask]] = {}

        for task in tasks:
            chat_ids = self._chat_targets_for_task(task)
            if not chat_ids:
                if fallback_chat:
                    chat_ids = (fallback_chat,)
                else:
                    LOGGER.warning(
                        "No Telegram chat mapping for assignee '%s' (task %s). Skipping.",
                        task.assignee,
                        task.task_id,
                    )
                    continue

            seen: set[str] = set()
            for chat_id in chat_ids:
                chat_candidate = str(chat_id).strip()
                if not chat_candidate or chat_candidate in seen:
                    continue
                deliveries.setdefault(chat_candidate, []).append(task)
                seen.add(chat_candidate)

        return deliveries

    def _dispatch_tasks_to_chat(self, chat_id: str, tasks: Sequence[ReminderTask]) -> None:
        if not chat_id:
            return

        self._ensure_default_chat(chat_id)

        if not tasks:
            self.send_plain_message(chat_id, "✅ На данный момент нет задач, требующих внимания.")
            return

        preface = (
            f"📌 Найдено задач: {len(tasks)}. "
            "Отметьте статус прямо в боте — выбор обновит задачу в ClickUp."
        )
        self.send_plain_message(chat_id, preface)

        for idx, task in enumerate(tasks, start=1):
            try:
                self.send_task_message(chat_id, task, idx)
            except Exception as exc:  # pragma: no cover - network guard
                LOGGER.error("Failed to send task %s to Telegram chat %s: %s", task.task_id, chat_id, exc)

    def send_reminders(
        self,
        chat_id: Optional[str] = None,
        limit: Optional[int] = None,
        broadcast_all: bool = False,
    ) -> List[ReminderTask]:
        tasks = self.fetch_pending_tasks(limit=limit)

        if chat_id is not None:
            target_chat = self._resolve_target_chat(chat_id)
            if not target_chat:
                raise ConfigurationError(
                    "Chat id not supplied. Отправьте /start боту или передайте --chat-id для send_telegram_reminders.py."
                )
            if broadcast_all:
                self._dispatch_tasks_to_chat(str(target_chat), tasks)
                return tasks

            fallback_global = self._resolve_target_chat()
            fallback_chat = (
                str(target_chat)
                if fallback_global is not None and str(fallback_global) == str(target_chat)
                else None
            )
            deliveries = self._group_tasks_by_chat(tasks, fallback_chat=fallback_chat)
            bucket = deliveries.get(str(target_chat), [])
            self._dispatch_tasks_to_chat(str(target_chat), bucket)
            return tasks

        if not tasks:
            fallback_chat = self._resolve_target_chat()
            if fallback_chat:
                self._dispatch_tasks_to_chat(fallback_chat, [])
            else:
                LOGGER.info("No pending tasks and no Telegram chat configured to notify.")
            return []

        fallback_chat = self._resolve_target_chat()
        deliveries = self._group_tasks_by_chat(tasks, fallback_chat=fallback_chat)
        if not deliveries:
            raise ConfigurationError(
                "Не удалось определить чаты Telegram для отправки напоминаний. "
                "Добавьте telegram.chat_id или telegram.assignee_chat_map в config.json."
            )

        for target_chat, bucket in deliveries.items():
            self._dispatch_tasks_to_chat(target_chat, bucket)

        return tasks

    def poll_updates_for(
        self,
        duration: float,
        poll_interval: float = 1.0,
        timeout: int = 10,
    ) -> int:
        """
        Poll Telegram updates for a limited duration to process callbacks/messages.

        Returns the amount of processed updates.
        """
        if duration <= 0:
            return 0

        deadline = time.monotonic() + duration
        offset: Optional[int] = None
        processed = 0

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            effective_timeout = min(timeout, max(1, int(remaining)))
            try:
                updates = self.get_updates(offset=offset, timeout=effective_timeout)
            except Exception as exc:  # pragma: no cover - network guard
                LOGGER.warning("Polling updates failed: %s", exc)
                sleep_for = min(poll_interval, max(0.1, deadline - time.monotonic()))
                time.sleep(sleep_for)
                continue

            if not updates:
                time.sleep(min(poll_interval, max(0.1, deadline - time.monotonic())))
                continue

            last_update_id: Optional[int] = offset
            for update in updates:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    last_update_id = update_id

                if "message" in update:
                    self.handle_message(update["message"])
                    processed += 1
                elif "callback_query" in update:
                    self.handle_callback(update["callback_query"])
                    processed += 1
                else:
                    LOGGER.debug("Ignored update keys: %s", list(update.keys()))

            if last_update_id is not None:
                offset = last_update_id + 1

        return processed

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

        actor = callback.get("from") or {}
        base_log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "task_id": task_id,
            "callback_id": callback_id,
            "action_code": action_code,
            "status_key": status_key,
            "chat_id": str(raw_chat_id) if raw_chat_id is not None else "",
            "message_id": message_id,
            "user_id": actor.get("id"),
            "username": actor.get("username"),
            "first_name": actor.get("first_name"),
            "last_name": actor.get("last_name"),
        }

        if chat_id:
            self.default_chat_id = chat_id
            self._persist_chat_id(chat_id)

        try:
            self.update_clickup_status(task_id, status_key)
        except Exception as exc:
            LOGGER.error("Failed to update task %s status: %s", task_id, exc)
            error_log = dict(base_log_entry)
            error_log["result"] = "error"
            error_log["error"] = str(exc)
            self._append_callback_log(error_log)
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

        success_log = dict(base_log_entry)
        success_log["chat_id"] = str(chat_id) if chat_id else success_log["chat_id"]
        success_log["result"] = "success"
        self._append_callback_log(success_log)

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
