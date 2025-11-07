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

try:  # pragma: no cover - support both package and script execution
    from .telephony import TwilioService
except ImportError:  # pragma: no cover - script mode fallback
    from telephony import TwilioService  # type: ignore

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
    assignee_id: Optional[str] = None


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
        "twilio_account_sid": os.getenv("TWILIO_ACCOUNT_SID"),
        "twilio_auth_token": os.getenv("TWILIO_AUTH_TOKEN"),
        "twilio_phone_number": os.getenv("TWILIO_PHONE_NUMBER"),
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
            "twilio_account_sid": (
                ("twilio", "account_sid"),
                ("twilio", "secrets", "account_sid"),
            ),
            "twilio_auth_token": (
                ("twilio", "auth_token"),
                ("twilio", "secrets", "auth_token"),
            ),
            "twilio_phone_number": (
                ("twilio", "phone_number"),
                ("twilio", "secrets", "phone_number"),
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


def _assignee_identity(task: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    assignees = task.get("assignees") or []
    if assignees and isinstance(assignees, list):
        first = assignees[0]
        name = first.get("username") or first.get("email") or first.get("name")
        if name:
            raw_id = first.get("id")
            assignee_id = str(raw_id).strip() if raw_id is not None else None
            return str(name), assignee_id or None

    watchers = task.get("watchers") or []
    if watchers and isinstance(watchers, list) and len(watchers) == 1:
        creator = task.get("creator") or {}
        creator_id = str(creator.get("id")).strip() if creator and creator.get("id") is not None else None
        for watcher in watchers:
            watcher_raw_id = watcher.get("id")
            watcher_id = str(watcher_raw_id).strip() if watcher_raw_id is not None else None
            if watcher_id and creator_id and watcher_id == creator_id:
                continue
            name = watcher.get("username") or watcher.get("email") or watcher.get("name")
            if name and name != "ClickBot":
                return str(name), watcher_id

    custom_fields = task.get("custom_fields") or []
    for field in custom_fields:
        if isinstance(field, dict) and field.get("name", "").lower() == "assignee":
            value = field.get("value")
            if isinstance(value, str) and value:
                return value, None

    return "—", None


class TelegramReminderService:
    """Business logic that orchestrates ClickUp and Telegram interactions."""

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

        configured_chat = credentials.get("telegram_chat_id")
        self._configured_default_chat = str(configured_chat) if configured_chat else None
        cached_chat = self._load_cached_chat_id()
        if self._configured_default_chat:
            self.default_chat_id = self._configured_default_chat
        else:
            self.default_chat_id = cached_chat

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
        self.assignee_chat_map_by_id = self._build_assignee_chat_map()
        self.status_actions = self._build_status_actions()
        self.status_action_map = {action["code"]: action for action in self.status_actions}
        self.status_action_by_key = {action["key"]: action for action in self.status_actions}
        for action in self.status_actions:
            if action["key"] not in self.status_mapping:
                normalized = action["key"].replace("_", " ").lower()
                LOGGER.warning(
                    "Status mapping key '%s' missing in config; defaulting to '%s'.",
                    action["key"],
                    normalized,
                )
                self.status_mapping[action["key"]] = normalized
        tz_name = config.get("working_hours", {}).get("timezone") or "UTC"
        self.timezone_name = tz_name
        self.callback_log_path = self._resolve_callback_log_path()
        self._processed_callback_ids: set[str] = self._load_processed_callback_ids()
        self.phone_mapping = self._build_phone_mapping()
        self.twilio_service: Optional[TwilioService] = None
        self.twilio_from_phone: Optional[str] = None
        self._init_twilio_service()

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
        mapping.setdefault("ПОСТАВЛЕНА", clickup_section.get("pending_status", "поставлена"))
        mapping.setdefault("НА_ДОРАБОТКЕ", clickup_section.get("callback_status", "на доработке"))
        mapping.setdefault("ОТМЕНЕНА", clickup_section.get("cancelled_status", "отменена"))
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

        ids_map: Dict[str, Tuple[str, ...]] = {}

        for raw_name, raw_chat_ids in mapping_cfg.items():
            if raw_name is None:
                continue

            if isinstance(raw_name, str):
                name_candidates = [part.strip() for part in raw_name.split("|") if part.strip()]
            else:
                name_candidates = [str(raw_name).strip()]

            extra_aliases: List[str] = []
            explicit_ids: List[str] = []
            chat_source = raw_chat_ids

            if isinstance(raw_chat_ids, dict):
                alias_field = raw_chat_ids.get("aliases")
                if isinstance(alias_field, str):
                    extra_aliases.extend(part.strip() for part in alias_field.split("|") if part.strip())
                elif isinstance(alias_field, (list, tuple, set)):
                    for alias in alias_field:
                        alias_str = str(alias).strip()
                        if alias_str:
                            extra_aliases.append(alias_str)

                id_field = raw_chat_ids.get("ids") or raw_chat_ids.get("assignee_ids") or raw_chat_ids.get("user_ids")
                if isinstance(id_field, str):
                    explicit_ids.extend(part.strip() for part in id_field.split("|") if part.strip())
                elif isinstance(id_field, (list, tuple, set)):
                    for entry in id_field:
                        entry_str = str(entry).strip()
                        if entry_str:
                            explicit_ids.append(entry_str)

                chat_candidates = (
                    raw_chat_ids.get("chat_ids")
                    or raw_chat_ids.get("chats")
                    or raw_chat_ids.get("telegram_ids")
                    or raw_chat_ids.get("chat_id")
                    or raw_chat_ids.get("telegram_id")
                )
                if isinstance(chat_candidates, (list, tuple, set)):
                    chat_source = chat_candidates
                elif chat_candidates is not None:
                    chat_source = (chat_candidates,)
                else:
                    chat_source = ()

            name_candidates.extend(extra_aliases)

            if not name_candidates:
                continue

            if isinstance(chat_source, (list, tuple, set)):
                chat_iterable = chat_source
            else:
                chat_iterable = (chat_source,)

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
                if not name:
                    continue
                token = name.strip()
                if not token:
                    continue
                lowered = token.lower()
                id_candidate: Optional[str] = None
                if token.isdigit():
                    id_candidate = token
                elif lowered.startswith("id:"):
                    _, _, suffix = token.partition(":")
                    candidate = suffix.strip()
                    if candidate:
                        id_candidate = candidate
                if id_candidate:
                    ids_map[id_candidate] = chat_tuple

            for identifier in explicit_ids:
                identifier_str = str(identifier).strip()
                if identifier_str:
                    ids_map[identifier_str] = chat_tuple

        return ids_map

    def _build_phone_mapping(self) -> Dict[str, str]:
        """
        Normalise phone mapping config so we can correlate assignees with numbers.

        Supports legacy keys like ``contacts`` and allows multiple aliases separated by ``|``.
        """
        raw_mapping = (
            self.config.get("phone_mapping")
            or self.config.get("contacts")
            or self.config.get("assignee_phones")
            or {}
        )
        if not isinstance(raw_mapping, dict):
            return {}

        mapping: Dict[str, str] = {}
        for raw_name, raw_phone in raw_mapping.items():
            if raw_phone is None:
                continue
            phone = str(raw_phone).strip()
            if not phone:
                continue

            if isinstance(raw_name, str):
                aliases = [part.strip() for part in raw_name.split("|") if part.strip()]
            else:
                aliases = [str(raw_name).strip()]

            for alias in aliases:
                normalized = self._normalize_assignee_name(alias)
                if not normalized or normalized in mapping:
                    continue
                mapping[normalized] = phone

        return mapping

    def _init_twilio_service(self) -> None:
        account_sid = str(self.credentials.get("twilio_account_sid") or "").strip()
        auth_token = str(self.credentials.get("twilio_auth_token") or "").strip()
        phone_number = str(self.credentials.get("twilio_phone_number") or "").strip()

        if not (account_sid and auth_token and phone_number):
            self.twilio_service = None
            self.twilio_from_phone = None
            return

        try:
            self.twilio_service = TwilioService(account_sid, auth_token)
            self.twilio_from_phone = phone_number
        except Exception as exc:  # pragma: no cover - network/sdk guard
            LOGGER.warning("Failed to initialise Twilio client: %s", exc)
            self.twilio_service = None
            self.twilio_from_phone = None

    def _generate_action_code(self, index: int, used: set[str]) -> str:
        base = f"a{index}"
        counter = 0
        candidate = base
        while candidate in used:
            counter += 1
            candidate = f"{base}{counter}"
        used.add(candidate)
        return candidate

    def _build_status_actions(self) -> List[Dict[str, str]]:
        telegram_cfg = self.config.get("telegram") or {}
        raw_actions = telegram_cfg.get("status_buttons") or telegram_cfg.get("status_actions")
        actions: List[Dict[str, str]] = []
        used_codes: set[str] = set()

        if isinstance(raw_actions, list):
            for idx, entry in enumerate(raw_actions):
                if isinstance(entry, str):
                    key = entry.strip().upper()
                    if not key:
                        continue
                    text = key.title()
                    code = self._generate_action_code(idx, used_codes)
                elif isinstance(entry, dict):
                    key = str(
                        entry.get("key")
                        or entry.get("status")
                        or entry.get("value")
                        or entry.get("name")
                        or ""
                    ).strip().upper()
                    if not key:
                        continue
                    text = str(entry.get("text") or entry.get("label") or key.title())
                    raw_code = entry.get("code")
                    if raw_code:
                        code = str(raw_code).strip()
                        if not code or code in used_codes:
                            code = self._generate_action_code(idx, used_codes)
                        else:
                            used_codes.add(code)
                    else:
                        code = self._generate_action_code(idx, used_codes)
                else:
                    continue
                actions.append({"code": code, "key": key, "text": text})

        if not actions:
            actions = [
                {"code": "d", "key": "ВЫПОЛНЕНО", "text": "✅ Выполнено"},
                {"code": "n", "key": "НЕ_ВЫПОЛНЕНО", "text": "❌ Не выполнено"},
                {"code": "p", "key": "В_РАБОТЕ", "text": "🔄 В работе"},
            ]
            used_codes.update(action["code"] for action in actions)

        return actions

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
        assignee_id = str(task.assignee_id).strip() if task.assignee_id else None
        if assignee_id:
            direct_by_id = self.assignee_chat_map_by_id.get(assignee_id)
            if direct_by_id:
                return direct_by_id

        return ()

    @staticmethod
    def _voice_prompt(task: ReminderTask) -> str:
        """Compose a short Russian summary for Twilio to read out."""
        due = task.due_human if task.due_human and task.due_human != "Не указан" else "без срока"
        status = task.status or "статус неизвестен"
        return f"Задача {task.name}. Статус: {status}. Срок: {due}."

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
            if entry.get("result") == "success":
                callback_id = entry.get("callback_id")
                if callback_id:
                    self._processed_callback_ids.add(str(callback_id))
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.debug("Failed to append callback log: %s", exc)

    def _load_processed_callback_ids(self) -> set[str]:
        if not self.callback_log_path:
            return set()
        try:
            lines = self.callback_log_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return set()
        except Exception as exc:
            LOGGER.debug("Failed to prime processed callback cache: %s", exc)
            return set()

        processed: set[str] = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("result") != "success":
                continue
            callback_id = entry.get("callback_id")
            if callback_id:
                processed.add(str(callback_id))
        return processed

    def _is_callback_processed(self, callback_id: Optional[str]) -> bool:
        if not callback_id:
            return False
        return str(callback_id) in self._processed_callback_ids

    def ensure_callback_comments(self, max_entries: int = 20) -> None:
        """
        Verify that recent successful callback entries have a corresponding ClickUp comment.

        Raises:
            RuntimeError: if any recent task lacks the expected audit comment.
        """
        if not self.callback_log_path:
            LOGGER.debug("Callback log path is not configured; skipping comment verification.")
            return

        try:
            lines = self.callback_log_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            LOGGER.debug("Callback log file %s not found; skipping comment verification.", self.callback_log_path)
            return
        except Exception as exc:
            LOGGER.warning("Failed to read callback log %s: %s", self.callback_log_path, exc)
            return

        if not lines:
            LOGGER.debug("Callback log file %s is empty; nothing to verify.", self.callback_log_path)
            return

        recent_entries: Dict[str, Dict[str, Any]] = {}
        for line in reversed(lines):
            if len(recent_entries) >= max_entries:
                break
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.debug("Skipping malformed callback log line: %s", line)
                continue
            if entry.get("result") != "success":
                continue
            task_id = str(entry.get("task_id") or "").strip()
            if not task_id:
                continue
            if task_id not in recent_entries:
                recent_entries[task_id] = entry

        if not recent_entries:
            LOGGER.debug("No successful callback entries found for verification.")
            return

        missing: List[str] = []
        expected_fragment = "Статус обновлен через Telegram-бота"

        for task_id, entry in recent_entries.items():
            matched = False
            for client in self.clickup_clients:
                try:
                    comments = client.fetch_comments(task_id)
                except Exception as exc:
                    LOGGER.debug("Failed to fetch comments for task %s via team %s: %s", task_id, client.team_id, exc)
                    continue
                for comment in comments:
                    comment_text = comment.get("comment_text") or comment.get("text") or ""
                    if expected_fragment in comment_text:
                        status_suffix = entry.get("status_key")
                        if not status_suffix or f"{expected_fragment}: {status_suffix}" in comment_text:
                            matched = True
                            break
                        if expected_fragment in comment_text and not status_suffix:
                            matched = True
                            break
                if matched:
                    break
            if not matched:
                missing.append(task_id)

        if missing:
            raise RuntimeError(
                "Не найдены комментарии с отметкой Telegram для задач: " + ", ".join(sorted(set(missing)))
            )

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

            assignee_name, assignee_id = _assignee_identity(task)

            pending.append(
                ReminderTask(
                    task_id=str(task["id"]),
                    name=str(task.get("name", "Без названия")),
                    status=status_obj.get("status") or status_obj.get("name") or "—",
                    due_human=_format_due(task.get("due_date"), self.timezone_name),
                    assignee=assignee_name,
                    assignee_id=assignee_id,
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
        telegram_cfg = self.config.get("telegram") or {}
        buttons_per_row = telegram_cfg.get("buttons_per_row", 3)
        try:
            buttons_per_row_int = int(buttons_per_row)
            if buttons_per_row_int <= 0:
                buttons_per_row_int = 3
        except (TypeError, ValueError):
            buttons_per_row_int = 3

        keyboard_buttons = [
            {
                "text": action["text"],
                "callback_data": f"s:{task.task_id}:{action['code']}",
            }
            for action in self.status_actions
        ]

        inline_keyboard: List[List[Dict[str, Any]]] = []
        for idx in range(0, len(keyboard_buttons), buttons_per_row_int):
            inline_keyboard.append(keyboard_buttons[idx : idx + buttons_per_row_int])

        reply_markup = {"inline_keyboard": inline_keyboard}
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
    ) -> Dict[str, List[ReminderTask]]:
        deliveries: Dict[str, List[ReminderTask]] = {}

        for task in tasks:
            chat_ids = self._chat_targets_for_task(task)
            if not chat_ids:
                LOGGER.warning(
                    "No Telegram chat mapping for assignee '%s' (id=%s, task %s). Skipping.",
                    task.assignee or "—",
                    task.assignee_id or "—",
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
            try:
                self.send_plain_message(chat_id, "✅ На данный момент нет задач, требующих внимания.")
            except Exception as exc:  # pragma: no cover - network guard
                LOGGER.error("Failed to send empty-state message to %s: %s", chat_id, exc)
            return

        preface = (
            f"📌 Найдено задач: {len(tasks)}. "
            "Отметьте статус прямо в боте — выбор обновит задачу в ClickUp."
        )
        try:
            self.send_plain_message(chat_id, preface)
        except Exception as exc:  # pragma: no cover - network guard
            LOGGER.error("Failed to send preface to chat %s: %s", chat_id, exc)
            return

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

            target_chat_str = str(target_chat)
            deliveries = self._group_tasks_by_chat(tasks)
            bucket = deliveries.get(target_chat_str, [])

            self._dispatch_tasks_to_chat(target_chat_str, bucket)
            return tasks

        if not tasks:
            LOGGER.info("No pending tasks to send.")
            return []

        deliveries = self._group_tasks_by_chat(tasks)
        if not deliveries:
            LOGGER.warning(
                "Не удалось сопоставить ни одну задачу с Telegram чатами исполнителей. "
                "Проверьте telegram.assignee_chat_map в config.json."
            )
            return []

        for target_chat, bucket in deliveries.items():
            self._dispatch_tasks_to_chat(target_chat, bucket)

        return tasks

    def send_voice_reminders(
        self,
        assignees: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Place Twilio voice calls for pending reminders grouped by phone mapping.

        Args:
            assignees: Optional iterable of assignee names to target (case-insensitive).
            limit: Optional max amount of tasks to pull from ClickUp.
            dry_run: When True, skip actual Twilio API calls and only log the plan.
        """
        if not self.twilio_service or not self.twilio_from_phone:
            raise ConfigurationError(
                "Twilio credentials are not configured. "
                "Set TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_PHONE_NUMBER or provide twilio.* in secrets."
            )
        if not self.phone_mapping:
            raise ConfigurationError("phone_mapping in config.json is empty — нечего прозванивать.")

        allowed_assignees: Optional[set[str]] = None
        if assignees:
            allowed_assignees = set()
            for name in assignees:
                normalized = self._normalize_assignee_name(name)
                if normalized:
                    allowed_assignees.add(normalized)
            if not allowed_assignees:
                raise ConfigurationError(
                    "Не удалось сопоставить указанных исполнителей с phone_mapping. Проверьте написание имён."
                )

        tasks = self.fetch_pending_tasks(limit=limit)
        if not tasks:
            LOGGER.info("Нет задач для голосовых напоминаний.")
            return []

        grouped: Dict[str, List[ReminderTask]] = {}
        skipped: List[str] = []

        for task in tasks:
            normalized = self._normalize_assignee_name(task.assignee)
            if allowed_assignees is not None and normalized not in allowed_assignees:
                continue

            phone = self.phone_mapping.get(normalized)
            if not phone:
                if task.assignee:
                    skipped.append(task.assignee)
                else:
                    skipped.append(task.task_id)
                continue

            grouped.setdefault(phone, []).append(task)

        if not grouped:
            LOGGER.info("Не найдено задач с телефонными номерами для голосовых напоминаний.")
            if skipped:
                LOGGER.debug(
                    "Пропущено %s задач без phone_mapping (пример: %s)",
                    len(skipped),
                    ", ".join(sorted(set(skipped))[:3]),
                )
            return []

        if skipped:
            LOGGER.debug(
                "Пропущено %s задач без phone_mapping (пример: %s)",
                len(skipped),
                ", ".join(sorted(set(skipped))[:3]),
            )

        deliveries: List[Dict[str, Any]] = []
        for phone, bucket in grouped.items():
            messages = [self._voice_prompt(task) for task in bucket]
            result = None
            if dry_run:
                LOGGER.info(
                    "Dry-run: пропущен звонок на %s (задач: %s)",
                    phone,
                    len(bucket),
                )
            else:
                result = self.twilio_service.make_call(
                    from_phone=self.twilio_from_phone,
                    to_phone=phone,
                    task_messages=messages,
                )
                status = "успех" if result.success else f"ошибка ({result.error or result.status})"
                LOGGER.info(
                    "Twilio звонок на %s — %s (задач: %s)",
                    phone,
                    status,
                    len(bucket),
                )

            deliveries.append(
                {
                    "phone": phone,
                    "assignees": sorted(
                        {task.assignee for task in bucket if task.assignee and task.assignee != "—"}
                    )
                    or ["—"],
                    "task_ids": [task.task_id for task in bucket],
                    "call_result": result,
                    "dry_run": dry_run,
                }
            )

        return deliveries

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

        if chat_id and not self._configured_default_chat:
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

        if self._is_callback_processed(callback_id):
            LOGGER.debug("Callback %s already processed; skipping duplicate update.", callback_id)
            return

        if not data.startswith("s:") or data.count(":") != 2:
            if callback_id:
                self.answer_callback(callback_id, "Неизвестное действие", show_alert=True)
            return

        _, task_id, action_code = data.split(":")
        action = self.status_action_map.get(action_code)
        if not action:
            if callback_id:
                self.answer_callback(callback_id, "Действие не поддерживается", show_alert=True)
            return
        status_key = action["key"]

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

        if chat_id and not self._configured_default_chat:
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
