#!/usr/bin/env python3
from __future__ import annotations

"""Utility script to list pending reminders from ClickUp."""

import os  # Работа с переменными окружения и путями
import sys  # Управление путями импорта
from datetime import datetime  # Преобразование времени дедлайна
from pathlib import Path  # Удобная работа с путями
from typing import Any, Dict, List, Mapping, Sequence  # Аннотации типов для наглядности
import requests  # HTTP-запросы к ClickUp API

# Ensure we can import project modules when the script is run standalone.
ROOT = Path(__file__).resolve().parents[1]  # Корень проекта
if str(ROOT) not in sys.path:  # Гарантируем доступ к локальным модулям
    sys.path.insert(0, str(ROOT))

try:
    from clickup import ClickUpClient
except Exception as exc:  # pragma: no cover - script mode convenience
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(2)


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load config JSON. Defaults to repo `config.json` if not overridden."""
    import json  # Локальный импорт, чтобы не грузить модуль раньше времени

    if path is None:
        env_override = os.getenv("CONFIG_PATH")  # Проверяем переопределение пути через переменную окружения
        if env_override:
            path = Path(env_override).expanduser()  # Используем указанный конфиг
        else:
            # fallback to a local config.json next to repo root
            path = Path(__file__).resolve().parents[1] / "config.json"

    if not path.exists():  # Если файла нет — сразу сообщаем об ошибке
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:  # Читаем конфигурацию
        return json.load(fh)  # Возвращаем словарь с настройками


def _extract(payload: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> str | None:
    """Return first matching string value from a nested secrets structure."""
    for path in paths:
        node: Any = payload
        for key in path:
            if isinstance(node, Mapping) and key in node:
                node = node[key]
            else:
                node = None
                break
        if node is None:
            continue
        # Support formats where the value is nested under { "value": ... }
        if isinstance(node, Mapping) and "value" in node:
            candidate = node["value"]
        else:
            candidate = node
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def try_load_clickup_from_secrets() -> tuple[str | None, str | None]:
    """Attempt to load ClickUp API key and team id from a secrets json file.

    Search order:
      1) $SECRETS_PATH (if set)
      2) <repo>/.venv/bin/secrets.json
      3) <repo>/../.venv/bin/secrets.json
      4) ~/.config/abacusai_auth_secrets.json
    """
    candidates: List[Path] = []  # Собираем список возможных путей к секретам
    env_path = os.getenv("SECRETS_PATH")  # Приоритет — переменная окружения
    if env_path:
        candidates.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root / ".venv/bin/secrets.json")  # Стандартный путь в проекте
    candidates.append(repo_root.parent / ".venv/bin/secrets.json")  # Путь уровнем выше (совместимость)
    candidates.append(Path.home() / ".config/abacusai_auth_secrets.json")  # Глобальный конфиг

    for path in candidates:  # Перебираем возможные пути
        try:
            if not path.exists():  # Пропускаем, если файла нет
                continue
            import json  # Импортируем модуль JSON локально

            with open(path, "r", encoding="utf-8") as fh:  # Открываем найденный secrets.json
                payload: Dict[str, Any] = json.load(fh)  # Загружаем содержимое

            api_key = _extract(
                payload,
                (
                    ("clickup", "api_key"),
                    ("telegram", "secrets", "clickup_api_key"),
                ),
            )
            team_id = _extract(
                payload,
                (
                    ("clickup", "team_id"),
                    ("telegram", "secrets", "clickup_team_id"),
                ),
            )
            if api_key and team_id:  # Если нашли оба значения — возвращаем
                return api_key, team_id
        except Exception:
            continue  # Игнорируем ошибки чтения и переходим к следующему пути
    return None, None  # Если ничего не нашли — возвращаем пустые значения


def resolve_team_id(api_key: str, team_id: str | None) -> str | None:
    """Validate or discover a usable team_id via ClickUp API.

    If the provided team_id is not found in the /team listing, pick the first
    accessible team id.
    """
    try:
        headers = {"Authorization": api_key, "Content-Type": "application/json"}  # Заголовки для ClickUp API
        resp = requests.get("https://api.clickup.com/api/v2/team", headers=headers, timeout=15)  # Получаем список команд
        resp.raise_for_status()  # Проверяем успешность ответа
        payload = resp.json()  # Разбираем JSON
        teams = payload.get("teams") or payload.get("teams", [])  # Берем массив команд
        if not isinstance(teams, list) or not teams:  # Если список пустой — возвращаем исходное значение
            return team_id

        ids = [str(t.get("id")) for t in teams if t.get("id")]  # Собираем доступные ID команд
        if team_id and str(team_id) in ids:  # Если текущий ID валиден
            return str(team_id)
        return ids[0] if ids else team_id  # Иначе берем первый доступный ID
    except Exception:
        return team_id  # В случае ошибки оставляем исходное значение


def human_due(ms: str | int | None) -> str:
    """Render ClickUp due timestamp (ms) in a human friendly format."""
    if not ms:
        return "—"
    try:
        ts = int(ms) / 1000
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ms)  # Если не удалось преобразовать — возвращаем исходное значение


def main() -> int:
    try:
        cfg = load_config()  # Загружаем конфигурацию напоминаний
    except Exception as e:
        print(f"Failed to load config: {e}", file=sys.stderr)  # Сообщаем о проблеме чтения конфига
        return 2  # Неверная конфигурация

    # Load ClickUp credentials from environment variables
    api_key = os.getenv("CLICKUP_API_KEY")  # Пытаемся взять ключ из окружения
    team_id = os.getenv("CLICKUP_TEAM_ID") or str(cfg.get("clickup_workspace_id") or "")  # Workspace по умолчанию из конфигурации

    # Secrets file can supply missing credentials when env vars are absent.
    file_api_key, file_team_id = try_load_clickup_from_secrets()  # Загружаем секреты из файла
    if not api_key and file_api_key:
        api_key = file_api_key  # Переопределяем ключ, если он не был задан
    if (not team_id) and file_team_id:
        team_id = file_team_id  # Переопределяем team_id, если нужно
    if not api_key or not team_id:
        print(
            "Missing CLICKUP_API_KEY and/or CLICKUP_TEAM_ID environment variables.",
            file=sys.stderr,  # Печатаем сообщение об отсутствующих переменных окружения
        )
        print(
            "Export them, e.g.:\n  export CLICKUP_API_KEY=...\n  export CLICKUP_TEAM_ID=...",
            file=sys.stderr,  # Подсказка по командам экспорта
        )
        # Also mention secrets path fallback
        print(
            "Or set SECRETS_PATH to your secrets json file (supports ClickUp structure).",
            file=sys.stderr,  # Напоминание про переменную SECRETS_PATH
        )
        return 2  # Выходим с ошибкой, если нет учётных данных

    list_name = (
        cfg.get("reminder_list_name")  # Современное имя поля
        or cfg.get("reminders_list_name")  # Обратная совместимость с legacy
        or "Напоминания"  # Значение по умолчанию
    )

    def collect_tags(config: Dict[str, Any]) -> List[str]:
        tags: List[str] = []

        def _extend(value: Any) -> None:
            if not value:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _extend(item)
                return
            if isinstance(value, str):
                candidate = value.strip()
            else:
                candidate = str(value).strip()
            if candidate and candidate not in tags:
                tags.append(candidate)

        clickup_cfg = config.get("clickup", {})
        _extend(clickup_cfg.get("reminder_tag"))
        _extend(clickup_cfg.get("reminder_tags"))
        _extend(config.get("reminder_tag"))
        _extend(config.get("reminder_tags"))

        if not tags:
            tags.append("#напоминание")

        return tags

    reminder_tags = collect_tags(cfg)

    # Ensure the ClickUp team id is valid for this token.
    team_id = resolve_team_id(api_key, team_id) or team_id  # Проверяем, что ID команды актуален

    client = ClickUpClient(api_key=api_key, team_id=team_id)  # Создаем клиента ClickUp

    tasks: List[Dict[str, Any]] = []

    try:
        if reminder_tags:
            tasks = client.fetch_tasks_by_tags(reminder_tags)
    except Exception as e:
        print(f"ClickUp API error while fetching tags {reminder_tags}: {e}", file=sys.stderr)
        return 1

    if not tasks:
        try:
            tasks = client.fetch_tasks(list_name)
        except Exception as e:
            print(f"ClickUp API error: {e}", file=sys.stderr)  # Сообщаем об ошибке запроса к ClickUp
            return 1

    if not tasks:
        print(
            f"No tasks with tags {', '.join(reminder_tags)} and no tasks in list '{list_name}'."
        )  # Сообщаем, что задач нет
        return 0

    print(
        f"Reminders for tags {', '.join(reminder_tags)} (fallback list '{list_name}'):"
    )  # Шапка списка напоминаний
    for t in tasks:  # Проходим по каждой задаче
        task_id = t.get("id")  # ID задачи в ClickUp
        name = t.get("name")  # Название задачи
        status = (t.get("status") or {}).get("status") or t.get("status")  # Текущий статус
        due = human_due(t.get("due_date"))  # Время дедлайна в читабельном виде
        print(f"- {name} | status: {status} | due: {due} | id: {task_id}")  # Выводим строку списка

    print(f"Total: {len(tasks)} task(s)")  # Счётчик задач
    return 0  # Выходим с кодом успеха


if __name__ == "__main__":
    raise SystemExit(main())  # Запускаем скрипт и пробрасываем код возврата в оболочку
