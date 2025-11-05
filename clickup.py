from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import requests


class ClickUpClient:
    """Tiny ClickUp API helper focused on the reminder workflow."""

    BASE_URL = "https://api.clickup.com/api/v2"

    def __init__(self, api_key: str, team_id: str):
        self.team_id = team_id
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": api_key,
                "Content-Type": "application/json",
            }
        )

    def fetch_tasks(
        self,
        list_name: str | None = None,
        list_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve tasks either by explicit list id or by name lookup."""
        resolved_list_id = list_id or (self._resolve_list_id(list_name) if list_name else None)
        if not resolved_list_id:
            return []

        return self._fetch_list_tasks(str(resolved_list_id))

    def fetch_tasks_by_tags(self, tags: Sequence[str]) -> List[Dict[str, Any]]:
        """Collect tasks across the workspace using ClickUp's tag filter."""

        unique_tags: List[str] = []
        seen: set[str] = set()
        for tag in tags:
            if tag is None:
                continue
            candidate = str(tag).strip()
            if not candidate:
                continue
            candidate = candidate.lstrip("#")
            normalized = candidate.casefold()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_tags.append(candidate)

        if not unique_tags:
            return []

        tasks: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        page = 0
        while True:
            params = [("page", page)] + [("tags[]", tag) for tag in unique_tags]
            response = self.session.get(
                f"{self.BASE_URL}/team/{self.team_id}/task", params=params
            )
            response.raise_for_status()
            payload = response.json() or {}
            chunk = payload.get("tasks") or []

            appended = False
            for task in chunk:
                if not isinstance(task, dict):
                    continue
                task_id_raw = task.get("id")
                task_id = str(task_id_raw) if task_id_raw is not None else ""
                if not task_id or task_id in seen_ids:
                    continue
                seen_ids.add(task_id)
                tasks.append(task)
                appended = True

            has_more = bool(payload.get("has_more"))
            if not has_more or not appended:
                break
            page += 1

        return tasks

    def fetch_task(self, task_id: str) -> Dict[str, Any]:
        """Retrieve a single task payload by id."""
        response = self.session.get(f"{self.BASE_URL}/task/{task_id}")
        response.raise_for_status()
        return response.json()

    def update_status(self, task_id: str, status: str) -> None:
        """Update ClickUp task status."""
        response = self.session.put(
            f"{self.BASE_URL}/task/{task_id}",
            json={"status": status},
        )
        response.raise_for_status()

    def add_comment(self, task_id: str, comment_text: str) -> None:
        """Attach a comment to the task if needed."""
        response = self.session.post(
            f"{self.BASE_URL}/task/{task_id}/comment",
            json={"comment_text": comment_text},
        )
        response.raise_for_status()

    @lru_cache
    def _resolve_list_id(self, list_name: str) -> Optional[str]:
        """Walk the ClickUp hierarchy to find the list id by name."""
        spaces = self._fetch_spaces()
        for space in spaces:
            space_id = space.get("id")
            if not space_id:
                continue
            for lst in self._fetch_lists(space_id):
                if lst.get("name") == list_name:
                    return lst.get("id")
        return None

    def _fetch_spaces(self) -> List[Dict[str, Any]]:
        response = self.session.get(f"{self.BASE_URL}/team/{self.team_id}/space")
        response.raise_for_status()
        payload = response.json()
        return payload.get("spaces", [])

    def _fetch_lists(self, space_id: str) -> List[Dict[str, Any]]:
        response = self.session.get(f"{self.BASE_URL}/space/{space_id}/list")
        response.raise_for_status()
        payload = response.json()
        return payload.get("lists", [])

    def _fetch_list_tasks(self, list_id: str) -> List[Dict[str, Any]]:
        response = self.session.get(f"{self.BASE_URL}/list/{list_id}/task")
        response.raise_for_status()
        payload = response.json()
        return payload.get("tasks", [])

