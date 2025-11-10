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

        params = {"include_assignees": "true"}
        response = self.session.get(f"{self.BASE_URL}/list/{resolved_list_id}/task", params=params)
        response.raise_for_status()
        payload = response.json()
        tasks = payload.get("tasks", [])
        return tasks

    def fetch_task(self, task_id: str) -> Dict[str, Any]:
        """Retrieve a single task payload by id."""
        response = self.session.get(f"{self.BASE_URL}/task/{task_id}")
        response.raise_for_status()
        return response.json()

    def fetch_tasks_by_tag(self, tag: str, space_ids: Sequence[str] | None = None) -> List[Dict[str, Any]]:
        """Return every task across the workspace that contains the provided tag."""

        base = str(tag or "").strip()
        if not base:
            return []

        tag_variants: List[str] = []
        if base not in tag_variants:
            tag_variants.append(base)
        if base.startswith("#"):
            stripped = base.lstrip("#").strip()
            if stripped and stripped not in tag_variants:
                tag_variants.append(stripped)
        else:
            with_hash = f"#{base}"
            if with_hash not in tag_variants:
                tag_variants.append(with_hash)

        overall: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        for variant in tag_variants:
            next_page: Optional[int] = 0
            while next_page is not None:
                params: Dict[str, Any] = {
                    "subtasks": "true",
                    "page": next_page,
                    "tags[]": variant,
                    "include_assignees": "true",
                }
                if space_ids:
                    params["space_ids[]"] = list(space_ids)
                response = self.session.get(f"{self.BASE_URL}/team/{self.team_id}/task", params=params)
                response.raise_for_status()

                payload = response.json()
                for task in payload.get("tasks", []):
                    task_id = str(task.get("id") or "").strip()
                    if not task_id or task_id in seen_ids:
                        continue
                    overall.append(task)
                    seen_ids.add(task_id)
                raw_next = payload.get("next_page")
                next_page = raw_next if isinstance(raw_next, int) else None

            if overall:
                break

        return overall

    def update_status(self, task_id: str, status: str) -> None:
        """Update ClickUp task status."""
        response = self.session.put(
            f"{self.BASE_URL}/task/{task_id}",
            json={"status": status},
        )
        response.raise_for_status()

    def update_task(self, task_id: str, payload: Dict[str, Any]) -> None:
        """Update arbitrary fields of a ClickUp task."""
        if not isinstance(payload, dict) or not payload:
            raise ValueError("payload must be a non-empty dict")

        response = self.session.put(
            f"{self.BASE_URL}/task/{task_id}",
            json=payload,
        )
        response.raise_for_status()

    def add_comment(self, task_id: str, comment_text: str) -> None:
        """Attach a comment to the task if needed."""
        response = self.session.post(
            f"{self.BASE_URL}/task/{task_id}/comment",
            json={"comment_text": comment_text},
        )
        response.raise_for_status()

    def fetch_comments(self, task_id: str) -> List[Dict[str, Any]]:
        """Fetch comments for a task to verify audit trail."""
        response = self.session.get(f"{self.BASE_URL}/task/{task_id}/comment")
        response.raise_for_status()
        payload = response.json()
        return payload.get("comments", [])

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
