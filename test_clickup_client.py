from __future__ import annotations

from typing import Any, Dict, List

from unittest.mock import MagicMock

import requests

from clickup import ClickUpClient


class DummyResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Dict[str, Any]:
        return self._payload


def make_client() -> ClickUpClient:
    client = ClickUpClient(api_key="token", team_id="42")
    client.session = MagicMock()
    return client


def test_fetch_tasks_by_tags_deduplicates_tags_and_tasks() -> None:
    client = make_client()
    session_mock = client.session

    first_page = DummyResponse({"tasks": [{"id": "1", "name": "Task A"}], "has_more": True})
    second_page = DummyResponse({
        "tasks": [{"id": "1", "name": "Task A"}, {"id": "2", "name": "Task B"}],
        "has_more": False,
    })

    session_mock.get.side_effect = [first_page, second_page]

    tasks = client.fetch_tasks_by_tags(["  #Напоминание", "#напоминание", "Другое", "другое", " "])

    assert [task["id"] for task in tasks] == ["1", "2"]

    assert session_mock.get.call_count == 2
    first_call_kwargs = session_mock.get.call_args_list[0].kwargs
    params: List[tuple[str, str]] = first_call_kwargs["params"]

    assert ("tags[]", "Напоминание") in params
    assert ("tags[]", "Другое") in params
    assert params[0] == ("page", 0)


def test_fetch_tasks_by_tags_returns_empty_without_valid_tags() -> None:
    client = make_client()
    tasks = client.fetch_tasks_by_tags(["", None, "   "])
    assert tasks == []
    client.session.get.assert_not_called()

