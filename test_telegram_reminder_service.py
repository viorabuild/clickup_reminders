#!/usr/bin/env python3
from __future__ import annotations

import copy
import unittest
from unittest.mock import MagicMock, patch

from telegram_reminder_service import ReminderTask, TelegramReminderService


class DummyResponse:
    def __init__(self, payload: dict | None = None):
        self._payload = payload or {"ok": True, "result": {}}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class DummySession:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return DummyResponse()


class TelegramReminderServiceTest(unittest.TestCase):
    def setUp(self):
        self.base_config = {
            "clickup": {
                "status_mapping": {
                    "ВЫПОЛНЕНО": "complete",
                    "НЕ_ВЫПОЛНЕНО": "todo",
                    "В_РАБОТЕ": "in progress",
                },
                "completed_status": "complete",
                "reminder_tag": "#напоминание",
            },
            "reminder_list_name": "Напоминания",
            "working_hours": {"timezone": "Europe/Lisbon"},
        }
        self.credentials = {
            "clickup_api_key": "key",
            "clickup_team_id": "123",
            "telegram_bot_token": "token",
            "telegram_chat_id": "42",
        }

    @patch("telegram_reminder_service.ClickUpClient")
    def test_send_reminders_distributes_tasks_by_assignee(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        config = copy.deepcopy(self.base_config)
        config.update(
            {
                "telegram": {
                    "chat_id": "42",
                    "assignee_chat_ids": {"Alex": "100", "Eve": 101},
                },
            }
        )
        session = DummySession()
        service = TelegramReminderService(config, self.credentials, session=session)

        tasks = [
            ReminderTask("t1", "Task 1", "todo", "2024-01-01", "Alex", "url1"),
            ReminderTask("t2", "Task 2", "todo", "2024-01-02", "Eve", "url2"),
            ReminderTask("t3", "Task 3", "todo", "2024-01-03", "—", "url3"),
        ]

        with patch.object(service, "fetch_pending_tasks", return_value=tasks):
            with patch.object(service, "send_task_message") as send_task, patch.object(
                service, "send_plain_message"
            ) as send_plain:
                service.send_reminders()

        sent_chats = [call.kwargs["chat_id"] if call.kwargs else call.args[0] for call in send_plain.call_args_list]
        self.assertIn("100", sent_chats)
        self.assertIn("101", sent_chats)
        self.assertIn("42", sent_chats)

        task_dispatch = [
            call.kwargs.get("chat_id", call.args[0])
            for call in send_task.call_args_list
        ]
        self.assertIn("100", task_dispatch)
        self.assertIn("101", task_dispatch)
        self.assertIn("42", task_dispatch)

    @patch("telegram_reminder_service.ClickUpClient")
    def test_fetch_pending_tasks_filters_completed(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.fetch_tasks_by_tags.return_value = [
            {
                "id": "task-complete",
                "name": "Done task",
                "status": {"status": "complete", "type": "done"},
                "due_date": None,
                "assignees": [{"username": "Alex"}],
                "tags": [{"name": "напоминание"}],
            },
            {
                "id": "task-pending",
                "name": "Pending task",
                "status": {"status": "in progress"},
                "due_date": "1710000000000",
                "assignees": [{"username": "Eve"}],
                "tags": [{"name": "напоминание"}],
            },
        ]
        mock_client_cls.return_value = mock_client

        service = TelegramReminderService(
            copy.deepcopy(self.base_config), self.credentials, session=DummySession()
        )
        tasks = service.fetch_pending_tasks()

        self.assertEqual(1, len(tasks))
        self.assertIsInstance(tasks[0], ReminderTask)
        self.assertEqual("task-pending", tasks[0].task_id)
        self.assertEqual("Pending task", tasks[0].name)
        mock_client.fetch_tasks_by_tags.assert_called_once_with(["#напоминание"])
        mock_client.fetch_tasks.assert_not_called()

    @patch("telegram_reminder_service.ClickUpClient")
    def test_fetch_pending_tasks_fallbacks_to_list_when_no_tagged(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.fetch_tasks_by_tags.return_value = []
        mock_client.fetch_tasks.return_value = [
            {
                "id": "task-from-list",
                "name": "List task",
                "status": {"status": "in progress"},
                "due_date": None,
                "assignees": [{"username": "Alex"}],
            }
        ]
        mock_client_cls.return_value = mock_client

        service = TelegramReminderService(
            copy.deepcopy(self.base_config), self.credentials, session=DummySession()
        )
        tasks = service.fetch_pending_tasks()

        self.assertEqual(1, len(tasks))
        mock_client.fetch_tasks_by_tags.assert_called_once_with(["#напоминание"])
        mock_client.fetch_tasks.assert_called_once()

    @patch("telegram_reminder_service.ClickUpClient")
    def test_handle_callback_updates_status_and_notifies(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        session = DummySession()
        service = TelegramReminderService(
            copy.deepcopy(self.base_config), self.credentials, session=session
        )

        with patch.object(service, "update_clickup_status") as update_status, patch.object(
            service, "fetch_task_details", return_value={"name": "Demo task"}
        ), patch.object(service, "answer_callback", wraps=service.answer_callback) as answer_cb, patch.object(
            service, "_persist_chat_id"
        ):
            callback = {
                "id": "cb-1",
                "data": "s:task-pending:d",
                "message": {"chat": {"id": 42}, "message_id": 99},
            }
            service.handle_callback(callback)

        update_status.assert_called_once_with("task-pending", "ВЫПОЛНЕНО")

        methods = [call["url"].split("/")[-1] for call in session.calls]
        self.assertIn("answerCallbackQuery", methods)
        self.assertIn("editMessageText", methods)
        self.assertNotIn("sendMessage", methods)

        edit_payload = next(call["json"] for call in session.calls if call["url"].endswith("editMessageText"))
        self.assertIn("Demo task", edit_payload["text"])
        self.assertIn("ВЫПОЛНЕНО", edit_payload["text"])
        self.assertIsNone(edit_payload["reply_markup"])
        answer_cb.assert_called_once()

    @patch("telegram_reminder_service.ClickUpClient")
    def test_handle_callback_without_chat_uses_default(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        session = DummySession()
        service = TelegramReminderService(
            copy.deepcopy(self.base_config), self.credentials, session=session
        )

        with patch.object(service, "update_clickup_status") as update_status, patch.object(
            service, "fetch_task_details", return_value={"name": "Demo task"}
        ), patch.object(service, "answer_callback", wraps=service.answer_callback) as answer_cb, patch.object(
            service, "_persist_chat_id"
        ):
            callback = {
                "id": "cb-2",
                "data": "s:task-pending:d",
                "message": {"message_id": 55},
            }
            service.handle_callback(callback)

        update_status.assert_called_once_with("task-pending", "ВЫПОЛНЕНО")
        answer_cb.assert_called_once()

        send_payload = next(call["json"] for call in session.calls if call["url"].endswith("sendMessage"))
        self.assertEqual("42", send_payload["chat_id"])


if __name__ == "__main__":
    unittest.main()
