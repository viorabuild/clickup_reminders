#!/usr/bin/env python3
from __future__ import annotations

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
        self.config = {
            "clickup": {
                "status_mapping": {
                    "ВЫПОЛНЕНО": "complete",
                    "НЕ_ВЫПОЛНЕНО": "todo",
                    "В_РАБОТЕ": "in progress",
                },
                "completed_status": "complete",
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
    def test_fetch_pending_tasks_filters_completed(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.fetch_tasks.return_value = [
            {
                "id": "task-complete",
                "name": "Done task",
                "status": {"status": "complete", "type": "done"},
                "due_date": None,
                "assignees": [{"username": "Alex"}],
            },
            {
                "id": "task-pending",
                "name": "Pending task",
                "status": {"status": "in progress"},
                "due_date": "1710000000000",
                "assignees": [{"username": "Eve"}],
            },
        ]
        mock_client_cls.return_value = mock_client

        service = TelegramReminderService(self.config, self.credentials, session=DummySession())
        tasks = service.fetch_pending_tasks()

        self.assertEqual(1, len(tasks))
        self.assertIsInstance(tasks[0], ReminderTask)
        self.assertEqual("task-pending", tasks[0].task_id)
        self.assertEqual("Pending task", tasks[0].name)

    @patch("telegram_reminder_service.ClickUpClient")
    def test_handle_callback_updates_status_and_notifies(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        session = DummySession()
        service = TelegramReminderService(self.config, self.credentials, session=session)

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
        self.assertIn("editMessageReplyMarkup", methods)
        self.assertIn("sendMessage", methods)

        send_payload = next(call["json"] for call in session.calls if call["url"].endswith("sendMessage"))
        self.assertIn("Demo task", send_payload["text"])
        self.assertIn("ВЫПОЛНЕНО", send_payload["text"])
        answer_cb.assert_called_once()

    @patch("telegram_reminder_service.ClickUpClient")
    def test_handle_callback_without_chat_uses_default(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        session = DummySession()
        service = TelegramReminderService(self.config, self.credentials, session=session)

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

    @patch("telegram_reminder_service.ClickUpClient")
    def test_fetch_pending_tasks_with_tags_deduplicates(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.fetch_tasks_by_tag.side_effect = [
            [
                {"id": "1", "name": "Task A", "status": {"status": "open"}, "due_date": None},
                {"id": "2", "name": "Task B", "status": {"status": "open"}, "due_date": None},
            ],
            [
                {"id": "2", "name": "Task B", "status": {"status": "open"}, "due_date": None},
                {"id": "3", "name": "Task C", "status": {"status": "done"}, "due_date": None},
            ],
        ]
        mock_client_cls.return_value = mock_client

        config = dict(self.config)
        config["clickup"] = dict(self.config["clickup"], reminder_tags=["#напоминание", "важно"])

        service = TelegramReminderService(config, self.credentials, session=DummySession())
        tasks = service.fetch_pending_tasks()

        self.assertEqual([task.task_id for task in tasks], ["1", "2"])
        mock_client.fetch_tasks_by_tag.assert_any_call("#напоминание")
        mock_client.fetch_tasks_by_tag.assert_any_call("важно")

    @patch("telegram_reminder_service.ClickUpClient")
    def test_callback_failure_reports_to_user(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        session = DummySession()
        service = TelegramReminderService(self.config, self.credentials, session=session)

        with patch.object(service, "update_clickup_status", side_effect=RuntimeError("boom")), patch.object(
            service, "_persist_chat_id"
        ):
            callback = {
                "id": "cb-err",
                "data": "s:task-err:d",
                "message": {"chat": {"id": 99}, "message_id": 77},
            }
            service.handle_callback(callback)

        methods = [call["url"].split("/")[-1] for call in session.calls]
        self.assertIn("answerCallbackQuery", methods)
        self.assertIn("sendMessage", methods)

        payloads = [call["json"] for call in session.calls if call["url"].endswith("sendMessage")]
        self.assertTrue(any("Не удалось обновить задачу" in payload["text"] for payload in payloads))


if __name__ == "__main__":
    unittest.main()
