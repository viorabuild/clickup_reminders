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
        complete_code = next(action["code"] for action in service.status_actions if action["key"] == "ВЫПОЛНЕНО")

        with patch.object(service, "update_clickup_status") as update_status, patch.object(
            service, "fetch_task_details", return_value={"name": "Demo task"}
        ), patch.object(service, "answer_callback", wraps=service.answer_callback) as answer_cb, patch.object(
            service, "_persist_chat_id"
        ), patch.object(service, "_append_callback_log") as log_callback:
            callback = {
                "id": "cb-1",
                "data": f"s:task-pending:{complete_code}",
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
        log_callback.assert_called_once()
        log_payload = log_callback.call_args[0][0]
        self.assertEqual(log_payload["task_id"], "task-pending")
        self.assertEqual(log_payload["result"], "success")

    @patch("telegram_reminder_service.ClickUpClient")
    def test_handle_callback_without_chat_uses_default(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        session = DummySession()
        service = TelegramReminderService(self.config, self.credentials, session=session)
        complete_code = next(action["code"] for action in service.status_actions if action["key"] == "ВЫПОЛНЕНО")

        with patch.object(service, "update_clickup_status") as update_status, patch.object(
            service, "fetch_task_details", return_value={"name": "Demo task"}
        ), patch.object(service, "answer_callback", wraps=service.answer_callback) as answer_cb, patch.object(
            service, "_persist_chat_id"
        ), patch.object(service, "_append_callback_log") as log_callback:
            callback = {
                "id": "cb-2",
                "data": f"s:task-pending:{complete_code}",
                "message": {"message_id": 55},
            }
            service.handle_callback(callback)

        update_status.assert_called_once_with("task-pending", "ВЫПОЛНЕНО")
        answer_cb.assert_called_once()
        log_callback.assert_called_once()
        log_payload = log_callback.call_args[0][0]
        self.assertEqual(log_payload["task_id"], "task-pending")
        self.assertEqual(log_payload["result"], "success")

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
        mock_client.fetch_tasks_by_tag.assert_any_call("#напоминание", space_ids=None)
        mock_client.fetch_tasks_by_tag.assert_any_call("важно", space_ids=None)

    @patch("telegram_reminder_service.ClickUpClient")
    def test_callback_failure_reports_to_user(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        session = DummySession()
        service = TelegramReminderService(self.config, self.credentials, session=session)
        complete_code = next(action["code"] for action in service.status_actions if action["key"] == "ВЫПОЛНЕНО")

        with patch.object(service, "update_clickup_status", side_effect=RuntimeError("boom")), patch.object(
            service, "_persist_chat_id"
        ), patch.object(service, "_append_callback_log") as log_callback:
            callback = {
                "id": "cb-err",
                "data": f"s:task-err:{complete_code}",
                "message": {"chat": {"id": 99}, "message_id": 77},
            }
            service.handle_callback(callback)

        methods = [call["url"].split("/")[-1] for call in session.calls]
        self.assertIn("answerCallbackQuery", methods)
        self.assertIn("sendMessage", methods)

        payloads = [call["json"] for call in session.calls if call["url"].endswith("sendMessage")]
        self.assertTrue(any("Не удалось обновить задачу" in payload["text"] for payload in payloads))
        log_callback.assert_called_once()
        log_payload = log_callback.call_args[0][0]
        self.assertEqual(log_payload["task_id"], "task-err")
        self.assertEqual(log_payload["result"], "error")
        self.assertIn("boom", log_payload["error"])

    @patch("telegram_reminder_service.ClickUpClient")
    def test_send_reminders_routes_by_assignee(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        session = DummySession()
        config = {
            "clickup": {
                "status_mapping": {
                    "ВЫПОЛНЕНО": "complete",
                    "НЕ_ВЫПОЛНЕНО": "todo",
                    "В_РАБОТЕ": "in progress",
                },
                "completed_status": "complete",
            },
            "working_hours": {"timezone": "Europe/Lisbon"},
            "telegram": {
                "assignee_chat_map": {
                    "Alex": "chat-alex",
                    "Eve|Ева": ["chat-eve-1", "chat-eve-2"],
                }
            },
        }
        credentials = {
            "clickup_api_key": "key",
            "clickup_team_id": "123",
            "telegram_bot_token": "token",
            "telegram_chat_id": "fallback-chat",
        }

        service = TelegramReminderService(config, credentials, session=session)
        tasks = [
            ReminderTask(task_id="1", name="Task Alex", status="todo", due_human="2024-01-01 10:00", assignee="Alex", url="url-1"),
            ReminderTask(task_id="2", name="Task Eve", status="todo", due_human="2024-01-02 10:00", assignee="Ева", url="url-2"),
            ReminderTask(task_id="3", name="Task Unknown", status="todo", due_human="2024-01-03 10:00", assignee="—", url="url-3"),
        ]

        with patch.object(service, "fetch_pending_tasks", return_value=tasks):
            service.send_reminders()

        send_calls = [call for call in session.calls if call["url"].endswith("sendMessage")]
        messages_by_chat: dict[str, list[str]] = {}
        for call in send_calls:
            chat = call["json"]["chat_id"]
            messages_by_chat.setdefault(chat, []).append(call["json"]["text"])

        expected_chats = {"chat-alex", "chat-eve-1", "chat-eve-2", "fallback-chat"}
        self.assertEqual(set(messages_by_chat.keys()), expected_chats)

        for chat_id in {"chat-alex", "chat-eve-1", "chat-eve-2"}:
            self.assertGreaterEqual(len(messages_by_chat[chat_id]), 1)

        self.assertEqual(len(messages_by_chat["chat-alex"]), 2)
        self.assertIn("Task Alex", messages_by_chat["chat-alex"][1])

        self.assertEqual(len(messages_by_chat["chat-eve-1"]), 2)
        self.assertIn("Task Eve", messages_by_chat["chat-eve-1"][1])

        self.assertEqual(len(messages_by_chat["chat-eve-2"]), 2)
        self.assertIn("Task Eve", messages_by_chat["chat-eve-2"][1])
        self.assertGreaterEqual(len(messages_by_chat["fallback-chat"]), 1)
        self.assertIn("Task Unknown", " ".join(messages_by_chat["fallback-chat"]))

    @patch("telegram_reminder_service.ClickUpClient")
    def test_send_reminders_broadcast_all_when_overridden(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        session = DummySession()
        config = {
            "clickup": {
                "status_mapping": {
                    "ВЫПОЛНЕНО": "complete",
                    "НЕ_ВЫПОЛНЕНО": "todo",
                    "В_РАБОТЕ": "in progress",
                },
                "completed_status": "complete",
            },
            "working_hours": {"timezone": "Europe/Lisbon"},
        }
        credentials = {
            "clickup_api_key": "key",
            "clickup_team_id": "123",
            "telegram_bot_token": "token",
            "telegram_chat_id": "fallback-chat",
        }

        service = TelegramReminderService(config, credentials, session=session)
        tasks = [
            ReminderTask(task_id="1", name="Task Alex", status="todo", due_human="2024-01-01 10:00", assignee="Alex", url="url-1"),
            ReminderTask(task_id="2", name="Task Eve", status="todo", due_human="2024-01-02 10:00", assignee="Ева", url="url-2"),
        ]

        with patch.object(service, "fetch_pending_tasks", return_value=tasks):
            service.send_reminders(chat_id="custom-chat", broadcast_all=True)

        send_calls = [call for call in session.calls if call["url"].endswith("sendMessage")]
        messages_by_chat: dict[str, list[str]] = {}
        for call in send_calls:
            chat = call["json"]["chat_id"]
            messages_by_chat.setdefault(chat, []).append(call["json"]["text"])

        self.assertEqual(set(messages_by_chat.keys()), {"custom-chat"})
        # Expect one preface + len(tasks) messages
        self.assertEqual(len(messages_by_chat["custom-chat"]), 1 + len(tasks))
        self.assertIn("Task Alex", " ".join(messages_by_chat["custom-chat"]))
        self.assertIn("Task Eve", " ".join(messages_by_chat["custom-chat"]))

    @patch("telegram_reminder_service.ClickUpClient")
    def test_send_reminders_chat_specific_excludes_fallback(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        session = DummySession()
        config = {
            "clickup": {
                "status_mapping": {
                    "ВЫПОЛНЕНО": "complete",
                    "НЕ_ВЫПОЛНЕНО": "todo",
                    "В_РАБОТЕ": "in progress",
                },
                "completed_status": "complete",
            },
            "working_hours": {"timezone": "Europe/Lisbon"},
            "telegram": {
                "assignee_chat_map": {
                    "Alex": "chat-alex",
                }
            },
        }
        credentials = {
            "clickup_api_key": "key",
            "clickup_team_id": "123",
            "telegram_bot_token": "token",
            "telegram_chat_id": "chat-alex",
        }

        service = TelegramReminderService(config, credentials, session=session)
        tasks = [
            ReminderTask(
                task_id="1",
                name="Task Alex",
                status="todo",
                due_human="2024-01-01 10:00",
                assignee="Alex",
                url="url-1",
            ),
            ReminderTask(
                task_id="2",
                name="Task Unknown",
                status="todo",
                due_human="2024-01-02 10:00",
                assignee="",
                url="url-2",
            ),
        ]

        with patch.object(service, "fetch_pending_tasks", return_value=tasks):
            service.send_reminders(chat_id="chat-alex")

        send_calls = [call for call in session.calls if call["url"].endswith("sendMessage")]
        self.assertTrue(send_calls)
        payloads = [call["json"] for call in send_calls if call["json"]["chat_id"] == "chat-alex"]
        self.assertTrue(payloads)

        combined_text = " ".join(payload["text"] for payload in payloads)
        self.assertIn("Task Alex", combined_text)
        self.assertNotIn("Task Unknown", combined_text)

    @patch("telegram_reminder_service.ClickUpClient")
    def test_send_reminders_chat_specific_with_only_unmapped_tasks(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        session = DummySession()
        config = {
            "clickup": {
                "status_mapping": {
                    "ВЫПОЛНЕНО": "complete",
                    "НЕ_ВЫПОЛНЕНО": "todo",
                    "В_РАБОТЕ": "in progress",
                },
                "completed_status": "complete",
            },
            "working_hours": {"timezone": "Europe/Lisbon"},
            "telegram": {
                "assignee_chat_map": {
                    "Alex": "chat-alex",
                }
            },
        }
        credentials = {
            "clickup_api_key": "key",
            "clickup_team_id": "123",
            "telegram_bot_token": "token",
            "telegram_chat_id": "chat-alex",
        }

        service = TelegramReminderService(config, credentials, session=session)
        tasks = [
            ReminderTask(
                task_id="1",
                name="Task Unknown",
                status="todo",
                due_human="2024-01-02 10:00",
                assignee="",
                url="url-2",
            ),
        ]

        with patch.object(service, "fetch_pending_tasks", return_value=tasks):
            service.send_reminders(chat_id="chat-alex")

        send_calls = [call for call in session.calls if call["url"].endswith("sendMessage")]
        self.assertTrue(send_calls)

        payloads = [call["json"] for call in send_calls if call["json"]["chat_id"] == "chat-alex"]
        self.assertEqual(len(payloads), 1)
        self.assertIn("На данный момент нет задач", payloads[0]["text"])
        self.assertNotIn("Task Unknown", payloads[0]["text"])

    @patch("telegram_reminder_service.ClickUpClient")
    def test_fetch_pending_tasks_multiple_team_ids(self, mock_client_cls):
        client_a = MagicMock()
        client_b = MagicMock()
        client_a.fetch_tasks_by_tag.return_value = [
            {"id": "1", "name": "Task A", "status": {"status": "open"}, "due_date": None},
        ]
        client_b.fetch_tasks_by_tag.return_value = [
            {"id": "2", "name": "Task B", "status": {"status": "open"}, "due_date": None},
            {"id": "1", "name": "Task A duplicate", "status": {"status": "open"}, "due_date": None},
        ]
        mock_client_cls.side_effect = [client_a, client_b]

        config = dict(self.config)
        config["clickup"] = dict(config["clickup"], reminder_tags=["#напоминание"], team_ids=["team-a", "team-b"])

        credentials = dict(self.credentials)
        credentials["clickup_team_ids"] = ["team-a", "team-b"]
        credentials["clickup_team_id"] = "team-a"

        service = TelegramReminderService(config, credentials, session=DummySession())
        tasks = service.fetch_pending_tasks()

        self.assertEqual([task.task_id for task in tasks], ["1", "2"])
        client_a.fetch_tasks_by_tag.assert_called_once_with("#напоминание", space_ids=None)
        client_b.fetch_tasks_by_tag.assert_called_once_with("#напоминание", space_ids=None)

    @patch("telegram_reminder_service.ClickUpClient")
    def test_fetch_pending_tasks_respects_space_ids(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.fetch_tasks_by_tag.return_value = [
            {"id": "1", "name": "Task A", "status": {"status": "open"}, "due_date": None},
        ]
        mock_client_cls.return_value = mock_client

        config = dict(self.config)
        config["clickup"] = dict(config["clickup"], reminder_tags=["#напоминание"], space_ids=["123"])

        service = TelegramReminderService(config, self.credentials, session=DummySession())
        service.fetch_pending_tasks()

        mock_client.fetch_tasks_by_tag.assert_called_once_with("#напоминание", space_ids=["123"])

    @patch("telegram_reminder_service.time")
    @patch("telegram_reminder_service.ClickUpClient")
    def test_poll_updates_for_processes_callbacks(self, mock_client_cls, mock_time):
        mock_client_cls.return_value = MagicMock()
        session = DummySession()
        service = TelegramReminderService(self.config, self.credentials, session=session)

        mock_time.monotonic.side_effect = [0, 0, 3]
        mock_time.sleep.return_value = None

        with patch.object(service, "get_updates", return_value=[{"update_id": 1, "callback_query": {"id": "cb"}}]) as get_updates, patch.object(
            service, "handle_callback"
        ) as handle_callback, patch.object(service, "handle_message") as handle_message:
            processed = service.poll_updates_for(duration=2, timeout=5)

        self.assertEqual(processed, 1)
        get_updates.assert_called_once()
        handle_callback.assert_called_once()
        handle_message.assert_not_called()


if __name__ == "__main__":
    unittest.main()
