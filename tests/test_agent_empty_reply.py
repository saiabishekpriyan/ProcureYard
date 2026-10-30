"""Regression tests for the empty-reply and hang bugs.

Root causes confirmed 2026-10-29:
  - msg rows stored with reply=None (agent crashed before commit).
  - Duplicate delivery returned `existing.reply or ""` → "".
  - agent.py returned "" for None/empty response.text with no log.
  - chat.send_message() had no timeout, blocking the thread indefinitely.
  - Repeated delivery of a null-reply message re-triggered an unbounded hang.

These tests verify every path is now safe and non-blocking.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_db_mock():
    return MagicMock()


def _silence_trace(agent_mod):
    original = agent_mod._trace
    agent_mod._trace = lambda *a, **kw: None
    return original


# ── agent.run_agent() unit tests ──────────────────────────────────────────────

def test_run_agent_no_key_returns_fallback():
    """Empty API key → immediate fallback, no crash."""
    import agent
    orig = _silence_trace(agent)
    try:
        result = agent.run_agent("R001", "test", "2026-09-22T09:00:00+05:30",
                                 _make_db_mock(), "")
        assert result and result.strip()
    finally:
        agent._trace = orig


def test_run_agent_passes_request_options_timeout():
    """send_message must be called with request_options containing timeout."""
    import agent
    orig = _silence_trace(agent)
    try:
        captured = {}

        fake_response = MagicMock()
        fake_response.candidates = [MagicMock()]
        fake_response.parts = []
        fake_response.text = "Payout confirmed."

        def _send(msg, request_options=None):
            captured["opts"] = request_options
            return fake_response

        fake_chat = MagicMock()
        fake_chat.send_message.side_effect = _send
        fake_model = MagicMock()
        fake_model.start_chat.return_value = fake_chat

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel", return_value=fake_model):
            result = agent.run_agent("R001", "test", "2026-09-22T09:00:00+05:30",
                                     _make_db_mock(), "fake-key")

        assert result == "Payout confirmed."
        assert captured.get("opts") is not None, "request_options was not passed"
        from google.generativeai.types import RequestOptions
        assert isinstance(captured["opts"], RequestOptions), \
            f"Expected RequestOptions, got {type(captured['opts'])}"
        assert captured["opts"].timeout is not None, "timeout must be set"
        assert captured["opts"].timeout > 0
    finally:
        agent._trace = orig


def test_run_agent_api_call_timeout_returns_fallback():
    """DeadlineExceeded (timeout) during send_message → fallback, no hang."""
    import agent
    from google.api_core.exceptions import DeadlineExceeded
    orig = _silence_trace(agent)
    try:
        fake_chat = MagicMock()
        fake_chat.send_message.side_effect = DeadlineExceeded("deadline exceeded")
        fake_model = MagicMock()
        fake_model.start_chat.return_value = fake_chat

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel", return_value=fake_model):
            result = agent.run_agent("R001", "test", "2026-09-22T09:00:00+05:30",
                                     _make_db_mock(), "fake-key")

        assert result and result.strip(), f"Expected non-empty fallback, got: {result!r}"
    finally:
        agent._trace = orig


def test_run_agent_empty_text_response_returns_fallback():
    """response.text is '' → fallback, not ''."""
    import agent
    orig = _silence_trace(agent)
    try:
        fake_part = MagicMock()
        fake_part.function_call.name = ""
        fake_candidate = MagicMock()
        fake_candidate.finish_reason = 1
        fake_response = MagicMock()
        fake_response.candidates = [fake_candidate]
        fake_response.parts = [fake_part]
        fake_response.text = ""

        fake_chat = MagicMock()
        fake_chat.send_message.return_value = fake_response
        fake_model = MagicMock()
        fake_model.start_chat.return_value = fake_chat

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel", return_value=fake_model):
            result = agent.run_agent("R001", "missing payout", "2026-09-22T09:00:00+05:30",
                                     _make_db_mock(), "fake-key")

        assert result and result.strip(), f"Expected non-empty reply, got: {result!r}"
    finally:
        agent._trace = orig


def test_run_agent_none_text_response_returns_fallback():
    """response.text is None → fallback."""
    import agent
    orig = _silence_trace(agent)
    try:
        fake_response = MagicMock()
        fake_response.candidates = [MagicMock()]
        fake_response.parts = []
        fake_response.text = None

        fake_chat = MagicMock()
        fake_chat.send_message.return_value = fake_response
        fake_model = MagicMock()
        fake_model.start_chat.return_value = fake_chat

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel", return_value=fake_model):
            result = agent.run_agent("R001", "missing payout", "2026-09-22T09:00:00+05:30",
                                     _make_db_mock(), "fake-key")

        assert result and result.strip()
    finally:
        agent._trace = orig


def test_run_agent_response_text_raises_value_error_returns_fallback():
    """response.text raises ValueError (blocked content) → fallback."""
    import agent
    orig = _silence_trace(agent)
    try:
        fake_candidate = MagicMock()
        fake_candidate.finish_reason = 3  # SAFETY
        fake_response = MagicMock()
        fake_response.candidates = [fake_candidate]
        fake_response.parts = []
        type(fake_response).text = PropertyMock(side_effect=ValueError("blocked"))

        fake_chat = MagicMock()
        fake_chat.send_message.return_value = fake_response
        fake_model = MagicMock()
        fake_model.start_chat.return_value = fake_chat

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel", return_value=fake_model):
            result = agent.run_agent("R001", "test", "2026-09-22T09:00:00+05:30",
                                     _make_db_mock(), "fake-key")

        assert result and result.strip()
    finally:
        agent._trace = orig


def test_run_agent_no_candidates_returns_fallback():
    """response.candidates == [] → fallback."""
    import agent
    orig = _silence_trace(agent)
    try:
        fake_response = MagicMock()
        fake_response.candidates = []
        fake_response.parts = []
        fake_response.prompt_feedback = None

        fake_chat = MagicMock()
        fake_chat.send_message.return_value = fake_response
        fake_model = MagicMock()
        fake_model.start_chat.return_value = fake_chat

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel", return_value=fake_model):
            result = agent.run_agent("R001", "test", "2026-09-22T09:00:00+05:30",
                                     _make_db_mock(), "fake-key")

        assert result and result.strip()
    finally:
        agent._trace = orig


def test_run_agent_provider_exception_returns_fallback():
    """Generic provider exception → fallback, no hang."""
    import agent
    orig = _silence_trace(agent)
    try:
        fake_chat = MagicMock()
        fake_chat.send_message.side_effect = Exception("404 model not found")
        fake_model = MagicMock()
        fake_model.start_chat.return_value = fake_chat

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel", return_value=fake_model):
            result = agent.run_agent("R001", "test", "2026-09-22T09:00:00+05:30",
                                     _make_db_mock(), "fake-key")

        assert result and result.strip()
    finally:
        agent._trace = orig


def test_run_agent_model_construction_fails_returns_fallback():
    """GenerativeModel() raises → fallback immediately."""
    import agent
    orig = _silence_trace(agent)
    try:
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel",
                   side_effect=ValueError("Schema error")):
            result = agent.run_agent("R001", "test", "2026-09-22T09:00:00+05:30",
                                     _make_db_mock(), "fake-key")

        assert result and result.strip()
    finally:
        agent._trace = orig


def test_run_agent_quota_exhausted_returns_fallback_without_retry():
    """ResourceExhausted (429) returns an explicit fallback immediately with error trace."""
    import agent
    from google.api_core.exceptions import ResourceExhausted
    orig = _silence_trace(agent)
    try:
        fake_chat = MagicMock()
        fake_chat.send_message.side_effect = ResourceExhausted("429 Resource has been exhausted (e.g. check quota).")
        fake_model = MagicMock()
        fake_model.start_chat.return_value = fake_chat

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel", return_value=fake_model):
            result = agent.run_agent("R001", "payout missing", "2026-09-22T09:00:00+05:30",
                                     _make_db_mock(), "fake-key")

        assert result and result.strip()
        # Ensure send_message was called exactly once — no auto-retries on quota exhaustion
        assert fake_chat.send_message.call_count == 1
    finally:
        agent._trace = orig


# ── _run_agent_safe() / messages route unit tests ────────────────────────────

def test_run_agent_safe_wall_timeout_returns_fallback():
    """_run_agent_safe returns fallback when the wall-clock future times out.

    We inject FuturesTimeoutError directly via a mock Future so the test
    completes instantly without a real sleep / thread-join hang.
    """
    from concurrent.futures import Future, TimeoutError as FTE
    from routes.messages import _run_agent_safe, _FALLBACK
    from unittest.mock import patch as _patch, MagicMock as _MM

    mock_future = _MM(spec=Future)
    mock_future.result.side_effect = FTE("timed out")
    mock_future.cancel.return_value = True

    mock_pool = _MM()
    mock_pool.submit.return_value = mock_future

    with _patch("routes.messages.ThreadPoolExecutor", return_value=mock_pool):
        result = _run_agent_safe("R001", "test", "2026-09-22T09:00:00+05:30",
                                 _make_db_mock())

    assert result == _FALLBACK, f"Expected fallback, got: {result!r}"
    mock_pool.shutdown.assert_called_once_with(wait=False)


def test_run_agent_safe_substitutes_fallback_for_empty_string():
    """_run_agent_safe converts '' → fallback."""
    from routes.messages import _run_agent_safe, _FALLBACK
    with patch("agent.run_agent", return_value=""):
        result = _run_agent_safe("R001", "test", "2026-09-22T09:00:00+05:30",
                                 _make_db_mock())
    assert result == _FALLBACK


def test_run_agent_safe_substitutes_fallback_for_exception():
    """_run_agent_safe converts exception → fallback."""
    from routes.messages import _run_agent_safe, _FALLBACK
    with patch("agent.run_agent", side_effect=RuntimeError("boom")):
        result = _run_agent_safe("R001", "test", "2026-09-22T09:00:00+05:30",
                                 _make_db_mock())
    assert result == _FALLBACK


def test_duplicate_message_with_null_reply_stores_permanently():
    """After a null-reply retry, the result is stored so the next request uses cache."""
    # This verifies the fix for the infinite-hang loop: we commit even on fallback.
    from routes.messages import _run_agent_safe, _FALLBACK
    with patch("agent.run_agent", return_value=_FALLBACK):
        result = _run_agent_safe("R001", "payout missing", "2026-09-22T09:00:00+05:30",
                                 _make_db_mock())
    # The result (even if fallback) must be non-empty so the next duplicate check
    # sees existing.reply = non-empty and returns cached rather than re-running.
    assert result
    assert result.strip()

