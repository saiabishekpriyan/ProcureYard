"""Verify that _TOOLS_GLM can be constructed without error using the installed SDK,
and that the model name is configurable via GEMINI_MODEL / settings.

These tests run without a real GEMINI_API_KEY — they only check schema validity
and configuration; no actual API calls are made.
"""
import os
import sys

# Allow importing from app/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


def test_tool_schema_construction():
    """_TOOLS_GLM must be a glm.Tool with all 7 function declarations."""
    import google.ai.generativelanguage as glm
    import agent  # noqa: F401 — import validates module-level glm construction

    assert isinstance(agent._TOOLS_GLM, glm.Tool), (
        "_TOOLS_GLM should be a glm.Tool instance"
    )
    declared_names = {fd.name for fd in agent._TOOLS_GLM.function_declarations}
    expected_names = {
        "lookup_rider",
        "lookup_trips",
        "lookup_payout_lines",
        "calculate_payout",
        "trigger_payout",
        "queue_approval",
        "queue_escalation",
    }
    assert declared_names == expected_names, (
        f"Tool names mismatch. Got: {declared_names}"
    )


def test_generative_model_accepts_tools():
    """GenerativeModel must accept _TOOLS_GLM without raising (schema check only)."""
    import google.generativeai as genai
    import agent  # noqa: F401
    from config import settings

    # Use a dummy key — we are only testing schema compatibility, not making API calls.
    genai.configure(api_key="DUMMY_KEY_FOR_SCHEMA_TEST")
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        tools=[agent._TOOLS_GLM],
    )
    assert model is not None


def test_settings_model_name_is_configurable():
    """settings.gemini_model must default to gemini-3.8-flash and be overridable."""
    import importlib
    import config as cfg_module

    # Default value
    assert cfg_module.settings.gemini_model == "gemini-3.8-flash", (
        f"Expected default 'gemini-3.8-flash', got '{cfg_module.settings.gemini_model}'"
    )

    # Env-var override (reload settings with patched env)
    os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"
    try:
        importlib.reload(cfg_module)
        assert cfg_module.settings.gemini_model == "gemini-2.5-flash"
    finally:
        del os.environ["GEMINI_MODEL"]
        importlib.reload(cfg_module)  # restore


def test_all_tool_types_are_enums():
    """Every Schema in _TOOLS_GLM must use glm.Type enum values, not strings."""
    import google.ai.generativelanguage as glm
    import agent  # noqa: F401

    def _check_schema(schema: glm.Schema, path: str) -> None:
        assert isinstance(schema.type, int), (
            f"{path}.type should be a glm.Type int enum, got {type(schema.type)}"
        )
        for prop_name, prop_schema in schema.properties.items():
            _check_schema(prop_schema, f"{path}.properties.{prop_name}")

    for fd in agent._TOOLS_GLM.function_declarations:
        _check_schema(fd.parameters, f"{fd.name}.parameters")


def test_run_agent_returns_fallback_without_api_key():
    """run_agent must return a valid non-empty reply immediately when no API key is set."""
    import agent
    from unittest.mock import MagicMock

    db = MagicMock()
    # Patch _trace so it doesn't hit the real DB
    original_trace = agent._trace
    agent._trace = lambda *a, **kw: None
    try:
        result = agent.run_agent(
            rider_id="R999",
            message_text="test",
            received_at="2026-09-22T09:00:00+05:30",
            db=db,
            gemini_api_key="",
        )
        assert result and result.strip()
    finally:
        agent._trace = original_trace
