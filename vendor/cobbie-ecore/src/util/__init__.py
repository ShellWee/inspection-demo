"""Public utility exports loaded on demand.

Keeping this package initializer dependency-light is important for focused
integrations such as ToG, which only needs ``baml_retry`` and must not import
the database, pandas, or code-execution stacks during startup.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "_create_function_from_source_code": (
        "._create_function_from_source_code",
        "_create_function_from_source_code",
    ),
    "_extract_function_metadata": (
        "._extract_function_metadata",
        "_extract_function_metadata",
    ),
    "setup_logger": (".setup_logger", "setup_logger"),
    "validate_type": (".validate_type", "validate_type"),
    "create_code_prefix": (".create_code_prefix", "create_code_prefix"),
    "save_new_tool": (".save_new_tool", "save_new_tool"),
    "get_function_code": (".get_function_code", "get_function_code"),
    "get_created_tools": (".get_created_tools", "get_created_tools"),
    "get_tools_description": (".get_created_tools", "get_tools_description"),
    "get_tools_names": (".get_created_tools", "get_tools_names"),
    "get_usage_openrouter": (".get_usage_openrouter", "get_usage_openrouter"),
    "generate_tools_docs": (".generate_tools_docs", "generate_tools_docs"),
    "delete_tool": (".delete_tool", "delete_tool"),
    "extract_tools_used": (".extract_tool_usage", "extract_tools_used"),
    "_execute_code_action": (".code_act_inner_loop", "_execute_code_action"),
    "call_baml_with_retry": (".baml_retry", "call_baml_with_retry"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
