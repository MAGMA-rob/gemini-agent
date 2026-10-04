from __future__ import annotations

from copy import deepcopy
import json
from typing import Any


BASE_INSTRUCTIONS = """You are MAGMA's robot commander. Decide the next response from the current input, task attributes, conversation history, permanent rules, and available tools.

Choose the next necessary, verifiable step. Use a declared environment function when an action or observation is needed. Multiple function calls are allowed only when the actions are independent, can safely start together, and target different robots, with at most one call per robot.

If required information is missing, obtain it with an available environment function when possible. If no tool can obtain it, ask the user one concise question in ordinary text. Otherwise, answer the user or briefly confirm completion only when no further action or clarification is required.

Call an environment function only when the current request or an ongoing task requires an action or observation. If the latest user message only provides facts, preferences, rules, or assignments and no task is already underway, briefly acknowledge the key information in text; do not call an environment function. A permanent rule describes how to handle a task when it arises; it does not itself request that task.

Use only declared functions and arguments grounded in the provided context. Select the executing robot with target_robot, which must be listed in known_robots. Do not invent objects, locations, states, quantities, robot names, or tool results. Respect partial observability.

Maintain the current task goal across function calls and make incremental progress toward completion. If an action is required and the necessary information is available, perform the next required environment action rather than returning a plan or progress report.

Do not repeat an operation whose successful result is already known. After a failed operation, use the new evidence to correct the arguments, observe relevant state, choose a valid alternative, or ask for missing information. Do not retry an identical failed call without new evidence.

Treat environment events and status messages as feedback about the ongoing task, not as new user requests. The most recent task attributes describe the current state.

Before declaring a physical task complete, use a relevant observation or detection function when available unless recent environment feedback already confirms the required final state.

Use native function calls for tools. Do not write a textual list of tool calls when you can call the functions. Do not expose hidden reasoning. When an environment function call is required, call the function directly without a user-facing preamble. Use a text-only response only when clarification, acknowledgement, or final completion is appropriate."""


def build_system_instruction(memory: dict[str, Any]) -> str:
    rules = memory.get("memory_list", [])
    if not isinstance(rules, list) or any(not isinstance(rule, str) for rule in rules):
        raise ValueError("memory.memory_list must be a list of strings")
    if not rules:
        return BASE_INSTRUCTIONS
    return BASE_INSTRUCTIONS + "\n\nPermanent rules:\n" + "\n".join(
        f"- {rule}" for rule in rules
    )


def clean_event(content: Any) -> str:
    value = content
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, dict):
        value = deepcopy(value)
        value.pop("previous_tool_call", None)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def format_current_input(attributes: dict[str, Any], instruction_type: str, content: str) -> str:
    label = "Instruction" if instruction_type == "user" else "Env Event"
    instruction = content if instruction_type == "user" else clean_event(content)
    return (
        "Attributs :\n"
        f"{json.dumps(attributes, ensure_ascii=False, sort_keys=True)}\n\n"
        f"{label} :\n{instruction}"
    )


def format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = ["Conversation antérieure (faits et décisions, par ordre chronologique) :"]
    for item in history:
        author = str(item.get("author", "USER")).upper()
        content = item.get("content", item.get("sentence", ""))
        if author in {"SYSTEM", "STATUS", "ENV"}:
            content = clean_event(content)
            label = "Env Event"
        else:
            label = "Assistant" if author in {"MODEL", "ASSISTANT"} else "User"
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False)
        lines.append(f"{label} :\n{content}")
    return "\n\n".join(lines)


def build_tools(functions: list[dict[str, Any]], attributes: dict[str, Any]) -> list[dict[str, Any]]:
    known_robots = attributes.get("known_robots", [])
    if not isinstance(known_robots, list):
        known_robots = []
    robot_names = [name for name in known_robots if isinstance(name, str) and name]
    declarations: list[dict[str, Any]] = []
    for function in functions:
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Each tool requires a non-empty name")
        original = function.get("parameters", function.get("arguments", {}))
        if not isinstance(original, dict):
            raise ValueError(f"Tool {name} parameters must be an object")
        schema_type = original.get("type")
        if schema_type == "object" and isinstance(original.get("properties"), dict):
            parameters = deepcopy(original)
            properties = parameters["properties"]
            required = list(parameters.get("required", []))
        else:
            optional = function.get("optional", [])
            if not isinstance(optional, list):
                raise ValueError(f"Tool {name} optional must be a list")
            properties: dict[str, Any] = {}
            required = []
            for parameter_name, specification in original.items():
                parameter = deepcopy(specification) if isinstance(specification, dict) else {"type": specification}
                type_name = str(parameter.get("type", "string")).lower()
                parameter["type"] = {
                    "str": "string", "int": "integer", "float": "number", "bool": "boolean",
                    "list": "array", "dict": "object",
                }.get(type_name, type_name)
                properties[parameter_name] = parameter
                if parameter_name not in optional:
                    required.append(parameter_name)
            parameters = {"type": "object", "properties": properties}
        if "target_robot" in properties:
            raise ValueError(f"Tool {name} already defines reserved target_robot")
        target: dict[str, Any] = {
            "type": "string", "description": "Robot that executes this function."
        }
        if robot_names:
            target["enum"] = robot_names
        properties["target_robot"] = target
        if "target_robot" not in required:
            required.append("target_robot")
        parameters["required"] = required
        declarations.append({
            "type": "function",
            "name": name,
            "description": str(function.get("description", "")),
            "parameters": parameters,
        })
    return declarations
