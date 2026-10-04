from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import logging
from typing import Any

import httpx
from magma_core.protocol.agent import (
    AgentDecision, AgentError, AgentInput, AgentOutput, AgentRequest, AgentResponse, ToolCall,
)

from .config import Settings
from .prompt import build_system_instruction, build_tools, format_current_input, format_history


INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
GEMINI_MEMORY_VERSION = 1
CONTEXT_ERROR_TERMS = (
    "context window", "context length", "context limit", "input token limit",
    "token limit", "too many tokens", "exceeds the maximum", "exceeded the maximum",
)


class ProviderFailure(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False,
                 rebuildable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.rebuildable = rebuildable


class Runtime:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client
        self.semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def process(self, request: AgentRequest) -> AgentResponse:
        candidates = [
            (entry, index)
            for entry in request.inputs
            for index in range(entry.num_outputs)
        ]
        outputs = await asyncio.gather(*(
            self._process_candidate(request.request_id, entry, index)
            for entry, index in candidates
        ))
        response = AgentResponse(outputs)
        response.validate_request(request)
        return response

    async def _process_candidate(
        self, request_id: str, entry: AgentInput, candidate_index: int
    ) -> AgentOutput:
        memory = deepcopy(entry.memory)
        steps: list[dict[str, Any]] = []
        try:
            history = memory.get("history", [])
            if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
                raise ValueError("memory.history must be a list of objects")
            state = memory.get("_gemini", {})
            if not isinstance(state, dict):
                raise ValueError("memory._gemini must be an object")
            if state and state.get("version") != GEMINI_MEMORY_VERSION:
                raise ValueError("Unsupported memory._gemini version")
            previous_id = state.get("interaction_id")
            if previous_id is not None and (not isinstance(previous_id, str) or not previous_id):
                raise ValueError("memory._gemini.interaction_id must be a non-empty string")
            pending_calls = state.get("pending_calls", [])
            if not isinstance(pending_calls, list) or any(
                not isinstance(call, dict)
                or not isinstance(call.get("name"), str)
                or not isinstance(call.get("call_id"), str)
                for call in pending_calls
            ):
                raise ValueError("memory._gemini.pending_calls must contain name/call_id objects")

            system_instruction = build_system_instruction(memory)
            tools = build_tools(entry.tools, entry.attributes)
            current_input = format_current_input(
                entry.attributes, entry.instruction.type, entry.instruction.content
            )
            reconstructed = not previous_id or (bool(pending_calls) and entry.instruction.type != "env")
            if reconstructed:
                transcript = format_history(history)
                input_value: str | list[dict[str, Any]] = (
                    f"{transcript}\n\n{current_input}" if transcript else current_input
                )
                previous_id = None
            elif pending_calls:
                input_value = [
                    {
                        "type": "function_result",
                        "name": call["name"],
                        "call_id": call["call_id"],
                        "result": [{"type": "text", "text": current_input}],
                    }
                    for call in pending_calls
                ]
            else:
                input_value = current_input

            payload: dict[str, Any] = {
                "model": self.settings.model,
                "store": True,
                "system_instruction": system_instruction,
                "generation_config": {
                    "thinking_level": self.settings.thinking_level,
                },
                "input": input_value,
            }
            if tools:
                payload["tools"] = tools
                payload["generation_config"]["tool_choice"] = "auto"
            if previous_id:
                payload["previous_interaction_id"] = previous_id

            trace: dict[str, Any] = {
                "id": "gemini-0",
                "component": "gemini",
                "origin": "model",
                "full_prompt": json.dumps(payload, ensure_ascii=False, indent=2),
                "input_elements": {
                    "context_source": (
                        "previous_interaction_id" if previous_id else "memory.history"
                    ),
                    "history": history,
                    "request": deepcopy(payload),
                },
                "output_raw": "",
            }
            steps.append(trace)
            try:
                interaction = await self._send_with_retries(payload, trace)
            except ProviderFailure as error:
                if not error.rebuildable or not previous_id:
                    raise
                trace["error"] = str(error)
                transcript = format_history(history)
                payload.pop("previous_interaction_id", None)
                payload["input"] = (
                    f"{transcript}\n\n{current_input}" if transcript else current_input
                )
                trace = {
                    "id": "gemini-1",
                    "component": "gemini",
                    "origin": "model",
                    "full_prompt": json.dumps(payload, ensure_ascii=False, indent=2),
                    "input_elements": {
                        "context_source": "memory.history",
                        "history": history,
                        "request": deepcopy(payload),
                        "reconstruction_reason": error.code,
                    },
                    "output_raw": "",
                }
                steps.append(trace)
                interaction = await self._send_with_retries(payload, trace)

            if not isinstance(interaction, dict):
                raise ValueError("Gemini response must be an object")
            interaction_id = interaction.get("id")
            if not isinstance(interaction_id, str) or not interaction_id:
                raise ValueError("Gemini response has no interaction id")

            raw_steps = interaction.get("steps", [])
            if not isinstance(raw_steps, list):
                raise ValueError("Gemini steps must be a list")
            visible_parts: list[str] = []
            calls: list[ToolCall] = []
            next_pending: list[dict[str, str]] = []
            for step in raw_steps:
                if not isinstance(step, dict):
                    raise ValueError("Gemini step must be an object")
                if step.get("type") == "model_output":
                    content = step.get("content", [])
                    if not isinstance(content, list):
                        raise ValueError("Gemini model output content must be a list")
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            value = part.get("text")
                            if isinstance(value, str):
                                visible_parts.append(value)
                elif step.get("type") == "function_call":
                    name = step.get("name")
                    arguments = step.get("arguments")
                    call_id = step.get("id")
                    if not isinstance(name, str) or not name:
                        raise ValueError("Gemini function call has no name")
                    if not isinstance(arguments, dict):
                        raise ValueError(f"Gemini function call {name} has invalid arguments")
                    if not isinstance(call_id, str) or not call_id:
                        raise ValueError(f"Gemini function call {name} has no id")
                    target_robot = arguments.get("target_robot")
                    if not isinstance(target_robot, str) or not target_robot:
                        raise ValueError(f"Gemini function call {name} has no target_robot")
                    tool_arguments = dict(arguments)
                    tool_arguments.pop("target_robot")
                    calls.append(ToolCall(
                        name=name, arguments=tool_arguments,
                        target_robot_name=target_robot,
                    ))
                    next_pending.append({"name": name, "call_id": call_id})

            say = "".join(visible_parts)
            decision = AgentDecision(say=say, tool_calls=calls)
            trace["output_formatted"] = decision.model_dump(mode="json")
            memory["_gemini"] = {
                "version": GEMINI_MEMORY_VERSION,
                "interaction_id": interaction_id,
                "pending_calls": next_pending,
            }
            action = [
                {call.target_robot_name: {"name": call.name, "arguments": call.arguments}}
                for call in calls
            ]
            memory.setdefault("history", []).extend([
                {
                    "author": "USER" if entry.instruction.type == "user" else "SYSTEM",
                    "content": entry.instruction.content,
                },
                {
                    "author": "MODEL",
                    "content": json.dumps({"say": say, "action": action}, ensure_ascii=False),
                },
            ])
            usage = interaction.get("usage", {})
            usage = usage if isinstance(usage, dict) else {}
            trace.update({
                "interaction_id": interaction_id,
                "total_input_tokens": usage.get("total_input_tokens"),
                "total_cached_tokens": usage.get("total_cached_tokens"),
                "total_thought_tokens": usage.get("total_thought_tokens"),
            })
            return AgentOutput(
                request_id=request_id,
                source_id=entry.id,
                candidate_index=candidate_index,
                status="completed",
                memory=memory,
                internal_steps=steps,
                output=decision,
            )
        except Exception as error:
            if isinstance(error, ProviderFailure):
                code = error.code
            elif isinstance(error, (TypeError, ValueError, KeyError)):
                code = "invalid_output"
            else:
                code = "internal_error"
                logging.exception("Gemini candidate processing failed")
            if steps:
                steps[-1]["error"] = str(error)
            else:
                steps.append({"component": "gemini", "origin": "model", "error": str(error)})
            return AgentOutput(
                request_id=request_id,
                source_id=entry.id,
                candidate_index=candidate_index,
                status="error",
                memory=deepcopy(entry.memory),
                internal_steps=steps,
                error=AgentError(code=code, message=str(error) or code, component="gemini"),
            )

    async def _send_with_retries(
        self, payload: dict[str, Any], trace: dict[str, Any]
    ) -> dict[str, Any]:
        for attempt in range(self.settings.max_retries + 1):
            try:
                async with self.semaphore:
                    response = await self.client.post(
                        INTERACTIONS_URL,
                        json=payload,
                        timeout=self.settings.request_timeout_seconds,
                    )
            except httpx.RequestError as error:
                failure = ProviderFailure("transport_error", str(error), retryable=True)
            else:
                trace["http_status"] = response.status_code
                trace["output_raw"] = response.text
                if response.is_success:
                    try:
                        interaction = response.json()
                    except ValueError as error:
                        raise ProviderFailure("invalid_response", "Gemini returned invalid JSON") from error
                    if not isinstance(interaction, dict):
                        raise ProviderFailure("invalid_response", "Gemini returned a non-object response")
                    status = interaction.get("status")
                    if status not in {"completed", "requires_action"}:
                        errors = interaction.get("errors", [])
                        message = (
                            json.dumps(errors, ensure_ascii=False)
                            if errors else f"Gemini status: {status}"
                        )
                        context_exceeded = any(
                            term in message.lower() for term in CONTEXT_ERROR_TERMS
                        )
                        raise ProviderFailure(
                            "context_exceeded" if context_exceeded else "provider_status",
                            message,
                            rebuildable=context_exceeded,
                        )
                    return interaction

                try:
                    body = response.json()
                except ValueError:
                    body = {}
                provider_error = body.get("error", {}) if isinstance(body, dict) else {}
                message = provider_error.get("message", "") if isinstance(provider_error, dict) else ""
                message = str(message or f"Gemini HTTP {response.status_code}")
                lowered = message.lower()
                context_exceeded = any(term in lowered for term in CONTEXT_ERROR_TERMS)
                invalid_id = bool(payload.get("previous_interaction_id")) and (
                    response.status_code == 404
                    or (response.status_code == 400 and "interaction" in lowered and (
                        "not found" in lowered or "invalid" in lowered or "expired" in lowered
                    ))
                )
                failure = ProviderFailure(
                    "context_exceeded" if context_exceeded else
                    "invalid_interaction" if invalid_id else
                    "provider_error",
                    message,
                    retryable=response.status_code == 429 or response.status_code >= 500,
                    rebuildable=context_exceeded or invalid_id,
                )
            if not failure.retryable or attempt == self.settings.max_retries:
                raise failure
            trace.setdefault("retries", []).append({
                "attempt": attempt + 1,
                "error": str(failure),
            })
            await asyncio.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Unreachable retry state")
