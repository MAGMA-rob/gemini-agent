# Gemini Agent

`gemini-agent` is an inference-only MAGMA protocol-v2 agent using Gemini Robotics ER 2
through the Gemini Interactions API. It does not implement coaching or export.

## Install and run

Use Python 3.12 and install the package in an environment with MAGMA Core:

```bash
python -m pip install -e agents/gemini-agent
export GEMINI_API_KEY='your-restricted-key'
gemini-agent --host 127.0.0.1 --port 8888
```

The API key is read only from `GEMINI_API_KEY`. Robotics ER requires a restricted
Gemini API key. Point MAGMA at `http://127.0.0.1:8888`.

The default model is `gemini-robotics-er-2-preview`. The agent defaults to four
concurrent API requests, two retries for temporary errors, a 90-second timeout
per attempt, and medium thinking. These can be set with the corresponding CLI
options or a JSON `--config` file. The agent uses the Interactions API's implicit
input cache and records token counts in each output's `internal_steps`.

## State and errors

Each output returns its own `memory._gemini` containing the last interaction ID
and any pending Gemini function call IDs. MAGMA must pass that memory back for
the next decision. `memory.history` is maintained as a readable fallback.

If an interaction ID expires or the context limit is reached, the agent tries
once to rebuild a new conversation from `memory.history`, without removing any
history entry. If that attempt fails, the candidate returns a protocol error.
The agent never executes robot tools; it returns native Gemini function calls
as MAGMA `tool_calls` for the client to handle.

## MAGMA-GEN debugging

Run MAGMA-GEN with `--no-coaching --magma-agent-address http://127.0.0.1:8888`.
Its default 360-second agent timeout covers normal retries; use
`--agent-timeout 600` if a context reconstruction must also be allowed to finish.
Start the viewer separately with `magma-gen viewer --output-dir ./output`.

Each candidate records the exact Interactions request body, the HTTP response
body, the converted MAGMA decision, the interaction ID, token usage, and errors
in `internal_steps`. The viewer shows these in the node detail panel. For a
continued interaction, the request contains `previous_interaction_id`; Google's
server retains the earlier context, so the exact internal context is not present
in that request body. The readable `memory.history` is shown alongside it.
The API key is sent only in the HTTP header and is not included in the trace.
