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
