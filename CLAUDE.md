# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
git config core.hooksPath .githooks   # once per clone: enables the commit-msg hook
```

Copy `.env.example` to `.env` (gitignored) and fill it in:
```
LLM_PROVIDER=anthropic              # anthropic (default) or openai: who runs the agents/chatbot
JUDGE_PROVIDER=openai               # optional: who judges the evals (defaults to LLM_PROVIDER)
ANTHROPIC_API_KEY=sk-ant-...        # needed when a provider is anthropic
OPENAI_API_KEY=sk-...               # needed when a provider is openai, and by test_agent_synthesized.py
VOYAGE_API_KEY=...                  # RAG agent embeddings (Voyage AI)
CONFIDENT_API_KEY=confident_us_...  # optional: stream traces to Confident AI dashboard
```

## Commits

- Every commit message follows [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/): `<type>[(scope)][!]: <description>`. Mark breaking changes with `!` and a `BREAKING CHANGE:` footer.
- Keep commits atomic: one logical change per commit, and every commit leaves the project runnable.
- `.githooks/commit-msg` rejects non-conforming messages. Enable it once per clone: `git config core.hooksPath .githooks`.

## Running

Run everything from the project root, as modules (`python -m agents.<name>`, `python -m evals.<name>`), so the root-level `config` and the `agents` and `evals` packages can be imported. Running a file by path (`python agents/plain.py`) fails with `ModuleNotFoundError`.

```bash
# Sanity-check the bots on their own
python -m agents.plain
python -m agents.instrumented
python -m agents.rag
python -m agents.chatbot                        # interactive, type 'quit' to exit

# Phase 1 — tool-calling support agent
python -m evals.test_agent_untraced             # TaskCompletion on a plain test case (no trace)
python -m evals.test_agent_tracing              # TaskCompletion + ToolCorrectness on traced goldens
python -m evals.test_agent_quality              # AnswerRelevancy + PromptAlignment + StepEfficiency
python -m evals.test_agent_geval                # GEval correctness against expected_output

# Phase 2 — RAG agent
python -m evals.test_rag_agent                  # ContextualPrecision + ContextualRecall + AnswerRelevancy + Faithfulness

# Phase 3 — multi-turn chatbot
python -m evals.test_chatbot                    # TurnRelevancy + KnowledgeRetention + ConversationCompleteness
python -m evals.test_chatbot_geval              # ConversationalGEval with custom criteria

# Safety evals on synthesized goldens
python -m evals.test_agent_synthesized          # Synthesizer + Bias + Toxicity + PIILeakage
```

Switch vendor per run without editing `.env`: `LLM_PROVIDER=openai python -m agents.plain` (bash) or `$env:LLM_PROVIDER="openai"; python -m agents.plain` (PowerShell).

## Project structure

| File | Purpose |
|---|---|
| `config.py` | The one place the LLM vendor is chosen. Reads `LLM_PROVIDER` / `JUDGE_PROVIDER`, holds the model IDs in `MODELS`, and builds chat models with `get_chat_model()` (`init_chat_model` under the hood). Stays at the root because both `agents/` and `evals/` import it. |
| `agents/` | The bots under evaluation. The evals import them as `agents.<module>`. |
| `agents/plain.py` | Baseline LangChain agent with the two shop tools. No eval code; kept as a baseline to diff against `agents/instrumented.py`. |
| `agents/instrumented.py` | The same agent plus DeepEval instrumentation and one extra tool loaded from `tools/mcp_server.py`. The agent evals import `support_agent` from here. |
| `agents/rag.py` | Agent with one tool, `search_policies`. Retrieved chunks are forwarded to the trace as `retrieval_context`. |
| `agents/chatbot.py` | Multi-turn chatbot that runs its own tool-call loop over a model with `bind_tools()`, returning `(reply, history, tools_called)` per turn. |
| `agents/tools/` | Every tool the bots call. New tools go here, not in the agent modules. |
| `agents/tools/shop.py` | The ShopEasy demo data (`ORDERS`, `REFUND_POLICIES`) and the two `@tool` functions `get_order_status` and `get_refund_policy`, shared by the agents and the chatbot. |
| `agents/tools/policy_search.py` | The RAG retriever: 9 policy documents in an `InMemoryVectorStore` with Voyage AI embeddings, the `search_policies` tool, and `retrieved_chunks()`, which pulls the chunks out of a run's `ToolMessage` artifacts. |
| `agents/tools/mcp_server.py` | MCP server exposing `get_shipping_options`, started over stdio by `agents/instrumented.py` by file path. |
| `policies.txt` | Source document for `test_agent_synthesized.py`. |
| `evals/common.py` | Shared eval helpers: `get_judge()`, `ASYNC_CONFIG`, `traced()`, and the chatbot conversation driver `run_conversation()`. |
| `evals/test_*.py` | The eval scripts listed under Running. |

## Architecture

**Dependency injection**: nothing outside `config.py` names a vendor or model. Each bot exposes a factory that takes the model as an optional argument and falls back to the default from `config.py`:
- `build_agent(llm=None)` in `agents/plain.py`, `agents/instrumented.py`, `agents/rag.py`
- `build_model(llm=None)` in `chatbot.py` (binds the shop tools)
- `get_judge(provider=JUDGE_PROVIDER)` in `evals/common.py` (`AnthropicModel` or `OpenAIModel`)

Each module keeps a default instance so the entry points (`support_agent`, `rag_support_agent`, `chat`) keep a simple signature. Anything that calls out is built on first use, not at import: `instrumented.default_agent()` (which starts the MCP server via `load_mcp_tools()`, in a worker thread so it never touches DeepEval's event loop) and `policy_search.get_vector_store()` (which embeds the documents). Both are `functools.cache`d. Replies are read with `message.text`, not `.content`, because Claude can return a list of content blocks.

**Agent instrumentation pattern** (`agents/instrumented.py` and `agents/rag.py`):
1. `CallbackHandler()` — captures every LangChain LLM/tool span automatically
2. `config={"callbacks": [deepeval_callback]}` on `.invoke()` — wires it in
3. `update_current_trace(output=reply)` (plus `retrieval_context` in the RAG agent) — sets the clean reply on the trace

The eval side adds the rest with `traced()` from `evals/common.py`: `@observe(name=...)` creates the top-level trace, and the wrapper copies `expected_output` and `expected_tools` from `get_current_golden()` into it.

**Why the golden is copied into the trace**: DeepEval metrics read `expected_tools`, `expected_output` and `retrieval_context` from the trace object, not directly from the golden.

**Judge vendor**: the judge defaults to the agent's vendor. Set `JUDGE_PROVIDER` to the other vendor to avoid a model grading its own answers. `ASYNC_CONFIG` runs Claude-judged evals synchronously and GPT-4o-judged ones concurrently.

**Every metric gets the judge explicitly** (`model=judge`). Metrics left without a model fall back to DeepEval's default OpenAI model, which breaks a Claude-only setup.

**Confident AI tracing**: adding `CONFIDENT_API_KEY` to `.env` is the only change needed to enable cloud tracing — no code changes required. DeepEval reads it automatically.

**Chatbot evaluation pattern** (`evals/test_chatbot*.py`):
- Uses `ConversationalTestCase` + `Turn` objects instead of `Golden`
- `run_conversation(chat, SUPPORT_CONVERSATION)` in `evals/common.py` drives `chat()` live over a scripted list of user messages and returns `Turn` objects; each assistant `Turn` carries the tools called on that turn as `tools_called`
- `ToolCorrectnessMetric` cannot be used with `ConversationalTestCase`; `test_chatbot_geval.py` judges tool use with `ConversationalGEval` and `MultiTurnParams.TOOLS_CALLED`
- Uses `evaluate()` (not `evals_iterator()`) since there is no golden to iterate over

## Metric reference

| Metric | Needs `expected_output`? | Needs `retrieval_context`? | LLM judge? |
|---|---|---|---|
| `TaskCompletionMetric` | No | No | Yes |
| `ToolCorrectnessMetric` | No | No | No (name match) |
| `AnswerRelevancyMetric` | No | No | Yes |
| `PromptAlignmentMetric` | No | No | Yes |
| `StepEfficiencyMetric` | No | No | Yes |
| `GEval` (Correctness) | **Yes** | No | Yes |
| `FaithfulnessMetric` | No | **Yes** | Yes |
| `ContextualPrecisionMetric` | **Yes** | **Yes** | Yes |
| `ContextualRecallMetric` | **Yes** | **Yes** | Yes |
| `KnowledgeRetentionMetric` | No | No | Yes |
| `TurnRelevancyMetric` | No | No | Yes |
| `ConversationCompletenessMetric` | No | No | Yes |
| `ConversationalGEval` | No | No | Yes |
| `BiasMetric` | No | No | Yes |
| `ToxicityMetric` | No | No | Yes |
| `PIILeakageMetric` | No | No | Yes |

## Known issues

- **`test_agent_untraced.py` scores low on purpose**: it passes a plain `LLMTestCase` with no trace, so the judge cannot see the tool call and assumes the answer is made up. `test_agent_tracing.py` runs the same check on a trace and passes. Kept as a baseline that shows why trace-level evals matter.
- **Synthesizer needs OpenAI**: DeepEval has no Anthropic or Voyage embedder, so `test_agent_synthesized.py` embeds `policies.txt` with OpenAI even when the judge is Claude.
- **Voyage free tier is 3 requests/minute**: `agents/tools/policy_search.py` embeds its 9 documents on the first search (one request) and each search embeds its query, so repeated RAG runs within a minute hit `RateLimitError` until a payment method is added.
- **Windows console encoding**: when stdout is piped, Python on Windows encodes with cp1252 and crashes on emoji in model replies. Set `PYTHONIOENCODING=utf-8`.
- **`create_react_agent` deprecation**: migrated to `from langchain.agents import create_agent` with `system_prompt=` replacing `prompt=`. Requires the `langchain` base package.
- **Old `_make_hashable` patch**: DeepEval 4.0.4 crashed `ToolCorrectnessMetric` on unhashable `ToolMessage` objects and was patched by hand in the venv. DeepEval 4.3.1 runs `test_agent_tracing.py` without the patch.
