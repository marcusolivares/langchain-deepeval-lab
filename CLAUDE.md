# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Use Python 3.13. Python 3.15 does not work: `langchain-voyageai==0.4.1` requires Python <3.15, so pip reports no matching version.

```bash
py -3.13 -m venv .venv                # Windows; elsewhere: python3.13 -m venv .venv
.venv\Scripts\Activate.ps1            # PowerShell; bash: source .venv/bin/activate (Windows: .venv/Scripts/activate)
pip install -r requirements.txt
git config core.hooksPath .githooks   # once per clone: enables the commit-msg hook
```

Copy `.env.example` to `.env` (gitignored) and fill it in:
```
LLM_PROVIDER=anthropic              # anthropic (default) or openai: who runs the agents/chatbot
JUDGE_PROVIDER=openai               # optional: who judges the evals (defaults to LLM_PROVIDER)
ANTHROPIC_API_KEY=sk-ant-...        # needed when a provider is anthropic
OPENAI_API_KEY=sk-...               # needed when a provider is openai, and by evals/synthesize_goldens.py
VOYAGE_API_KEY=...                  # RAG agent embeddings (Voyage AI)
CONFIDENT_API_KEY=confident_us_...  # optional: stream traces to Confident AI dashboard
```

## Commits

- Every commit message follows [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/): `<type>[(scope)][!]: <description>`. Mark breaking changes with `!` and a `BREAKING CHANGE:` footer.
- Keep commits atomic: one logical change per commit, and every commit leaves the project runnable.
- `.githooks/commit-msg` rejects non-conforming messages. Enable it once per clone: `git config core.hooksPath .githooks`.

## Running

Run everything from the project root. Run the bots as modules (`python -m agents.<name>`) so the root-level `config` and the `agents` package can be imported; running a file by path (`python agents/plain.py`) fails with `ModuleNotFoundError`.

```bash
# Sanity-check the bots on their own
python -m agents.plain
python -m agents.instrumented
python -m agents.rag
python -m agents.chatbot                        # interactive, type 'quit' to exit
```

The offline tests in `tests/` swap every chat model and embedding for a fake, so they are free, take a few seconds and need no API keys. Plain `pytest` runs them (`testpaths = tests`). `tests/conftest.py` sets fake API keys before `load_dotenv()` runs, so a test that reaches a real API by mistake fails with an auth error instead of spending money.

```bash
pytest                                          # offline tests, free
```

The evals are pytest tests, run with DeepEval's runner. A metric below its threshold fails its test, and any failure makes the run exit non-zero. Every eval makes paid API calls; a full run costs about $0.50.

```bash
deepeval test run evals/                        # full run
deepeval test run evals/ -m smoke               # one suite per bot, about half the cost
deepeval test run evals/ -m rag                 # one bot: agent, rag or chatbot
deepeval test run evals/test_agent_quality.py   # one file

python -m evals.synthesize_goldens              # regenerate the goldens test_agent_synthesized scores (paid, needs OPENAI_API_KEY)
```

| File | Marker | Metrics |
|---|---|---|
| `test_agent_tracing.py` | agent, smoke | TaskCompletion + ToolCorrectness on traced goldens |
| `test_agent_quality.py` | agent | AnswerRelevancy + PromptAlignment + StepEfficiency |
| `test_agent_geval.py` | agent | GEval correctness against expected_output |
| `test_agent_untraced.py` | agent | TaskCompletion on a plain test case, no trace (expected to fail) |
| `test_agent_synthesized.py` | agent | Bias + Toxicity + PIILeakage on the saved synthesized goldens (skipped until generated) |
| `test_rag_agent.py` | rag, smoke | ContextualPrecision + ContextualRecall + AnswerRelevancy + Faithfulness |
| `test_chatbot.py` | chatbot, smoke | TurnRelevancy + KnowledgeRetention + ConversationCompleteness |
| `test_chatbot_geval.py` | chatbot | ConversationalGEval with custom criteria |

`pytest evals/` is refused by `evals/conftest.py` before any test runs: the traced tests call `assert_test(golden=...)`, which only works under `deepeval test run`. `pytest --collect-only evals/` is allowed and makes no API calls.

Switch vendor per run without editing `.env`: `LLM_PROVIDER=openai python -m agents.plain` (bash) or `$env:LLM_PROVIDER="openai"; python -m agents.plain` (PowerShell).

## Project structure

| File | Purpose |
|---|---|
| `config.py` | The one place the LLM vendor is chosen. Reads `LLM_PROVIDER` / `JUDGE_PROVIDER`, holds the model IDs in `MODELS`, and builds chat models with `get_chat_model()` (`init_chat_model` under the hood). Stays at the root because both `agents/` and `evals/` import it. |
| `agents/` | The bots under evaluation. The evals import them as `agents.<module>`. |
| `agents/plain.py` | Baseline LangChain agent: the shop tools and `SYSTEM_PROMPT`. No eval code. `build_agent(llm=None, extra_tools=())` is also how `agents/instrumented.py` builds its agent. |
| `agents/instrumented.py` | `plain.build_agent()` plus the tools loaded from `tools/mcp_server.py`, with DeepEval instrumentation. The agent evals import `support_agent` from here. |
| `agents/rag.py` | Agent with one tool, `search_policies`. Retrieved chunks are forwarded to the trace as `retrieval_context`. |
| `agents/chatbot.py` | Multi-turn chatbot that runs its own tool-call loop over a model with `bind_tools()`, returning `(reply, history, tools_called)` per turn. |
| `agents/tools/` | Every tool the bots call. New tools go here, not in the agent modules. |
| `agents/tools/shop.py` | The ShopEasy demo data (`ORDERS`, `REFUND_POLICIES`) and the two `@tool` functions `get_order_status` and `get_refund_policy`, listed in `SHOP_TOOLS`, which the agents and the chatbot are built with. |
| `agents/tools/policy_search.py` | The RAG retriever: 9 policy documents in an `InMemoryVectorStore` with Voyage AI embeddings, the `search_policies` tool, and `retrieved_chunks()`, which pulls the chunks out of a run's `ToolMessage` artifacts. |
| `agents/tools/mcp_server.py` | MCP server exposing `get_shipping_options`, started over stdio by `agents/instrumented.py` by file path. |
| `policies.txt` | Source document for `evals/synthesize_goldens.py`. |
| `pytest.ini` | Points plain `pytest` at `tests/` and registers the `smoke`, `agent`, `rag` and `chatbot` markers (`--strict-markers`, so a typo fails). |
| `tests/conftest.py` | Sets fake API keys and opts out of DeepEval telemetry, before any project module loads `.env`. |
| `tests/fakes.py` | `FakeToolCallingModel` (a `GenericFakeChatModel` with a no-op `bind_tools()`), `fake_model(*replies)` and `tool_call(name, **args)` for scripting a model's replies. |
| `tests/test_*.py` | Offline tests of the tools, the retriever, the agent loops (fake model, real tools and MCP server) and `run_conversation()`. |
| `evals/common.py` | Shared eval helpers: `get_judge()`, `RUN_ASYNC`, `traced()` and the traced runners built with it (`run_support_agent`, `run_rag_support_agent`), `SYNTHESIZED_GOLDENS`, and the chatbot conversation driver `run_conversation()`. |
| `evals/conftest.py` | Refuses plain `pytest` runs, and provides the session-scoped `support_conversation` fixture so the two chatbot evals share one paid conversation. |
| `evals/synthesize_goldens.py` | Generates goldens from `policies.txt` with DeepEval's Synthesizer and saves them to `evals/datasets/synthesized_goldens.json`. |
| `evals/test_*.py` | The evals listed under Running. |

## Architecture

**Dependency injection**: nothing outside `config.py` names a vendor or model. Each bot exposes a factory that takes the model as an optional argument and falls back to the default from `config.py`:
- `build_agent(llm=None)` in `agents/plain.py` (which also takes `extra_tools`), `agents/instrumented.py`, `agents/rag.py`
- `build_model(llm=None)` in `chatbot.py` (binds the shop tools)
- `get_judge(provider=JUDGE_PROVIDER)` in `evals/common.py` (`AnthropicModel` or `OpenAIModel`), `cache`d so every eval shares one judge

Each module builds a default instance on first use, so the entry points (`support_agent`, `rag_support_agent`, `chat`) keep a simple signature and importing a module builds nothing: `default_agent()` in `agents/plain.py`, `agents/instrumented.py` and `agents/rag.py`, and `default_model()` in `agents/chatbot.py`, all `functools.cache`d. `chat()` also takes `model=`, so a test can pass a fake. Two defaults call out when built: `instrumented.default_agent()` starts the MCP server via `load_mcp_tools()`, in a worker thread so it never touches DeepEval's event loop (`with_sync_call()` gives each MCP tool a sync implementation, since `langchain-mcp-adapters` builds them async-only and the agent runs with the sync `invoke()`), and `policy_search.get_vector_store()`, also `cache`d, embeds the documents on the first search. Replies are read with `message.text`, not `.content`, because Claude can return a list of content blocks.

**Agent instrumentation pattern** (`agents/instrumented.py` and `agents/rag.py`):
1. `CallbackHandler()` — captures every LangChain LLM/tool span automatically
2. `config={"callbacks": [deepeval_callback]}` on `.invoke()` — wires it in
3. `update_current_trace(output=reply)` (plus `retrieval_context` in the RAG agent) — sets the clean reply on the trace

The eval side adds the rest with `traced()` from `evals/common.py`: it wraps the agent in `@observe(name=...)` and returns a function that takes the golden, copies its `expected_output` and `expected_tools` onto the trace, and runs the agent on its input. `evals/common.py` builds one such runner per agent, `run_support_agent` and `run_rag_support_agent`, and the evals import those. Each test then calls `assert_test(golden=golden, metrics=[...])`, which scores the trace the test just produced.

**Why the golden is copied into the trace**: DeepEval metrics read `expected_tools`, `expected_output` and `retrieval_context` from the trace object, not directly from the golden.

**Judge vendor**: the judge defaults to the agent's vendor. Set `JUDGE_PROVIDER` to the other vendor to avoid a model grading its own answers. `RUN_ASYNC`, passed to `assert_test()` on plain and conversational test cases, scores metrics one at a time with a Claude judge and concurrently with GPT-4o.

**Every metric gets the judge explicitly** (`model=judge`). Metrics left without a model fall back to DeepEval's default OpenAI model, which breaks a Claude-only setup.

**Claude effort, not temperature**: Claude Sonnet 5.5 always thinks and rejects a non-default `temperature`, so token spend is tuned with `effort`, set to `low` everywhere: the bots via the `get_chat_model()` default (pass `effort=` to raise it) and the judge via `generation_kwargs` in `get_judge()`. The bots and the judge only do short lookups and classification, where `low` mostly skips thinking. GPT-4o has no effort setting and keeps `temperature=0`. DeepEval only knows Sonnet 5.5's prices from 4.3.3 on; older versions run it without cost tracking.

**Confident AI tracing**: adding `CONFIDENT_API_KEY` to `.env` is the only change needed to enable cloud tracing — no code changes required. DeepEval reads it automatically.

**Chatbot evaluation pattern** (`evals/test_chatbot*.py`):
- Uses `ConversationalTestCase` + `Turn` objects instead of `Golden`
- `run_conversation(chat, SUPPORT_CONVERSATION)` in `evals/common.py` drives `chat()` live over a scripted list of user messages and returns `Turn` objects; each assistant `Turn` carries the tools called on that turn as `tools_called`
- The `support_conversation` fixture in `evals/conftest.py` runs that conversation once per session; both chatbot evals score the same run
- `ToolCorrectnessMetric` cannot be used with `ConversationalTestCase`; `test_chatbot_geval.py` judges tool use with `ConversationalGEval` and `MultiTurnParams.TOOLS_CALLED`
- Calls `assert_test(test_case, metrics)` with the `ConversationalTestCase`, since there is no golden or trace to score

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

- **`test_agent_untraced.py` scores low on purpose**: it passes a plain `LLMTestCase` with no trace, so the judge cannot see the tool call and assumes the answer is made up. `test_agent_tracing.py` runs the same check on a trace and passes. Kept as a baseline that shows why trace-level evals matter, and marked `xfail` so it does not fail the run.
- **Known agent bug, marked `xfail`**: in `test_agent_quality.py`, the ORD-2099 return question fails because `get_order_status` does not return the order's category, so the agent cannot look up the refund policy. The `xfail` is non-strict: once the tool is fixed the case reports `XPASS`; remove the mark then. `tests/test_shop_tools.py` checks the same gap for free with a strict `xfail`, which fails the run once the tool is fixed.- **Synthesizer needs OpenAI**: DeepEval has no Anthropic or Voyage embedder, so `evals/synthesize_goldens.py` embeds `policies.txt` with OpenAI even when the judge is Claude. The evals themselves only read the saved goldens.
- **Voyage free tier is 3 requests/minute**: `agents/tools/policy_search.py` embeds its 9 documents on the first search (one request) and each search embeds its query, so repeated RAG runs within a minute hit `RateLimitError` until a payment method is added.
- **Windows console encoding**: when stdout is piped, Python on Windows encodes with cp1252 and crashes on emoji in model replies. Set `PYTHONIOENCODING=utf-8`.
- **`create_react_agent` deprecation**: migrated to `from langchain.agents import create_agent` with `system_prompt=` replacing `prompt=`. Requires the `langchain` base package.
- **Old `_make_hashable` patch**: DeepEval 4.0.4 crashed `ToolCorrectnessMetric` on unhashable `ToolMessage` objects and was patched by hand in the venv. DeepEval 4.3.1 runs `test_agent_tracing.py` without the patch.
